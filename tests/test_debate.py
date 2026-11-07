import json

import pytest

from agent_debate.debate import DebateError, parse_verdict, run_debate


class FunctionCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = FunctionCall(name, arguments)


class Message:
    def __init__(self, content: str | None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class Response:
    def __init__(self, message: Message) -> None:
        self.choices = [type("Choice", (), {"message": message})()]


def chat(content: str, tool_calls: list | None = None) -> Response:
    return Response(Message(content, tool_calls))


class ScriptedClient:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("scripted client ran out of responses")
        return self.responses.pop(0)


def system_text(call: dict) -> str:
    return call["messages"][0]["content"]


def first_user(call: dict) -> str:
    for message in call["messages"]:
        if message["role"] == "user":
            return message["content"]
    return ""


def dumped(call: dict) -> str:
    return json.dumps(call)


VERDICT = {
    "verdict": "true",
    "confidence": 0.75,
    "reasoning": "The quoted passage supports the claim.",
}


def test_debate_order_hides_retrieval_from_the_judge():
    searches: list[str] = []

    def search(query: str) -> str:
        searches.append(query)
        return "SEARCH_MARKER secret passage"

    client = ScriptedClient(
        [
            chat(
                None,
                [
                    ToolCall(
                        "call-1",
                        "search_knowledge_base",
                        json.dumps({"query": "lighthouse date"}),
                    )
                ],
            ),
            chat(
                "PROP_OPEN",
                [
                    ToolCall(
                        "call-2",
                        "search_knowledge_base",
                        json.dumps({"query": "should not run"}),
                    )
                ],
            ),
            chat("OPP_OPEN"),
            chat("PROP_REB"),
            chat("OPP_REB"),
            chat("not json"),
            chat(json.dumps(VERDICT)),
        ]
    )

    result = run_debate(
        claim="The tower was first lit in 1882.",
        client=client,
        chat_model="test-chat",
        search=search,
        max_searches=1,
    )

    assert searches == ["lighthouse date"]
    assert [(turn.role, turn.phase, turn.argument) for turn in result.transcript] == [
        ("proponent", "opening", "PROP_OPEN"),
        ("opponent", "opening", "OPP_OPEN"),
        ("proponent", "rebuttal", "PROP_REB"),
        ("opponent", "rebuttal", "OPP_REB"),
    ]
    assert result.verdict.verdict == "true"
    assert result.verdict.confidence == 0.75
    assert "SEARCH_MARKER" not in " ".join(turn.argument for turn in result.transcript)

    opponent_opening = next(
        call
        for call in client.calls
        if "You are the opponent" in system_text(call)
        and "opening argument" in first_user(call)
    )
    assert "PROP_OPEN" not in dumped(opponent_opening)
    assert "SEARCH_MARKER" not in dumped(opponent_opening)

    proponent_rebuttal = next(
        call
        for call in client.calls
        if "You are the proponent" in system_text(call)
        and "Write your rebuttal" in first_user(call)
    )
    rebuttal_user = first_user(proponent_rebuttal)
    assert "PROP_OPEN" in rebuttal_user
    assert "OPP_OPEN" in rebuttal_user
    assert "PROP_REB" not in rebuttal_user

    opponent_rebuttal = next(
        call
        for call in client.calls
        if "You are the opponent" in system_text(call)
        and "You speak last" in first_user(call)
    )
    assert "PROP_REB" in first_user(opponent_rebuttal)

    judge_calls = [
        call
        for call in client.calls
        if system_text(call).startswith("You are the judge")
    ]
    assert len(judge_calls) == 2
    for call in judge_calls:
        assert "tools" not in call
        assert "SEARCH_MARKER" not in dumped(call)
        assert "PROP_OPEN" in dumped(call)
        assert "OPP_REB" in dumped(call)
    assert "tools" in client.calls[0]
    assert "tools" not in client.calls[1]


def test_judge_failure_after_retry_raises():
    client = ScriptedClient(
        [
            chat("PROP_OPEN"),
            chat("OPP_OPEN"),
            chat("PROP_REB"),
            chat("OPP_REB"),
            chat("nope"),
            chat("still nope"),
        ]
    )
    with pytest.raises(DebateError, match="invalid verdict"):
        run_debate(
            claim="A claim.",
            client=client,
            chat_model="test-chat",
            search=lambda query: "unused",
            max_searches=0,
        )


def test_parse_verdict_accepts_a_fence_and_rejects_bad_confidence():
    parsed = parse_verdict(
        '```json\n{"verdict": "FALSE", "confidence": "0.25", "reasoning": "The opponent quoted the record."}\n```'
    )
    assert parsed.verdict == "false"
    assert parsed.confidence == 0.25

    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_verdict(
            '{"verdict": "uncertain", "confidence": 1.2, "reasoning": "Too sure."}'
        )
