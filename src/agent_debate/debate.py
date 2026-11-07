"""Run a two-round debate and ask a judge who sees only the transcript."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for passages relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for in the knowledge base.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

JUDGE_SYSTEM = """You are the judge of a debate about whether a claim is true.
You did not see the knowledge base and you cannot search it. Use only the transcript.
Passage ids cited by the debaters are claims about documents you cannot check. Treat a citation as unsupported when the argument does not quote or explain what that passage says.
If the evidence is missing, contradictory, or only asserted, return uncertain.
Reply with JSON only, with no markdown fence, using exactly this shape:
{"verdict": "true" | "false" | "uncertain", "confidence": 0.0, "reasoning": "short explanation"}
confidence is a number from 0 to 1."""

JUDGE_RETRY = (
    "That reply was not valid. Reply with JSON only, using this shape: "
    '{"verdict": "true" | "false" | "uncertain", "confidence": 0.0, "reasoning": "short explanation"}'
)

TURN_LABELS = {
    ("proponent", "opening"): "Proponent opening",
    ("opponent", "opening"): "Opponent opening",
    ("proponent", "rebuttal"): "Proponent rebuttal",
    ("opponent", "rebuttal"): "Opponent rebuttal",
}

SearchFn = Callable[[str], str]


class DebateError(Exception):
    """A debate turn or the judge verdict could not be completed."""


@dataclass(frozen=True)
class Turn:
    role: str
    phase: str
    argument: str


@dataclass(frozen=True)
class Verdict:
    verdict: str
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class DebateResult:
    claim: str
    transcript: list[Turn]
    verdict: Verdict


def run_debate(
    *,
    claim: str,
    client,
    chat_model: str,
    search: SearchFn,
    max_searches: int = 3,
    on_turn: Callable[[Turn], None] | None = None,
) -> DebateResult:
    """Proponent and opponent open independently, rebut once each, then the judge decides."""
    if max_searches < 0:
        raise ValueError("max_searches must be at least 0")

    claim = claim.strip()
    if not claim:
        raise DebateError("Claim is empty")

    transcript: list[Turn] = []

    def publish(role: str, phase: str, argument: str) -> None:
        if not argument.strip():
            label = TURN_LABELS[(role, phase)]
            raise DebateError(f"{label} was empty")
        turn = Turn(role=role, phase=phase, argument=argument.strip())
        transcript.append(turn)
        if on_turn is not None:
            on_turn(turn)

    publish(
        "proponent",
        "opening",
        _run_turn(
            client,
            chat_model,
            system=_debater_system("proponent", max_searches),
            user=_opening_user(claim),
            search=search,
            max_searches=max_searches,
        ),
    )
    publish(
        "opponent",
        "opening",
        _run_turn(
            client,
            chat_model,
            system=_debater_system("opponent", max_searches),
            user=_opening_user(claim),
            search=search,
            max_searches=max_searches,
        ),
    )
    publish(
        "proponent",
        "rebuttal",
        _run_turn(
            client,
            chat_model,
            system=_debater_system("proponent", max_searches),
            user=_rebuttal_user(claim, transcript, "proponent"),
            search=search,
            max_searches=max_searches,
        ),
    )
    publish(
        "opponent",
        "rebuttal",
        _run_turn(
            client,
            chat_model,
            system=_debater_system("opponent", max_searches),
            user=_rebuttal_user(claim, transcript, "opponent"),
            search=search,
            max_searches=max_searches,
        ),
    )
    verdict = _judge(client, chat_model, claim, transcript)
    return DebateResult(claim=claim, transcript=transcript, verdict=verdict)


def result_to_dict(result: DebateResult) -> dict:
    return {
        "claim": result.claim,
        "transcript": [
            {"role": turn.role, "phase": turn.phase, "argument": turn.argument}
            for turn in result.transcript
        ],
        "verdict": {
            "verdict": result.verdict.verdict,
            "confidence": result.verdict.confidence,
            "reasoning": result.verdict.reasoning,
        },
    }


def parse_verdict(text: str) -> Verdict:
    data = _load_json_object(text)
    try:
        verdict = str(data["verdict"]).strip().lower()
        confidence = float(data["confidence"])
        reasoning = str(data["reasoning"]).strip()
    except KeyError as exc:
        raise ValueError(f"missing field {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number from 0 to 1") from exc
    if verdict not in {"true", "false", "uncertain"}:
        raise ValueError("verdict must be true, false, or uncertain")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not reasoning:
        raise ValueError("reasoning is required")
    return Verdict(verdict=verdict, confidence=confidence, reasoning=reasoning)


def _debater_system(role: str, max_searches: int) -> str:
    stance = "TRUE" if role == "proponent" else "FALSE"
    if max_searches > 0:
        search_rules = (
            "Use the search_knowledge_base tool to find evidence before you argue. "
            f"You may search at most {max_searches} times. "
            "Cite passage ids exactly as returned, for example [notes.md#0]. "
            "Do not invent sources, passage ids, or quotations."
        )
    else:
        search_rules = (
            "You cannot search the knowledge base on this turn. "
            "Argue only from the claim and the transcript you were given."
        )
    return (
        f"You are the {role} in a debate about whether a claim is true.\n"
        f"Argue that the claim is {stance}.\n"
        f"{search_rules}\n"
        "Prefer evidence over assertion. Concede points the evidence does not support.\n"
        "Write in the same language as the claim.\n"
        "When you are finished searching, reply with the argument only and do not call a tool."
    )


def _opening_user(claim: str) -> str:
    return (
        f"Claim:\n{claim}\n\n"
        "Write your opening argument. The other side has not spoken yet."
    )


def _rebuttal_user(claim: str, transcript: list[Turn], role: str) -> str:
    lines = [f"Claim:\n{claim}", "", "Transcript so far:"]
    for turn in transcript:
        lines.append("")
        lines.append(f"{TURN_LABELS[(turn.role, turn.phase)]}:")
        lines.append(turn.argument)
    if role == "proponent":
        instruction = "Write your rebuttal. Respond to the opponent's opening."
    else:
        instruction = (
            "Write your rebuttal. You speak last. "
            "Respond to the proponent's rebuttal and defend your position."
        )
    lines.append("")
    lines.append(instruction)
    return "\n".join(lines)


def _run_turn(
    client, chat_model: str, system: str, user: str, search: SearchFn, max_searches: int
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    searches_used = 0
    nudges = 0
    # One step per search, plus a final answer, plus one nudge if the answer is empty.
    for _ in range(max_searches + 2):
        kwargs: dict = {"model": chat_model, "messages": messages}
        if searches_used < max_searches:
            kwargs["tools"] = [SEARCH_TOOL]
            kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])
        if tool_calls and searches_used < max_searches:
            messages.append(_assistant_dict(message))
            for call in tool_calls:
                if searches_used >= max_searches:
                    result = "Search limit reached. Write your argument from the passages you already have."
                else:
                    searches_used += 1
                    result = _execute_search(call, search)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )
            continue

        text = (message.content or "").strip()
        if text:
            return text
        if nudges >= 1:
            return ""
        nudges += 1
        messages.append(_assistant_dict(message))
        messages.append(
            {
                "role": "user",
                "content": "Write your argument now. Do not call tools.",
            }
        )
    return ""


def _execute_search(call, search: SearchFn) -> str:
    name = getattr(getattr(call, "function", None), "name", "")
    if name != "search_knowledge_base":
        return "Unknown tool. Use search_knowledge_base."
    query = _search_query(call)
    if not query:
        return "Search failed: provide a non-empty query string."
    return search(query)


def _search_query(call) -> str:
    raw = getattr(getattr(call, "function", None), "arguments", "")
    try:
        data = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    query = data.get("query", "")
    if not isinstance(query, str):
        return ""
    return query.strip()


def _assistant_dict(message) -> dict:
    payload: dict = {"role": "assistant", "content": message.content}
    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments
                    if isinstance(call.function.arguments, str)
                    else json.dumps(call.function.arguments),
                },
            }
        )
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _judge(client, chat_model: str, claim: str, transcript: list[Turn]) -> Verdict:
    messages: list[dict] = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": _judge_user(claim, transcript)},
    ]
    last_error = "the reply was not valid JSON"
    for attempt in range(2):
        response = client.chat.completions.create(model=chat_model, messages=messages)
        text = (response.choices[0].message.content or "").strip()
        try:
            return parse_verdict(text)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == 1:
                break
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": JUDGE_RETRY})
    raise DebateError(f"Judge returned an invalid verdict: {last_error}")


def _judge_user(claim: str, transcript: list[Turn]) -> str:
    parts = [f"Claim:\n{claim}"]
    for turn in transcript:
        parts.append(f"{TURN_LABELS[(turn.role, turn.phase)]}:\n{turn.argument}")
    return "\n\n".join(parts)


def _load_json_object(text: str) -> dict:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(candidate[start : end + 1])
    if not isinstance(data, dict):
        raise TypeError("verdict must be a JSON object")
    return data
