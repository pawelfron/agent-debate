import json

import pytest

from agent_debate.cli import main, parse_args, render_verdict, write_report
from agent_debate.debate import DebateResult, Turn, Verdict


class EmbedItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class EmbedResponse:
    def __init__(self, data: list[EmbedItem]) -> None:
        self.data = data


class Message:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class ChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": Message(content)})()]


class FakeClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.embed_calls: list[dict] = []
        self.embeddings = self
        self.chat = self
        self.completions = self
        self._arguments: list[str] = [
            "The notes say the tower was first lit in 1882.",
            "The notes do not settle the designer.",
            "The opening already quoted the lighting date.",
            "The designer is not identified in the quoted passage.",
        ]

    def create(self, **kwargs):
        if "input" in kwargs:
            self.embed_calls.append(kwargs)
            texts = kwargs["input"]
            return EmbedResponse(
                [
                    EmbedItem(index, [float(len(text)), 1.0])
                    for index, text in enumerate(texts)
                ]
            )
        self.chat_calls.append(kwargs)
        if len(self.chat_calls) <= 4:
            return ChatResponse(self._arguments[len(self.chat_calls) - 1])
        return ChatResponse(
            json.dumps(
                {
                    "verdict": "uncertain",
                    "confidence": 0.4,
                    "reasoning": "The transcript quotes a date but not enough to confirm the whole claim.",
                }
            )
        )


def test_parse_args_uses_openai_defaults():
    args = parse_args(["verify", "A claim.", "--kb", "examples/kb"])
    assert args.base_url == "https://api.openai.com/v1"
    assert args.chat_model == "gpt-4o-mini"
    assert args.embed_model == "text-embedding-3-small"
    assert args.top_k == 4
    assert args.max_searches == 3
    assert args.output is None


def test_main_requires_an_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["verify", "A claim.", "--kb", str(tmp_path)])
    assert code == 1


def test_main_rejects_a_missing_knowledge_base(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    code = main(["verify", "A claim.", "--kb", str(tmp_path / "missing")])
    assert code == 1
    assert "sk-test-secret" not in capsys.readouterr().err


def test_main_prints_transcript_and_writes_a_report_without_the_key(
    tmp_path, monkeypatch, capsys
):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "note.md").write_text(
        "The current tower was first lit in 1882.", encoding="utf-8"
    )
    report = tmp_path / "report.json"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setattr("agent_debate.cli.OpenAI", lambda **kwargs: FakeClient())

    code = main(
        [
            "verify",
            "The tower was first lit in 1882.",
            "--kb",
            str(kb),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(report),
            "--max-searches",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Proponent — opening" in captured.out
    assert "Opponent — rebuttal" in captured.out
    assert "Verdict: uncertain" in captured.out
    assert "sk-test-secret" not in captured.out
    assert "sk-test-secret" not in captured.err

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdict"]["verdict"] == "uncertain"
    assert [turn["phase"] for turn in payload["transcript"]] == [
        "opening",
        "opening",
        "rebuttal",
        "rebuttal",
    ]
    assert "sk-test-secret" not in report.read_text(encoding="utf-8")
    assert "tools" not in json.dumps(payload)


def test_write_report_shape(tmp_path):
    result = DebateResult(
        claim="A claim.",
        transcript=[Turn("proponent", "opening", "It follows.")],
        verdict=Verdict("uncertain", 0.5, "Both sides asserted the point."),
    )
    path = tmp_path / "report.json"
    write_report(path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["claim"] == "A claim."
    assert payload["verdict"]["confidence"] == 0.5
    rendered = render_verdict(result.verdict)
    assert "Verdict: uncertain" in rendered
    assert "Confidence: 0.50" in rendered


def test_top_k_must_be_positive():
    with pytest.raises(SystemExit):
        parse_args(["verify", "A claim.", "--kb", "kb", "--top-k", "0"])
