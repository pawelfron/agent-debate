"""Command line interface for the debate verifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from agent_debate.debate import (
    DebateError,
    DebateResult,
    Turn,
    Verdict,
    result_to_dict,
    run_debate,
)
from agent_debate.knowledge import (
    KnowledgeBase,
    KnowledgeError,
    default_cache_dir,
    format_passages,
    local_embed,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_CHAT_MODEL = "deepseek-flash"
DEFAULT_EMBED_MODEL = "local-hash"
LOCAL_EMBED_URL = "local"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent-debate",
        description="Verify a claim with a two-agent debate over a local knowledge base.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    verify = subcommands.add_parser(
        "verify", help="Debate a claim and judge the transcript"
    )
    verify.add_argument("claim", help="Claim to verify")
    verify.add_argument(
        "--kb",
        required=True,
        type=Path,
        help="Directory of .txt, .md, and .markdown documents",
    )
    verify.add_argument(
        "--api-key",
        default=None,
        help="API key (default: DEEPSEEK_API_KEY from the environment or a .env file)",
    )
    verify.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    verify.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"Chat model used by all three agents (default: {DEFAULT_CHAT_MODEL})",
    )
    verify.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=(
            "Local embedding id stored in the retrieval cache "
            f"(default: {DEFAULT_EMBED_MODEL}). Retrieval does not call the API."
        ),
    )
    verify.add_argument(
        "--top-k",
        type=_positive_int,
        default=4,
        help="Passages returned per search (default: 4)",
    )
    verify.add_argument(
        "--max-searches",
        type=_non_negative_int,
        default=3,
        help="Searches each debater may run during one turn (default: 3)",
    )
    verify.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the transcript and verdict as JSON",
    )
    verify.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Embedding cache directory (default: .cache/agent-debate in the working directory)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1

    if args.command != "verify":
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    load_dotenv(Path.cwd() / ".env")
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print(
            "Missing API key. Set DEEPSEEK_API_KEY, add it to .env, or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    kb_dir = args.kb.expanduser().resolve()
    if not kb_dir.is_dir():
        print(f"Knowledge base is not a directory: {args.kb}", file=sys.stderr)
        return 1

    try:
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        result = verify(
            claim=args.claim,
            kb_dir=kb_dir,
            client=client,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
            top_k=args.top_k,
            max_searches=args.max_searches,
            cache_dir=args.cache_dir,
            on_turn=_print_turn,
        )
    except (KnowledgeError, DebateError, OpenAIError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(render_verdict(result.verdict), end="")
    if args.output is not None:
        try:
            write_report(args.output, result)
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


def verify(
    *,
    claim: str,
    kb_dir: Path,
    client,
    chat_model: str,
    embed_model: str,
    top_k: int = 4,
    max_searches: int = 3,
    cache_dir: Path | None = None,
    on_turn=None,
) -> DebateResult:
    knowledge = KnowledgeBase(
        kb_dir,
        local_embed,
        embed_model=embed_model,
        base_url=LOCAL_EMBED_URL,
        cache_dir=cache_dir if cache_dir is not None else default_cache_dir(),
        top_k=top_k,
    )

    def search(query: str) -> str:
        return format_passages(knowledge.search(query))

    return run_debate(
        claim=claim,
        client=client,
        chat_model=chat_model,
        search=search,
        max_searches=max_searches,
        on_turn=on_turn,
    )


def write_report(path: Path, result: DebateResult) -> None:
    payload = result_to_dict(result)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def render_turn(turn: Turn) -> str:
    return f"{turn.role.capitalize()} — {turn.phase}\n\n{turn.argument}\n"


def render_verdict(verdict: Verdict) -> str:
    return (
        f"Verdict: {verdict.verdict}\n"
        f"Confidence: {verdict.confidence:.2f}\n"
        f"Reasoning: {verdict.reasoning}\n"
    )


def _print_turn(turn: Turn) -> None:
    print(render_turn(turn))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
