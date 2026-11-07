# Agent debate

Verify a claim by debating it against a local knowledge base. A proponent argues that the claim is true and an opponent argues that it is false. Both can search the documents. A judge, who sees only the published transcript, decides whether the claim is true, false, or still uncertain.

Openings are independent. Each side then makes one rebuttal, and the opponent speaks last. Retrieval stays inside the debater's turn: the judge never receives the knowledge base, the search tool, or the raw passages.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

Set `OPENAI_API_KEY`, or pass `--api-key`. The default endpoint is the public OpenAI API. Point `--base-url`, `--chat-model`, and `--embed-model` at any OpenAI-compatible server.

```bash
agent-debate verify "The current Eddystone Lighthouse was first lit in 1882." \
  --kb ./examples/kb
```

Another endpoint:

```bash
agent-debate verify "The current Eddystone Lighthouse was first lit in 1882." \
  --kb ./examples/kb \
  --base-url "$OPENAI_BASE_URL" \
  --chat-model "$OPENAI_CHAT_MODEL" \
  --embed-model "$OPENAI_EMBED_MODEL"
```

The knowledge base is a directory of `.txt`, `.md`, and `.markdown` files, read recursively. Other files are skipped. Embeddings are cached in `.cache/agent-debate/` in the working directory.

Each turn prints as it finishes. The verdict is `true`, `false`, or `uncertain`.

```bash
agent-debate verify "Smeaton designed the lighthouse that was first lit in 1882." \
  --kb ./examples/kb \
  --output report.json
```

`--output` writes the claim, the transcript, and the verdict as JSON. The API key is never printed or written into that file.

| Flag | Default |
| --- | --- |
| `--base-url` | `https://api.openai.com/v1` |
| `--chat-model` | `gpt-4o-mini` |
| `--embed-model` | `text-embedding-3-small` |
| `--top-k` | `4` passages per search |
| `--max-searches` | `3` searches per turn |
| `--cache-dir` | `.cache/agent-debate` |

The sample documents in `examples/kb` describe the Eddystone Lighthouse, Plymouth Hoe, and an unrelated library note.

## Tests

```bash
pytest
```

Tests use fake clients. They do not call a live API.
