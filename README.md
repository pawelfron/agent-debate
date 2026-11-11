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

Put `DEEPSEEK_API_KEY` in a `.env` file in the working directory, export it, or pass `--api-key`. A value already set in the environment is left as-is. The default endpoint is the DeepSeek API, which speaks the OpenAI chat format. Point `--base-url` and `--chat-model` at any compatible server.

```bash
# .env
DEEPSEEK_API_KEY=sk-...
```

```bash
agent-debate verify "The current Eddystone Lighthouse was first lit in 1882." \
  --kb ./examples/kb
```

Another endpoint:

```bash
agent-debate verify "The current Eddystone Lighthouse was first lit in 1882." \
  --kb ./examples/kb \
  --base-url "$DEEPSEEK_BASE_URL" \
  --chat-model "$DEEPSEEK_CHAT_MODEL"
```

The knowledge base is a directory of `.txt`, `.md`, and `.markdown` files, read recursively. Other files are skipped. DeepSeek does not provide an embeddings API, so retrieval uses a local token hash. Those vectors are cached in `.cache/agent-debate/` in the working directory. `--embed-model` only names that cache.

Each turn prints as it finishes. The verdict is `true`, `false`, or `uncertain`.

```bash
agent-debate verify "Smeaton designed the lighthouse that was first lit in 1882." \
  --kb ./examples/kb \
  --output report.json
```

`--output` writes the claim, the transcript, and the verdict as JSON. The API key is never printed or written into that file.

| Flag | Default |
| --- | --- |
| `--base-url` | `https://api.deepseek.com` |
| `--chat-model` | `deepseek-flash` |
| `--embed-model` | `local-hash` |
| `--top-k` | `4` passages per search |
| `--max-searches` | `3` searches per turn |
| `--cache-dir` | `.cache/agent-debate` |

The sample documents in `examples/kb` describe the Eddystone Lighthouse, Plymouth Hoe, and an unrelated library note.

## Tests

```bash
pytest
```

Tests use fake clients. They do not call a live API.
