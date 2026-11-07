"""Load a local knowledge base, embed it, and search it by cosine similarity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}
EMBED_BATCH_SIZE = 64

EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]
WarnFn = Callable[[str], None]


class KnowledgeError(Exception):
    """The knowledge base could not be loaded or searched."""


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    index: int
    text: str
    score: float


@dataclass
class _Chunk:
    id: str
    source: str
    index: int
    text: str
    content_hash: str
    vector: list[float] | None = None


def default_cache_dir() -> Path:
    return Path.cwd() / ".cache" / "agent-debate"


def default_warn(message: str) -> None:
    print(message, file=sys.stderr)


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into chunks of about `size` characters with `overlap`.

    Paragraph boundaries are kept when a paragraph fits in a chunk. A paragraph
    longer than `size` is split into overlapping windows.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than the chunk size")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]

    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()
    ]
    blocks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            blocks.append(paragraph)
        else:
            blocks.extend(_window(paragraph, size, overlap))

    chunks: list[str] = []
    current: list[str] = []

    for block in blocks:
        if current and _packed_len(current) + 2 + len(block) > size:
            chunks.append("\n\n".join(current))
            current = _overlap_carry(current, overlap, len(block), size)
        current.append(block)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _window(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return chunks


def _packed_len(parts: list[str]) -> int:
    if not parts:
        return 0
    return sum(len(part) for part in parts) + 2 * (len(parts) - 1)


def _overlap_carry(
    parts: list[str],
    overlap: int,
    next_len: int,
    size: int,
) -> list[str]:
    carry: list[str] = []
    for part in reversed(parts):
        trial = [part, *carry]
        if _packed_len(trial) > overlap:
            break
        carry = trial
    if carry and _packed_len(carry) + 2 + next_len > size:
        return []
    return carry


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    scores = np.zeros(matrix.shape[0], dtype=np.float64)
    np.divide(matrix @ query, denom, out=scores, where=denom > 0)
    return scores


def format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "No passages found."
    blocks = [
        f"[{passage.id}] source: {passage.source}\n{passage.text}"
        for passage in passages
    ]
    return "\n\n".join(blocks)


class KnowledgeBase:
    """Chunked documents and their embedding matrix, loaded from a directory."""

    def __init__(
        self,
        kb_dir: Path,
        embed: EmbedFn,
        *,
        embed_model: str,
        base_url: str,
        cache_dir: Path | None = None,
        top_k: int = 4,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        warn: WarnFn = default_warn,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.kb_dir = kb_dir
        self.embed = embed
        self.embed_model = embed_model
        self.base_url = base_url
        self.cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.warn = warn
        self._chunks = self._index()
        self._matrix = np.asarray(
            [chunk.vector for chunk in self._chunks], dtype=np.float64
        )

    def search(self, query: str, top_k: int | None = None) -> list[Passage]:
        query = query.strip()
        if not query:
            return []
        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be at least 1")
        vectors = _embed_batched(self.embed, [query])
        scores = cosine_scores(np.asarray(vectors[0], dtype=np.float64), self._matrix)
        limit = min(k, len(self._chunks))
        order = np.argsort(-scores)[:limit]
        return [
            Passage(
                id=self._chunks[int(index)].id,
                source=self._chunks[int(index)].source,
                index=self._chunks[int(index)].index,
                text=self._chunks[int(index)].text,
                score=float(scores[int(index)]),
            )
            for index in order
        ]

    def _index(self) -> list[_Chunk]:
        sources = _load_sources(self.kb_dir, self.warn)
        cached = _read_cache(self._cache_path(), self._cache_meta())
        chunks: list[_Chunk] = []
        missing: list[_Chunk] = []

        for source, content_hash, text in sources:
            pieces = chunk_text(text, self.chunk_size, self.overlap)
            reused = _reusable_vectors(cached.get(source), content_hash, pieces)
            for index, piece in enumerate(pieces):
                chunk = _Chunk(
                    id=f"{source}#{index}",
                    source=source,
                    index=index,
                    text=piece,
                    content_hash=content_hash,
                    vector=None if reused is None else reused[index],
                )
                chunks.append(chunk)
                if chunk.vector is None:
                    missing.append(chunk)

        if not chunks:
            raise KnowledgeError(
                f"Knowledge base is empty: no .txt, .md, or .markdown files in {self.kb_dir}"
            )

        if missing:
            print(f"Embedding {len(missing)} passages...", file=sys.stderr)
            vectors = _embed_batched(self.embed, [chunk.text for chunk in missing])
            width = len(vectors[0])
            for chunk, vector in zip(missing, vectors, strict=True):
                if len(vector) != width:
                    raise KnowledgeError(
                        "Embedding model returned vectors of different lengths"
                    )
                chunk.vector = vector
            for chunk in chunks:
                if chunk.vector is not None and len(chunk.vector) != width:
                    raise KnowledgeError(
                        "Cached embeddings do not match the current embedding model"
                    )

        self._write_cache(chunks)
        return chunks

    def _cache_meta(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "embed_model": self.embed_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }

    def _cache_path(self) -> Path:
        raw = f"{self.base_url}\n{self.embed_model}\n{self.chunk_size}\n{self.overlap}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _write_cache(self, chunks: list[_Chunk]) -> None:
        files: dict[str, dict[str, object]] = {}
        for chunk in chunks:
            entry = files.setdefault(
                chunk.source,
                {"content_hash": chunk.content_hash, "chunks": []},
            )
            chunk_list = entry["chunks"]
            assert isinstance(chunk_list, list)
            chunk_list.append(
                {"index": chunk.index, "text": chunk.text, "vector": chunk.vector}
            )

        payload = {**self._cache_meta(), "files": files}
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)


def _load_sources(kb_dir: Path, warn: WarnFn) -> list[tuple[str, str, str]]:
    if not kb_dir.is_dir():
        raise KnowledgeError(f"Knowledge base is not a directory: {kb_dir}")

    sources: list[tuple[str, str, str]] = []
    for path in _iter_files(kb_dir):
        relative = path.relative_to(kb_dir).as_posix()
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            warn(f"Skipping unsupported file: {relative}")
            continue
        try:
            data = path.read_bytes()
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            warn(f"Skipping {relative}: not valid UTF-8")
            continue
        if not text.strip():
            continue
        content_hash = hashlib.sha256(data).hexdigest()
        sources.append((relative, content_hash, text))
    return sources


def _iter_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            found.append(Path(dirpath) / name)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def _read_cache(
    path: Path, expected_meta: dict[str, object]
) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if any(data.get(key) != value for key, value in expected_meta.items()):
        return {}
    files = data.get("files")
    if not isinstance(files, dict):
        return {}
    return {key: value for key, value in files.items() if isinstance(value, dict)}


def _reusable_vectors(
    cached_file: dict[str, object] | None,
    content_hash: str,
    pieces: list[str],
) -> list[list[float]] | None:
    if not cached_file or cached_file.get("content_hash") != content_hash:
        return None
    cached_chunks = cached_file.get("chunks")
    if not isinstance(cached_chunks, list) or len(cached_chunks) != len(pieces):
        return None
    vectors: list[list[float]] = []
    for piece, cached in zip(pieces, cached_chunks, strict=True):
        if not isinstance(cached, dict) or cached.get("text") != piece:
            return None
        vector = cached.get("vector")
        if not isinstance(vector, list) or not vector:
            return None
        try:
            vectors.append([float(item) for item in vector])
        except (TypeError, ValueError):
            return None
    return vectors


def _embed_batched(
    embed: EmbedFn, texts: list[str], batch_size: int = EMBED_BATCH_SIZE
) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        received = list(embed(batch))
        if len(received) != len(batch):
            raise KnowledgeError(
                f"Embedding model returned {len(received)} vectors for {len(batch)} inputs"
            )
        for vector in received:
            values = [float(item) for item in vector]
            if not values:
                raise KnowledgeError("Embedding model returned an empty vector")
            vectors.append(values)
    return vectors
