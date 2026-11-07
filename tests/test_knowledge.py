import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from agent_debate.knowledge import (
    KnowledgeBase,
    KnowledgeError,
    chunk_text,
    cosine_scores,
    format_passages,
)


def hashed_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = np.zeros(64, dtype=np.float64)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = hashlib.sha256(token.encode("utf-8")).digest()[0] % 64
            vector[bucket] += 1.0
        vectors.append(vector.tolist())
    return vectors


def test_short_text_is_one_chunk():
    assert chunk_text("A short note.") == ["A short note."]
    assert chunk_text("   \n") == []


def test_long_text_overlaps_by_the_requested_window():
    text = "".join(f"{index:04d}" for index in range(600))
    chunks = chunk_text(text, size=1200, overlap=200)

    assert len(chunks) == 3
    assert chunks[0] == text[0:1200]
    assert chunks[1] == text[1000:2200]
    assert chunks[2] == text[2000:]
    assert chunks[0][-200:] == chunks[1][:200]
    assert all(len(chunk) <= 1200 for chunk in chunks)


def test_chunks_break_on_paragraphs_and_carry_overlap():
    paragraphs = [f"P{index:02d} " + ("x" * 90) for index in range(6)]
    chunks = chunk_text("\n\n".join(paragraphs), size=250, overlap=100)

    assert all(len(chunk) <= 250 for chunk in chunks)
    assert len(chunks) > 1
    assert chunks[1].startswith(chunks[0].split("\n\n")[-1])


def test_paragraphs_larger_than_the_overlap_are_not_merged():
    paragraphs = ["A" * 800, "B" * 800, "C" * 800]
    text = "\n\n".join(paragraphs)
    assert chunk_text(text, size=1200, overlap=200) == paragraphs


def test_search_ranks_the_matching_document(tmp_path: Path):
    (tmp_path / "cats.md").write_text(
        "Cats are mammals that purr and hunt mice.", encoding="utf-8"
    )
    (tmp_path / "france.md").write_text(
        "The capital of France is Paris, a city on the Seine.",
        encoding="utf-8",
    )
    (tmp_path / "notes.pdf").write_text("ignore me", encoding="utf-8")
    (tmp_path / ".secret.md").write_text(
        "The secret capital is hidden.", encoding="utf-8"
    )
    warnings: list[str] = []

    knowledge = KnowledgeBase(
        tmp_path,
        hashed_embed,
        embed_model="test-embed",
        base_url="http://example.test/v1",
        cache_dir=tmp_path / "cache",
        warn=warnings.append,
    )
    hits = knowledge.search("capital of france paris")

    assert hits[0].source == "france.md"
    assert hits[0].id == "france.md#0"
    assert hits[0].score > hits[-1].score
    rendered = format_passages(hits)
    assert "[france.md#0]" in rendered
    assert "Paris" in rendered
    assert warnings == ["Skipping unsupported file: notes.pdf"]
    assert all(hit.source != ".secret.md" for hit in hits)


def test_unchanged_files_are_not_reembedded(tmp_path: Path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "cats.md").write_text("Cats are mammals that purr.", encoding="utf-8")
    (kb_dir / "france.md").write_text(
        "The capital of France is Paris.", encoding="utf-8"
    )
    cache_dir = tmp_path / "cache"
    batches: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        batches.append(list(texts))
        return hashed_embed(texts)

    settings = {
        "embed_model": "test-embed",
        "base_url": "http://example.test/v1",
        "cache_dir": cache_dir,
    }
    KnowledgeBase(kb_dir, embed, **settings).search("paris")
    recorded = len(batches)

    KnowledgeBase(kb_dir, embed, **settings).search("paris")

    assert batches[recorded:] == [["paris"]]

    (kb_dir / "france.md").write_text(
        "A completely different note about oranges.", encoding="utf-8"
    )
    recorded = len(batches)
    KnowledgeBase(kb_dir, embed, **settings).search("oranges")
    reembedded = [text for batch in batches[recorded:] for text in batch]

    assert any("oranges" in text for text in reembedded)
    assert not any("mammals" in text for text in reembedded)


def test_empty_directory_raises(tmp_path: Path):
    with pytest.raises(KnowledgeError, match="empty"):
        KnowledgeBase(
            tmp_path,
            hashed_embed,
            embed_model="test-embed",
            base_url="http://example.test/v1",
            cache_dir=tmp_path / "cache",
        )


def test_cosine_scores_ignore_zero_vectors():
    matrix = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    scores = cosine_scores(np.asarray([1.0, 0.0]), matrix)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
