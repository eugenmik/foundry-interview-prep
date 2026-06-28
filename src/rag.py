"""Retrieval-augmented generation helpers and a tiny persistent vector store.

Two features live here:

* ``retrieve_context`` – chunk the resume, embed the chunks, and return the
  pieces most relevant to a query (e.g. the target role). This focuses the
  question generator on the parts of the resume that matter (Medium: RAG).

* ``SeenStore`` – a minimal on-disk vector store of previously generated
  interview-prep items. Before showing new questions we can check whether
  very similar ones were produced before and ask the model to diversify
  (Hard: vector DB to detect already-seen prep data).

Embeddings come from OpenRouter; similarity is plain cosine via numpy so we
avoid a heavy vector-DB dependency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import openrouter_client as orc

STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"
STORE_PATH = STORE_DIR / "seen.json"
SIM_THRESHOLD = 0.86  # cosine similarity above which two items are "the same"


def chunk_text(text: str, chunk_size: int = 90, overlap: int = 20) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve_context(resume_text: str, query: str, top_k: int = 4) -> str | None:
    """Return the resume chunks most relevant to ``query`` as a text block.

    Returns None if embedding is unavailable or the resume is short enough
    that retrieval adds nothing.
    """
    chunks = chunk_text(resume_text)
    if len(chunks) <= top_k:
        return None  # nothing to gain from retrieval
    try:
        vectors = orc.embed(chunks + [query])
    except orc.OpenRouterError:
        return None
    if len(vectors) != len(chunks) + 1:
        return None
    chunk_vecs = [np.array(v) for v in vectors[:-1]]
    query_vec = np.array(vectors[-1])
    scored = sorted(
        zip(chunks, chunk_vecs),
        key=lambda cv: _cosine(cv[1], query_vec),
        reverse=True,
    )
    top = [c for c, _ in scored[:top_k]]
    return "\n---\n".join(top)


class SeenStore:
    """A tiny persistent vector store to detect already-seen prep items."""

    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = path
        self.items: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.items = []

    def _save(self) -> None:
        os.makedirs(self.path.parent, exist_ok=True)
        self.path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")

    def check_and_add(self, texts: list[str]) -> list[str]:
        """Embed ``texts``, return those that are NEW (not seen before), and store them.

        On any embedding failure it fails open: treats everything as new and
        skips persistence so the app keeps working offline.
        """
        if not texts:
            return []
        try:
            new_vecs = orc.embed(texts)
        except orc.OpenRouterError:
            return texts
        existing = [np.array(it["vec"]) for it in self.items]
        fresh: list[str] = []
        for text, vec in zip(texts, new_vecs):
            v = np.array(vec)
            seen = any(_cosine(v, e) >= SIM_THRESHOLD for e in existing)
            if not seen:
                fresh.append(text)
                self.items.append({"text": text, "vec": vec})
                existing.append(v)
        try:
            self._save()
        except OSError:
            pass
        return fresh
