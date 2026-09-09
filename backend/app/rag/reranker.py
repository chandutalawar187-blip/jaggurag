"""Cross-encoder reranker — re-scores query-document pairs."""

from __future__ import annotations

from typing import List

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import RetrievedChunk

log = get_logger(__name__)

_reranker_model = None


def _load_reranker():
    global _reranker_model
    if _reranker_model is None:
        log.info("Loading cross-encoder reranker...")
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        log.info("✓ Reranker loaded")
    return _reranker_model


def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    top_k: int | None = None,
) -> List[RetrievedChunk]:
    """Re-rank chunks using cross-encoder. Returns top_k results."""
    if not settings.RERANKER_ENABLED:
        log.info("Reranker disabled, returning original order")
        return chunks[:top_k or settings.FINAL_TOP_K]

    if not chunks:
        return []

    top_k = top_k or settings.FINAL_TOP_K

    try:
        model = _load_reranker()

        # Create query-document pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)

        # Update scores
        for i, chunk in enumerate(chunks):
            chunk.score = float(scores[i])

        # Sort by reranker score (descending)
        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)

        # Calibrate each score independently. Min-max normalization is unsafe
        # here because it makes every top result look like a perfect match.
        import math
        for c in reranked:
            c.score = round(1.0 / (1.0 + math.exp(-float(c.score))), 4)

        result = reranked[:top_k]
        log.info("Reranked %d → %d chunks", len(chunks), len(result))
        return result

    except Exception as e:
        log.error("Reranker failed: %s — falling back to original order", e)
        return chunks[:top_k]


def is_reranker_available() -> bool:
    try:
        _load_reranker()
        return True
    except Exception:
        return False
