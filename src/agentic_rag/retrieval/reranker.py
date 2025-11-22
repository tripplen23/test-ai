"""Reranking implementation using cross-encoder models."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from langsmith import traceable
from sentence_transformers import CrossEncoder

from agentic_rag.settings import get_settings

from .base import BaseReranker
from .schemas import Query, RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers cross-encoder."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize CrossEncoderReranker.

        Args:
            model_name: Optional cross-encoder model name.
                      Defaults to ms-marco-MiniLM-L-6-v2
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.reranker_model

        logger.info(f"Loading cross-encoder model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("Cross-encoder model loaded successfully")

    @traceable(name="crossencoder_rerank")
    def rerank(
        self,
        query: Query,
        candidates: Iterable[RetrievedChunk],
        *,
        k: int = 5
    ) -> Sequence[RetrievedChunk]:
        """
        Rerank retrieved candidates using cross-encoder.

        Args:
            query: Query object
            candidates: Retrieved chunks to rerank
            k: Number of results to return after reranking

        Returns:
            Top-k reranked chunks sorted by relevance score (descending)
        """
        # Convert to list if needed
        candidates_list = list(candidates)

        if not candidates_list:
            logger.debug("No candidates to rerank")
            return []

        logger.debug(f"Reranking {len(candidates_list)} candidates for query: {query.text[:50]}...")

        # Prepare query-document pairs for cross-encoder
        pairs = [(query.text, chunk.text) for chunk in candidates_list]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Update chunks with new scores
        reranked_chunks = []
        for chunk, score in zip(candidates_list, scores):
            # Create new chunk with updated score
            reranked_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),  # Convert numpy float to Python float
                metadata=chunk.metadata,
            )
            reranked_chunks.append(reranked_chunk)

        # Sort by score (descending) and take top-k
        reranked_chunks.sort(key=lambda x: x.score, reverse=True)
        results = reranked_chunks[:k]

        logger.info(
            f"Reranked {len(candidates_list)} -> {len(results)} chunks, "
            f"top score: {results[0].score:.3f}" if results else "no results"
        )

        return results
