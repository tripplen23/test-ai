"""Tests for CrossEncoderReranker."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from agentic_rag.retrieval.schemas import Query, RetrievedChunk


@pytest.fixture
def sample_chunks():
    """Sample retrieved chunks for testing."""
    return [
        RetrievedChunk(
            chunk_id="chunk1",
            text="WordPress is a content management system.",
            score=0.8,
            metadata={"title": "What is WordPress"},
        ),
        RetrievedChunk(
            chunk_id="chunk2",
            text="To install WordPress, download the files from wordpress.org.",
            score=0.75,
            metadata={"title": "WordPress Installation"},
        ),
        RetrievedChunk(
            chunk_id="chunk3",
            text="WordPress themes control the design of your site.",
            score=0.7,
            metadata={"title": "WordPress Themes"},
        ),
        RetrievedChunk(
            chunk_id="chunk4",
            text="Unrelated content about something else.",
            score=0.65,
            metadata={"title": "Other Topic"},
        ),
    ]


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker functionality."""

    def test_rerank_returns_top_k(self, sample_chunks):
        """Test that rerank returns k results."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        # Mock the cross-encoder model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.85, 0.6, 0.3]

        with patch("agentic_rag.retrieval.reranker.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker()
            query = Query(text="How to install WordPress")
            results = reranker.rerank(query, sample_chunks, k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_rerank_scores_updated(self, sample_chunks):
        """Test that reranker updates scores from cross-encoder."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        # Return scores in different order than original
        mock_model.predict.return_value = [0.5, 0.95, 0.7, 0.3]

        with patch("agentic_rag.retrieval.reranker.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker()
            query = Query(text="How to install WordPress")
            results = reranker.rerank(query, sample_chunks, k=3)

        # Should be sorted by new scores (descending)
        assert results[0].chunk_id == "chunk2"  # highest score 0.95
        assert results[1].chunk_id == "chunk3"  # second highest 0.7
        assert results[2].chunk_id == "chunk1"  # third highest 0.5

    def test_rerank_preserves_metadata(self, sample_chunks):
        """Test that metadata is preserved during reranking."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.85, 0.6, 0.3]

        with patch("agentic_rag.retrieval.reranker.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker()
            query = Query(text="test query")
            results = reranker.rerank(query, sample_chunks, k=2)

        assert all(r.metadata is not None for r in results)
        assert results[0].metadata["title"] == "What is WordPress"

    def test_rerank_with_empty_candidates(self):
        """Test reranking with empty candidate list."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        with patch("agentic_rag.retrieval.reranker.CrossEncoder"):
            reranker = CrossEncoderReranker()
            query = Query(text="test query")
            results = reranker.rerank(query, [], k=5)

        assert len(results) == 0
        assert isinstance(results, list)

    def test_rerank_k_larger_than_candidates(self, sample_chunks):
        """Test when k is larger than number of candidates."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.85, 0.6, 0.3]

        with patch("agentic_rag.retrieval.reranker.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker()
            query = Query(text="test query")
            results = reranker.rerank(query, sample_chunks, k=10)

        # Should return all candidates
        assert len(results) == 4

    def test_rerank_calls_cross_encoder(self, sample_chunks):
        """Test that cross-encoder is called with correct inputs."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.85, 0.6, 0.3]

        with patch("agentic_rag.retrieval.reranker.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker()
            query = Query(text="How to install WordPress")
            reranker.rerank(query, sample_chunks, k=2)

        # Should call predict with query-document pairs
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 4  # 4 chunks
        assert all(isinstance(pair, tuple) for pair in call_args)
        assert all(pair[0] == "How to install WordPress" for pair in call_args)

    def test_reranker_initialization_with_custom_model(self):
        """Test initializing reranker with custom model."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        with patch("agentic_rag.retrieval.reranker.CrossEncoder") as mock_ce:
            reranker = CrossEncoderReranker(model_name="custom-model")
            mock_ce.assert_called_once_with("custom-model")

    def test_rerank_single_candidate(self):
        """Test reranking with single candidate."""
        from agentic_rag.retrieval.reranker import CrossEncoderReranker

        single_chunk = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Test content",
                score=0.8,
                metadata={},
            )
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95]

        with patch("agentic_rag.retrieval.reranker.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker()
            query = Query(text="test")
            results = reranker.rerank(query, single_chunk, k=5)

        assert len(results) == 1
        assert results[0].score == 0.95
