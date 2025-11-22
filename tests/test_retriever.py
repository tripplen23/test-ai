"""Tests for PGVectorRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from agentic_rag.retrieval.schemas import Query, RetrievedChunk


@pytest.fixture
def mock_vector_store():
    """Mock PGVector store."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 1536  # Mock embedding vector
    return mock


class TestPGVectorRetriever:
    """Test PGVectorRetriever functionality."""

    def test_search_returns_retrieved_chunks(self, mock_vector_store, mock_embeddings):
        """Test that search returns RetrievedChunk objects."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        # Mock search results
        mock_doc1 = Mock()
        mock_doc1.page_content = "Test content 1"
        mock_doc1.metadata = {"chunk_id": "doc1_chunk_0", "score": 0.9}

        mock_doc2 = Mock()
        mock_doc2.page_content = "Test content 2"
        mock_doc2.metadata = {"chunk_id": "doc2_chunk_0", "score": 0.8}

        mock_vector_store.similarity_search_with_relevance_scores.return_value = [
            (mock_doc1, 0.9),
            (mock_doc2, 0.8),
        ]

        with patch("agentic_rag.retrieval.retriever.PGVector", return_value=mock_vector_store):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings", return_value=mock_embeddings):
                retriever = PGVectorRetriever()
                query = Query(text="test query")
                results = retriever.search(query, k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].text == "Test content 1"
        assert results[0].score == 0.9
        assert results[1].text == "Test content 2"
        assert results[1].score == 0.8

    def test_search_respects_k_parameter(self, mock_vector_store, mock_embeddings):
        """Test that k parameter limits results."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        # Mock more results than k
        mock_docs = []
        for i in range(10):
            mock_doc = Mock()
            mock_doc.page_content = f"Content {i}"
            mock_doc.metadata = {"chunk_id": f"doc{i}_chunk_0", "score": 1.0 - i * 0.1}
            mock_docs.append((mock_doc, 1.0 - i * 0.1))

        mock_vector_store.similarity_search_with_relevance_scores.return_value = mock_docs

        with patch("agentic_rag.retrieval.retriever.PGVector", return_value=mock_vector_store):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings", return_value=mock_embeddings):
                retriever = PGVectorRetriever()
                query = Query(text="test query")
                results = retriever.search(query, k=3)

        # Should call with k=3
        mock_vector_store.similarity_search_with_relevance_scores.assert_called_once()
        call_kwargs = mock_vector_store.similarity_search_with_relevance_scores.call_args[1]
        assert call_kwargs["k"] == 3

    def test_search_with_score_threshold(self, mock_vector_store, mock_embeddings):
        """Test filtering by score threshold."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        mock_doc1 = Mock()
        mock_doc1.page_content = "High score content"
        mock_doc1.metadata = {"chunk_id": "doc1_chunk_0"}

        mock_doc2 = Mock()
        mock_doc2.page_content = "Low score content"
        mock_doc2.metadata = {"chunk_id": "doc2_chunk_0"}

        mock_vector_store.similarity_search_with_relevance_scores.return_value = [
            (mock_doc1, 0.9),
            (mock_doc2, 0.3),
        ]

        with patch("agentic_rag.retrieval.retriever.PGVector", return_value=mock_vector_store):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings", return_value=mock_embeddings):
                retriever = PGVectorRetriever(score_threshold=0.5)
                query = Query(text="test query")
                results = retriever.search(query, k=5)

        # Should only return results above threshold
        assert len(results) == 1
        assert results[0].score == 0.9
        assert results[0].text == "High score content"

    def test_search_with_metadata_filter(self, mock_vector_store, mock_embeddings):
        """Test metadata filtering."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        with patch("agentic_rag.retrieval.retriever.PGVector", return_value=mock_vector_store):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings", return_value=mock_embeddings):
                retriever = PGVectorRetriever()
                query = Query(text="test query", metadata={"category": "wordpress"})
                retriever.search(query, k=5)

        # Should pass metadata filter to search
        call_kwargs = mock_vector_store.similarity_search_with_relevance_scores.call_args[1]
        assert "filter" in call_kwargs
        assert call_kwargs["filter"] == {"category": "wordpress"}

    def test_search_empty_results(self, mock_vector_store, mock_embeddings):
        """Test handling of empty search results."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        mock_vector_store.similarity_search_with_relevance_scores.return_value = []

        with patch("agentic_rag.retrieval.retriever.PGVector", return_value=mock_vector_store):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings", return_value=mock_embeddings):
                retriever = PGVectorRetriever()
                query = Query(text="test query")
                results = retriever.search(query, k=5)

        assert len(results) == 0
        assert isinstance(results, list)

    def test_search_handles_missing_chunk_id(self, mock_vector_store, mock_embeddings):
        """Test handling of documents without chunk_id in metadata."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        mock_doc = Mock()
        mock_doc.page_content = "Content without chunk_id"
        mock_doc.metadata = {}  # No chunk_id

        mock_vector_store.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.9)]

        with patch("agentic_rag.retrieval.retriever.PGVector", return_value=mock_vector_store):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings", return_value=mock_embeddings):
                retriever = PGVectorRetriever()
                query = Query(text="test query")
                results = retriever.search(query, k=5)

        # Should generate a chunk_id or handle gracefully
        assert len(results) == 1
        assert results[0].chunk_id is not None

    def test_retriever_initialization(self):
        """Test retriever initializes with correct settings."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        with patch("agentic_rag.retrieval.retriever.PGVector"):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings"):
                retriever = PGVectorRetriever(score_threshold=0.7)
                assert retriever.score_threshold == 0.7

    def test_search_preserves_metadata(self, mock_vector_store, mock_embeddings):
        """Test that metadata from documents is preserved."""
        from agentic_rag.retrieval.retriever import PGVectorRetriever

        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {
            "chunk_id": "doc1_chunk_0",
            "original_title": "Test Title",
            "source": "corpus.jsonl",
        }

        mock_vector_store.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.9)]

        with patch("agentic_rag.retrieval.retriever.PGVector", return_value=mock_vector_store):
            with patch("agentic_rag.retrieval.retriever.OpenAIEmbeddings", return_value=mock_embeddings):
                retriever = PGVectorRetriever()
                query = Query(text="test query")
                results = retriever.search(query, k=5)

        assert len(results) == 1
        assert results[0].metadata is not None
        assert results[0].metadata["original_title"] == "Test Title"
        assert results[0].metadata["source"] == "corpus.jsonl"
