"""Tests for agent tools."""

from unittest.mock import MagicMock, patch
from agentic_rag.agent.tools import rag_search_tool

def test_rag_search_tool_success():
    """Test rag_search_tool with successful retrieval and reranking."""
    
    # Mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Chunk 1 content"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "Chunk 2 content"
    
    with patch("agentic_rag.agent.tools.PGVectorRetriever") as MockRetriever:
        with patch("agentic_rag.agent.tools.CrossEncoderReranker") as MockReranker:
            # Setup mocks
            retriever_instance = MockRetriever.return_value
            retriever_instance.search.return_value = [mock_chunk1, mock_chunk2]
            
            reranker_instance = MockReranker.return_value
            reranker_instance.rerank.return_value = [mock_chunk1] # Return top 1
            
            # Run tool
            result = rag_search_tool.invoke("test query")
            
            # Verify interactions
            retriever_instance.search.assert_called_once()
            call_kwargs = retriever_instance.search.call_args[1]
            assert call_kwargs["k"] == 10 # Check if k increased
            
            reranker_instance.rerank.assert_called_once()
            
            # Verify output
            assert "Chunk 1 content" in result
            assert "Chunk 2 content" not in result # Should be filtered out by reranker mock

def test_rag_search_tool_no_results():
    """Test rag_search_tool when no results found."""
    
    with patch("agentic_rag.agent.tools.PGVectorRetriever") as MockRetriever:
        with patch("agentic_rag.agent.tools.CrossEncoderReranker") as MockReranker:
            retriever_instance = MockRetriever.return_value
            retriever_instance.search.return_value = []
            
            result = rag_search_tool.invoke("test query")
            
            assert result == ""
            MockReranker.return_value.rerank.assert_not_called()
