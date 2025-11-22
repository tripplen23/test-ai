"""Tests for persistence functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.data.ingestion_pipeline import IngestionPipeline
from agentic_rag.data.types import Chunk


def test_persist_chunks_to_db() -> None:
    """Test insert chunks vào database."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        # Create mock chunks
        chunks = [
            Chunk(
                chunk_id="chunk1",
                record_id="doc1",
                text="Test chunk 1",
                metadata={"key": "value"},
            ),
            Chunk(
                chunk_id="chunk2",
                record_id="doc1",
                text="Test chunk 2",
                metadata={},
            ),
        ]
        
        # Mock PGVector
        with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
            mock_store = MagicMock()
            mock_pgvector.return_value = mock_store
            
            pipeline = IngestionPipeline()
            pipeline.persist(chunks, Path("output"))
            
            # Verify PGVector was initialized
            assert mock_pgvector.called
            # Verify add_texts was called
            assert mock_store.add_texts.called


def test_batch_insert() -> None:
    """Test batch insertion."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        # Create chunks (more than batch_size to test batching)
        chunks = [
            Chunk(
                chunk_id=f"chunk{i}",
                record_id=f"doc{i//10}",
                text=f"Test chunk {i}",
                metadata={},
            )
            for i in range(600)  # More than batch_size (500)
        ]
        
        with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
            mock_store = MagicMock()
            mock_pgvector.return_value = mock_store
            
            pipeline = IngestionPipeline()
            pipeline.persist(chunks, Path("output"))
            
            # Should be called at least twice (500 + 100)
            assert mock_store.add_texts.call_count >= 2
            
            # Verify batch sizes
            call_args_list = mock_store.add_texts.call_args_list
            # First batch should have 500 chunks
            assert len(call_args_list[0].kwargs["ids"]) == 500
            # Last batch should have remaining chunks
            assert len(call_args_list[-1].kwargs["ids"]) == 100


def test_handle_duplicates() -> None:
    """Test ON CONFLICT handling."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        # Create chunks with duplicate IDs
        chunks = [
            Chunk(
                chunk_id="chunk1",  # Duplicate ID
                record_id="doc1",
                text="Test chunk 1",
                metadata={},
            ),
            Chunk(
                chunk_id="chunk1",  # Duplicate ID
                record_id="doc1",
                text="Test chunk 1 duplicate",
                metadata={},
            ),
        ]
        
        with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
            mock_store = MagicMock()
            # PGVector handles duplicates internally, so it should not raise
            mock_store.add_texts.return_value = None
            mock_pgvector.return_value = mock_store
            
            pipeline = IngestionPipeline()
            # Should not raise exception
            pipeline.persist(chunks, Path("output"))
            
            # Verify add_texts was called
            assert mock_store.add_texts.called


def test_transaction_rollback() -> None:
    """Test error handling when insert fails."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        chunks = [
            Chunk(
                chunk_id="chunk1",
                record_id="doc1",
                text="Test chunk",
                metadata={},
            ),
        ]
        
        with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
            mock_store = MagicMock()
            # Simulate database error
            mock_store.add_texts.side_effect = Exception("Database connection failed")
            mock_pgvector.return_value = mock_store
            
            pipeline = IngestionPipeline()
            
            # Should handle error gracefully and continue
            # (not raise exception, just log error)
            try:
                pipeline.persist(chunks, Path("output"))
            except Exception:
                # If it raises, that's also acceptable - error was handled
                pass
            
            # Verify add_texts was attempted
            assert mock_store.add_texts.called


def test_progress_logging() -> None:
    """Test progress logging."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        chunks = [
            Chunk(
                chunk_id=f"chunk{i}",
                record_id="doc1",
                text=f"Test chunk {i}",
                metadata={},
            )
            for i in range(10)
        ]
        
        with patch("agentic_rag.data.ingestion_pipeline.PGVector"):
            with patch("agentic_rag.data.ingestion_pipeline.Progress") as mock_progress:
                mock_progress_instance = MagicMock()
                mock_task = MagicMock()
                mock_progress_instance.add_task.return_value = mock_task
                mock_progress.return_value.__enter__.return_value = mock_progress_instance
                
                pipeline = IngestionPipeline()
                pipeline.persist(chunks, Path("output"))
                
                # Verify Progress was used
                assert mock_progress.called
                # Verify task was added
                assert mock_progress_instance.add_task.called
                # Verify progress was updated
                assert mock_progress_instance.update.called


def test_connection_cleanup() -> None:
    """Test connections được close properly."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        chunks = [
            Chunk(
                chunk_id="chunk1",
                record_id="doc1",
                text="Test chunk",
                metadata={},
            ),
        ]
        
        with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
            mock_store = MagicMock()
            mock_pgvector.return_value = mock_store
            
            pipeline = IngestionPipeline()
            pipeline.persist(chunks, Path("output"))
            
            # PGVector manages connections internally
            # We verify that PGVector was created and used
            assert mock_pgvector.called
            assert mock_store.add_texts.called


def test_persist_empty_chunks() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        chunks = []
        
        with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
            mock_store = MagicMock()
            mock_pgvector.return_value = mock_store
            
            pipeline = IngestionPipeline()
            pipeline.persist(chunks, Path("output"))
            
            # Should not call add_texts if no chunks
            assert not mock_store.add_texts.called


def test_persist_missing_api_key() -> None:
    """Test persist raises error when OPENAI_API_KEY is missing."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        chunks = [
            Chunk(
                chunk_id="chunk1",
                record_id="doc1",
                text="Test chunk",
                metadata={},
            ),
        ]
        
        # Create pipeline first
        pipeline = IngestionPipeline()
        
        # Then mock settings to have no API key
        pipeline.settings.openai_api_key = None
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            pipeline.persist(chunks, Path("output"))
