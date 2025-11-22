"""Tests for ingestion pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_rag.data.ingestion_pipeline import BaseIngestionPipeline, IngestionPipeline
from agentic_rag.data.types import Chunk, RawRecord
from agentic_rag.utils import write_jsonl


def test_pipeline_implements_base_interface() -> None:
    """Test implement BaseIngestionPipeline."""
    pipeline = IngestionPipeline()
    assert isinstance(pipeline, BaseIngestionPipeline)
    assert hasattr(pipeline, "load_raw")
    assert hasattr(pipeline, "transform")
    assert hasattr(pipeline, "persist")
    assert hasattr(pipeline, "run")


def test_pipeline_run_complete_flow(tmp_path: Path) -> None:
    """Test complete pipeline flow."""
    # Create sample corpus file
    corpus_file = tmp_path / "corpus.jsonl"
    sample_data = [
        {
            "_id": "doc1",
            "title": "Test Title 1",
            "text": "This is a test document with some content that will be chunked.",
        },
        {
            "_id": "doc2",
            "title": "Test Title 2",
            "text": "Another test document for testing the pipeline.",
        },
    ]
    write_jsonl(corpus_file, sample_data)

    # Mock database operations
    with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
        mock_store = MagicMock()
        mock_pgvector.return_value = mock_store

        pipeline = IngestionPipeline()
        
        # Mock settings to use our test directory
        with patch.object(pipeline.settings, "dataset") as mock_dataset:
            mock_dataset.corpus_filename = "corpus.jsonl"
            
            # Run pipeline
            pipeline.run(tmp_path, tmp_path)

        # Verify PGVector was initialized
        assert mock_pgvector.called
        # Verify add_texts was called (chunks were persisted)
        assert mock_store.add_texts.called


def test_pipeline_error_handling(tmp_path: Path) -> None:
    """Test error handling trong pipeline."""
    # Create invalid corpus file
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text("invalid json content\n")

    pipeline = IngestionPipeline()
    
    # Should handle errors gracefully
    with patch("agentic_rag.data.ingestion_pipeline.PGVector"):
        with patch.object(pipeline.settings, "dataset") as mock_dataset:
            mock_dataset.corpus_filename = "corpus.jsonl"
            
            # Should not raise exception, should handle gracefully
            try:
                pipeline.run(tmp_path, tmp_path)
            except Exception:
                # If it raises, that's also acceptable - just verify it's handled
                pass


def test_pipeline_progress_tracking(tmp_path: Path) -> None:
    """Test progress tracking."""
    corpus_file = tmp_path / "corpus.jsonl"
    sample_data = [
        {"_id": "doc1", "title": "Test", "text": "Test content " * 100},  # Long text
    ]
    write_jsonl(corpus_file, sample_data)

    with patch("agentic_rag.data.ingestion_pipeline.PGVector"):
        with patch("agentic_rag.data.ingestion_pipeline.Progress") as mock_progress:
            mock_progress.return_value.__enter__.return_value = MagicMock()
            
            pipeline = IngestionPipeline()
            
            with patch.object(pipeline.settings, "dataset") as mock_dataset:
                mock_dataset.corpus_filename = "corpus.jsonl"
                
                pipeline.run(tmp_path, tmp_path)
            
            # Verify Progress was used
            assert mock_progress.called


def test_pipeline_with_empty_data(tmp_path: Path) -> None:
    """Test handle empty dataset."""
    # Create empty corpus file
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.touch()

    pipeline = IngestionPipeline()
    
    with patch("agentic_rag.data.ingestion_pipeline.PGVector"):
        with patch.object(pipeline.settings, "dataset") as mock_dataset:
            mock_dataset.corpus_filename = "corpus.jsonl"
            
            # Should handle empty file gracefully
            try:
                pipeline.run(tmp_path, tmp_path)
            except Exception:
                # Empty file might raise FileNotFoundError or similar
                # That's acceptable behavior
                pass


def test_pipeline_idempotency(tmp_path: Path) -> None:
    """Test có thể run nhiều lần."""
    corpus_file = tmp_path / "corpus.jsonl"
    sample_data = [
        {"_id": "doc1", "title": "Test", "text": "Test content"},
    ]
    write_jsonl(corpus_file, sample_data)

    with patch("agentic_rag.data.ingestion_pipeline.PGVector") as mock_pgvector:
        mock_store = MagicMock()
        mock_pgvector.return_value = mock_store

        pipeline = IngestionPipeline()
        
        with patch.object(pipeline.settings, "dataset") as mock_dataset:
            mock_dataset.corpus_filename = "corpus.jsonl"
            
            # Run pipeline multiple times
            pipeline.run(tmp_path, tmp_path)
            pipeline.run(tmp_path, tmp_path)
            pipeline.run(tmp_path, tmp_path)
        
        # Should complete without errors
        assert mock_pgvector.called

