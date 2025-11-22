"""Tests for Hugging Face dataset download functionality."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.download_dataset import download


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from agentic_rag.settings.schema import DatasetConfig

    class MockSettings:
        dataset = DatasetConfig(name="mteb/cqadupstack-wordpress")

    return MockSettings()


def test_download_dataset_from_huggingface(temp_output_dir: Path, mock_settings) -> None:
    """Test download dataset từ Hugging Face."""
    with patch("scripts.download_dataset.get_settings", return_value=mock_settings):
        with patch("scripts.download_dataset.load_dataset") as mock_load:
            from datasets import Dataset

            # Mock dataset với sample data
            mock_corpus = Dataset.from_dict(
                {
                    "_id": ["1", "2"],
                    "title": ["Title 1", "Title 2"],
                    "text": ["Text 1", "Text 2"],
                }
            )
            mock_queries = Dataset.from_dict(
                {
                    "_id": ["q1", "q2"],
                    "text": ["Query 1", "Query 2"],
                }
            )
            mock_qrels = Dataset.from_dict(
                {
                    "query-id": ["q1", "q1"],
                    "corpus-id": ["1", "2"],
                    "score": [1.0, 1.0],
                }
            )

            def load_dataset_side_effect(name: str, config: str, split: str, token: str | None = None):
                if config == "corpus":
                    return mock_corpus
                elif config == "queries":
                    return mock_queries
                elif config == "default":
                    return mock_qrels
                return mock_corpus

            mock_load.side_effect = load_dataset_side_effect

            download(temp_output_dir)

            # Verify files were created
            assert (temp_output_dir / "corpus.jsonl").exists()
            assert (temp_output_dir / "queries.jsonl").exists()
            assert (temp_output_dir / "qrels.jsonl").exists()

            # Verify load_dataset was called with correct parameters
            assert mock_load.call_count == 3


def test_download_with_hf_token(temp_output_dir: Path, mock_settings) -> None:
    """Test download với HF_TOKEN authentication."""
    test_token = "hf_test_token_12345"

    with patch("scripts.download_dataset.get_settings", return_value=mock_settings):
        with patch("scripts.download_dataset.load_dataset") as mock_load:
            from datasets import Dataset

            mock_dataset = Dataset.from_dict({"_id": ["1"], "title": ["Test"], "text": ["Test text"]})

            def load_dataset_side_effect(name: str, config: str, split: str, token: str | None = None):
                if token:
                    assert token == test_token
                return mock_dataset

            mock_load.side_effect = load_dataset_side_effect

            # Set HF_TOKEN environment variable
            with patch.dict(os.environ, {"HF_TOKEN": test_token}):
                download(temp_output_dir)

            # Verify load_dataset was called
            assert mock_load.called


def test_download_generates_corpus_jsonl(temp_output_dir: Path, mock_settings) -> None:
    """Test generate corpus.jsonl file với đúng structure."""
    with patch("scripts.download_dataset.get_settings", return_value=mock_settings):
        with patch("scripts.download_dataset.load_dataset") as mock_load:
            from datasets import Dataset

            mock_corpus = Dataset.from_dict(
                {
                    "_id": ["doc1", "doc2"],
                    "title": ["Title 1", "Title 2"],
                    "text": ["Text content 1", "Text content 2"],
                }
            )

            def load_dataset_side_effect(name: str, config: str, split: str, token: str | None = None):
                if config == "corpus":
                    return mock_corpus
                return Dataset.from_dict({})

            mock_load.side_effect = load_dataset_side_effect

            download(temp_output_dir)

            corpus_file = temp_output_dir / "corpus.jsonl"
            assert corpus_file.exists()

            # Verify file content structure
            from agentic_rag.utils import read_jsonl

            records = list(read_jsonl(corpus_file))
            assert len(records) == 2
            assert "_id" in records[0]
            assert "title" in records[0]
            assert "text" in records[0]


def test_download_generates_queries_jsonl(temp_output_dir: Path, mock_settings) -> None:
    """Test generate queries.jsonl file."""
    with patch("scripts.download_dataset.get_settings", return_value=mock_settings):
        with patch("scripts.download_dataset.load_dataset") as mock_load:
            from datasets import Dataset

            mock_queries = Dataset.from_dict(
                {
                    "_id": ["q1", "q2"],
                    "text": ["Query 1", "Query 2"],
                }
            )

            def load_dataset_side_effect(name: str, config: str, split: str, token: str | None = None):
                if config == "queries":
                    return mock_queries
                return Dataset.from_dict({})

            mock_load.side_effect = load_dataset_side_effect

            download(temp_output_dir)

            queries_file = temp_output_dir / "queries.jsonl"
            assert queries_file.exists()


def test_download_generates_qrels_jsonl(temp_output_dir: Path, mock_settings) -> None:
    """Test generate qrels.jsonl file."""
    with patch("scripts.download_dataset.get_settings", return_value=mock_settings):
        with patch("scripts.download_dataset.load_dataset") as mock_load:
            from datasets import Dataset

            mock_qrels = Dataset.from_dict(
                {
                    "query-id": ["q1", "q2"],
                    "corpus-id": ["doc1", "doc2"],
                    "score": [1.0, 1.0],
                }
            )

            def load_dataset_side_effect(name: str, config: str, split: str, token: str | None = None):
                if config == "default":
                    return mock_qrels
                return Dataset.from_dict({})

            mock_load.side_effect = load_dataset_side_effect

            download(temp_output_dir)

            qrels_file = temp_output_dir / "qrels.jsonl"
            assert qrels_file.exists()


def test_download_handles_missing_token(temp_output_dir: Path, mock_settings) -> None:
    """Test handle missing HF_TOKEN gracefully."""
    with patch("scripts.download_dataset.get_settings", return_value=mock_settings):
        with patch("scripts.download_dataset.load_dataset") as mock_load:
            from datasets import Dataset

            mock_dataset = Dataset.from_dict({"_id": ["1"], "title": ["Test"], "text": ["Test"]})

            def load_dataset_side_effect(name: str, config: str, split: str, token: str | None = None):
                # Should work without token for public datasets
                return mock_dataset

            mock_load.side_effect = load_dataset_side_effect

            # Ensure HF_TOKEN is not set
            with patch.dict(os.environ, {}, clear=True):
                download(temp_output_dir)

            # Should complete without errors
            assert mock_load.called


def test_dataset_structure(temp_output_dir: Path, mock_settings) -> None:
    """Test verify dataset structure (fields: _id, title, text)."""
    with patch("scripts.download_dataset.get_settings", return_value=mock_settings):
        with patch("scripts.download_dataset.load_dataset") as mock_load:
            from datasets import Dataset

            mock_corpus = Dataset.from_dict(
                {
                    "_id": ["doc1"],
                    "title": ["Test Title"],
                    "text": ["Test text content"],
                }
            )

            def load_dataset_side_effect(name: str, config: str, split: str, token: str | None = None):
                if config == "corpus":
                    return mock_corpus
                return Dataset.from_dict({})

            mock_load.side_effect = load_dataset_side_effect

            download(temp_output_dir)

            # Verify structure in generated file
            from agentic_rag.utils import read_jsonl

            corpus_file = temp_output_dir / "corpus.jsonl"
            records = list(read_jsonl(corpus_file))

            assert len(records) > 0
            record = records[0]
            assert "_id" in record
            assert "title" in record
            assert "text" in record

