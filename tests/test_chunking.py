"""Tests for chunking functionality."""

from __future__ import annotations

from agentic_rag.data.types import Chunk, RawRecord


def test_recursive_character_splitter() -> None:
    """Test RecursiveCharacterTextSplitter."""
    from agentic_rag.data.chunking import ChunkingStrategy

    record = RawRecord(
        identifier="doc1",
        title="Test Title",
        body="This is a test document with some content.",
    )
    
    strategy = ChunkingStrategy(chunk_size=20, chunk_overlap=5)
    chunks = strategy.chunk(record)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunk_size_respected() -> None:
    """Test chunk size is respected."""
    from agentic_rag.data.chunking import ChunkingStrategy

    # Create a long text
    long_text = " ".join(["word"] * 100)  # ~500 characters
    record = RawRecord(
        identifier="doc1",
        title="Test",
        body=long_text,
    )
    
    strategy = ChunkingStrategy(chunk_size=50, chunk_overlap=10)
    chunks = strategy.chunk(record)
    
    # Each chunk should be approximately chunk_size (allowing some flexibility)
    for chunk in chunks:
        # chunk_size (50) + title overhead ("Title: Test\nContent: " = 21 chars) = 71
        assert len(chunk.text) <= 80


def test_chunk_overlap() -> None:
    """Test overlap between chunks."""
    from agentic_rag.data.chunking import ChunkingStrategy

    text = " ".join(["sentence"] * 50)
    record = RawRecord(identifier="doc1", title="Test", body=text)
    
    strategy = ChunkingStrategy(chunk_size=50, chunk_overlap=10)
    chunks = strategy.chunk(record)
    
    # 450 chars body / (50 - 10) step ~= 11-12 chunks + overlap overhead
    # Previously 14 with title included in split text, now 13 with just body split
    assert len(chunks) == 13


def test_preserve_metadata() -> None:
    """Test metadata được preserve trong mỗi chunk."""
    from agentic_rag.data.chunking import ChunkingStrategy

    record = RawRecord(
        identifier="doc1",
        title="Test Title",
        body="Test content",
        metadata={"key1": "value1", "key2": 123},
    )
    
    strategy = ChunkingStrategy(chunk_size=100, chunk_overlap=0)
    chunks = strategy.chunk(record)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert "key1" in chunk.metadata
        assert chunk.metadata["key1"] == "value1"
        assert chunk.metadata["key2"] == 123
        assert "original_title" in chunk.metadata


def test_generate_unique_chunk_ids() -> None:
    """Test unique chunk_id generation."""
    from agentic_rag.data.chunking import ChunkingStrategy

    record = RawRecord(
        identifier="doc1",
        title="Test",
        body="This is a test that will be split into chunks.",
    )
    
    strategy = ChunkingStrategy(chunk_size=20, chunk_overlap=0)
    chunks = strategy.chunk(record)
    
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(chunk_id.startswith("doc1_chunk_") for chunk_id in chunk_ids)


def test_empty_text_handling() -> None:
    """Test handle empty text."""
    from agentic_rag.data.chunking import ChunkingStrategy

    record = RawRecord(identifier="doc1", title="", body="")
    strategy = ChunkingStrategy(chunk_size=100, chunk_overlap=0)
    chunks = strategy.chunk(record)
    
    # Should handle empty text gracefully
    assert isinstance(chunks, list)


def test_very_long_text() -> None:
    """Test handle very long text."""
    from agentic_rag.data.chunking import ChunkingStrategy

    # Create very long text
    very_long_text = " ".join(["word"] * 1000)
    record = RawRecord(identifier="doc1", title="Test", body=very_long_text)
    
    strategy = ChunkingStrategy(chunk_size=100, chunk_overlap=20)
    chunks = strategy.chunk(record)
    
    # Should split into multiple chunks
    assert len(chunks) > 1
    # All chunks should be valid
    assert all(isinstance(chunk, Chunk) for chunk in chunks)