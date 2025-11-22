"""Chunking strategies for text splitting."""

from __future__ import annotations

from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.data.types import Chunk, RawRecord
from agentic_rag.settings import get_settings


class ChunkingStrategy:
    """Chunking strategy using LangChain RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        """
        Initialize chunking strategy.
        
        Args:
            chunk_size: Size of chunks in characters (defaults to settings.chunk_size)
            chunk_overlap: Overlap between chunks in characters (defaults to settings.chunk_overlap)
        """
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        
        # Ensure overlap is not larger than chunk_size
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size - 1)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def chunk(self, record: RawRecord) -> list[Chunk]:
        """
        Chunk a RawRecord into multiple Chunks.
        
        Args:
            record: RawRecord to chunk
            
        Returns:
            List of Chunk objects
        """
        # Split the body text directly
        text_chunks = self.splitter.split_text(record.body)
        
        # Convert to Chunk objects
        chunks = []
        for idx, text_chunk in enumerate(text_chunks):
            chunk_id = f"{record.identifier}_chunk_{idx}"
            
            # Prepend title to every chunk for better context
            contextualized_text = f"Title: {record.title}\nContent: {text_chunk}"
            
            # Preserve metadata from original record
            chunk_metadata = dict(record.metadata)
            chunk_metadata["original_title"] = record.title
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["total_chunks"] = len(text_chunks)
            
            chunk = Chunk(
                chunk_id=chunk_id,
                record_id=record.identifier,
                text=contextualized_text,
                metadata=chunk_metadata,
                created_at=datetime.now(),
            )
            chunks.append(chunk)
        
        return chunks