"""Main ingestion pipeline implementation."""

from __future__ import annotations

import abc
import logging
from collections.abc import Iterable
from pathlib import Path

from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from rich.progress import Progress, SpinnerColumn, TextColumn

from agentic_rag.data.chunking import ChunkingStrategy
from agentic_rag.data.cleaning import clean_text, validate_record
from agentic_rag.data.db import get_connection_string
from agentic_rag.data.types import Chunk, RawRecord
from agentic_rag.settings import get_settings
from agentic_rag.utils import read_jsonl

logger = logging.getLogger(__name__)


class BaseIngestionPipeline(abc.ABC):
    """Blueprint for dataset ingestion pipelines."""

    @abc.abstractmethod
    def load_raw(self, raw_dir: Path) -> Iterable[RawRecord]:
        """Load raw corpus rows from `raw_dir`."""

    @abc.abstractmethod
    def transform(self, records: Iterable[RawRecord]) -> Iterable[Chunk]:
        """Transform raw records into clean, chunked units."""

    @abc.abstractmethod
    def persist(self, chunks: Iterable[Chunk], output_dir: Path) -> None:
        """Write processed chunks or index artifacts to disk/backend."""

    def run(self, raw_dir: Path, output_dir: Path) -> None:
        """Run the complete ingestion pipeline."""
        records = self.load_raw(raw_dir)
        chunks = self.transform(records)
        self.persist(chunks, output_dir)


class IngestionPipeline(BaseIngestionPipeline):
    """Ingestion pipeline for processing and storing documents."""

    def __init__(self) -> None:
        """Initialize ingestion pipeline with components."""
        self.settings = get_settings()
        self.chunking_strategy = ChunkingStrategy(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def load_raw(self, raw_dir: Path) -> Iterable[RawRecord]:
        """
        Load raw corpus rows from raw_dir.
        
        Args:
            raw_dir: Directory containing corpus.jsonl
            
        Yields:
            RawRecord objects
        """
        corpus_file = raw_dir / self.settings.dataset.corpus_filename
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        for record_data in read_jsonl(corpus_file):
            try:
                # Extract fields
                identifier = str(record_data.get("_id", ""))
                title = str(record_data.get("title", ""))
                body = str(record_data.get("text", record_data.get("body", "")))
                
                # Extract metadata (all other fields)
                excluded_fields = ("_id", "title", "text", "body")
                metadata = {k: v for k, v in record_data.items() if k not in excluded_fields}
                
                record = RawRecord(
                    identifier=identifier,
                    title=title,
                    body=body,
                    metadata=metadata,
                )
                
                yield record
            except Exception as e:
                # Skip invalid records but log error
                logger.warning(f"Skipping invalid record: {e}", exc_info=True)
                continue

    def transform(self, records: Iterable[RawRecord]) -> Iterable[Chunk]:
        """
        Transform raw records into clean, chunked units.
        
        Note: Embeddings are computed by PGVector during persistence, not here.
        
        Args:
            records: RawRecord objects to transform
            
        Yields:
            Chunk objects ready for persistence
        """
        for record in records:
            # Validate record
            if not validate_record(record):
                continue
            
            # Clean text
            cleaned_title = clean_text(record.title)
            cleaned_body = clean_text(record.body)
            
            # Create cleaned record
            cleaned_record = RawRecord(
                identifier=record.identifier,
                title=cleaned_title,
                body=cleaned_body,
                metadata=record.metadata,
            )
            
            # Chunk the record and yield chunks
            yield from self.chunking_strategy.chunk(cleaned_record)

    def persist(self, chunks: Iterable[Chunk], output_dir: Path) -> None:
        """
        Write processed chunks to pgvector database.
        
        Args:
            chunks: Chunk objects to persist
            output_dir: Output directory (not used for DB persistence, but required by interface)
        """
        if not self.settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in your environment or .env file."
            )
        
        logger.info("Initializing PGVector connection")
        vector_store = self._create_vector_store()
        
        batch_size = 500
        stats = {"total": 0, "successful_batches": 0, "failed_batches": 0, "failed_chunks": 0}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Persisting chunks...", total=None)
            
            batch = self._prepare_batch(chunks, batch_size)
            
            for batch_chunks, batch_texts, batch_metadatas, is_final in batch:
                success = self._persist_batch(
                    vector_store, batch_chunks, batch_texts, batch_metadatas, is_final
                )
                
                if success:
                    stats["total"] += len(batch_chunks)
                    stats["successful_batches"] += 1
                    progress.update(
                        task,
                        description=(
                            f"Persisted {stats['total']} chunks "
                            f"({stats['successful_batches']} batches)..."
                        ),
                    )
                    if is_final:
                        logger.info(
                            f"Successfully persisted final batch of {len(batch_chunks)} chunks"
                        )
                else:
                    stats["failed_batches"] += 1
                    stats["failed_chunks"] += len(batch_chunks)
                    progress.update(
                        task,
                        description=f"Error persisting batch ({stats['failed_batches']} failed)...",
                    )
                    if is_final:
                        logger.error(
                            f"FAILED to persist final batch of {len(batch_chunks)} chunks. "
                            f"Check error logs above for details."
                        )
        
        self._log_persistence_stats(stats)
    
    def _create_vector_store(self) -> PGVector:
        """Create and return PGVector instance."""
        embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )
        
        logger.info(f"Connecting to database collection: {self.settings.vector_store.collection}")
        return PGVector(
            connection_string=get_connection_string(),
            embedding_function=embeddings,
            collection_name=self.settings.vector_store.collection,
        )
    
    def _prepare_batch(
        self, chunks: Iterable[Chunk], batch_size: int
    ) -> Iterable[tuple[list[Chunk], list[str], list[dict], bool]]:
        """
        Prepare chunks into batches for persistence.
        
        Yields:
            Tuples of (chunks, texts, metadatas, is_final)
        """
        chunk_batch: list[Chunk] = []
        texts_batch: list[str] = []
        metadatas_batch: list[dict] = []
        
        for chunk in chunks:
            if not chunk.chunk_id or not chunk.text:
                logger.warning(f"Skipping invalid chunk: {chunk.chunk_id}")
                continue
            
            chunk_batch.append(chunk)
            texts_batch.append(chunk.text)
            metadatas_batch.append(self._prepare_chunk_metadata(chunk))
            
            if len(chunk_batch) >= batch_size:
                yield chunk_batch.copy(), texts_batch.copy(), metadatas_batch.copy(), False
                chunk_batch.clear()
                texts_batch.clear()
                metadatas_batch.clear()
        
        # Yield final batch if any remaining
        if chunk_batch:
            logger.info(f"Processing final batch of {len(chunk_batch)} chunks")
            yield chunk_batch, texts_batch, metadatas_batch, True
    
    def _prepare_chunk_metadata(self, chunk: Chunk) -> dict:
        """Prepare metadata for a chunk, excluding internal fields."""
        metadata = {k: v for k, v in chunk.metadata.items() if k != "_embedding"}
        metadata["chunk_id"] = chunk.chunk_id
        metadata["record_id"] = chunk.record_id
        return metadata
    
    def _persist_batch(
        self,
        vector_store: PGVector,
        batch_chunks: list[Chunk],
        batch_texts: list[str],
        batch_metadatas: list[dict],
        is_final: bool,
    ) -> bool:
        """Persist a batch of chunks to the database. Returns True if successful."""
        try:
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=[chunk.chunk_id for chunk in batch_chunks],
            )
            batch_type = "final" if is_final else "regular"
            logger.debug(f"Successfully persisted {batch_type} batch of {len(batch_chunks)} chunks")
            return True
        except Exception as e:
            batch_type = "final" if is_final else "regular"
            logger.error(
                f"Failed to persist {batch_type} batch of {len(batch_chunks)} chunks: {e}",
                exc_info=True,
            )
            return False
    
    def _log_persistence_stats(self, stats: dict[str, int]) -> None:
        """Log final persistence statistics."""
        logger.info(
            f"Persistence completed: {stats['total']} chunks persisted successfully, "
            f"{stats['failed_chunks']} chunks failed ({stats['failed_batches']} failed batches)"
        )
        
        if stats["failed_chunks"] > 0:
            logger.warning(
                f"Some chunks failed to persist. Total failed: {stats['failed_chunks']} "
                f"across {stats['failed_batches']} batches. Check logs for details."
            )