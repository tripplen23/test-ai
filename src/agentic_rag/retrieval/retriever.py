"""Vector-based retrieval implementation using PGVector."""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from agentic_rag.data.db import get_connection_string
from agentic_rag.settings import get_settings

from .base import BaseRetriever
from .schemas import Query, RetrievedChunk

logger = logging.getLogger(__name__)


class PGVectorRetriever(BaseRetriever):
    """Retriever implementation using PGVector for similarity search."""

    def __init__(self, score_threshold: float | None = None):
        """
        Initialize PGVectorRetriever.

        Args:
            score_threshold: Minimum similarity score for results (0.0-1.0).
                           Results below this threshold will be filtered out.
        """
        self.settings = get_settings()
        self.score_threshold = score_threshold or self.settings.retrieval_score_threshold
        
        # Initialize embeddings with same model as ingestion
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )
        
        # Initialize vector store connection
        self.vector_store = self._init_vector_store()
        logger.info(
            f"Initialized PGVectorRetriever with model={self.settings.embedding_model}, "
            f"threshold={self.score_threshold}"
        )

    def _init_vector_store(self) -> PGVector:
        """Initialize PGVector store connection."""
        connection_string = get_connection_string()
        
        return PGVector(
            collection_name=self.settings.vector_store.collection,
            connection_string=connection_string,
            embedding_function=self.embeddings,
        )

    @traceable(name="pgvector_search")
    def search(self, query: Query, *, k: int = 5) -> Sequence[RetrievedChunk]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Query object with text and optional metadata filters
            k: Number of results to retrieve (before threshold filtering)

        Returns:
            List of RetrievedChunk objects sorted by relevance score (descending)
        """
        logger.debug(f"Searching for query: {query.text[:100]}... (k={k})")
        
        # Prepare search kwargs
        search_kwargs = {"k": k}
        
        # Add metadata filter if provided
        if query.metadata:
            search_kwargs["filter"] = query.metadata
            logger.debug(f"Applying metadata filter: {query.metadata}")
        
        # Perform similarity search with relevance scores (normalized 0-1)
        results = self.vector_store.similarity_search_with_relevance_scores(
            query.text,
            **search_kwargs
        )
        
        logger.info(f"Retrieved {len(results)} results before filtering")
        
        # Convert to RetrievedChunk objects and filter by threshold
        chunks = []
        for doc, score in results:
            # Filter by score threshold
            if score < self.score_threshold:
                logger.debug(f"Filtering out result with score {score:.3f} < {self.score_threshold}")
                continue
            
            # Get chunk_id from metadata or generate one
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                # Generate a fallback chunk_id if not present
                chunk_id = f"chunk_{hash(doc.page_content)}"
                logger.warning(f"Document missing chunk_id, generated: {chunk_id}")
            
            # Create RetrievedChunk
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                score=score,
                metadata=doc.metadata if doc.metadata else None,
            )
            chunks.append(chunk)
        
        logger.info(f"Returning {len(chunks)} results after threshold filtering")
        return chunks
