"""Database schema management for pgvector."""

from __future__ import annotations

import logging

from langchain_community.vectorstores import PGVector
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from agentic_rag.data.db import get_connection_string
from agentic_rag.settings import get_settings

logger = logging.getLogger(__name__)

def create_schema(
    embeddings: Embeddings,
    collection_name: str | None = None,
    pre_delete_collection: bool = False,
) -> PGVector:
    settings = get_settings()
    collection = collection_name or settings.vector_store.collection
    connection_string = get_connection_string()
    
    return PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name=collection,
        pre_delete_collection=pre_delete_collection,
    )


def schema_exists(collection_name: str | None = None) -> bool:
    try:
        import psycopg2
        
        settings = get_settings()
        collection = collection_name or settings.vector_store.collection
        connection_string = get_connection_string()
        
        # Connect to database and check if collection exists
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Check if langchain_pg_collection table exists first
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'langchain_pg_collection'
                    );
                    """
                )
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    # Table doesn't exist, so collection doesn't exist
                    return False
                
                # Check if collection exists in langchain_pg_collection table
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM langchain_pg_collection 
                        WHERE name = %s
                    );
                    """,
                    (collection,),
                )
                result = cur.fetchone()
                return result[0] if result else False
                
    except ImportError:
        logger.warning("psycopg2 not available, cannot check schema existence")
        # Return True as PGVector will create if needed
        return True
    except Exception as e:
        logger.warning(f"Could not check schema existence: {e}")
        # Return False on error to be safe
        return False


def drop_schema(collection_name: str | None = None) -> None:
    settings = get_settings()
    collection = collection_name or settings.vector_store.collection
    
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set, cannot drop schema")
        return
    
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    
    create_schema(
        embeddings=embeddings,
        collection_name=collection,
        pre_delete_collection=True,
    )
    
    logger.info(f"Dropped collection: {collection}")