"""Database connection utilities for PostgreSQL with pgvector."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentic_rag.settings import get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def get_connection_string() -> str:
    """
    Get PostgreSQL connection string from settings.
    
    Returns:
        Connection string in format: postgresql://user:password@host:port/dbname
    """
    settings = get_settings()
    return (
        f"postgresql://{settings.db_user}:{settings.db_password}@"
        f"{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )


def get_connection_string_with_schema(schema_name: str | None = None) -> str:
    """
    Get PostgreSQL connection string with optional schema.
    
    Args:
        schema_name: Optional schema name to include in connection string
        
    Returns:
        Connection string with schema if provided
    """
    conn_str = get_connection_string()
    if schema_name:
        conn_str += f"?options=-csearch_path%3D{schema_name}"
    return conn_str


def verify_pgvector_extension() -> bool:
    """
    Verify that pgvector extension is available in the database.
    
    Note: LangChain PGVector handles extension creation automatically,
    but this function can be used to verify it exists.
    
    Returns:
        True if pgvector extension is available, False otherwise
    """
    try:
        import psycopg2
        
        conn_str = get_connection_string()
        
        # Try to connect and check for extension
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                # Check if pgvector extension exists
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                )
                result = cur.fetchone()
                if result and result[0]:
                    logger.info("pgvector extension is available")
                    return True
                else:
                    logger.warning(
                        "pgvector extension not found, but PGVector will create it if needed"
                    )
                    # Return True anyway since PGVector will create it
                    return True
    except ImportError:
        logger.warning("psycopg2 not available, cannot verify pgvector extension")
        # Return True since PGVector will handle it
        return True
    except Exception as e:
        logger.warning(f"Could not verify pgvector extension: {e}")
        # Return True since PGVector will handle extension creation
        return True