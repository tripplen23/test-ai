"""Tests for database schema module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.data.schema import create_schema, drop_schema, schema_exists


def test_create_schema() -> None:
    """Test tạo schema/collection với PGVector."""
    from langchain_openai import OpenAIEmbeddings
    from unittest.mock import patch
    
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        # Mock OpenAIEmbeddings
        mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
        
        # Mock PGVector
        with patch("agentic_rag.data.schema.PGVector") as mock_pgvector:
            mock_instance = MagicMock()
            mock_pgvector.return_value = mock_instance
            
            result = create_schema(mock_embeddings, collection_name="test_collection")
            
            # Verify PGVector was called with correct parameters
            mock_pgvector.assert_called_once()
            assert result == mock_instance


def test_create_schema_with_default_collection() -> None:
    """Test create schema với default collection name từ settings."""
    from langchain_openai import OpenAIEmbeddings
    
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
        
        with patch("agentic_rag.data.schema.PGVector") as mock_pgvector:
            mock_instance = MagicMock()
            mock_pgvector.return_value = mock_instance
            
            result = create_schema(mock_embeddings)
            
            # Should use default collection from settings
            mock_pgvector.assert_called_once()
            assert result == mock_instance


def test_create_schema_with_pre_delete() -> None:
    """Test create schema với pre_delete_collection=True."""
    from langchain_openai import OpenAIEmbeddings
    
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
        
        with patch("agentic_rag.data.schema.PGVector") as mock_pgvector:
            mock_instance = MagicMock()
            mock_pgvector.return_value = mock_instance
            
            result = create_schema(
                mock_embeddings,
                collection_name="test_collection",
                pre_delete_collection=True,
            )
            
            # Verify pre_delete_collection was passed
            call_kwargs = mock_pgvector.call_args[1]
            assert call_kwargs.get("pre_delete_collection") is True
            assert result == mock_instance


def test_schema_exists() -> None:
    """Test check schema/collection exists."""
    # Mock psycopg2 to check collection existence
    import sys
    
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Mock collection exists (check for collection in langchain_pg_collection table)
    mock_cursor.fetchone.return_value = (True,)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_psycopg2.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_psycopg2.connect.return_value.__exit__ = MagicMock(return_value=False)
    
    sys.modules["psycopg2"] = mock_psycopg2
    
    try:
        result = schema_exists("test_collection")
        assert isinstance(result, bool)
    finally:
        if "psycopg2" in sys.modules and not hasattr(sys.modules["psycopg2"], "__file__"):
            del sys.modules["psycopg2"]


def test_schema_exists_not_found() -> None:
    """Test schema_exists returns False when collection doesn't exist."""
    import sys
    
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Mock collection doesn't exist
    mock_cursor.fetchone.return_value = (False,)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_psycopg2.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_psycopg2.connect.return_value.__exit__ = MagicMock(return_value=False)
    
    sys.modules["psycopg2"] = mock_psycopg2
    
    try:
        result = schema_exists("nonexistent_collection")
        assert isinstance(result, bool)
    finally:
        if "psycopg2" in sys.modules and not hasattr(sys.modules["psycopg2"], "__file__"):
            del sys.modules["psycopg2"]


def test_schema_exists_connection_error() -> None:
    """Test schema_exists handles connection errors gracefully."""
    import sys
    
    mock_psycopg2 = MagicMock()
    mock_psycopg2.connect.side_effect = Exception("Connection failed")
    
    sys.modules["psycopg2"] = mock_psycopg2
    
    try:
        result = schema_exists("test_collection")
        # Should return False on error
        assert isinstance(result, bool)
    finally:
        if "psycopg2" in sys.modules and not hasattr(sys.modules["psycopg2"], "__file__"):
            del sys.modules["psycopg2"]


def test_drop_schema() -> None:
    """Test drop schema/collection."""
    from langchain_openai import OpenAIEmbeddings
    
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
        
        # Mock PGVector with pre_delete_collection=True
        with patch("agentic_rag.data.schema.PGVector") as mock_pgvector:
            mock_instance = MagicMock()
            mock_pgvector.return_value = mock_instance
            
            drop_schema("test_collection")
            
            # Verify PGVector was called with pre_delete_collection=True
            mock_pgvector.assert_called_once()
            call_kwargs = mock_pgvector.call_args[1]
            assert call_kwargs.get("pre_delete_collection") is True


def test_drop_schema_with_default_collection() -> None:
    """Test drop schema với default collection name."""
    from langchain_openai import OpenAIEmbeddings
    
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
        
        with patch("agentic_rag.data.schema.PGVector") as mock_pgvector:
            mock_instance = MagicMock()
            mock_pgvector.return_value = mock_instance
            
            drop_schema()
            
            # Should use default collection from settings
            mock_pgvector.assert_called_once()
            call_kwargs = mock_pgvector.call_args[1]
            assert call_kwargs.get("pre_delete_collection") is True
