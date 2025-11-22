"""Tests for database connection module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.data.db import (
    get_connection_string,
    get_connection_string_with_schema,
    verify_pgvector_extension,
)


def test_get_connection_string() -> None:
    """Test create connection string from settings."""
    from agentic_rag.settings import get_settings

    settings = get_settings()
    
    assert hasattr(settings, "db_host")
    assert hasattr(settings, "db_port")
    assert hasattr(settings, "db_user")
    assert hasattr(settings, "db_password")
    assert hasattr(settings, "db_name")
    
    # Get connection string
    connection_string = get_connection_string()
    
    # Verify format
    assert "postgresql://" in connection_string
    assert settings.db_user in connection_string
    assert settings.db_host in connection_string
    assert str(settings.db_port) in connection_string
    assert settings.db_name in connection_string


def test_get_connection_string_with_schema() -> None:
    """Test connection string with schema."""
    schema_name = "test_schema"
    conn_str = get_connection_string_with_schema(schema_name)
    
    assert "postgresql://" in conn_str
    assert schema_name in conn_str or "search_path" in conn_str
    
    conn_str_no_schema = get_connection_string_with_schema(None)
    assert conn_str_no_schema == get_connection_string()


def test_get_connection_string_custom_settings() -> None:
    """Test connection string with custom settings."""
    with patch("agentic_rag.data.db.get_settings") as mock_get_settings:
        mock_settings = MagicMock()
        mock_settings.db_user = "test_user"
        mock_settings.db_password = "test_pass"
        mock_settings.db_host = "test_host"
        mock_settings.db_port = 5433
        mock_settings.db_name = "test_db"
        mock_get_settings.return_value = mock_settings
        
        conn_str = get_connection_string()
        assert "test_user" in conn_str
        assert "test_pass" in conn_str
        assert "test_host" in conn_str
        assert "5433" in conn_str
        assert "test_db" in conn_str


def test_verify_pgvector_extension() -> None:
    result = verify_pgvector_extension()
    assert isinstance(result, bool)


def test_verify_pgvector_extension_with_connection() -> None:
    """Test verify pgvector extension với actual connection."""
    # Mock psycopg2 connection to test extension check
    # psycopg2 is imported inside the function, so we need to mock it at import time
    import sys
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (True,)  # Extension exists
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_psycopg2.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_psycopg2.connect.return_value.__exit__ = MagicMock(return_value=False)
    
    # Temporarily add psycopg2 to sys.modules
    sys.modules["psycopg2"] = mock_psycopg2
    
    try:
        result = verify_pgvector_extension()
        assert result is True
    finally:
        # Clean up
        if "psycopg2" in sys.modules and not hasattr(sys.modules["psycopg2"], "__file__"):
            del sys.modules["psycopg2"]


def test_verify_pgvector_extension_connection_error() -> None:
    """Test verify pgvector extension handles connection errors."""
    # Mock connection error
    import sys
    mock_psycopg2 = MagicMock()
    mock_psycopg2.connect.side_effect = Exception("Connection failed")
    
    # Temporarily add psycopg2 to sys.modules
    sys.modules["psycopg2"] = mock_psycopg2
    
    try:
        # Should handle error gracefully
        result = verify_pgvector_extension()
        # Should return True (graceful handling)
        assert isinstance(result, bool)
    finally:
        # Clean up
        if "psycopg2" in sys.modules and not hasattr(sys.modules["psycopg2"], "__file__"):
            del sys.modules["psycopg2"]


def test_connection_string_format() -> None:
    """Test connection string format is correct."""
    conn_str = get_connection_string()
    
    # Should start with postgresql://
    assert conn_str.startswith("postgresql://")
    
    # Should contain all required components
    parts = conn_str.replace("postgresql://", "").split("/")
    assert len(parts) == 2
    
    # Should have @ separator for credentials
    assert "@" in parts[0]

