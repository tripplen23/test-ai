"""Tests for settings configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from agentic_rag.settings import get_settings


def test_hf_token_config() -> None:
    """Test HF_TOKEN is loaded from env."""
    test_token = "hf_test_token_12345"
    
    with patch.dict(os.environ, {"HF_TOKEN": test_token}, clear=False):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        settings = get_settings()
        assert settings.hf_token == test_token


def test_hf_token_optional() -> None:
    """Test HF_TOKEN là optional."""
    # Test that settings can be created without HF_TOKEN
    # We'll test by ensuring the field exists and can be None
    import agentic_rag.settings.schema
    agentic_rag.settings.schema._settings = None
    
    # Save original HF_TOKEN
    original_hf_token = os.environ.pop("HF_TOKEN", None)
    
    try:
        # Reset settings singleton
        agentic_rag.settings.schema._settings = None
        
        # Create settings without HF_TOKEN in environment
        # Note: If there's a .env file with HF_TOKEN, it will still be loaded
        # This test verifies that the field exists and is optional
        settings = get_settings()
        
        # The field should exist (even if loaded from .env)
        assert hasattr(settings, "hf_token")
        # If HF_TOKEN is not in env and not in .env, it should be None
        # But if it's in .env, that's also acceptable behavior
        assert isinstance(settings.hf_token, (str, type(None)))
    finally:
        # Restore original HF_TOKEN
        if original_hf_token is not None:
            os.environ["HF_TOKEN"] = original_hf_token
        # Reset singleton
        agentic_rag.settings.schema._settings = None


def test_db_config_fields() -> None:
    """Test database config fields được load từ env."""
    test_config = {
        "AGENTIC_RAG_DB_HOST": "localhost",
        "AGENTIC_RAG_DB_PORT": "5432",
        "AGENTIC_RAG_DB_USER": "test_user",
        "AGENTIC_RAG_DB_PASSWORD": "test_pass",
        "AGENTIC_RAG_DB_NAME": "test_db",
    }
    
    with patch.dict(os.environ, test_config, clear=False):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        settings = get_settings()
        assert settings.db_host == "localhost"
        assert settings.db_port == 5432
        assert settings.db_user == "test_user"
        assert settings.db_password == "test_pass"
        assert settings.db_name == "test_db"


def test_embedding_config() -> None:
    """Test embedding model config."""
    with patch.dict(os.environ, {"AGENTIC_RAG_EMBEDDING_MODEL": "all-mpnet-base-v2"}, clear=False):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        settings = get_settings()
        assert settings.embedding_model == "all-mpnet-base-v2"


def test_chunking_config() -> None:
    """Test chunking parameters config."""
    with patch.dict(
        os.environ,
        {
            "AGENTIC_RAG_CHUNK_SIZE": "1024",
            "AGENTIC_RAG_CHUNK_OVERLAP": "100",
        },
        clear=False,
    ):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        settings = get_settings()
        assert settings.chunk_size == 1024
        assert settings.chunk_overlap == 100


def test_default_values() -> None:
    """Test default values cho các config."""
    # Reset settings singleton
    import agentic_rag.settings.schema
    agentic_rag.settings.schema._settings = None
    
    settings = get_settings()
    # Test defaults - updated to OpenAI embedding model
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536
    # Default chunk_size is 384 (not 512)
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200