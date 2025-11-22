"""Tests for OpenAI configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

from agentic_rag.settings import get_settings


def test_openai_api_key_config() -> None:
    """Test OPENAI_API_KEY được load từ env."""
    test_key = "sk-test-key-12345"
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": test_key}, clear=False):
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        settings = get_settings()
        assert settings.openai_api_key == test_key


def test_openai_api_key_optional() -> None:
    """Test OPENAI_API_KEY là optional."""
    # Save original OPENAI_API_KEY if it exists
    original_key = os.environ.get("OPENAI_API_KEY")
    
    # Remove OPENAI_API_KEY from environment for this test
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    try:
        # Reset settings singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None
        
        settings = get_settings()
        # Field should exist and can be None
        assert hasattr(settings, "openai_api_key")
        assert isinstance(settings.openai_api_key, (str, type(None)))
    finally:
        # Restore original OPENAI_API_KEY if it existed
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
        # Reset singleton
        import agentic_rag.settings.schema
        agentic_rag.settings.schema._settings = None

