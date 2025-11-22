"""Tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from agentic_rag.agent.types import Message, Role


@pytest.fixture
def mock_agent():
    """Mock AgentController for testing."""
    agent = Mock()
    agent.run.return_value = Message(
        role=Role.ASSISTANT,
        content="This is a test response from the agent."
    )
    return agent


@pytest.fixture
def client(mock_agent):
    """Create TestClient with mocked agent."""
    with patch("agentic_rag.api._agent", mock_agent):
        from agentic_rag.api import app
        yield TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "agent_initialized" in data


def test_chat_endpoint_success(client, mock_agent):
    """Test successful chat request."""
    request_data = {
        "messages": [
            {"role": "user", "content": "What is WordPress?"}
        ]
    }
    
    response = client.post("/chat", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] == "This is a test response from the agent."
    # No metadata field anymore
    
    # Verify agent.run was called
    mock_agent.run.assert_called_once()
    call_args = mock_agent.run.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].role == Role.USER
    assert call_args[0].content == "What is WordPress?"


def test_chat_endpoint_with_history(client, mock_agent):
    """Test chat with conversation history."""
    request_data = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is WordPress?"}
        ]
    }
    
    response = client.post("/chat", json=request_data)
    
    assert response.status_code == 200
    
    # Verify history was passed correctly
    call_args = mock_agent.run.call_args[0][0]
    assert len(call_args) == 3
    assert call_args[0].role == Role.USER
    assert call_args[1].role == Role.ASSISTANT
    assert call_args[2].role == Role.USER


def test_chat_endpoint_empty_messages(client):
    """Test chat with empty messages."""
    request_data = {"messages": []}
    
    response = client.post("/chat", json=request_data)
    
    # Should fail validation (min_length=1)
    assert response.status_code == 422


def test_chat_endpoint_invalid_role(client):
    """Test chat with invalid role."""
    request_data = {
        "messages": [
            {"role": "invalid", "content": "Hello"}
        ]
    }
    
    response = client.post("/chat", json=request_data)
    
    # Should fail validation
    assert response.status_code == 422


def test_chat_endpoint_agent_error(client, mock_agent):
    """Test chat when agent raises an error."""
    mock_agent.run.side_effect = Exception("Agent error")
    
    request_data = {
        "messages": [
            {"role": "user", "content": "What is WordPress?"}
        ]
    }
    
    response = client.post("/chat", json=request_data)
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Agent error" in data["detail"]


def test_openapi_docs(client):
    """Test that OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_schema = response.json()
    assert openapi_schema["info"]["title"] == "Agentic RAG API"
    assert "/chat" in openapi_schema["paths"]
    assert "/health" in openapi_schema["paths"]
