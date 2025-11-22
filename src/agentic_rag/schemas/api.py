"""API request/response schemas for FastAPI endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""
    
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Role of the message sender"
    )
    content: str = Field(
        ...,
        description="Content of the message"
    )


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    
    messages: list[ChatMessage] = Field(
        ...,
        description="Conversation history including the current user message",
        min_length=1
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    
    response: str = Field(
        ...,
        description="Assistant's response message"
    )
