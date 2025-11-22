"""FastAPI application for RAG agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.agent import AgentController
from agentic_rag.agent.types import Message, Role
from agentic_rag.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

# Global agent instance
_agent: AgentController | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global _agent
    
    # Startup: Initialize agent
    logger.info("🚀 Initializing AgentController...")
    _agent = AgentController()
    logger.info("✅ AgentController initialized")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")
    _agent = None


# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="REST API for WordPress Q&A Agent powered by LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_initialized": _agent is not None
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the RAG agent.
    
    Args:
        request: Chat request with conversation history
        
    Returns:
        ChatResponse with assistant's message
        
    Raises:
        HTTPException: If agent is not initialized or processing fails
    """
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # Convert API messages to internal Message format
        history = []
        for msg in request.messages:
            role = Role.USER if msg.role == "user" else Role.ASSISTANT
            history.append(Message(role=role, content=msg.content))
        
        # Run agent
        logger.info(f"💬 Processing chat request with {len(history)} messages")
        response_message = _agent.run(history)
        
        return ChatResponse(response=response_message.content)
        
    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )