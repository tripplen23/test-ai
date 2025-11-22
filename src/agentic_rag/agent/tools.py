"""Tools for the RAG agent."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool
from langsmith import traceable
from tavily import TavilyClient

from agentic_rag.retrieval.retriever import PGVectorRetriever
from agentic_rag.retrieval.reranker import CrossEncoderReranker
from agentic_rag.retrieval.schemas import Query
from agentic_rag.settings import get_settings

logger = logging.getLogger(__name__)


@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    settings = get_settings()
    if not settings.tavily_api_key:
        return "TAVILY_API_KEY not configured."
        
    try:
        tavily = TavilyClient(api_key=settings.tavily_api_key)
        result = tavily.search(query=query, search_depth="basic")

        # Extract and format the results from Tavily response
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")

            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"WEB_ERROR::{e}"


@tool
@traceable(name="rag_search_tool")
def rag_search_tool(query: str) -> str:
    """Top-3 chunks from KB (empty string if none)"""
    try:
        # Initialize retriever and reranker
        retriever = PGVectorRetriever()
        reranker = CrossEncoderReranker()
        
        # Create query object
        query_obj = Query(text=query)
        
        # 1. Retrieve more candidates (e.g., top 10)
        candidates = retriever.search(query_obj, k=10)
        
        if not candidates:
            return ""
            
        # 2. Rerank to get top 3
        top_chunks = reranker.rerank(query_obj, candidates, k=3)
        
        return "\n\n".join(chunk.text for chunk in top_chunks) if top_chunks else ""
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"RAG_ERROR::{e}"