"""LangGraph implementation for Advanced RAG agent."""

from __future__ import annotations

import logging
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from agentic_rag.agent.prompts import ROUTER_PROMPT, JUDGE_PROMPT, get_system_prompt
from agentic_rag.agent.tools import rag_search_tool, web_search_tool
from agentic_rag.settings import get_settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    context: str
    router_decision: str
    judge_decision: str


def router(state: AgentState) -> dict:
    """Route the user query."""
    settings = get_settings()
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content
    
    logger.info(f"🧭 Routing query: {query[:50]}...")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )
    
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=query)
    ])
    
    decision = response.content.strip().lower()
    # Fallback if LLM returns something else
    if "needs_kb" in decision:
        decision = "needs_kb"
    elif "greeting" in decision:
        decision = "greeting"
    else:
        decision = "direct_answer"
        
    logger.info(f"🧭 Router decision: {decision}")
    return {"router_decision": decision, "query": query}


def rag_lookup(state: AgentState) -> dict:
    """Retrieve from Knowledge Base."""
    query = state["query"]
    logger.info("📚 Searching Knowledge Base...")
    
    context = rag_search_tool.invoke(query)
    
    return {"context": context}


def judge(state: AgentState) -> dict:
    """Judge if context is sufficient."""
    settings = get_settings()
    query = state["query"]
    context = state.get("context", "")
    
    if not context:
        logger.info("⚖️ No context found in KB. Judging: insufficient.")
        return {"judge_decision": "no"}
    
    logger.info("⚖️ Judging context sufficiency...")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )
    
    prompt = JUDGE_PROMPT.format(context=context, question=query)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    decision = response.content.strip().lower()
    if "yes" in decision:
        decision = "yes"
    else:
        decision = "no"
        
    logger.info(f"⚖️ Judge decision: {decision}")
    return {"judge_decision": decision}


def web_search(state: AgentState) -> dict:
    """Search the web."""
    query = state["query"]
    logger.info("🌐 Searching Web...")
    
    context = web_search_tool.invoke(query)
    
    return {"context": context}


def answer(state: AgentState) -> dict:
    """Generate final answer."""
    settings = get_settings()
    messages = state["messages"]
    query = state["query"]
    context = state.get("context", "")
    router_decision = state.get("router_decision", "direct_answer")
    
    logger.info("✍️ Generating answer...")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=settings.openai_api_key,
    )
    
    if router_decision == "direct_answer":
        # Just answer directly without context
        response = llm.invoke(messages)
    else:
        # Answer with context
        system_prompt = get_system_prompt()
        user_prompt = f"""Context:
{context}

Question: {query}

Answer the question based on the context provided."""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
    return {"messages": [response]}


def greeting(state: AgentState) -> dict:
    """Handle greetings."""
    logger.info("👋 Generating greeting...")
    return {"messages": [AIMessage(content="Hello! How can I help you with WordPress today?")]}


def build_graph():
    """Build the Advanced LangGraph agent."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("rag_lookup", rag_lookup)
    workflow.add_node("judge", judge)
    workflow.add_node("web_search", web_search)
    workflow.add_node("answer", answer)
    workflow.add_node("greeting", greeting)
    
    # Add edges
    workflow.set_entry_point("router")
    
    # Conditional edges from router
    def route_decision(state):
        return state["router_decision"]
    
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "direct_answer": "answer",
            "greeting": "greeting",
            "needs_kb": "rag_lookup",
        }
    )
    
    # Edge from rag_lookup to judge
    workflow.add_edge("rag_lookup", "judge")
    
    # Conditional edges from judge
    def judge_decision(state):
        return state["judge_decision"]
    
    workflow.add_conditional_edges(
        "judge",
        judge_decision,
        {
            "yes": "answer",
            "no": "web_search",
        }
    )
    
    # Edge from web_search to answer
    workflow.add_edge("web_search", "answer")
    
    # Edges to END
    workflow.add_edge("answer", END)
    workflow.add_edge("greeting", END)
    
    # Compile
    return workflow.compile()

# Export the graph for LangGraph Studio
graph = build_graph()
