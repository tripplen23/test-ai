"""Agent controller implementation for RAG system with LangGraph integration."""

from __future__ import annotations

import logging
from typing import Sequence, Any

from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

from agentic_rag.agent.graph import graph
from agentic_rag.settings import get_settings

from .controller import BaseAgentController
from .types import Message, PlanStep, Role

logger = logging.getLogger(__name__)


class AgentController(BaseAgentController):
    """RAG agent controller using LangGraph."""

    def __init__(self):
        """Initialize the agent controller with LangGraph."""
        self.settings = get_settings()
        self.graph = graph
        logger.info("AgentController initialized with LangGraph")

    def plan(self, history: Sequence[Message]) -> Sequence[PlanStep]:
        """
        Produce a plan (kept for interface compatibility).
        """
        return [
            PlanStep(name="retrieve", depends_on=[]),
            PlanStep(name="generate", depends_on=["retrieve"]),
        ]

    @traceable(name="agent_run")
    def run(self, history: Sequence[Message]) -> Message:
        """
        Execute the RAG pipeline using LangGraph.

        Args:
            history: Conversation history

        Returns:
            Assistant's response message
        """
        # Convert history to LangChain messages
        lc_messages = []
        for msg in history:
            if msg.role == Role.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
        
        if not lc_messages:
            return Message(
                role=Role.ASSISTANT,
                content="Please ask a question.",
            )
            
        # Invoke graph
        logger.info("🚀 Invoking LangGraph...")
        inputs = {"messages": lc_messages}
        
        try:
            # Run the graph
            final_state = self.graph.invoke(inputs)
            
            # Extract response
            last_message = final_state["messages"][-1]
            response_content = last_message.content
            
            # Log retrieved docs (if available in state)
            if "documents" in final_state:
                docs = final_state["documents"]
                logger.info(f"✅ Graph finished. Used {len(docs)} documents.")
            
            return Message(role=Role.ASSISTANT, content=response_content)
            
        except Exception as e:
            logger.error(f"❌ Error in LangGraph execution: {e}", exc_info=True)
            return Message(
                role=Role.ASSISTANT,
                content=f"I encountered an error: {str(e)}"
            )

    def serve(self) -> None:
        """Run interactive CLI for the agent."""
        print("=" * 60)
        print("WordPress QA Agent (LangGraph Powered 🦜🕸️)")
        print("=" * 60)
        print("Ask questions about WordPress. Type 'exit' or 'quit' to stop.")
        print("=" * 60)
        print()
        
        conversation_history: list[Message] = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Add user message to history
                user_message = Message(role=Role.USER, content=user_input)
                conversation_history.append(user_message)
                
                # Get response
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.run(conversation_history)
                print(response.content)
                
                # Add assistant response to history
                conversation_history.append(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in serve loop: {e}")
                print(f"\n❌ Error: {e}")