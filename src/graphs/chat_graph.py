"""
Chat-style graph entry point for LangGraph Studio.
This provides a simple interface for testing user queries.
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolExecutor
import logging

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """State for chat-style interactions."""
    message: str
    response: str
    session_id: str
    user_id: str
    conversation_history: list
    metadata: Dict[str, Any]


async def chat_processor(state: ChatState) -> ChatState:
    """
    Process a chat message using the main orchestrator.
    
    Args:
        state: Current chat state
        
    Returns:
        Updated state with response
    """
    from main import invoke_graph
    
    try:
        # Call the main orchestrator
        result = await invoke_graph(
            user_input=state["message"],
            user_id=state["user_id"],
            session_id=state["session_id"],
            conversation_history=state["conversation_history"]
        )
        
        return {
            **state,
            "response": result.get("response", "I processed your request but didn't get a specific response."),
            "session_id": result.get("session_id", state["session_id"]),
            "conversation_history": result.get("conversation_history", state["conversation_history"]),
            "metadata": result.get("metadata", {})
        }
        
    except Exception as e:
        logger.error(f"Error in chat processor: {e}")
        return {
            **state,
            "response": f"Sorry, I encountered an error: {str(e)}",
            "metadata": {"error": str(e)}
        }


# Create the chat graph
chat_graph = StateGraph(ChatState)

# Add the chat processor node
chat_graph.add_node("chat_processor", chat_processor)

# Set the entry and exit points
chat_graph.set_entry_point("chat_processor")
chat_graph.set_finish_point("chat_processor")

# Compile the graph
chat_graph_compiled = chat_graph.compile()

# Export the compiled graph
__all__ = ["chat_graph_compiled"] 