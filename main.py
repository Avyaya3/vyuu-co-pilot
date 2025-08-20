"""
Main entry point for the LangGraph-based intent orchestration system.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
from uuid import uuid4

from src.orchestrator import MainOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def invoke_graph(
    user_input: str = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_history: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for LangGraph Studio to invoke the orchestrator graph.
    
    This function supports both chat-style input (simple message string) and
    structured input (YAML/JSON with all parameters).
    
    Args:
        user_input: The user's message (can be passed directly or via kwargs)
        user_id: Optional user ID for session tracking
        session_id: Optional session ID for conversation continuity
        conversation_history: Optional list of previous messages
        **kwargs: Additional parameters passed from Studio
        
    Returns:
        Dictionary containing the graph execution result
    """
    try:
        # Handle different input formats
        if user_input is None:
            # Check if input was passed as a string in kwargs (Studio chat mode)
            if isinstance(kwargs.get('input'), str):
                user_input = kwargs['input']
            elif 'user_input' in kwargs:
                user_input = kwargs['user_input']
            elif 'message' in kwargs:
                user_input = kwargs['message']
            elif 'text' in kwargs:
                user_input = kwargs['text']
            elif 'query' in kwargs:
                user_input = kwargs['query']
            else:
                # Try to extract from the first positional argument or kwargs
                for key, value in kwargs.items():
                    if isinstance(value, str) and key not in ['user_id', 'session_id', 'conversation_history']:
                        user_input = value
                        break
        
        if not user_input:
            raise ValueError("No user input provided. Please provide a message to process.")
        
        # Use test user for Studio if no user_id provided
        if not user_id:
            user_id = os.getenv("STUDIO_TEST_USER_ID")
            if not user_id:
                raise ValueError("STUDIO_TEST_USER_ID environment variable must be set for Studio")
            logger.info(f"[Studio] Using test user: {user_id[:8]}...")
        
        # Create orchestrator instance with in-memory sessions for Studio
        orchestrator = MainOrchestrator(use_database=False)
        
        logger.info(f"[Studio] Invoking orchestrator for session {session_id[:8] if session_id else 'new'} with input: '{user_input[:50]}...'")
        
        # Call the orchestrator (this handles user isolation and session management)
        result = await orchestrator.process_user_message(
            user_input=user_input,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history
        )
        
        # Add Studio-specific metadata
        result["metadata"] = {
            **result.get("metadata", {}),
            "source": "langgraph_studio",
            "studio_session": True,
            "input_format": "chat" if isinstance(kwargs.get('input'), str) else "structured"
        }
        
        logger.info(f"[Studio] Orchestrator execution completed for session {result.get('session_id', 'unknown')[:8]}")
        return result
        
    except Exception as e:
        logger.error(f"[Studio] Error invoking orchestrator: {e}")
        return {
            "error": str(e),
            "session_id": session_id or "unknown",
            "user_id": user_id or "unknown",
            "response": f"Error: {str(e)}",
            "conversation_history": conversation_history or [],
            "status": "error"
        }


async def chat_endpoint(message: str, **kwargs) -> Dict[str, Any]:
    """
    Chat-style endpoint for LangGraph Studio that accepts simple message strings.
    
    This function provides a simple interface for testing user queries in Studio.
    
    Args:
        message: The user's message/query
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing the response and metadata
    """
    logger.info(f"[Studio Chat] Processing message: '{message[:50]}...'")
    
    # Call the main invoke_graph function with the message
    result = await invoke_graph(user_input=message, **kwargs)
    
    # Format the response for chat interface
    if result.get("status") == "error":
        return {
            "response": f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}",
            "status": "error",
            "session_id": result.get("session_id"),
            "metadata": result.get("metadata", {})
        }
    
    # Extract the main response
    response = result.get("response", "I processed your request but didn't get a specific response.")
    
    return {
        "response": response,
        "status": "success",
        "session_id": result.get("session_id"),
        "conversation_history": result.get("conversation_history", []),
        "metadata": result.get("metadata", {})
    }


def main():
    """
    Main function to initialize and run the LangGraph system.
    """
    print("LangGraph Intent Orchestration System")
    print("Starting application...")
    
    # Test the orchestrator with a simple invocation
    async def test_invocation():
        test_input = "Hello, I need help with my finances"
        result = await invoke_graph(test_input)
        print(f"Test result: {result.get('response', 'No response')}")
        print(f"Session ID: {result.get('session_id', 'No session')}")
        print(f"Status: {result.get('status', 'Unknown')}")
    
    # Run the test
    asyncio.run(test_invocation())


if __name__ == "__main__":
    main() 