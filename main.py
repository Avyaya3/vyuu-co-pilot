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
    user_input: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_history: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for LangGraph Studio to invoke the orchestrator graph.
    
    Args:
        user_input: The user's message
        user_id: Optional user ID for session tracking
        session_id: Optional session ID for conversation continuity
        conversation_history: Optional list of previous messages
        **kwargs: Additional parameters passed from Studio
        
    Returns:
        Dictionary containing the graph execution result
    """
    try:
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
            "studio_session": True
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