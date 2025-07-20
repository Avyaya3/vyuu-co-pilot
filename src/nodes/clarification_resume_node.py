"""
Clarification Resume Node for LangGraph Intent Orchestration System.

This node handles resuming the clarification flow when a user responds to a
clarification question. It processes the user response and prepares the state
for continued clarification processing.
"""

import logging
from typing import Dict, Any
from src.schemas.state_schemas import ClarificationState, MessageManager

logger = logging.getLogger(__name__)


async def clarification_resume_node(state: ClarificationState) -> ClarificationState:
    """
    Resume clarification processing after receiving user response.
    
    This node is called when the clarification subgraph resumes after a user
    has responded to a clarification question. It processes the user response
    and prepares the state for continued clarification.
    
    Args:
        state: ClarificationState with user response in user_input
        
    Returns:
        Updated ClarificationState ready for response processing
    """
    node_name = "clarification_resume_node"
    
    try:
        logger.info(f"[ClarificationResume] Resuming clarification for session {state.session_id[:8]}")
        
        # Get the user response from the current user_input
        user_response = state.user_input
        
        # Find the last question that was asked (should be the pending question)
        last_question = state.last_question_asked or state.pending_question
        
        if not last_question:
            logger.warning(f"[ClarificationResume] No last question found for session {state.session_id[:8]}")
            last_question = "Unknown question"
        
        # Update the last clarification history entry with the user response
        updated_history = state.clarification_history.copy()
        if updated_history:
            # Update the last entry with the user response
            last_entry = updated_history[-1]
            last_entry["user_response"] = user_response
            last_entry["response_timestamp"] = "now"
            updated_history[-1] = last_entry
        
        # Add system message for tracking
        state = MessageManager.add_system_message(
            state,
            f"Resuming clarification processing - user responded: '{user_response[:50]}...'",
            node_name
        )
        
        # Update state to resume processing
        updated_state = state.model_copy(update={
            "waiting_for_response": False,
            "clarification_phase": "processing",
            "pending_question": None,  # Clear the pending question
            "clarification_history": updated_history,
            "metadata": {
                **state.metadata,
                "clarification_status": "processing_user_response",
                "resumed_from_pause": True,
                "last_user_response": user_response,
                "last_question_asked": last_question
            }
        })
        
        logger.info(f"[ClarificationResume] Successfully resumed clarification for session {state.session_id[:8]}")
        return updated_state
        
    except Exception as e:
        logger.error(f"[ClarificationResume] Failed to resume clarification for session {state.session_id[:8]}: {e}")
        
        # Create fallback state
        fallback_state = state.model_copy(update={
            "waiting_for_response": False,
            "clarification_phase": "processing",
            "pending_question": None,
            "metadata": {
                **state.metadata,
                "clarification_status": "error_during_resume",
                "error": f"Resume failed: {str(e)}",
                "resumed_from_pause": True
            }
        })
        
        return fallback_state 