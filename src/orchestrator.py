"""
Main Orchestrator Class for LangGraph Intent Orchestration System.

This module provides the high-level interface for the complete intent orchestration
system. It manages graph execution, session handling, error recovery, and provides
a clean API for external consumers (like FastAPI endpoints).

Features:
- Complete graph orchestration and execution
- In-memory session management (Phase 1)
- Comprehensive error handling and recovery
- Structured logging and monitoring
- Clean API interface for chat applications
- Conversation context preservation
- Performance tracking and metrics
"""

import logging
import time
from typing import Dict, Any, Optional, List
from uuid import uuid4
from datetime import datetime, timezone

from src.schemas.state_schemas import MainState, MessageManager, ConversationContext
from src.graphs.main_orchestrator_graph import main_orchestrator_graph

logger = logging.getLogger(__name__)


class SessionManager:
    """
    In-memory session management for development and testing.
    
    This will be replaced with database-backed session management in Phase 2.
    """
    
    def __init__(self):
        self._sessions: Dict[str, MainState] = {}
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        
    def save_session_state(self, state: MainState) -> bool:
        """
        Save session state to in-memory storage.
        
        Args:
            state: MainState to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._sessions[state.session_id] = state
            self._session_metadata[state.session_id] = {
                "created_at": state.timestamp,
                "last_updated": datetime.now(timezone.utc),
                "message_count": len(state.messages),
                "user_id": state.metadata.get("user_id")
            }
            logger.debug(f"[SessionManager] Saved session {state.session_id[:8]}")
            return True
        except Exception as e:
            logger.error(f"[SessionManager] Failed to save session {state.session_id[:8]}: {e}")
            return False
    
    def load_session_state(self, session_id: str) -> Optional[MainState]:
        """
        Load session state from in-memory storage.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            MainState if found, None otherwise
        """
        try:
            if session_id in self._sessions:
                logger.debug(f"[SessionManager] Loaded session {session_id[:8]}")
                return self._sessions[session_id]
            else:
                logger.debug(f"[SessionManager] Session {session_id[:8]} not found")
                return None
        except Exception as e:
            logger.error(f"[SessionManager] Failed to load session {session_id[:8]}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from in-memory storage.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if session_id in self._sessions:
                del self._sessions[session_id]
                del self._session_metadata[session_id]
                logger.info(f"[SessionManager] Deleted session {session_id[:8]}")
                return True
            return False
        except Exception as e:
            logger.error(f"[SessionManager] Failed to delete session {session_id[:8]}: {e}")
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current sessions.
        
        Returns:
            Dictionary with session statistics
        """
        return {
            "total_sessions": len(self._sessions),
            "session_ids": list(self._sessions.keys()),
            "total_messages": sum(len(state.messages) for state in self._sessions.values())
        }


class MainOrchestrator:
    """
    Main Orchestrator class for the complete intent orchestration system.
    
    This class provides the high-level interface for processing user requests
    through the complete LangGraph workflow with session management and error handling.
    """
    
    def __init__(self, use_database: bool = False):
        """
        Initialize the MainOrchestrator.
        
        Args:
            use_database: Whether to use database session management
        """
        self.graph = main_orchestrator_graph
        self.use_database = use_database
        
        # Initialize session manager
        if use_database:
            try:
                from src.utils.database_session_manager import DatabaseSessionManager
                self.session_manager = DatabaseSessionManager(session_expiry_hours=24)
                logger.info("[MainOrchestrator] Initialized with database session management")
            except Exception as e:
                logger.error(f"[MainOrchestrator] Failed to initialize database session manager: {e}")
                logger.warning("[MainOrchestrator] Falling back to in-memory session management")
                self.session_manager = SessionManager()
                self.use_database = False
        else:
            self.session_manager = SessionManager()
            logger.info("[MainOrchestrator] Initialized with in-memory session management")
    
    async def process_user_message(
        self,
        user_input: str,
        user_id: str = None,
        session_id: str = None,
        conversation_history: List[Dict] = None,
        financial_data: Dict[str, Any] = None ) -> Dict[str, Any]:
        """
        Main API endpoint for processing user messages.
        
        This is the primary method that external consumers (like FastAPI endpoints)
        will call to process user requests through the complete orchestration workflow.
        
        Args:
            user_input: User's message/request
            user_id: Optional user ID for session association
            session_id: Optional existing session ID for conversation continuity
            conversation_history: Optional previous conversation messages
            
        Returns:
            Dictionary with response, session info, and metadata
        """
        start_time = time.time()
        processing_metadata = {
            "start_time": start_time,
            "user_id": user_id,
            "input_length": len(user_input),
            "has_existing_session": session_id is not None
        }
        
        try:
            logger.info(f"[MainOrchestrator] Processing user message - Session: {session_id[:8] if session_id else 'new'}, User: {user_id or 'anonymous'}")
            logger.info(f"[MainOrchestrator] Financial data present: {financial_data is not None}")
            if financial_data:
                logger.info(f"[MainOrchestrator] Financial data keys: {list(financial_data.keys())}")
            else:
                logger.warning("[MainOrchestrator] No financial data provided!")
            
            # Load or create session state
            state = await self._prepare_session_state(
                user_input=user_input,
                user_id=user_id,
                session_id=session_id,
                conversation_history=conversation_history,
                financial_data=financial_data
            )
            
            # Add Supabase JWT token for MCP calls if user_id is available
            if user_id:
                try:
                    from src.utils.auth import get_auth_manager
                    auth_manager = get_auth_manager()
                    supabase_jwt = auth_manager.get_supabase_jwt_for_mcp(user_id)
                    state.metadata["supabase_jwt_token"] = supabase_jwt
                    logger.debug(f"[MainOrchestrator] Added Supabase JWT token for MCP calls for user {user_id}")
                except Exception as e:
                    logger.warning(f"[MainOrchestrator] Failed to create Supabase JWT token for user {user_id}: {e}")
                    # Continue without Supabase JWT - MCP tools will need to handle this gracefully
            
            # Check if this is a response to a clarification question
            if self._is_clarification_response(state, user_input):
                logger.info(f"[MainOrchestrator] Detected clarification response for session {state.session_id[:8]}")
                state = await self._prepare_clarification_resume(state, user_input)
            
            # Process through main graph
            logger.info(f"[MainOrchestrator] Executing main graph for session {state.session_id[:8]}")
            logger.info(f"[MainOrchestrator] State metadata before graph execution: {list(state.metadata.keys())}")
            logger.info(f"[MainOrchestrator] Financial data in state before graph execution: {'financial_data' in state.metadata}")
            raw_result = await self.graph.ainvoke(state)
            
            # Convert result to proper MainState if it's an AddableValuesDict
            if hasattr(raw_result, 'get') and not hasattr(raw_result, 'session_id'):
                # LangGraph returned AddableValuesDict, convert to MainState
                final_state = MainState(**dict(raw_result))
            else:
                final_state = raw_result
            
            # Save session state
            save_success = self.session_manager.save_session_state(final_state)
            if not save_success:
                logger.warning(f"[MainOrchestrator] Failed to save session state for {final_state.session_id[:8]}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_metadata.update({
                "processing_time_seconds": processing_time,
                "final_message_count": len(final_state.messages),
                "session_saved": save_success
            })
            
            # Check if we're paused for clarification
            if self._is_paused_for_clarification(final_state):
                logger.info(f"[MainOrchestrator] Paused for clarification - returning question for session {final_state.session_id[:8]}")
                return self._format_clarification_question_response(final_state, processing_metadata)
            
            # Format successful response
            response = {
                "response": final_state.response,
                "session_id": final_state.session_id,
                "conversation_history": [msg.model_dump() for msg in final_state.messages],
                "status": "success",
                "metadata": {
                    **final_state.metadata,
                    "processing_metadata": processing_metadata
                }
            }
            
            logger.info(f"[MainOrchestrator] Successfully processed message for session {final_state.session_id[:8]} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            # Handle errors gracefully
            processing_time = time.time() - start_time
            error_response = await self._handle_processing_error(
                error=e,
                user_input=user_input,
                session_id=session_id,
                processing_metadata=processing_metadata,
                processing_time=processing_time
            )
            
            logger.error(f"[MainOrchestrator] Error processing message: {e}")
            return error_response
    
    async def _prepare_session_state(
        self,
        user_input: str,
        user_id: str = None,
        session_id: str = None,
        conversation_history: List[Dict] = None,
        financial_data: Dict[str, Any] = None ) -> MainState:
        """
        Prepare the session state for graph processing.
        
        Args:
            user_input: User's message
            user_id: Optional user ID
            session_id: Optional existing session ID
            conversation_history: Optional conversation history
            
        Returns:
            MainState ready for graph processing
        """
        # Load existing session or create new one
        if session_id:
            existing_state = self.session_manager.load_session_state(session_id)
            if existing_state:
                # Update existing session with new user input
                logger.debug(f"[MainOrchestrator] Loaded existing session {session_id[:8]}")
                
                # Update metadata with new financial data if provided
                updated_metadata = {
                    **existing_state.metadata,
                    "turn_count": existing_state.metadata.get("turn_count", 0) + 1
                }
                
                if financial_data:
                    updated_metadata["financial_data"] = financial_data
                    logger.debug(f"[MainOrchestrator] Updated financial data in existing session")
                
                return existing_state.model_copy(update={
                    "user_input": user_input,
                    "timestamp": datetime.now(timezone.utc),
                    "metadata": updated_metadata
                })
            else:
                logger.warning(f"[MainOrchestrator] Session {session_id[:8]} not found, creating new session")
                session_id = None  # Force new session creation
        
        # Create new session
        if not session_id:
            session_id = str(uuid4())
            
        # Prepare messages from conversation history
        messages = []
        if conversation_history:
            # Convert conversation history to Message objects
            for msg in conversation_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(MessageManager._create_message(
                        content=msg["content"],
                        role=msg["role"],
                        node_name=msg.get("node_name", "conversation_history")
                    ))
        
        # Create new MainState
        metadata = {
            "user_id": user_id,
            "session_created": datetime.now(timezone.utc).isoformat(),
            "turn_count": 1,
            "orchestrator_version": "1.0.0"
        }
        
        # Add financial data to metadata if provided
        if financial_data:
            metadata["financial_data"] = financial_data
            logger.info(f"[MainOrchestrator] Added financial data to session metadata")
            logger.info(f"[MainOrchestrator] Financial data keys: {list(financial_data.keys())}")
        else:
            logger.warning("[MainOrchestrator] No financial data to add to metadata")
        
        state = MainState(
            user_input=user_input,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            messages=messages,
            metadata=metadata
        )
        
        logger.info(f"[MainOrchestrator] Created new session {session_id[:8]} for user {user_id or 'anonymous'}")
        return state
    
    async def _handle_processing_error(
        self,
        error: Exception,
        user_input: str,
        session_id: str = None,
        processing_metadata: Dict[str, Any] = None,
        processing_time: float = 0 ) -> Dict[str, Any]:
        """
        Handle processing errors gracefully.
        
        Args:
            error: The exception that occurred
            user_input: Original user input
            session_id: Session ID if available
            processing_metadata: Processing metadata
            processing_time: Time spent processing
            
        Returns:
            Error response dictionary
        """
        error_id = str(uuid4())[:8]
        
        # Create fallback response
        fallback_response = "I apologize, but I encountered an error while processing your request. Please try again."
        
        # Prepare error metadata
        error_metadata = {
            "error_id": error_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "user_input": user_input[:100],  # Truncated for logging
            "session_id": session_id,
            "processing_time_seconds": processing_time,
            **(processing_metadata or {})
        }
        
        # Create error response
        error_response = {
            "response": fallback_response,
            "session_id": session_id or str(uuid4()),
            "conversation_history": [],
            "status": "error",
            "metadata": {
                "error": True,
                "error_metadata": error_metadata
            }
        }
        
        logger.error(f"[MainOrchestrator] Processing error {error_id}: {error}")
        return error_response
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific session.
        
        Args:
            session_id: Session ID to query
            
        Returns:
            Session information dictionary or None if not found
        """
        state = self.session_manager.load_session_state(session_id)
        if state:
            return {
                "session_id": state.session_id,
                "created_at": state.timestamp,
                "message_count": len(state.messages),
                "user_id": state.metadata.get("user_id"),
                "turn_count": state.metadata.get("turn_count", 0),
                "last_intent": state.intent.value if state.intent else None,
                "last_confidence": state.confidence
            }
        return None
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics and health information.
        
        Returns:
            Dictionary with orchestrator statistics
        """
        session_stats = self.session_manager.get_session_stats()
        
        return {
            "status": "healthy",
            "session_management": "in-memory" if not self.use_database else "database",
            "graph_compiled": self.graph is not None,
            "session_stats": session_stats,
            "version": "1.0.0"
        }
    
    async def cleanup_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old sessions.
        
        Args:
            max_age_hours: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        if self.use_database:
            # Convert hours to days for database cleanup
            max_age_days = max_age_hours // 24
            return self.session_manager.cleanup_old_sessions(max_age_days)
        else:
            # TODO: Implement session cleanup based on age for in-memory storage
            logger.info("[MainOrchestrator] Session cleanup not yet implemented for in-memory storage")
            return 0
    
    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get sessions for a specific user.
        
        Args:
            user_id: User ID to query sessions for
            limit: Maximum number of sessions to return
            
        Returns:
            List of session information dictionaries
        """
        if self.use_database:
            return self.session_manager.get_user_sessions(user_id, limit)
        else:
            # For in-memory storage, filter sessions by user_id
            sessions = []
            for session_id, metadata in self.session_manager._session_metadata.items():
                if metadata.get("user_id") == user_id:
                    session_state = self.session_manager.load_session_state(session_id)
                    if session_state:
                        sessions.append({
                            "session_id": session_id,
                            "created_at": metadata.get("created_at"),
                            "last_updated": metadata.get("last_updated"),
                            "message_count": metadata.get("message_count", 0),
                            "last_intent": session_state.intent.value if session_state.intent else None,
                            "last_confidence": session_state.confidence,
                            "is_active": True
                        })
            
            # Sort by last_updated descending and limit
            sessions.sort(key=lambda x: x.get("last_updated", datetime.min), reverse=True)
            return sessions[:limit]
    
    def cleanup_user_sessions(self, user_id: str, keep_recent: int = 10) -> int:
        """
        Clean up old sessions for a specific user, keeping only the most recent ones.
        
        Args:
            user_id: User ID to clean up sessions for
            keep_recent: Number of recent sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        if self.use_database:
            # Get user sessions
            user_sessions = self.get_user_sessions(user_id, limit=1000)  # Get all sessions
            
            if len(user_sessions) <= keep_recent:
                return 0  # No cleanup needed
            
            # Sort by last_updated and get sessions to delete
            user_sessions.sort(key=lambda x: x.get("last_updated", datetime.min), reverse=True)
            sessions_to_delete = user_sessions[keep_recent:]
            
            # Delete old sessions
            deleted_count = 0
            for session_info in sessions_to_delete:
                if self.delete_session(session_info["session_id"]):
                    deleted_count += 1
            
            logger.info(f"[MainOrchestrator] Cleaned up {deleted_count} old sessions for user {user_id}")
            return deleted_count
        else:
            # For in-memory storage, implement similar logic
            user_sessions = self.get_user_sessions(user_id, limit=1000)
            
            if len(user_sessions) <= keep_recent:
                return 0
            
            # Sort and delete old sessions
            user_sessions.sort(key=lambda x: x.get("last_updated", datetime.min), reverse=True)
            sessions_to_delete = user_sessions[keep_recent:]
            
            deleted_count = 0
            for session_info in sessions_to_delete:
                if self.delete_session(session_info["session_id"]):
                    deleted_count += 1
            
            logger.info(f"[MainOrchestrator] Cleaned up {deleted_count} old sessions for user {user_id}")
            return deleted_count
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        return self.session_manager.delete_session(session_id)

    def _is_clarification_response(self, state: MainState, user_input: str) -> bool:
        """
        Check if this input is a response to a pending clarification question.
        
        Args:
            state: Current session state
            user_input: User's input message
            
        Returns:
            True if this is a response to a clarification question
        """
        # Check if we have a clarification state with pending question
        if hasattr(state, 'waiting_for_response') and state.waiting_for_response:
            return True
        
        # Check metadata for clarification status
        clarification_status = state.metadata.get("clarification_status")
        return clarification_status == "waiting_for_user_response"
    
    def _is_paused_for_clarification(self, state: MainState) -> bool:
        """
        Check if the graph is paused waiting for user response.
        
        Args:
            state: Final state from graph execution
            
        Returns:
            True if paused for clarification
        """
        # Check metadata for pending question
        pending_question = state.metadata.get("pending_question")
        if pending_question:
            return True
        
        # Check metadata for clarification status
        clarification_status = state.metadata.get("clarification_status")
        return clarification_status == "waiting_for_user_response"
    
    def _format_clarification_question_response(self, state: MainState, processing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format response when paused for clarification.
        
        Args:
            state: State with pending clarification question
            processing_metadata: Processing metadata
            
        Returns:
            Formatted response dictionary
        """
        # Get the pending question from metadata
        pending_question = state.metadata.get("pending_question")
        if not pending_question:
            pending_question = "I need some additional information to help you. Could you provide more details?"
        
        return {
            "response": pending_question,
            "session_id": state.session_id,
            "conversation_history": [msg.model_dump() for msg in state.messages],
            "status": "waiting_for_clarification",
            "metadata": {
                **state.metadata,
                "processing_metadata": processing_metadata,
                "response_type": "clarification_question",
                "waiting_for_response": True,
                "clarification_attempts": state.metadata.get("clarification_attempts", 0)
            }
        }
    
    async def _prepare_clarification_resume(self, state: MainState, user_response: str) -> MainState:
        """
        Prepare state to resume clarification processing.
        
        Args:
            state: Current session state
            user_response: User's response to clarification question
            
        Returns:
            Updated state ready for clarification resume
        """
        logger.info(f"[MainOrchestrator] Preparing clarification resume for session {state.session_id[:8]}")
        
        # If this is a MainState, we need to convert it to ClarificationState for resume
        if isinstance(state, MainState):
            # Convert to ClarificationState for resume processing
            from src.schemas.state_schemas import ClarificationState
            
            # Create ClarificationState with resume information
            clarification_state = ClarificationState(
                user_input=user_response,  # Set current input to response
                session_id=state.session_id,
                intent=state.intent,
                confidence=state.confidence,
                messages=state.messages,
                timestamp=state.timestamp,
                metadata=state.metadata,
                parameters=state.parameters,
                execution_results=state.execution_results,
                response=state.response,
                # Preserve clarification state from session
                missing_params=getattr(state, 'missing_params', []),
                missing_critical_params=getattr(state, 'missing_critical_params', []),
                parameter_priorities=getattr(state, 'parameter_priorities', []),
                normalization_suggestions=getattr(state, 'normalization_suggestions', {}),
                ambiguity_flags=getattr(state, 'ambiguity_flags', {}),
                clarification_history=getattr(state, 'clarification_history', []),
                clarification_attempts=getattr(state, 'clarification_attempts', 0),
                max_attempts=getattr(state, 'max_attempts', 3),
                extracted_parameters=getattr(state, 'extracted_parameters', {}),
                # Resume state
                waiting_for_response=False,
                clarification_phase="processing",
                pending_question=None,
                last_question_asked=getattr(state, 'last_question_asked', None)
            )
            
            # Update metadata for resume
            clarification_state = clarification_state.model_copy(update={
                "metadata": {
                    **clarification_state.metadata,
                    "clarification_status": "processing_user_response",
                    "resumed_from_pause": True,
                    "last_user_response": user_response
                }
            })
            
            return clarification_state
        else:
            # Already a ClarificationState, just update the user input
            return state.model_copy(update={
                "user_input": user_response,
                "waiting_for_response": False,
                "clarification_phase": "processing",
                "pending_question": None,
                "metadata": {
                    **state.metadata,
                    "clarification_status": "processing_user_response",
                    "resumed_from_pause": True,
                    "last_user_response": user_response
                }
            })


# Create the main orchestrator instance
main_orchestrator = MainOrchestrator(use_database=False)


# Export for use by API endpoints
__all__ = [
    "MainOrchestrator",
    "SessionManager",
    "main_orchestrator"
] 