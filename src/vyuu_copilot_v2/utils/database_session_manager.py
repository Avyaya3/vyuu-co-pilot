"""
Database Session Manager for LangGraph State Persistence.

This module provides database-backed session management to replace
in-memory storage, enabling session persistence across server restarts.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from uuid import uuid4

from ..schemas.state_schemas import MainState
from ..repositories.conversation_session_repository import ConversationSessionRepository
from ..repositories.base_repository import DatabaseOperationError

logger = logging.getLogger(__name__)


class DatabaseSessionManager:
    """
    Database-backed session manager for LangGraph state persistence.
    
    This class provides session management operations that persist
    conversation state to the database, enabling session continuity
    across server restarts and user-specific session management.
    """
    
    def __init__(self, session_expiry_hours: int = 24):
        """
        Initialize the database session manager.
        
        Args:
            session_expiry_hours: Hours after which sessions expire
        """
        self.repository = ConversationSessionRepository()
        self.session_expiry_hours = session_expiry_hours
        logger.info(f"[DatabaseSessionManager] Initialized with {session_expiry_hours}h session expiry")
    
    def save_session_state(self, state: MainState) -> bool:
        """
        Save session state to database storage.
        
        Args:
            state: MainState to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate expiration time
            expires_at = datetime.now(timezone.utc) + timedelta(hours=self.session_expiry_hours)
            
            # Extract session metadata
            user_id = state.metadata.get("user_id")
            message_count = len(state.messages)
            last_intent = state.intent.value if state.intent else None
            last_confidence = state.confidence
            
            # Serialize state data
            state_data = {
                "user_input": state.user_input,
                "session_id": state.session_id,
                "timestamp": state.timestamp.isoformat(),
                "messages": [msg.model_dump() for msg in state.messages],
                "metadata": state.metadata,
                "intent": state.intent.value if state.intent else None,
                "confidence": state.confidence,
                "parameters": state.parameters,
                "execution_results": state.execution_results,
                "response": state.response
            }
            
            # Check if session exists
            existing_session = self.repository.get_by_id(state.session_id)
            
            if existing_session:
                # Update existing session
                from ..schemas.database_models import ConversationSessionUpdate
                update_data = ConversationSessionUpdate(
                    state_data=state_data,
                    expires_at=expires_at,
                    is_active=True,
                    message_count=message_count,
                    last_intent=last_intent,
                    last_confidence=last_confidence
                )
                
                updated_session = self.repository.update(state.session_id, update_data)
                if updated_session:
                    logger.debug(f"[DatabaseSessionManager] Updated session {state.session_id[:8]}")
                    return True
                else:
                    logger.error(f"[DatabaseSessionManager] Failed to update session {state.session_id[:8]}")
                    return False
            else:
                # Create new session
                from ..schemas.database_models import ConversationSessionCreate
                create_data = ConversationSessionCreate(
                    session_id=state.session_id,
                    user_id=user_id,
                    state_data=state_data,
                    expires_at=expires_at
                )
                
                created_session = self.repository.create(create_data)
                if created_session:
                    logger.debug(f"[DatabaseSessionManager] Created session {state.session_id[:8]}")
                    return True
                else:
                    logger.error(f"[DatabaseSessionManager] Failed to create session {state.session_id[:8]}")
                    return False
                    
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to save session {state.session_id[:8]}: {e}")
            return False
    
    def load_session_state(self, session_id: str) -> Optional[MainState]:
        """
        Load session state from database storage.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            MainState if found, None otherwise
        """
        try:
            session = self.repository.get_by_id(session_id)
            
            if not session:
                logger.debug(f"[DatabaseSessionManager] Session {session_id[:8]} not found")
                return None
            
            if not session.is_active:
                logger.debug(f"[DatabaseSessionManager] Session {session_id[:8]} is inactive")
                return None
            
            # Check if session is expired
            if session.expires_at and session.expires_at < datetime.now(timezone.utc):
                logger.debug(f"[DatabaseSessionManager] Session {session_id[:8]} has expired")
                # Mark as inactive
                self._deactivate_session(session_id)
                return None
            
            # Deserialize state data
            state_data = session.state_data
            
            # Reconstruct MainState
            from ..schemas.state_schemas import MessageManager
            
            # Reconstruct messages
            from ..schemas.state_schemas import Message, MessageRole
            messages = []
            for msg_data in state_data.get("messages", []):
                message = Message(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    metadata=msg_data.get("metadata", {})
                )
                messages.append(message)
            
            # Reconstruct MainState
            from ..schemas.state_schemas import IntentType
            state = MainState(
                user_input=state_data.get("user_input", ""),
                session_id=state_data.get("session_id", session_id),
                timestamp=datetime.fromisoformat(state_data.get("timestamp", datetime.now(timezone.utc).isoformat())),
                messages=messages,
                metadata=state_data.get("metadata", {}),
                intent=IntentType(state_data.get("intent", "unknown")),
                confidence=state_data.get("confidence"),
                parameters=state_data.get("parameters", {}),
                execution_results=state_data.get("execution_results", {}),
                response=state_data.get("response", "")
            )
            
            logger.debug(f"[DatabaseSessionManager] Loaded session {session_id[:8]}")
            return state
            
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to load session {session_id[:8]}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from database storage.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.repository.delete(session_id)
            if success:
                logger.info(f"[DatabaseSessionManager] Deleted session {session_id[:8]}")
            else:
                logger.warning(f"[DatabaseSessionManager] Session {session_id[:8]} not found for deletion")
            return success
            
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to delete session {session_id[:8]}: {e}")
            return False
    
    def _deactivate_session(self, session_id: str) -> bool:
        """
        Deactivate a session (mark as inactive).
        
        Args:
            session_id: Session ID to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from ..schemas.database_models import ConversationSessionUpdate
            update_data = ConversationSessionUpdate(is_active=False)
            
            updated_session = self.repository.update(session_id, update_data)
            if updated_session:
                logger.debug(f"[DatabaseSessionManager] Deactivated session {session_id[:8]}")
                return True
            else:
                logger.warning(f"[DatabaseSessionManager] Session {session_id[:8]} not found for deactivation")
                return False
                
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to deactivate session {session_id[:8]}: {e}")
            return False
    
    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get sessions for a specific user.
        
        Args:
            user_id: User ID to query sessions for
            limit: Maximum number of sessions to return
            
        Returns:
            List of session information dictionaries
        """
        try:
            sessions = self.repository.get_by_user_id(user_id, limit=limit)
            
            session_info = []
            for session in sessions:
                session_info.append({
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "message_count": session.message_count,
                    "last_intent": session.last_intent,
                    "last_confidence": session.last_confidence,
                    "is_active": session.is_active
                })
            
            logger.info(f"[DatabaseSessionManager] Found {len(session_info)} sessions for user {user_id}")
            return session_info
            
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to get sessions for user {user_id}: {e}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            cleaned_count = self.repository.cleanup_expired_sessions()
            logger.info(f"[DatabaseSessionManager] Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to cleanup expired sessions: {e}")
            return 0
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Clean up old sessions.
        
        Args:
            max_age_days: Maximum age of sessions to keep (in days)
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            cleaned_count = self.repository.cleanup_old_sessions(max_age_days)
            logger.info(f"[DatabaseSessionManager] Cleaned up {cleaned_count} old sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to cleanup old sessions: {e}")
            return 0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        try:
            stats = self.repository.get_session_stats()
            logger.info(f"[DatabaseSessionManager] Session stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"[DatabaseSessionManager] Failed to get session stats: {e}")
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "user_sessions": 0,
                "anonymous_sessions": 0,
                "avg_message_count": 0,
                "last_activity": None
            }
