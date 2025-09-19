"""
Conversation Session Repository for Database Session Management.

This module provides database operations for conversation sessions,
enabling persistent session storage across server restarts.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import asyncpg
import json

from ..repositories.base_repository import BaseRepository, DatabaseOperationError, EntityNotFoundError
from ..schemas.database_models import (
    ConversationSession, 
    ConversationSessionCreate, 
    ConversationSessionUpdate
)

logger = logging.getLogger(__name__)


class ConversationSessionRepository(BaseRepository[ConversationSession, ConversationSessionCreate, ConversationSessionUpdate, str]):
    """
    Repository for conversation session entity operations.
    
    Provides specialized session management operations including user-specific
    session retrieval, session cleanup, and expiration handling.
    """
    
    def __init__(self):
        """Initialize the conversation session repository."""
        super().__init__(ConversationSession, "conversation_sessions")
    
    def _row_to_model(self, row: Optional[asyncpg.Record]) -> Optional[ConversationSession]:
        """
        Convert database row to ConversationSession model.
        
        Args:
            row: Database row record
            
        Returns:
            ConversationSession model or None
        """
        if not row:
            return None
        
        try:
            # Parse state_data JSON
            state_data = row['state_data']
            if isinstance(state_data, str):
                state_data = json.loads(state_data)
            
            return ConversationSession(
                session_id=row['session_id'],
                user_id=row['user_id'],
                state_data=state_data,
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                expires_at=row['expires_at'],
                is_active=row['is_active'],
                message_count=row['message_count'],
                last_intent=row['last_intent'],
                last_confidence=row['last_confidence']
            )
        except Exception as e:
            self._logger.error(f"Failed to convert row to ConversationSession: {e}")
            raise EntityNotFoundError(f"Failed to create ConversationSession: {e}")
    
    async def create(self, session_data: ConversationSessionCreate) -> ConversationSession:
        """
        Create a new conversation session.
        
        Args:
            session_data: Session creation data
            
        Returns:
            Created conversation session entity
            
        Raises:
            EntityValidationError: If validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Creating conversation session: {session_data.session_id[:8]}")
        
        try:
            # Convert Pydantic model to dict and handle JSON fields
            session_dict = session_data.model_dump()
            
            query = """
                INSERT INTO conversation_sessions (
                    session_id, user_id, state_data, created_at, updated_at, 
                    expires_at, is_active, message_count, last_intent, last_confidence
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING *
            """
            
            now = datetime.now(timezone.utc)
            result = await self._execute_query(
                query,
                session_dict['session_id'],
                session_dict.get('user_id'),
                json.dumps(session_dict['state_data']),  # Serialize state_data
                now,  # created_at
                now,  # updated_at
                session_dict.get('expires_at'),
                True,  # is_active
                0,     # message_count
                None,  # last_intent
                None,  # last_confidence
                fetch_one=True
            )
            
            created_session = self._row_to_model(result)
            self._logger.info(f"Conversation session created successfully: {created_session.session_id[:8]}")
            return created_session
            
        except Exception as e:
            self._logger.error(f"Failed to create conversation session: {e}")
            raise DatabaseOperationError(f"Session creation failed: {e}")
    
    async def update(self, session_id: str, session_data: ConversationSessionUpdate) -> Optional[ConversationSession]:
        """
        Update an existing conversation session.
        
        Args:
            session_id: Session ID to update
            session_data: Session update data
            
        Returns:
            Updated conversation session entity or None if not found
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Updating conversation session: {session_id[:8]}")
        
        try:
            # Build dynamic update query
            update_fields = []
            values = []
            param_count = 1
            
            session_dict = session_data.model_dump(exclude_unset=True)
            
            if 'state_data' in session_dict:
                update_fields.append(f"state_data = ${param_count}")
                values.append(json.dumps(session_dict['state_data']))
                param_count += 1
            
            if 'expires_at' in session_dict:
                update_fields.append(f"expires_at = ${param_count}")
                values.append(session_dict['expires_at'])
                param_count += 1
            
            if 'is_active' in session_dict:
                update_fields.append(f"is_active = ${param_count}")
                values.append(session_dict['is_active'])
                param_count += 1
            
            if 'message_count' in session_dict:
                update_fields.append(f"message_count = ${param_count}")
                values.append(session_dict['message_count'])
                param_count += 1
            
            if 'last_intent' in session_dict:
                update_fields.append(f"last_intent = ${param_count}")
                values.append(session_dict['last_intent'])
                param_count += 1
            
            if 'last_confidence' in session_dict:
                update_fields.append(f"last_confidence = ${param_count}")
                values.append(session_dict['last_confidence'])
                param_count += 1
            
            # Always update the updated_at timestamp
            update_fields.append(f"updated_at = ${param_count}")
            values.append(datetime.now(timezone.utc))
            param_count += 1
            
            if not update_fields:
                self._logger.warning(f"No fields to update for session {session_id[:8]}")
                return await self.get_by_id(session_id)
            
            # Add session_id parameter
            values.append(session_id)
            
            query = f"""
                UPDATE conversation_sessions 
                SET {', '.join(update_fields)}
                WHERE session_id = ${param_count}
                RETURNING *
            """
            
            result = await self._execute_query(query, *values, fetch_one=True)
            
            if result:
                updated_session = self._row_to_model(result)
                self._logger.info(f"Conversation session updated successfully: {session_id[:8]}")
                return updated_session
            else:
                self._logger.warning(f"Conversation session not found for update: {session_id[:8]}")
                return None
                
        except Exception as e:
            self._logger.error(f"Failed to update conversation session: {e}")
            raise DatabaseOperationError(f"Session update failed: {e}")
    
    async def get_by_user_id(self, user_id: str, limit: int = 50, offset: int = 0) -> List[ConversationSession]:
        """
        Get conversation sessions for a specific user.
        
        Args:
            user_id: User ID to query sessions for
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of conversation sessions for the user
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Getting conversation sessions for user: {user_id}")
        
        try:
            query = """
                SELECT * FROM conversation_sessions 
                WHERE user_id = $1 AND is_active = true
                ORDER BY updated_at DESC
                LIMIT $2 OFFSET $3
            """
            
            results = await self._execute_query(query, user_id, limit, offset, fetch_all=True)
            
            sessions = []
            for row in results:
                session = self._row_to_model(row)
                if session:
                    sessions.append(session)
            
            self._logger.info(f"Found {len(sessions)} conversation sessions for user {user_id}")
            return sessions
            
        except Exception as e:
            self._logger.error(f"Failed to get sessions for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get user sessions: {e}")
    
    async def get_active_sessions(self, limit: int = 100) -> List[ConversationSession]:
        """
        Get all active conversation sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of active conversation sessions
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info("Getting all active conversation sessions")
        
        try:
            query = """
                SELECT * FROM conversation_sessions 
                WHERE is_active = true
                ORDER BY updated_at DESC
                LIMIT $1
            """
            
            results = await self._execute_query(query, limit, fetch_all=True)
            
            sessions = []
            for row in results:
                session = self._row_to_model(row)
                if session:
                    sessions.append(session)
            
            self._logger.info(f"Found {len(sessions)} active conversation sessions")
            return sessions
            
        except Exception as e:
            self._logger.error(f"Failed to get active sessions: {e}")
            raise DatabaseOperationError(f"Failed to get active sessions: {e}")
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired conversation sessions.
        
        Returns:
            Number of sessions cleaned up
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info("Cleaning up expired conversation sessions")
        
        try:
            query = """
                UPDATE conversation_sessions 
                SET is_active = false, updated_at = NOW()
                WHERE is_active = true 
                AND expires_at IS NOT NULL 
                AND expires_at < NOW()
                RETURNING session_id
            """
            
            results = await self._execute_query(query, fetch_all=True)
            cleaned_count = len(results)
            
            self._logger.info(f"Cleaned up {cleaned_count} expired conversation sessions")
            return cleaned_count
            
        except Exception as e:
            self._logger.error(f"Failed to cleanup expired sessions: {e}")
            raise DatabaseOperationError(f"Failed to cleanup expired sessions: {e}")
    
    async def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Clean up old conversation sessions.
        
        Args:
            max_age_days: Maximum age of sessions to keep (in days)
            
        Returns:
            Number of sessions cleaned up
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Cleaning up conversation sessions older than {max_age_days} days")
        
        try:
            query = """
                UPDATE conversation_sessions 
                SET is_active = false, updated_at = NOW()
                WHERE is_active = true 
                AND updated_at < NOW() - INTERVAL '%s days'
                RETURNING session_id
            """ % max_age_days
            
            results = await self._execute_query(query, fetch_all=True)
            cleaned_count = len(results)
            
            self._logger.info(f"Cleaned up {cleaned_count} old conversation sessions")
            return cleaned_count
            
        except Exception as e:
            self._logger.error(f"Failed to cleanup old sessions: {e}")
            raise DatabaseOperationError(f"Failed to cleanup old sessions: {e}")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get conversation session statistics.
        
        Returns:
            Dictionary with session statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info("Getting conversation session statistics")
        
        try:
            query = """
                SELECT 
                    COUNT(*) as total_sessions,
                    COUNT(*) FILTER (WHERE is_active = true) as active_sessions,
                    COUNT(*) FILTER (WHERE user_id IS NOT NULL) as user_sessions,
                    COUNT(*) FILTER (WHERE user_id IS NULL) as anonymous_sessions,
                    AVG(message_count) as avg_message_count,
                    MAX(updated_at) as last_activity
                FROM conversation_sessions
            """
            
            result = await self._execute_query(query, fetch_one=True)
            
            stats = {
                "total_sessions": result['total_sessions'],
                "active_sessions": result['active_sessions'],
                "user_sessions": result['user_sessions'],
                "anonymous_sessions": result['anonymous_sessions'],
                "avg_message_count": float(result['avg_message_count']) if result['avg_message_count'] else 0,
                "last_activity": result['last_activity']
            }
            
            self._logger.info(f"Session stats: {stats}")
            return stats
            
        except Exception as e:
            self._logger.error(f"Failed to get session stats: {e}")
            raise DatabaseOperationError(f"Failed to get session stats: {e}")
    
    async def get_by_id(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get a conversation session by its ID.
        
        Args:
            session_id: Session ID to query
            
        Returns:
            ConversationSession if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Getting conversation session: {session_id[:8]}")
        
        try:
            query = "SELECT * FROM conversation_sessions WHERE session_id = $1"
            result = await self._execute_query(query, session_id, fetch_one=True)
            
            if result:
                session = self._row_to_model(result)
                self._logger.info(f"Found conversation session: {session_id[:8]}")
                return session
            else:
                self._logger.info(f"Conversation session not found: {session_id[:8]}")
                return None
                
        except Exception as e:
            self._logger.error(f"Failed to get conversation session {session_id[:8]}: {e}")
            raise DatabaseOperationError(f"Failed to get session: {e}")
    
    async def delete(self, session_id: str) -> bool:
        """
        Delete a conversation session by its ID.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Deleting conversation session: {session_id[:8]}")
        
        try:
            query = "DELETE FROM conversation_sessions WHERE session_id = $1 RETURNING session_id"
            result = await self._execute_query(query, session_id, fetch_one=True)
            
            if result:
                self._logger.info(f"Deleted conversation session: {session_id[:8]}")
                return True
            else:
                self._logger.info(f"Conversation session not found for deletion: {session_id[:8]}")
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to delete conversation session {session_id[:8]}: {e}")
            raise DatabaseOperationError(f"Failed to delete session: {e}")
    
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[ConversationSession]:
        """
        List all conversation sessions with optional pagination.
        
        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of conversation sessions
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info("Listing all conversation sessions")
        
        try:
            query = "SELECT * FROM conversation_sessions ORDER BY updated_at DESC"
            params = []
            
            if limit is not None:
                query += f" LIMIT ${len(params) + 1}"
                params.append(limit)
            
            if offset is not None:
                query += f" OFFSET ${len(params) + 1}"
                params.append(offset)
            
            results = await self._execute_query(query, *params, fetch_all=True)
            
            sessions = []
            for row in results:
                session = self._row_to_model(row)
                if session:
                    sessions.append(session)
            
            self._logger.info(f"Found {len(sessions)} conversation sessions")
            return sessions
            
        except Exception as e:
            self._logger.error(f"Failed to list conversation sessions: {e}")
            raise DatabaseOperationError(f"Failed to list sessions: {e}")
