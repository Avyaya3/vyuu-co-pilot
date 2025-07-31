"""
User Repository for User Management Operations.

This module provides domain-specific repository operations for user entities,
including user creation, authentication support, profile management, and
user-specific business logic.

Features:
- User CRUD operations with validation
- Email uniqueness enforcement
- User settings management
- User profile operations
- Efficient user lookup methods
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
import asyncpg

from ..repositories.base_repository import BaseRepository, DatabaseOperationError, EntityNotFoundError
from ..schemas.database_models import User, UserCreate, UserUpdate

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository[User, UserCreate, UserUpdate, UUID]):
    """
    Repository for user entity operations.
    
    Provides specialized user management operations including email uniqueness
    validation, profile management, and user settings operations.
    """
    
    def __init__(self):
        """Initialize the user repository."""
        super().__init__(User, "users")
    
    def _row_to_model(self, row: Optional[asyncpg.Record]) -> Optional[User]:
        """
        Convert a database row to a User model with JSON field handling.
        
        Args:
            row: Database row record
            
        Returns:
            User model instance or None
            
        Raises:
            EntityValidationError: If model validation fails
        """
        if not row:
            return None
        
        try:
            # Convert row to dict
            row_dict = dict(row)
            
            # Handle JSON fields that might be stored as strings
            if 'settings' in row_dict and isinstance(row_dict['settings'], str):
                import json
                try:
                    row_dict['settings'] = json.loads(row_dict['settings'])
                except json.JSONDecodeError:
                    row_dict['settings'] = {}
            
            return self.model_class.model_validate(row_dict)
        except Exception as e:
            self._logger.error(f"Model validation failed: {e}")
            raise EntityValidationError(f"Failed to create {self.model_class.__name__}: {e}")
    
    async def create(self, user_data: UserCreate) -> User:
        """
        Create a new user with email uniqueness validation.
        
        Args:
            user_data: User creation data
            
        Returns:
            Created user entity
            
        Raises:
            EntityValidationError: If email already exists or validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Creating user with email: {user_data.email}")
        
        # Check email uniqueness
        existing_user = await self.get_by_email(user_data.email)
        if existing_user:
            raise EntityValidationError(f"User with email {user_data.email} already exists")
        
        try:
            # Convert Pydantic model to dict and handle JSON fields
            user_dict = user_data.model_dump()
            
            query = """
                INSERT INTO users (name, email, profile_pic_url, settings)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            """
            
            result = await self._execute_query(
                query,
                user_dict['name'],
                user_dict['email'],
                user_dict.get('profile_pic_url'),
                user_dict.get('settings', {}),
                fetch_one=True
            )
            
            created_user = self._row_to_model(result)
            self._logger.info(f"User created successfully with ID: {created_user.id}")
            return created_user
            
        except Exception as e:
            self._logger.error(f"Failed to create user: {e}")
            raise DatabaseOperationError(f"User creation failed: {e}")
    
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Get a user by their ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = "SELECT * FROM users WHERE id = $1"
            result = await self._execute_query(query, user_id, fetch_one=True)
            return self._row_to_model(result)
        except Exception as e:
            self._logger.error(f"Failed to get user by ID {user_id}: {e}")
            raise DatabaseOperationError(f"Get user by ID failed: {e}")
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by their email address.
        
        Args:
            email: User email address
            
        Returns:
            User if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            # Normalize email for consistent lookup
            normalized_email = email.lower().strip()
            query = "SELECT * FROM users WHERE email = $1"
            result = await self._execute_query(query, normalized_email, fetch_one=True)
            return self._row_to_model(result)
        except Exception as e:
            self._logger.error(f"Failed to get user by email {email}: {e}")
            raise DatabaseOperationError(f"Get user by email failed: {e}")
    
    async def update(self, user_id: UUID, user_update: UserUpdate) -> Optional[User]:
        """
        Update a user's information.
        
        Args:
            user_id: User identifier
            user_update: Update data
            
        Returns:
            Updated user if found, None otherwise
            
        Raises:
            EntityValidationError: If validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Updating user {user_id}")
        
        try:
            # Check if user exists
            if not await self.exists(user_id):
                return None
            
            # Build dynamic update query based on provided fields
            update_fields = []
            params = []
            param_idx = 1
            
            update_dict = user_update.model_dump(exclude_unset=True)
            
            for field, value in update_dict.items():
                if value is not None:
                    update_fields.append(f"{field} = ${param_idx}")
                    params.append(value)
                    param_idx += 1
            
            if not update_fields:
                # No fields to update, return current user
                return await self.get_by_id(user_id)
            
            # Add user_id parameter
            params.append(user_id)
            
            query = f"""
                UPDATE users 
                SET {', '.join(update_fields)}
                WHERE id = ${param_idx}
                RETURNING *
            """
            
            result = await self._execute_query(query, *params, fetch_one=True)
            updated_user = self._row_to_model(result)
            
            if updated_user:
                self._logger.info(f"User {user_id} updated successfully")
            
            return updated_user
            
        except Exception as e:
            self._logger.error(f"Failed to update user {user_id}: {e}")
            raise DatabaseOperationError(f"User update failed: {e}")
    
    async def delete(self, user_id: UUID) -> bool:
        """
        Delete a user by their ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Deleting user {user_id}")
        
        try:
            query = "DELETE FROM users WHERE id = $1"
            result = await self._execute_query(query, user_id, fetch_one=False, fetch_all=False)
            
            # Check if any rows were affected
            deleted = result == "DELETE 1"
            
            if deleted:
                self._logger.info(f"User {user_id} deleted successfully")
            else:
                self._logger.warning(f"User {user_id} not found for deletion")
            
            return deleted
            
        except Exception as e:
            self._logger.error(f"Failed to delete user {user_id}: {e}")
            raise DatabaseOperationError(f"User deletion failed: {e}")
    
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[User]:
        """
        List all users with optional pagination.
        
        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            
        Returns:
            List of users
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            if limit is not None and offset is not None:
                query = "SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2"
                result = await self._execute_query(query, limit, offset, fetch_all=True)
            elif limit is not None:
                query = "SELECT * FROM users ORDER BY created_at DESC LIMIT $1"
                result = await self._execute_query(query, limit, fetch_all=True)
            else:
                query = "SELECT * FROM users ORDER BY created_at DESC"
                result = await self._execute_query(query, fetch_all=True)
            
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to list users: {e}")
            raise DatabaseOperationError(f"List users failed: {e}")
    
    # User-specific domain methods
    
    async def update_settings(self, user_id: UUID, settings: Dict[str, any]) -> Optional[User]:
        """
        Update user settings.
        
        Args:
            user_id: User identifier
            settings: New settings dictionary
            
        Returns:
            Updated user if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                UPDATE users 
                SET settings = $1
                WHERE id = $2
                RETURNING *
            """
            
            result = await self._execute_query(query, settings, user_id, fetch_one=True)
            return self._row_to_model(result)
            
        except Exception as e:
            self._logger.error(f"Failed to update settings for user {user_id}: {e}")
            raise DatabaseOperationError(f"Settings update failed: {e}")
    
    async def merge_settings(self, user_id: UUID, new_settings: Dict[str, any]) -> Optional[User]:
        """
        Merge new settings with existing user settings.
        
        Args:
            user_id: User identifier
            new_settings: Settings to merge
            
        Returns:
            Updated user if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            # Get current user to access existing settings
            current_user = await self.get_by_id(user_id)
            if not current_user:
                return None
            
            # Merge settings
            merged_settings = {**current_user.settings, **new_settings}
            
            return await self.update_settings(user_id, merged_settings)
            
        except Exception as e:
            self._logger.error(f"Failed to merge settings for user {user_id}: {e}")
            raise DatabaseOperationError(f"Settings merge failed: {e}")
    
    async def search_by_name(self, name_query: str, limit: int = 50) -> List[User]:
        """
        Search users by name (case-insensitive partial match).
        
        Args:
            name_query: Name search query
            limit: Maximum number of results
            
        Returns:
            List of matching users
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT * FROM users 
                WHERE name ILIKE $1 
                ORDER BY name 
                LIMIT $2
            """
            
            search_pattern = f"%{name_query.strip()}%"
            result = await self._execute_query(query, search_pattern, limit, fetch_all=True)
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to search users by name '{name_query}': {e}")
            raise DatabaseOperationError(f"User name search failed: {e}")
    
    async def get_users_created_after(self, after_date: datetime, limit: Optional[int] = None) -> List[User]:
        """
        Get users created after a specific date.
        
        Args:
            after_date: Date threshold
            limit: Maximum number of users to return
            
        Returns:
            List of users created after the date
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            if limit:
                query = """
                    SELECT * FROM users 
                    WHERE created_at > $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                """
                result = await self._execute_query(query, after_date, limit, fetch_all=True)
            else:
                query = """
                    SELECT * FROM users 
                    WHERE created_at > $1 
                    ORDER BY created_at DESC
                """
                result = await self._execute_query(query, after_date, fetch_all=True)
            
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to get users created after {after_date}: {e}")
            raise DatabaseOperationError(f"Get users by creation date failed: {e}")
    
    async def count_active_users(self) -> int:
        """
        Count users who have accounts (active users).
        
        Returns:
            Number of active users
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT COUNT(DISTINCT u.id) 
                FROM users u
                INNER JOIN accounts a ON u.id = a.user_id
            """
            
            result = await self._execute_query(query, fetch_one=True)
            return result['count'] if result else 0
            
        except Exception as e:
            self._logger.error(f"Failed to count active users: {e}")
            raise DatabaseOperationError(f"Count active users failed: {e}")
    
    async def get_user_summary_stats(self, user_id: UUID) -> Optional[Dict[str, any]]:
        """
        Get summary statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User summary statistics or None if user not found
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT 
                    u.id,
                    u.name,
                    u.email,
                    u.created_at,
                    COUNT(DISTINCT a.id) as account_count,
                    COUNT(DISTINCT t.id) as transaction_count,
                    COUNT(DISTINCT g.id) as goal_count,
                    COALESCE(SUM(a.current_balance), 0) as total_balance
                FROM users u
                LEFT JOIN accounts a ON u.id = a.user_id
                LEFT JOIN transactions t ON u.id = t.user_id
                LEFT JOIN goals g ON u.id = g.user_id AND g.is_active = true
                WHERE u.id = $1
                GROUP BY u.id, u.name, u.email, u.created_at
            """
            
            result = await self._execute_query(query, user_id, fetch_one=True)
            
            if result:
                return {
                    'user_id': result['id'],
                    'name': result['name'],
                    'email': result['email'],
                    'created_at': result['created_at'],
                    'account_count': result['account_count'],
                    'transaction_count': result['transaction_count'],
                    'goal_count': result['goal_count'],
                    'total_balance': float(result['total_balance'])
                }
            
            return None
            
        except Exception as e:
            self._logger.error(f"Failed to get user summary for {user_id}: {e}")
            raise DatabaseOperationError(f"Get user summary failed: {e}")
    
    async def check_email_availability(self, email: str) -> bool:
        """
        Check if an email is available for registration.
        
        Args:
            email: Email to check
            
        Returns:
            True if email is available, False if taken
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            user = await self.get_by_email(email)
            return user is None
        except Exception as e:
            self._logger.error(f"Failed to check email availability for {email}: {e}")
            raise DatabaseOperationError(f"Email availability check failed: {e}") 