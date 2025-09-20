"""
Goal Repository for Financial Goal Management.

This module provides domain-specific repository operations for goal entities,
including goal tracking, progress calculations, and goal achievement analytics.

Features:
- Goal CRUD operations with validation
- Progress tracking and calculations
- Goal achievement analytics
- Active/inactive goal management
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID

from ..repositories.base_repository import BaseRepository, DatabaseOperationError, EntityValidationError
from ..schemas.database_models import Goal, GoalCreate, GoalUpdate

logger = logging.getLogger(__name__)


class GoalRepository(BaseRepository[Goal, GoalCreate, GoalUpdate, UUID]):
    """
    Repository for goal entity operations.
    
    Provides specialized goal management operations including progress tracking,
    goal analytics, and achievement calculations.
    """
    
    def __init__(self):
        """Initialize the goal repository."""
        super().__init__(Goal, "goals")
    
    async def create(self, goal_data: GoalCreate) -> Goal:
        """
        Create a new goal with validation.
        
        Args:
            goal_data: Goal creation data
            
        Returns:
            Created goal entity
            
        Raises:
            EntityValidationError: If validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Creating goal for user {goal_data.user_id}: {goal_data.title}")
        
        try:
            goal_dict = goal_data.model_dump()
            
            query = """
                INSERT INTO goals (user_id, title, target_amount, current_amount, is_active)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
            """
            
            result = await self._execute_query(
                query,
                goal_dict['user_id'],
                goal_dict['title'],
                goal_dict['target_amount'],
                goal_dict['current_amount'],
                goal_dict['is_active'],
                fetch_one=True
            )
            
            created_goal = self._row_to_model(result)
            self._logger.info(f"Goal created successfully with ID: {created_goal.id}")
            return created_goal
            
        except Exception as e:
            self._logger.error(f"Failed to create goal: {e}")
            raise DatabaseOperationError(f"Goal creation failed: {e}")
    
    async def get_by_id(self, goal_id: UUID) -> Optional[Goal]:
        """Get a goal by its ID."""
        try:
            query = "SELECT * FROM goals WHERE id = $1"
            result = await self._execute_query(query, goal_id, fetch_one=True)
            return self._row_to_model(result)
        except Exception as e:
            self._logger.error(f"Failed to get goal by ID {goal_id}: {e}")
            raise DatabaseOperationError(f"Get goal by ID failed: {e}")
    
    async def update(self, goal_id: UUID, goal_update: GoalUpdate) -> Optional[Goal]:
        """Update a goal's information."""
        try:
            if not await self.exists(goal_id):
                return None
            
            update_fields = []
            params = []
            param_idx = 1
            
            update_dict = goal_update.model_dump(exclude_unset=True)
            
            for field, value in update_dict.items():
                if value is not None:
                    update_fields.append(f"{field} = ${param_idx}")
                    params.append(value)
                    param_idx += 1
            
            if not update_fields:
                return await self.get_by_id(goal_id)
            
            params.append(goal_id)
            
            query = f"""
                UPDATE goals 
                SET {', '.join(update_fields)}
                WHERE id = ${param_idx}
                RETURNING *
            """
            
            result = await self._execute_query(query, *params, fetch_one=True)
            return self._row_to_model(result)
            
        except Exception as e:
            self._logger.error(f"Failed to update goal {goal_id}: {e}")
            raise DatabaseOperationError(f"Goal update failed: {e}")
    
    async def delete(self, goal_id: UUID) -> bool:
        """Delete a goal by its ID."""
        try:
            query = "DELETE FROM goals WHERE id = $1"
            result = await self._execute_query(query, goal_id, fetch_one=False, fetch_all=False)
            return result == "DELETE 1"
        except Exception as e:
            self._logger.error(f"Failed to delete goal {goal_id}: {e}")
            raise DatabaseOperationError(f"Goal deletion failed: {e}")
    
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Goal]:
        """List all goals with optional pagination."""
        try:
            if limit is not None and offset is not None:
                query = "SELECT * FROM goals ORDER BY created_at DESC LIMIT $1 OFFSET $2"
                result = await self._execute_query(query, limit, offset, fetch_all=True)
            elif limit is not None:
                query = "SELECT * FROM goals ORDER BY created_at DESC LIMIT $1"
                result = await self._execute_query(query, limit, fetch_all=True)
            else:
                query = "SELECT * FROM goals ORDER BY created_at DESC"
                result = await self._execute_query(query, fetch_all=True)
            
            return self._rows_to_models(result or [])
        except Exception as e:
            self._logger.error(f"Failed to list goals: {e}")
            raise DatabaseOperationError(f"List goals failed: {e}")
    
    # Goal-specific domain methods
    
    async def get_user_goals(self, user_id: UUID, active_only: bool = False) -> List[Goal]:
        """Get all goals for a specific user."""
        try:
            if active_only:
                query = """
                    SELECT * FROM goals 
                    WHERE user_id = $1 AND is_active = true
                    ORDER BY created_at DESC
                """
            else:
                query = """
                    SELECT * FROM goals 
                    WHERE user_id = $1 
                    ORDER BY created_at DESC
                """
            
            result = await self._execute_query(query, user_id, fetch_all=True)
            return self._rows_to_models(result or [])
        except Exception as e:
            self._logger.error(f"Failed to get goals for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get user goals failed: {e}")
    
    async def update_goal_progress(self, goal_id: UUID, new_amount: Decimal) -> Optional[Goal]:
        """Update the current amount for a goal."""
        try:
            query = """
                UPDATE goals 
                SET current_amount = $1
                WHERE id = $2
                RETURNING *
            """
            
            result = await self._execute_query(query, new_amount, goal_id, fetch_one=True)
            return self._row_to_model(result)
        except Exception as e:
            self._logger.error(f"Failed to update goal progress {goal_id}: {e}")
            raise DatabaseOperationError(f"Goal progress update failed: {e}")
    
    async def get_goal_progress_percentage(self, goal_id: UUID) -> Optional[float]:
        """Get the progress percentage for a goal."""
        try:
            goal = await self.get_by_id(goal_id)
            if not goal:
                return None
            
            if goal.target_amount <= 0:
                return 0.0
            
            progress = float(goal.current_amount) / float(goal.target_amount) * 100
            return min(progress, 100.0)  # Cap at 100%
        except Exception as e:
            self._logger.error(f"Failed to get goal progress for {goal_id}: {e}")
            raise DatabaseOperationError(f"Get goal progress failed: {e}")
    
    async def get_achieved_goals(self, user_id: UUID) -> List[Goal]:
        """Get all achieved goals for a user."""
        try:
            query = """
                SELECT * FROM goals 
                WHERE user_id = $1 AND current_amount >= target_amount
                ORDER BY created_at DESC
            """
            
            result = await self._execute_query(query, user_id, fetch_all=True)
            return self._rows_to_models(result or [])
        except Exception as e:
            self._logger.error(f"Failed to get achieved goals for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get achieved goals failed: {e}")
    
    async def deactivate_goal(self, goal_id: UUID) -> Optional[Goal]:
        """Deactivate a goal."""
        try:
            query = """
                UPDATE goals 
                SET is_active = false
                WHERE id = $1
                RETURNING *
            """
            
            result = await self._execute_query(query, goal_id, fetch_one=True)
            return self._row_to_model(result)
        except Exception as e:
            self._logger.error(f"Failed to deactivate goal {goal_id}: {e}")
            raise DatabaseOperationError(f"Goal deactivation failed: {e}")
    
    async def get_user_goal_summary(self, user_id: UUID) -> Dict[str, any]:
        """Get goal summary statistics for a user."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_goals,
                    COUNT(*) FILTER (WHERE is_active = true) as active_goals,
                    COUNT(*) FILTER (WHERE current_amount >= target_amount) as achieved_goals,
                    COALESCE(SUM(target_amount), 0) as total_target,
                    COALESCE(SUM(current_amount), 0) as total_current
                FROM goals
                WHERE user_id = $1
            """
            
            result = await self._execute_query(query, user_id, fetch_one=True)
            
            if result:
                return {
                    'total_goals': result['total_goals'],
                    'active_goals': result['active_goals'],
                    'achieved_goals': result['achieved_goals'],
                    'total_target_amount': float(result['total_target']),
                    'total_current_amount': float(result['total_current']),
                    'overall_progress': (
                        float(result['total_current']) / float(result['total_target']) * 100
                        if result['total_target'] > 0 else 0.0
                    )
                }
            
            return {
                'total_goals': 0,
                'active_goals': 0,
                'achieved_goals': 0,
                'total_target_amount': 0.0,
                'total_current_amount': 0.0,
                'overall_progress': 0.0
            }
        except Exception as e:
            self._logger.error(f"Failed to get goal summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get goal summary failed: {e}") 