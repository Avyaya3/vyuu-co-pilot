"""
Savings Repository for Financial Management System.

This module provides data access operations for Savings entities,
including CRUD operations, filtering, and aggregation queries.

Features:
- Full CRUD operations for savings
- Advanced filtering and search capabilities
- Aggregation queries for financial analysis
- Transaction support and error handling
- Connection pooling and health monitoring
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

import asyncpg

from .base_repository import BaseRepository, RepositoryError, EntityNotFoundError
from ..schemas.database_models import Savings, SavingsCreate, SavingsUpdate, SavingsFilters
from ..utils.database import DatabaseConnectionError

logger = logging.getLogger(__name__)


class SavingsRepository(BaseRepository[Savings, SavingsCreate, SavingsUpdate, str]):
    """
    Repository for Savings entity operations.
    
    Provides comprehensive data access methods for managing savings accounts,
    including creation, updates, queries, and financial analysis.
    """
    
    def __init__(self):
        """Initialize the savings repository."""
        super().__init__(Savings, "savings")
    
    async def create_savings(self, savings_data: SavingsCreate) -> Savings:
        """
        Create a new savings account.
        
        Args:
            savings_data: Savings creation data
            
        Returns:
            Created savings account
            
        Raises:
            EntityValidationError: If savings data is invalid
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO savings (
                        id, userId, name, type, currentBalance, interestRate,
                        maturityDate, monthlyContribution, targetAmount, description
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                    ) RETURNING *
                """
                
                # Generate cuid for the savings account
                savings_id = await self._generate_cuid(conn)
                
                values = (
                    savings_id,
                    savings_data.userId,
                    savings_data.name,
                    savings_data.type,
                    savings_data.currentBalance,
                    savings_data.interestRate,
                    savings_data.maturityDate,
                    savings_data.monthlyContribution,
                    savings_data.targetAmount,
                    savings_data.description
                )
                
                row = await conn.fetchrow(query, *values)
                return Savings(**dict(row))
                
        except asyncpg.UniqueViolationError as e:
            self._logger.error(f"Savings creation failed - unique constraint violation: {e}")
            raise EntityValidationError("Savings account with this name already exists for this user")
        except Exception as e:
            self._logger.error(f"Savings creation failed: {e}")
            raise DatabaseOperationError(f"Failed to create savings account: {e}")
    
    async def get_savings_by_id(self, savings_id: str) -> Savings:
        """
        Get savings account by ID.
        
        Args:
            savings_id: Savings account ID
            
        Returns:
            Savings account entity
            
        Raises:
            EntityNotFoundError: If savings account not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM savings WHERE id = $1"
                row = await conn.fetchrow(query, savings_id)
                
                if not row:
                    raise EntityNotFoundError(f"Savings account with ID {savings_id} not found")
                
                return Savings(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get savings account by ID {savings_id}: {e}")
            raise DatabaseOperationError(f"Failed to get savings account: {e}")
    
    async def get_savings_by_user(self, user_id: str, filters: Optional[SavingsFilters] = None) -> List[Savings]:
        """
        Get all savings accounts for a user with optional filtering.
        
        Args:
            user_id: User ID
            filters: Optional filters to apply
            
        Returns:
            List of savings accounts
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build query with filters
                query_parts = ["SELECT * FROM savings WHERE userId = $1"]
                params = [user_id]
                param_count = 1
                
                if filters:
                    if filters.type:
                        param_count += 1
                        query_parts.append(f"AND type = ${param_count}")
                        params.append(filters.type)
                    
                    if filters.minBalance is not None:
                        param_count += 1
                        query_parts.append(f"AND currentBalance >= ${param_count}")
                        params.append(filters.minBalance)
                    
                    if filters.maxBalance is not None:
                        param_count += 1
                        query_parts.append(f"AND currentBalance <= ${param_count}")
                        params.append(filters.maxBalance)
                
                # Add ordering
                order_by = "name"
                order_direction = "ASC"
                if filters and filters.orderBy:
                    order_by = filters.orderBy
                if filters and filters.orderDirection:
                    order_direction = filters.orderDirection
                
                query_parts.append(f"ORDER BY {order_by} {order_direction}")
                
                # Add pagination
                if filters and filters.limit:
                    param_count += 1
                    query_parts.append(f"LIMIT ${param_count}")
                    params.append(filters.limit)
                
                if filters and filters.offset:
                    param_count += 1
                    query_parts.append(f"OFFSET ${param_count}")
                    params.append(filters.offset)
                
                query = " ".join(query_parts)
                rows = await conn.fetch(query, *params)
                
                return [Savings(**dict(row)) for row in rows]
                
        except Exception as e:
            self._logger.error(f"Failed to get savings accounts for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get savings accounts: {e}")
    
    async def update_savings(self, savings_id: str, update_data: SavingsUpdate) -> Savings:
        """
        Update a savings account.
        
        Args:
            savings_id: Savings account ID
            update_data: Savings update data
            
        Returns:
            Updated savings account
            
        Raises:
            EntityNotFoundError: If savings account not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build dynamic update query
                update_fields = []
                params = []
                param_count = 0
                
                if update_data.name is not None:
                    param_count += 1
                    update_fields.append(f"name = ${param_count}")
                    params.append(update_data.name)
                
                if update_data.type is not None:
                    param_count += 1
                    update_fields.append(f"type = ${param_count}")
                    params.append(update_data.type)
                
                if update_data.currentBalance is not None:
                    param_count += 1
                    update_fields.append(f"currentBalance = ${param_count}")
                    params.append(update_data.currentBalance)
                
                if update_data.interestRate is not None:
                    param_count += 1
                    update_fields.append(f"interestRate = ${param_count}")
                    params.append(update_data.interestRate)
                
                if update_data.maturityDate is not None:
                    param_count += 1
                    update_fields.append(f"maturityDate = ${param_count}")
                    params.append(update_data.maturityDate)
                
                if update_data.monthlyContribution is not None:
                    param_count += 1
                    update_fields.append(f"monthlyContribution = ${param_count}")
                    params.append(update_data.monthlyContribution)
                
                if update_data.targetAmount is not None:
                    param_count += 1
                    update_fields.append(f"targetAmount = ${param_count}")
                    params.append(update_data.targetAmount)
                
                if update_data.description is not None:
                    param_count += 1
                    update_fields.append(f"description = ${param_count}")
                    params.append(update_data.description)
                
                if not update_fields:
                    # No fields to update, just return the current savings account
                    return await self.get_savings_by_id(savings_id)
                
                # Add updatedAt timestamp
                param_count += 1
                update_fields.append(f"updatedAt = ${param_count}")
                params.append(datetime.utcnow())
                
                # Add savings_id to params
                param_count += 1
                params.append(savings_id)
                
                query = f"""
                    UPDATE savings 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    raise EntityNotFoundError(f"Savings account with ID {savings_id} not found")
                
                return Savings(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to update savings account {savings_id}: {e}")
            raise DatabaseOperationError(f"Failed to update savings account: {e}")
    
    async def delete_savings(self, savings_id: str) -> bool:
        """
        Delete a savings account.
        
        Args:
            savings_id: Savings account ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            EntityNotFoundError: If savings account not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "DELETE FROM savings WHERE id = $1 RETURNING id"
                row = await conn.fetchrow(query, savings_id)
                
                if not row:
                    raise EntityNotFoundError(f"Savings account with ID {savings_id} not found")
                
                return True
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to delete savings account {savings_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete savings account: {e}")
    
    async def get_savings_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get savings summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Savings summary statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_savings,
                        SUM(currentBalance) as total_balance,
                        SUM(monthlyContribution) as total_monthly_contribution,
                        AVG(interestRate) as avg_interest_rate,
                        MIN(currentBalance) as min_balance,
                        MAX(currentBalance) as max_balance
                    FROM savings 
                    WHERE userId = $1
                """
                
                row = await conn.fetchrow(query, user_id)
                
                return {
                    "totalSavings": row["total_savings"] or 0,
                    "totalBalance": row["total_balance"] or 0,
                    "totalMonthlyContribution": row["total_monthly_contribution"] or 0,
                    "avgInterestRate": row["avg_interest_rate"] or 0,
                    "minBalance": row["min_balance"] or 0,
                    "maxBalance": row["max_balance"] or 0
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get savings summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get savings summary: {e}")
    
    async def get_savings_by_type(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get savings accounts grouped by type.
        
        Args:
            user_id: User ID
            
        Returns:
            List of type summaries with savings accounts
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        type,
                        COUNT(*) as savings_count,
                        SUM(currentBalance) as total_balance,
                        SUM(monthlyContribution) as total_monthly_contribution,
                        AVG(interestRate) as avg_interest_rate
                    FROM savings 
                    WHERE userId = $1
                    GROUP BY type
                    ORDER BY total_balance DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "type": row["type"],
                        "savingsCount": row["savings_count"],
                        "totalBalance": row["total_balance"],
                        "totalMonthlyContribution": row["total_monthly_contribution"],
                        "avgInterestRate": row["avg_interest_rate"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get savings by type for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get savings by type: {e}")
    
    async def _generate_cuid(self, conn: asyncpg.Connection) -> str:
        """
        Generate a cuid for new savings accounts.
        
        Args:
            conn: Database connection
            
        Returns:
            Generated cuid
            
        Raises:
            DatabaseOperationError: If cuid generation fails
        """
        try:
            # Use cuid() function from database
            result = await conn.fetchval("SELECT cuid()")
            return result
        except Exception as e:
            self._logger.error(f"Failed to generate cuid: {e}")
            raise DatabaseOperationError(f"Failed to generate cuid: {e}")
