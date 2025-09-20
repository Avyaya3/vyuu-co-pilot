"""
Income Repository for Financial Management System.

This module provides data access operations for Income entities,
including CRUD operations, filtering, and aggregation queries.

Features:
- Full CRUD operations for income
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
from ..schemas.database_models import Income, IncomeCreate, IncomeUpdate, IncomeFilters
from ..utils.database import DatabaseConnectionError

logger = logging.getLogger(__name__)


class IncomeRepository(BaseRepository[Income, IncomeCreate, IncomeUpdate, str]):
    """
    Repository for Income entity operations.
    
    Provides comprehensive data access methods for managing income records,
    including creation, updates, queries, and financial analysis.
    """
    
    def __init__(self):
        """Initialize the income repository."""
        super().__init__(Income, "incomes")
    
    async def create_income(self, income_data: IncomeCreate) -> Income:
        """
        Create a new income record.
        
        Args:
            income_data: Income creation data
            
        Returns:
            Created income record
            
        Raises:
            EntityValidationError: If income data is invalid
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO incomes (
                        id, userId, source, amount, frequency, 
                        category, date, description
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8
                    ) RETURNING *
                """
                
                # Generate cuid for the income record
                income_id = await self._generate_cuid(conn)
                
                values = (
                    income_id,
                    income_data.userId,
                    income_data.source,
                    income_data.amount,
                    income_data.frequency,
                    income_data.category,
                    income_data.date,
                    income_data.description
                )
                
                row = await conn.fetchrow(query, *values)
                return Income(**dict(row))
                
        except asyncpg.UniqueViolationError as e:
            self._logger.error(f"Income creation failed - unique constraint violation: {e}")
            raise EntityValidationError("Income record with this source already exists for this user")
        except Exception as e:
            self._logger.error(f"Income creation failed: {e}")
            raise DatabaseOperationError(f"Failed to create income record: {e}")
    
    async def get_income_by_id(self, income_id: str) -> Income:
        """
        Get income record by ID.
        
        Args:
            income_id: Income record ID
            
        Returns:
            Income entity
            
        Raises:
            EntityNotFoundError: If income record not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM incomes WHERE id = $1"
                row = await conn.fetchrow(query, income_id)
                
                if not row:
                    raise EntityNotFoundError(f"Income record with ID {income_id} not found")
                
                return Income(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get income record by ID {income_id}: {e}")
            raise DatabaseOperationError(f"Failed to get income record: {e}")
    
    async def get_incomes_by_user(self, user_id: str, filters: Optional[IncomeFilters] = None) -> List[Income]:
        """
        Get all income records for a user with optional filtering.
        
        Args:
            user_id: User ID
            filters: Optional filters to apply
            
        Returns:
            List of income records
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build query with filters
                query_parts = ["SELECT * FROM incomes WHERE userId = $1"]
                params = [user_id]
                param_count = 1
                
                if filters:
                    if filters.source:
                        param_count += 1
                        query_parts.append(f"AND source = ${param_count}")
                        params.append(filters.source)
                    
                    if filters.category:
                        param_count += 1
                        query_parts.append(f"AND category = ${param_count}")
                        params.append(filters.category)
                    
                    if filters.frequency:
                        param_count += 1
                        query_parts.append(f"AND frequency = ${param_count}")
                        params.append(filters.frequency)
                    
                    if filters.minAmount is not None:
                        param_count += 1
                        query_parts.append(f"AND amount >= ${param_count}")
                        params.append(filters.minAmount)
                    
                    if filters.maxAmount is not None:
                        param_count += 1
                        query_parts.append(f"AND amount <= ${param_count}")
                        params.append(filters.maxAmount)
                    
                    if filters.startDate:
                        param_count += 1
                        query_parts.append(f"AND date >= ${param_count}")
                        params.append(filters.startDate)
                    
                    if filters.endDate:
                        param_count += 1
                        query_parts.append(f"AND date <= ${param_count}")
                        params.append(filters.endDate)
                
                # Add ordering
                order_by = "date"
                order_direction = "DESC"
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
                
                return [Income(**dict(row)) for row in rows]
                
        except Exception as e:
            self._logger.error(f"Failed to get income records for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get income records: {e}")
    
    async def update_income(self, income_id: str, update_data: IncomeUpdate) -> Income:
        """
        Update an income record.
        
        Args:
            income_id: Income record ID
            update_data: Income update data
            
        Returns:
            Updated income record
            
        Raises:
            EntityNotFoundError: If income record not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build dynamic update query
                update_fields = []
                params = []
                param_count = 0
                
                if update_data.source is not None:
                    param_count += 1
                    update_fields.append(f"source = ${param_count}")
                    params.append(update_data.source)
                
                if update_data.amount is not None:
                    param_count += 1
                    update_fields.append(f"amount = ${param_count}")
                    params.append(update_data.amount)
                
                if update_data.frequency is not None:
                    param_count += 1
                    update_fields.append(f"frequency = ${param_count}")
                    params.append(update_data.frequency)
                
                if update_data.category is not None:
                    param_count += 1
                    update_fields.append(f"category = ${param_count}")
                    params.append(update_data.category)
                
                if update_data.date is not None:
                    param_count += 1
                    update_fields.append(f"date = ${param_count}")
                    params.append(update_data.date)
                
                if update_data.description is not None:
                    param_count += 1
                    update_fields.append(f"description = ${param_count}")
                    params.append(update_data.description)
                
                if not update_fields:
                    # No fields to update, just return the current income record
                    return await self.get_income_by_id(income_id)
                
                # Add updatedAt timestamp
                param_count += 1
                update_fields.append(f"updatedAt = ${param_count}")
                params.append(datetime.utcnow())
                
                # Add income_id to params
                param_count += 1
                params.append(income_id)
                
                query = f"""
                    UPDATE incomes 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    raise EntityNotFoundError(f"Income record with ID {income_id} not found")
                
                return Income(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to update income record {income_id}: {e}")
            raise DatabaseOperationError(f"Failed to update income record: {e}")
    
    async def delete_income(self, income_id: str) -> bool:
        """
        Delete an income record.
        
        Args:
            income_id: Income record ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            EntityNotFoundError: If income record not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "DELETE FROM incomes WHERE id = $1 RETURNING id"
                row = await conn.fetchrow(query, income_id)
                
                if not row:
                    raise EntityNotFoundError(f"Income record with ID {income_id} not found")
                
                return True
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to delete income record {income_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete income record: {e}")
    
    async def get_income_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get income summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Income summary statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_incomes,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount,
                        MIN(amount) as min_amount,
                        MAX(amount) as max_amount
                    FROM incomes 
                    WHERE userId = $1
                """
                
                row = await conn.fetchrow(query, user_id)
                
                return {
                    "totalIncomes": row["total_incomes"] or 0,
                    "totalAmount": row["total_amount"] or 0,
                    "avgAmount": row["avg_amount"] or 0,
                    "minAmount": row["min_amount"] or 0,
                    "maxAmount": row["max_amount"] or 0
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get income summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get income summary: {e}")
    
    async def get_incomes_by_category(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get income records grouped by category.
        
        Args:
            user_id: User ID
            
        Returns:
            List of category summaries with income records
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        category,
                        COUNT(*) as income_count,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount
                    FROM incomes 
                    WHERE userId = $1
                    GROUP BY category
                    ORDER BY total_amount DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "category": row["category"],
                        "incomeCount": row["income_count"],
                        "totalAmount": row["total_amount"],
                        "avgAmount": row["avg_amount"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get incomes by category for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get incomes by category: {e}")
    
    async def get_incomes_by_frequency(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get income records grouped by frequency.
        
        Args:
            user_id: User ID
            
        Returns:
            List of frequency summaries with income records
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        frequency,
                        COUNT(*) as income_count,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount
                    FROM incomes 
                    WHERE userId = $1
                    GROUP BY frequency
                    ORDER BY total_amount DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "frequency": row["frequency"],
                        "incomeCount": row["income_count"],
                        "totalAmount": row["total_amount"],
                        "avgAmount": row["avg_amount"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get incomes by frequency for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get incomes by frequency: {e}")
    
    async def _generate_cuid(self, conn: asyncpg.Connection) -> str:
        """
        Generate a cuid for new income records.
        
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
