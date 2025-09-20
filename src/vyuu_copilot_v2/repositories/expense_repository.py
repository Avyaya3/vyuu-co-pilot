"""
Expense Repository for Financial Management System.

This module provides data access operations for Expense entities,
including CRUD operations, filtering, and aggregation queries.

Features:
- Full CRUD operations for expenses
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
from ..schemas.database_models import Expense, ExpenseCreate, ExpenseUpdate, ExpenseFilters
from ..utils.database import DatabaseConnectionError

logger = logging.getLogger(__name__)


class ExpenseRepository(BaseRepository[Expense, ExpenseCreate, ExpenseUpdate, str]):
    """
    Repository for Expense entity operations.
    
    Provides comprehensive data access methods for managing expense records,
    including creation, updates, queries, and financial analysis.
    """
    
    def __init__(self):
        """Initialize the expense repository."""
        super().__init__(Expense, "expenses")
    
    async def create_expense(self, expense_data: ExpenseCreate) -> Expense:
        """
        Create a new expense record.
        
        Args:
            expense_data: Expense creation data
            
        Returns:
            Created expense record
            
        Raises:
            EntityValidationError: If expense data is invalid
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO expenses (
                        id, userId, category, subcategory, amount, 
                        date, description, paymentMethod
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8
                    ) RETURNING *
                """
                
                # Generate cuid for the expense record
                expense_id = await self._generate_cuid(conn)
                
                values = (
                    expense_id,
                    expense_data.userId,
                    expense_data.category,
                    expense_data.subcategory,
                    expense_data.amount,
                    expense_data.date,
                    expense_data.description,
                    expense_data.paymentMethod
                )
                
                row = await conn.fetchrow(query, *values)
                return Expense(**dict(row))
                
        except asyncpg.UniqueViolationError as e:
            self._logger.error(f"Expense creation failed - unique constraint violation: {e}")
            raise EntityValidationError("Expense record with this description already exists for this user")
        except Exception as e:
            self._logger.error(f"Expense creation failed: {e}")
            raise DatabaseOperationError(f"Failed to create expense record: {e}")
    
    async def get_expense_by_id(self, expense_id: str) -> Expense:
        """
        Get expense record by ID.
        
        Args:
            expense_id: Expense record ID
            
        Returns:
            Expense entity
            
        Raises:
            EntityNotFoundError: If expense record not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM expenses WHERE id = $1"
                row = await conn.fetchrow(query, expense_id)
                
                if not row:
                    raise EntityNotFoundError(f"Expense record with ID {expense_id} not found")
                
                return Expense(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get expense record by ID {expense_id}: {e}")
            raise DatabaseOperationError(f"Failed to get expense record: {e}")
    
    async def get_expenses_by_user(self, user_id: str, filters: Optional[ExpenseFilters] = None) -> List[Expense]:
        """
        Get all expense records for a user with optional filtering.
        
        Args:
            user_id: User ID
            filters: Optional filters to apply
            
        Returns:
            List of expense records
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build query with filters
                query_parts = ["SELECT * FROM expenses WHERE userId = $1"]
                params = [user_id]
                param_count = 1
                
                if filters:
                    if filters.category:
                        param_count += 1
                        query_parts.append(f"AND category = ${param_count}")
                        params.append(filters.category)
                    
                    if filters.subcategory:
                        param_count += 1
                        query_parts.append(f"AND subcategory = ${param_count}")
                        params.append(filters.subcategory)
                    
                    if filters.paymentMethod:
                        param_count += 1
                        query_parts.append(f"AND paymentMethod = ${param_count}")
                        params.append(filters.paymentMethod)
                    
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
                
                return [Expense(**dict(row)) for row in rows]
                
        except Exception as e:
            self._logger.error(f"Failed to get expense records for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get expense records: {e}")
    
    async def update_expense(self, expense_id: str, update_data: ExpenseUpdate) -> Expense:
        """
        Update an expense record.
        
        Args:
            expense_id: Expense record ID
            update_data: Expense update data
            
        Returns:
            Updated expense record
            
        Raises:
            EntityNotFoundError: If expense record not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build dynamic update query
                update_fields = []
                params = []
                param_count = 0
                
                if update_data.category is not None:
                    param_count += 1
                    update_fields.append(f"category = ${param_count}")
                    params.append(update_data.category)
                
                if update_data.subcategory is not None:
                    param_count += 1
                    update_fields.append(f"subcategory = ${param_count}")
                    params.append(update_data.subcategory)
                
                if update_data.amount is not None:
                    param_count += 1
                    update_fields.append(f"amount = ${param_count}")
                    params.append(update_data.amount)
                
                if update_data.date is not None:
                    param_count += 1
                    update_fields.append(f"date = ${param_count}")
                    params.append(update_data.date)
                
                if update_data.description is not None:
                    param_count += 1
                    update_fields.append(f"description = ${param_count}")
                    params.append(update_data.description)
                
                if update_data.paymentMethod is not None:
                    param_count += 1
                    update_fields.append(f"paymentMethod = ${param_count}")
                    params.append(update_data.paymentMethod)
                
                if not update_fields:
                    # No fields to update, just return the current expense record
                    return await self.get_expense_by_id(expense_id)
                
                # Add updatedAt timestamp
                param_count += 1
                update_fields.append(f"updatedAt = ${param_count}")
                params.append(datetime.utcnow())
                
                # Add expense_id to params
                param_count += 1
                params.append(expense_id)
                
                query = f"""
                    UPDATE expenses 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    raise EntityNotFoundError(f"Expense record with ID {expense_id} not found")
                
                return Expense(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to update expense record {expense_id}: {e}")
            raise DatabaseOperationError(f"Failed to update expense record: {e}")
    
    async def delete_expense(self, expense_id: str) -> bool:
        """
        Delete an expense record.
        
        Args:
            expense_id: Expense record ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            EntityNotFoundError: If expense record not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "DELETE FROM expenses WHERE id = $1 RETURNING id"
                row = await conn.fetchrow(query, expense_id)
                
                if not row:
                    raise EntityNotFoundError(f"Expense record with ID {expense_id} not found")
                
                return True
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to delete expense record {expense_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete expense record: {e}")
    
    async def get_expense_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get expense summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Expense summary statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_expenses,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount,
                        MIN(amount) as min_amount,
                        MAX(amount) as max_amount
                    FROM expenses 
                    WHERE userId = $1
                """
                
                row = await conn.fetchrow(query, user_id)
                
                return {
                    "totalExpenses": row["total_expenses"] or 0,
                    "totalAmount": row["total_amount"] or 0,
                    "avgAmount": row["avg_amount"] or 0,
                    "minAmount": row["min_amount"] or 0,
                    "maxAmount": row["max_amount"] or 0
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get expense summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get expense summary: {e}")
    
    async def get_expenses_by_category(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get expense records grouped by category.
        
        Args:
            user_id: User ID
            
        Returns:
            List of category summaries with expense records
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        category,
                        COUNT(*) as expense_count,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount
                    FROM expenses 
                    WHERE userId = $1
                    GROUP BY category
                    ORDER BY total_amount DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "category": row["category"],
                        "expenseCount": row["expense_count"],
                        "totalAmount": row["total_amount"],
                        "avgAmount": row["avg_amount"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get expenses by category for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get expenses by category: {e}")
    
    async def get_expenses_by_subcategory(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get expense records grouped by subcategory.
        
        Args:
            user_id: User ID
            
        Returns:
            List of subcategory summaries with expense records
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        category,
                        subcategory,
                        COUNT(*) as expense_count,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount
                    FROM expenses 
                    WHERE userId = $1
                    GROUP BY category, subcategory
                    ORDER BY total_amount DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "category": row["category"],
                        "subcategory": row["subcategory"],
                        "expenseCount": row["expense_count"],
                        "totalAmount": row["total_amount"],
                        "avgAmount": row["avg_amount"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get expenses by subcategory for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get expenses by subcategory: {e}")
    
    async def get_expenses_by_payment_method(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get expense records grouped by payment method.
        
        Args:
            user_id: User ID
            
        Returns:
            List of payment method summaries with expense records
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        paymentMethod,
                        COUNT(*) as expense_count,
                        SUM(amount) as total_amount,
                        AVG(amount) as avg_amount
                    FROM expenses 
                    WHERE userId = $1
                    GROUP BY paymentMethod
                    ORDER BY total_amount DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "paymentMethod": row["paymentMethod"],
                        "expenseCount": row["expense_count"],
                        "totalAmount": row["total_amount"],
                        "avgAmount": row["avg_amount"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get expenses by payment method for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get expenses by payment method: {e}")
    
    async def _generate_cuid(self, conn: asyncpg.Connection) -> str:
        """
        Generate a cuid for new expense records.
        
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
