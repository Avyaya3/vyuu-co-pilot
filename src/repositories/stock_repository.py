"""
Stock Repository for Financial Management System.

This module provides data access operations for Stock entities,
including CRUD operations, filtering, and aggregation queries.

Features:
- Full CRUD operations for stocks
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
from ..schemas.database_models import Stock, StockCreate, StockUpdate, StockFilters
from ..utils.database import DatabaseConnectionError

logger = logging.getLogger(__name__)


class StockRepository(BaseRepository[Stock, StockCreate, StockUpdate, str]):
    """
    Repository for Stock entity operations.
    
    Provides comprehensive data access methods for managing stock investments,
    including creation, updates, queries, and financial analysis.
    """
    
    def __init__(self):
        """Initialize the stock repository."""
        super().__init__(Stock, "stocks")
    
    async def create_stock(self, stock_data: StockCreate) -> Stock:
        """
        Create a new stock investment.
        
        Args:
            stock_data: Stock creation data
            
        Returns:
            Created stock investment
            
        Raises:
            EntityValidationError: If stock data is invalid
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO stocks (
                        id, userId, name, type, amount, currentValue, 
                        purchaseDate, returns, description
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9
                    ) RETURNING *
                """
                
                # Generate cuid for the stock investment
                stock_id = await self._generate_cuid(conn)
                
                values = (
                    stock_id,
                    stock_data.userId,
                    stock_data.name,
                    stock_data.type,
                    stock_data.amount,
                    stock_data.currentValue,
                    stock_data.purchaseDate,
                    stock_data.returns,
                    stock_data.description
                )
                
                row = await conn.fetchrow(query, *values)
                return Stock(**dict(row))
                
        except asyncpg.UniqueViolationError as e:
            self._logger.error(f"Stock creation failed - unique constraint violation: {e}")
            raise EntityValidationError("Stock investment with this name already exists for this user")
        except Exception as e:
            self._logger.error(f"Stock creation failed: {e}")
            raise DatabaseOperationError(f"Failed to create stock investment: {e}")
    
    async def get_stock_by_id(self, stock_id: str) -> Stock:
        """
        Get stock investment by ID.
        
        Args:
            stock_id: Stock investment ID
            
        Returns:
            Stock entity
            
        Raises:
            EntityNotFoundError: If stock investment not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM stocks WHERE id = $1"
                row = await conn.fetchrow(query, stock_id)
                
                if not row:
                    raise EntityNotFoundError(f"Stock investment with ID {stock_id} not found")
                
                return Stock(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get stock investment by ID {stock_id}: {e}")
            raise DatabaseOperationError(f"Failed to get stock investment: {e}")
    
    async def get_stocks_by_user(self, user_id: str, filters: Optional[StockFilters] = None) -> List[Stock]:
        """
        Get all stock investments for a user with optional filtering.
        
        Args:
            user_id: User ID
            filters: Optional filters to apply
            
        Returns:
            List of stock investments
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build query with filters
                query_parts = ["SELECT * FROM stocks WHERE userId = $1"]
                params = [user_id]
                param_count = 1
                
                if filters:
                    if filters.name:
                        param_count += 1
                        query_parts.append(f"AND name = ${param_count}")
                        params.append(filters.name)
                    
                    if filters.type:
                        param_count += 1
                        query_parts.append(f"AND type = ${param_count}")
                        params.append(filters.type)
                    
                    if filters.minAmount is not None:
                        param_count += 1
                        query_parts.append(f"AND amount >= ${param_count}")
                        params.append(filters.minAmount)
                    
                    if filters.maxAmount is not None:
                        param_count += 1
                        query_parts.append(f"AND amount <= ${param_count}")
                        params.append(filters.maxAmount)
                    
                    if filters.minCurrentValue is not None:
                        param_count += 1
                        query_parts.append(f"AND currentValue >= ${param_count}")
                        params.append(filters.minCurrentValue)
                    
                    if filters.maxCurrentValue is not None:
                        param_count += 1
                        query_parts.append(f"AND currentValue <= ${param_count}")
                        params.append(filters.maxCurrentValue)
                
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
                
                return [Stock(**dict(row)) for row in rows]
                
        except Exception as e:
            self._logger.error(f"Failed to get stock investments for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get stock investments: {e}")
    
    async def update_stock(self, stock_id: str, update_data: StockUpdate) -> Stock:
        """
        Update a stock investment.
        
        Args:
            stock_id: Stock investment ID
            update_data: Stock update data
            
        Returns:
            Updated stock investment
            
        Raises:
            EntityNotFoundError: If stock investment not found
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
                
                if update_data.amount is not None:
                    param_count += 1
                    update_fields.append(f"amount = ${param_count}")
                    params.append(update_data.amount)
                
                if update_data.currentValue is not None:
                    param_count += 1
                    update_fields.append(f"currentValue = ${param_count}")
                    params.append(update_data.currentValue)
                
                if update_data.purchaseDate is not None:
                    param_count += 1
                    update_fields.append(f"purchaseDate = ${param_count}")
                    params.append(update_data.purchaseDate)
                
                if update_data.returns is not None:
                    param_count += 1
                    update_fields.append(f"returns = ${param_count}")
                    params.append(update_data.returns)
                
                if update_data.description is not None:
                    param_count += 1
                    update_fields.append(f"description = ${param_count}")
                    params.append(update_data.description)
                
                if not update_fields:
                    # No fields to update, just return the current stock investment
                    return await self.get_stock_by_id(stock_id)
                
                # Add updatedAt timestamp
                param_count += 1
                update_fields.append(f"updatedAt = ${param_count}")
                params.append(datetime.utcnow())
                
                # Add stock_id to params
                param_count += 1
                params.append(stock_id)
                
                query = f"""
                    UPDATE stocks 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    raise EntityNotFoundError(f"Stock investment with ID {stock_id} not found")
                
                return Stock(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to update stock investment {stock_id}: {e}")
            raise DatabaseOperationError(f"Failed to update stock investment: {e}")
    
    async def delete_stock(self, stock_id: str) -> bool:
        """
        Delete a stock investment.
        
        Args:
            stock_id: Stock investment ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            EntityNotFoundError: If stock investment not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "DELETE FROM stocks WHERE id = $1 RETURNING id"
                row = await conn.fetchrow(query, stock_id)
                
                if not row:
                    raise EntityNotFoundError(f"Stock investment with ID {stock_id} not found")
                
                return True
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to delete stock investment {stock_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete stock investment: {e}")
    
    async def get_stock_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get stock investment summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Stock summary statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_stocks,
                        SUM(amount) as total_investment,
                        SUM(currentValue) as total_current_value,
                        AVG(returns) as avg_returns,
                        MIN(amount) as min_investment,
                        MAX(amount) as max_investment
                    FROM stocks 
                    WHERE userId = $1
                """
                
                row = await conn.fetchrow(query, user_id)
                
                total_investment = row["total_investment"] or 0
                total_current_value = row["total_current_value"] or 0
                
                return {
                    "totalStocks": row["total_stocks"] or 0,
                    "totalInvestment": total_investment,
                    "totalCurrentValue": total_current_value,
                    "totalGain": total_current_value - total_investment,
                    "avgReturns": row["avg_returns"] or 0,
                    "minInvestment": row["min_investment"] or 0,
                    "maxInvestment": row["max_investment"] or 0
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get stock summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get stock summary: {e}")
    
    async def get_stocks_by_type(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get stock investments grouped by type.
        
        Args:
            user_id: User ID
            
        Returns:
            List of type summaries with stock investments
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        type,
                        COUNT(*) as stock_count,
                        SUM(amount) as total_investment,
                        SUM(currentValue) as total_current_value,
                        AVG(returns) as avg_returns
                    FROM stocks 
                    WHERE userId = $1
                    GROUP BY type
                    ORDER BY total_investment DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "type": row["type"],
                        "stockCount": row["stock_count"],
                        "totalInvestment": row["total_investment"],
                        "totalCurrentValue": row["total_current_value"],
                        "avgReturns": row["avg_returns"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get stocks by type for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get stocks by type: {e}")
    
    async def _generate_cuid(self, conn: asyncpg.Connection) -> str:
        """
        Generate a cuid for new stock investments.
        
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
