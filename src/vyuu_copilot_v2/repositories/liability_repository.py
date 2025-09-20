"""
Liability Repository for Financial Management System.

This module provides data access operations for Liability entities,
including CRUD operations, filtering, and aggregation queries.

Features:
- Full CRUD operations for liabilities
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
from ..schemas.database_models import Liability, LiabilityCreate, LiabilityUpdate, LiabilityFilters
from ..utils.database import DatabaseConnectionError

logger = logging.getLogger(__name__)


class LiabilityRepository(BaseRepository[Liability, LiabilityCreate, LiabilityUpdate, str]):
    """
    Repository for Liability entity operations.
    
    Provides comprehensive data access methods for managing financial liabilities,
    including creation, updates, queries, and financial analysis.
    """
    
    def __init__(self):
        """Initialize the liability repository."""
        super().__init__(Liability, "liabilities")
    
    async def create_liability(self, liability_data: LiabilityCreate) -> Liability:
        """
        Create a new liability.
        
        Args:
            liability_data: Liability creation data
            
        Returns:
            Created liability
            
        Raises:
            EntityValidationError: If liability data is invalid
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO liabilities (
                        id, userId, name, type, amount, emi, 
                        interestRate, startDate, endDate, description
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                    ) RETURNING *
                """
                
                # Generate cuid for the liability
                liability_id = await self._generate_cuid(conn)
                
                values = (
                    liability_id,
                    liability_data.userId,
                    liability_data.name,
                    liability_data.type,
                    liability_data.amount,
                    liability_data.emi,
                    liability_data.interestRate,
                    liability_data.startDate,
                    liability_data.endDate,
                    liability_data.description
                )
                
                row = await conn.fetchrow(query, *values)
                return Liability(**dict(row))
                
        except asyncpg.UniqueViolationError as e:
            self._logger.error(f"Liability creation failed - unique constraint violation: {e}")
            raise EntityValidationError("Liability with this name already exists for this user")
        except Exception as e:
            self._logger.error(f"Liability creation failed: {e}")
            raise DatabaseOperationError(f"Failed to create liability: {e}")
    
    async def get_liability_by_id(self, liability_id: str) -> Liability:
        """
        Get liability by ID.
        
        Args:
            liability_id: Liability ID
            
        Returns:
            Liability entity
            
        Raises:
            EntityNotFoundError: If liability not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM liabilities WHERE id = $1"
                row = await conn.fetchrow(query, liability_id)
                
                if not row:
                    raise EntityNotFoundError(f"Liability with ID {liability_id} not found")
                
                return Liability(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get liability by ID {liability_id}: {e}")
            raise DatabaseOperationError(f"Failed to get liability: {e}")
    
    async def get_liabilities_by_user(self, user_id: str, filters: Optional[LiabilityFilters] = None) -> List[Liability]:
        """
        Get all liabilities for a user with optional filtering.
        
        Args:
            user_id: User ID
            filters: Optional filters to apply
            
        Returns:
            List of liabilities
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build query with filters
                query_parts = ["SELECT * FROM liabilities WHERE userId = $1"]
                params = [user_id]
                param_count = 1
                
                if filters:
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
                
                return [Liability(**dict(row)) for row in rows]
                
        except Exception as e:
            self._logger.error(f"Failed to get liabilities for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get liabilities: {e}")
    
    async def update_liability(self, liability_id: str, update_data: LiabilityUpdate) -> Liability:
        """
        Update a liability.
        
        Args:
            liability_id: Liability ID
            update_data: Liability update data
            
        Returns:
            Updated liability
            
        Raises:
            EntityNotFoundError: If liability not found
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
                
                if update_data.emi is not None:
                    param_count += 1
                    update_fields.append(f"emi = ${param_count}")
                    params.append(update_data.emi)
                
                if update_data.interestRate is not None:
                    param_count += 1
                    update_fields.append(f"interestRate = ${param_count}")
                    params.append(update_data.interestRate)
                
                if update_data.startDate is not None:
                    param_count += 1
                    update_fields.append(f"startDate = ${param_count}")
                    params.append(update_data.startDate)
                
                if update_data.endDate is not None:
                    param_count += 1
                    update_fields.append(f"endDate = ${param_count}")
                    params.append(update_data.endDate)
                
                if update_data.description is not None:
                    param_count += 1
                    update_fields.append(f"description = ${param_count}")
                    params.append(update_data.description)
                
                if not update_fields:
                    # No fields to update, just return the current liability
                    return await self.get_liability_by_id(liability_id)
                
                # Add updatedAt timestamp
                param_count += 1
                update_fields.append(f"updatedAt = ${param_count}")
                params.append(datetime.utcnow())
                
                # Add liability_id to params
                param_count += 1
                params.append(liability_id)
                
                query = f"""
                    UPDATE liabilities 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    raise EntityNotFoundError(f"Liability with ID {liability_id} not found")
                
                return Liability(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to update liability {liability_id}: {e}")
            raise DatabaseOperationError(f"Failed to update liability: {e}")
    
    async def delete_liability(self, liability_id: str) -> bool:
        """
        Delete a liability.
        
        Args:
            liability_id: Liability ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            EntityNotFoundError: If liability not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "DELETE FROM liabilities WHERE id = $1 RETURNING id"
                row = await conn.fetchrow(query, liability_id)
                
                if not row:
                    raise EntityNotFoundError(f"Liability with ID {liability_id} not found")
                
                return True
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to delete liability {liability_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete liability: {e}")
    
    async def get_liability_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get liability summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Liability summary statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_liabilities,
                        SUM(amount) as total_amount,
                        SUM(emi) as total_emi,
                        AVG(interestRate) as avg_interest_rate,
                        MIN(amount) as min_amount,
                        MAX(amount) as max_amount
                    FROM liabilities 
                    WHERE userId = $1
                """
                
                row = await conn.fetchrow(query, user_id)
                
                return {
                    "totalLiabilities": row["total_liabilities"] or 0,
                    "totalAmount": row["total_amount"] or 0,
                    "totalEmi": row["total_emi"] or 0,
                    "avgInterestRate": row["avg_interest_rate"] or 0,
                    "minAmount": row["min_amount"] or 0,
                    "maxAmount": row["max_amount"] or 0
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get liability summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get liability summary: {e}")
    
    async def get_liabilities_by_type(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get liabilities grouped by type.
        
        Args:
            user_id: User ID
            
        Returns:
            List of type summaries with liabilities
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        type,
                        COUNT(*) as liability_count,
                        SUM(amount) as total_amount,
                        SUM(emi) as total_emi,
                        AVG(interestRate) as avg_interest_rate
                    FROM liabilities 
                    WHERE userId = $1
                    GROUP BY type
                    ORDER BY total_amount DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "type": row["type"],
                        "liabilityCount": row["liability_count"],
                        "totalAmount": row["total_amount"],
                        "totalEmi": row["total_emi"],
                        "avgInterestRate": row["avg_interest_rate"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get liabilities by type for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get liabilities by type: {e}")
    
    async def _generate_cuid(self, conn: asyncpg.Connection) -> str:
        """
        Generate a cuid for new liabilities.
        
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
