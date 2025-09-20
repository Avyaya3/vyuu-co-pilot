"""
Insurance Repository for Financial Management System.

This module provides data access operations for Insurance entities,
including CRUD operations, filtering, and aggregation queries.

Features:
- Full CRUD operations for insurance policies
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
from ..schemas.database_models import Insurance, InsuranceCreate, InsuranceUpdate, InsuranceFilters
from ..utils.database import DatabaseConnectionError

logger = logging.getLogger(__name__)


class InsuranceRepository(BaseRepository[Insurance, InsuranceCreate, InsuranceUpdate, str]):
    """
    Repository for Insurance entity operations.
    
    Provides comprehensive data access methods for managing insurance policies,
    including creation, updates, queries, and financial analysis.
    """
    
    def __init__(self):
        """Initialize the insurance repository."""
        super().__init__(Insurance, "insurances")
    
    async def create_insurance(self, insurance_data: InsuranceCreate) -> Insurance:
        """
        Create a new insurance policy.
        
        Args:
            insurance_data: Insurance creation data
            
        Returns:
            Created insurance policy
            
        Raises:
            EntityValidationError: If insurance data is invalid
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO insurances (
                        id, userId, type, provider, policyNumber, premium, 
                        coverage, startDate, endDate, description
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                    ) RETURNING *
                """
                
                # Generate cuid for the insurance policy
                insurance_id = await self._generate_cuid(conn)
                
                values = (
                    insurance_id,
                    insurance_data.userId,
                    insurance_data.type,
                    insurance_data.provider,
                    insurance_data.policyNumber,
                    insurance_data.premium,
                    insurance_data.coverage,
                    insurance_data.startDate,
                    insurance_data.endDate,
                    insurance_data.description
                )
                
                row = await conn.fetchrow(query, *values)
                return Insurance(**dict(row))
                
        except asyncpg.UniqueViolationError as e:
            self._logger.error(f"Insurance creation failed - unique constraint violation: {e}")
            raise EntityValidationError("Insurance policy with this policy number already exists for this user")
        except Exception as e:
            self._logger.error(f"Insurance creation failed: {e}")
            raise DatabaseOperationError(f"Failed to create insurance policy: {e}")
    
    async def get_insurance_by_id(self, insurance_id: str) -> Insurance:
        """
        Get insurance policy by ID.
        
        Args:
            insurance_id: Insurance policy ID
            
        Returns:
            Insurance entity
            
        Raises:
            EntityNotFoundError: If insurance policy not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM insurances WHERE id = $1"
                row = await conn.fetchrow(query, insurance_id)
                
                if not row:
                    raise EntityNotFoundError(f"Insurance policy with ID {insurance_id} not found")
                
                return Insurance(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get insurance policy by ID {insurance_id}: {e}")
            raise DatabaseOperationError(f"Failed to get insurance policy: {e}")
    
    async def get_insurances_by_user(self, user_id: str, filters: Optional[InsuranceFilters] = None) -> List[Insurance]:
        """
        Get all insurance policies for a user with optional filtering.
        
        Args:
            user_id: User ID
            filters: Optional filters to apply
            
        Returns:
            List of insurance policies
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build query with filters
                query_parts = ["SELECT * FROM insurances WHERE userId = $1"]
                params = [user_id]
                param_count = 1
                
                if filters:
                    if filters.type:
                        param_count += 1
                        query_parts.append(f"AND type = ${param_count}")
                        params.append(filters.type)
                    
                    if filters.provider:
                        param_count += 1
                        query_parts.append(f"AND provider = ${param_count}")
                        params.append(filters.provider)
                    
                    if filters.minPremium is not None:
                        param_count += 1
                        query_parts.append(f"AND premium >= ${param_count}")
                        params.append(filters.minPremium)
                    
                    if filters.maxPremium is not None:
                        param_count += 1
                        query_parts.append(f"AND premium <= ${param_count}")
                        params.append(filters.maxPremium)
                    
                    if filters.minCoverage is not None:
                        param_count += 1
                        query_parts.append(f"AND coverage >= ${param_count}")
                        params.append(filters.minCoverage)
                    
                    if filters.maxCoverage is not None:
                        param_count += 1
                        query_parts.append(f"AND coverage <= ${param_count}")
                        params.append(filters.maxCoverage)
                
                # Add ordering
                order_by = "type"
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
                
                return [Insurance(**dict(row)) for row in rows]
                
        except Exception as e:
            self._logger.error(f"Failed to get insurance policies for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get insurance policies: {e}")
    
    async def update_insurance(self, insurance_id: str, update_data: InsuranceUpdate) -> Insurance:
        """
        Update an insurance policy.
        
        Args:
            insurance_id: Insurance policy ID
            update_data: Insurance update data
            
        Returns:
            Updated insurance policy
            
        Raises:
            EntityNotFoundError: If insurance policy not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build dynamic update query
                update_fields = []
                params = []
                param_count = 0
                
                if update_data.type is not None:
                    param_count += 1
                    update_fields.append(f"type = ${param_count}")
                    params.append(update_data.type)
                
                if update_data.provider is not None:
                    param_count += 1
                    update_fields.append(f"provider = ${param_count}")
                    params.append(update_data.provider)
                
                if update_data.policyNumber is not None:
                    param_count += 1
                    update_fields.append(f"policyNumber = ${param_count}")
                    params.append(update_data.policyNumber)
                
                if update_data.premium is not None:
                    param_count += 1
                    update_fields.append(f"premium = ${param_count}")
                    params.append(update_data.premium)
                
                if update_data.coverage is not None:
                    param_count += 1
                    update_fields.append(f"coverage = ${param_count}")
                    params.append(update_data.coverage)
                
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
                    # No fields to update, just return the current insurance policy
                    return await self.get_insurance_by_id(insurance_id)
                
                # Add updatedAt timestamp
                param_count += 1
                update_fields.append(f"updatedAt = ${param_count}")
                params.append(datetime.utcnow())
                
                # Add insurance_id to params
                param_count += 1
                params.append(insurance_id)
                
                query = f"""
                    UPDATE insurances 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    raise EntityNotFoundError(f"Insurance policy with ID {insurance_id} not found")
                
                return Insurance(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to update insurance policy {insurance_id}: {e}")
            raise DatabaseOperationError(f"Failed to update insurance policy: {e}")
    
    async def delete_insurance(self, insurance_id: str) -> bool:
        """
        Delete an insurance policy.
        
        Args:
            insurance_id: Insurance policy ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            EntityNotFoundError: If insurance policy not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "DELETE FROM insurances WHERE id = $1 RETURNING id"
                row = await conn.fetchrow(query, insurance_id)
                
                if not row:
                    raise EntityNotFoundError(f"Insurance policy with ID {insurance_id} not found")
                
                return True
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to delete insurance policy {insurance_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete insurance policy: {e}")
    
    async def get_insurance_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get insurance summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Insurance summary statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_policies,
                        SUM(premium) as total_premium,
                        SUM(coverage) as total_coverage,
                        AVG(premium) as avg_premium,
                        MIN(premium) as min_premium,
                        MAX(premium) as max_premium
                    FROM insurances 
                    WHERE userId = $1
                """
                
                row = await conn.fetchrow(query, user_id)
                
                return {
                    "totalPolicies": row["total_policies"] or 0,
                    "totalPremium": row["total_premium"] or 0,
                    "totalCoverage": row["total_coverage"] or 0,
                    "avgPremium": row["avg_premium"] or 0,
                    "minPremium": row["min_premium"] or 0,
                    "maxPremium": row["max_premium"] or 0
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get insurance summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get insurance summary: {e}")
    
    async def get_insurances_by_type(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get insurance policies grouped by type.
        
        Args:
            user_id: User ID
            
        Returns:
            List of type summaries with insurance policies
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        type,
                        COUNT(*) as policy_count,
                        SUM(premium) as total_premium,
                        SUM(coverage) as total_coverage,
                        AVG(premium) as avg_premium
                    FROM insurances 
                    WHERE userId = $1
                    GROUP BY type
                    ORDER BY total_premium DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "type": row["type"],
                        "policyCount": row["policy_count"],
                        "totalPremium": row["total_premium"],
                        "totalCoverage": row["total_coverage"],
                        "avgPremium": row["avg_premium"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get insurances by type for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get insurances by type: {e}")
    
    async def get_insurances_by_provider(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get insurance policies grouped by provider.
        
        Args:
            user_id: User ID
            
        Returns:
            List of provider summaries with insurance policies
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        provider,
                        COUNT(*) as policy_count,
                        SUM(premium) as total_premium,
                        SUM(coverage) as total_coverage,
                        AVG(premium) as avg_premium
                    FROM insurances 
                    WHERE userId = $1
                    GROUP BY provider
                    ORDER BY total_premium DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "provider": row["provider"],
                        "policyCount": row["policy_count"],
                        "totalPremium": row["total_premium"],
                        "totalCoverage": row["total_coverage"],
                        "avgPremium": row["avg_premium"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get insurances by provider for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get insurances by provider: {e}")
    
    async def _generate_cuid(self, conn: asyncpg.Connection) -> str:
        """
        Generate a cuid for new insurance policies.
        
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
