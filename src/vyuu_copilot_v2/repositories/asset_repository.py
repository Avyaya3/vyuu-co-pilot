"""
Asset Repository for Financial Management System.

This module provides data access operations for Asset entities,
including CRUD operations, filtering, and aggregation queries.

Features:
- Full CRUD operations for assets
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
from ..schemas.database_models import Asset, AssetCreate, AssetUpdate, AssetFilters
from ..utils.database import DatabaseConnectionError

logger = logging.getLogger(__name__)


class AssetRepository(BaseRepository[Asset, AssetCreate, AssetUpdate, str]):
    """
    Repository for Asset entity operations.
    
    Provides comprehensive data access methods for managing financial assets,
    including creation, updates, queries, and financial analysis.
    """
    
    def __init__(self):
        """Initialize the asset repository."""
        super().__init__(Asset, "assets")
    
    async def create_asset(self, asset_data: AssetCreate) -> Asset:
        """
        Create a new asset.
        
        Args:
            asset_data: Asset creation data
            
        Returns:
            Created asset
            
        Raises:
            EntityValidationError: If asset data is invalid
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO assets (
                        id, userId, name, category, subcategory, 
                        currentValue, purchaseValue, purchaseDate, description
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9
                    ) RETURNING *
                """
                
                # Generate cuid for the asset
                asset_id = await self._generate_cuid(conn)
                
                values = (
                    asset_id,
                    asset_data.userId,
                    asset_data.name,
                    asset_data.category,
                    asset_data.subcategory,
                    asset_data.currentValue,
                    asset_data.purchaseValue,
                    asset_data.purchaseDate,
                    asset_data.description
                )
                
                row = await conn.fetchrow(query, *values)
                return Asset(**dict(row))
                
        except asyncpg.UniqueViolationError as e:
            self._logger.error(f"Asset creation failed - unique constraint violation: {e}")
            raise EntityValidationError("Asset with this name already exists for this user")
        except Exception as e:
            self._logger.error(f"Asset creation failed: {e}")
            raise DatabaseOperationError(f"Failed to create asset: {e}")
    
    async def get_asset_by_id(self, asset_id: str) -> Asset:
        """
        Get asset by ID.
        
        Args:
            asset_id: Asset ID
            
        Returns:
            Asset entity
            
        Raises:
            EntityNotFoundError: If asset not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM assets WHERE id = $1"
                row = await conn.fetchrow(query, asset_id)
                
                if not row:
                    raise EntityNotFoundError(f"Asset with ID {asset_id} not found")
                
                return Asset(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get asset by ID {asset_id}: {e}")
            raise DatabaseOperationError(f"Failed to get asset: {e}")
    
    async def get_assets_by_user(self, user_id: str, filters: Optional[AssetFilters] = None) -> List[Asset]:
        """
        Get all assets for a user with optional filtering.
        
        Args:
            user_id: User ID
            filters: Optional filters to apply
            
        Returns:
            List of assets
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                # Build query with filters
                query_parts = ["SELECT * FROM assets WHERE userId = $1"]
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
                    
                    if filters.minValue is not None:
                        param_count += 1
                        query_parts.append(f"AND currentValue >= ${param_count}")
                        params.append(filters.minValue)
                    
                    if filters.maxValue is not None:
                        param_count += 1
                        query_parts.append(f"AND currentValue <= ${param_count}")
                        params.append(filters.maxValue)
                
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
                
                return [Asset(**dict(row)) for row in rows]
                
        except Exception as e:
            self._logger.error(f"Failed to get assets for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get assets: {e}")
    
    async def update_asset(self, asset_id: str, update_data: AssetUpdate) -> Asset:
        """
        Update an asset.
        
        Args:
            asset_id: Asset ID
            update_data: Asset update data
            
        Returns:
            Updated asset
            
        Raises:
            EntityNotFoundError: If asset not found
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
                
                if update_data.category is not None:
                    param_count += 1
                    update_fields.append(f"category = ${param_count}")
                    params.append(update_data.category)
                
                if update_data.subcategory is not None:
                    param_count += 1
                    update_fields.append(f"subcategory = ${param_count}")
                    params.append(update_data.subcategory)
                
                if update_data.currentValue is not None:
                    param_count += 1
                    update_fields.append(f"currentValue = ${param_count}")
                    params.append(update_data.currentValue)
                
                if update_data.purchaseValue is not None:
                    param_count += 1
                    update_fields.append(f"purchaseValue = ${param_count}")
                    params.append(update_data.purchaseValue)
                
                if update_data.purchaseDate is not None:
                    param_count += 1
                    update_fields.append(f"purchaseDate = ${param_count}")
                    params.append(update_data.purchaseDate)
                
                if update_data.description is not None:
                    param_count += 1
                    update_fields.append(f"description = ${param_count}")
                    params.append(update_data.description)
                
                if not update_fields:
                    # No fields to update, just return the current asset
                    return await self.get_asset_by_id(asset_id)
                
                # Add updatedAt timestamp
                param_count += 1
                update_fields.append(f"updatedAt = ${param_count}")
                params.append(datetime.utcnow())
                
                # Add asset_id to params
                param_count += 1
                params.append(asset_id)
                
                query = f"""
                    UPDATE assets 
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                    RETURNING *
                """
                
                row = await conn.fetchrow(query, *params)
                
                if not row:
                    raise EntityNotFoundError(f"Asset with ID {asset_id} not found")
                
                return Asset(**dict(row))
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to update asset {asset_id}: {e}")
            raise DatabaseOperationError(f"Failed to update asset: {e}")
    
    async def delete_asset(self, asset_id: str) -> bool:
        """
        Delete an asset.
        
        Args:
            asset_id: Asset ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            EntityNotFoundError: If asset not found
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = "DELETE FROM assets WHERE id = $1 RETURNING id"
                row = await conn.fetchrow(query, asset_id)
                
                if not row:
                    raise EntityNotFoundError(f"Asset with ID {asset_id} not found")
                
                return True
                
        except EntityNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to delete asset {asset_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete asset: {e}")
    
    async def get_asset_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get asset summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Asset summary statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_assets,
                        SUM(currentValue) as total_value,
                        AVG(currentValue) as avg_value,
                        MIN(currentValue) as min_value,
                        MAX(currentValue) as max_value
                    FROM assets 
                    WHERE userId = $1
                """
                
                row = await conn.fetchrow(query, user_id)
                
                return {
                    "totalAssets": row["total_assets"] or 0,
                    "totalValue": row["total_value"] or 0,
                    "avgValue": row["avg_value"] or 0,
                    "minValue": row["min_value"] or 0,
                    "maxValue": row["max_value"] or 0
                }
                
        except Exception as e:
            self._logger.error(f"Failed to get asset summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get asset summary: {e}")
    
    async def get_assets_by_category(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get assets grouped by category.
        
        Args:
            user_id: User ID
            
        Returns:
            List of category summaries with assets
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT 
                        category,
                        COUNT(*) as asset_count,
                        SUM(currentValue) as total_value,
                        AVG(currentValue) as avg_value
                    FROM assets 
                    WHERE userId = $1
                    GROUP BY category
                    ORDER BY total_value DESC
                """
                
                rows = await conn.fetch(query, user_id)
                
                return [
                    {
                        "category": row["category"],
                        "assetCount": row["asset_count"],
                        "totalValue": row["total_value"],
                        "avgValue": row["avg_value"]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            self._logger.error(f"Failed to get assets by category for user {user_id}: {e}")
            raise DatabaseOperationError(f"Failed to get assets by category: {e}")
    
    async def _generate_cuid(self, conn: asyncpg.Connection) -> str:
        """
        Generate a cuid for new assets.
        
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
