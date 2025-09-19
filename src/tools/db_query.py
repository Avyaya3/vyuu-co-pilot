"""
MCP-based Database query tool for read-only operations.

This tool uses Supabase MCP directly instead of the repository layer.
Handles queries like:
- Asset information retrieval
- Liability information retrieval
- Savings account information
- Income and expense data
- Stock investment data
- Insurance policy information
- User financial summary
"""

import time
import logging
from typing import Dict, Any, Literal, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from .base import ToolResponse

logger = logging.getLogger(__name__)


class DbQueryParams(BaseModel):
    """
    Parameters for database query operations.
    """
    
    operation: Literal[
        # Asset operations
        "get_user_assets",
        "get_asset_by_id",
        "get_asset_summary",
        "get_assets_by_category",
        
        # Liability operations
        "get_user_liabilities", 
        "get_liability_by_id",
        "get_liability_summary",
        "get_liabilities_by_type",
        
        # Savings operations
        "get_user_savings",
        "get_savings_by_id",
        "get_savings_summary",
        "get_savings_by_type",
        
        # Income operations
        "get_user_incomes",
        "get_income_by_id",
        "get_income_summary",
        "get_incomes_by_category",
        "get_incomes_by_frequency",
        
        # Expense operations
        "get_user_expenses",
        "get_expense_by_id",
        "get_expense_summary",
        "get_expenses_by_category",
        "get_expenses_by_subcategory",
        "get_expenses_by_payment_method",
        
        # Stock operations
        "get_user_stocks",
        "get_stock_by_id",
        "get_stock_summary",
        "get_stocks_by_type",
        
        # Insurance operations
        "get_user_insurances",
        "get_insurance_by_id",
        "get_insurance_summary",
        "get_insurances_by_type",
        "get_insurances_by_provider",
        
        # Goal operations
        "get_user_goals",
        "get_goal_by_id",
        "get_goals_by_category",
        "get_goals_by_priority",
        
        # Financial summary
        "get_financial_summary"
    ] = Field(description="Type of query operation to perform")
    
    # Common parameters
    user_id: Optional[str] = Field(None, description="User ID for user-specific queries")
    entity_id: Optional[str] = Field(None, description="Specific entity ID for single record queries")
    
    # Filter parameters
    category: Optional[str] = Field(None, description="Category filter")
    subcategory: Optional[str] = Field(None, description="Subcategory filter")
    type: Optional[str] = Field(None, description="Type filter")
    provider: Optional[str] = Field(None, description="Provider filter")
    name: Optional[str] = Field(None, description="Name filter")
    source: Optional[str] = Field(None, description="Source filter")
    frequency: Optional[str] = Field(None, description="Frequency filter")
    payment_method: Optional[str] = Field(None, description="Payment method filter")
    priority: Optional[str] = Field(None, description="Priority filter")
    
    # Amount filters
    min_amount: Optional[int] = Field(None, description="Minimum amount filter")
    max_amount: Optional[int] = Field(None, description="Maximum amount filter")
    min_premium: Optional[int] = Field(None, description="Minimum premium filter")
    max_premium: Optional[int] = Field(None, description="Maximum premium filter")
    min_coverage: Optional[int] = Field(None, description="Minimum coverage filter")
    max_coverage: Optional[int] = Field(None, description="Maximum coverage filter")
    min_current_value: Optional[int] = Field(None, description="Minimum current value filter")
    max_current_value: Optional[int] = Field(None, description="Maximum current value filter")
    min_target: Optional[int] = Field(None, description="Minimum target filter")
    max_target: Optional[int] = Field(None, description="Maximum target filter")
    
    # Date filters
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    
    # Pagination and ordering
    limit: Optional[int] = Field(50, ge=1, le=1000, description="Maximum number of records to return")
    offset: Optional[int] = Field(0, ge=0, description="Number of records to skip")
    order_by: Optional[str] = Field(None, description="Field to order by")
    order_direction: Optional[str] = Field("desc", description="Order direction (asc/desc)")
    
    class Config:
        extra = "ignore"  # Allow extra parameters but ignore them


class DbQueryTool:
    """
    MCP-based Database query tool for read-only operations.
    
    Uses Supabase MCP directly for database access instead of repository layer.
    Provides standardized interface for querying financial data.
    """
    
    name: Literal["db_query"] = "db_query"
    schema = DbQueryParams
    
    def __init__(self):
        # Supported entity types and their corresponding table names
        self.entity_tables = {
            "assets": "assets",
            "liabilities": "liabilities", 
            "savings": "savings",
            "incomes": "incomes",
            "expenses": "expenses",
            "stocks": "stocks",
            "insurances": "insurances",
            "goals": "goals"
        }
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a database query operation using Supabase MCP.
        
        Args:
            params: Query parameters matching DbQueryParams schema
            
        Returns:
            ToolResponse with query results or error information
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            validated_params = self.schema(**params)
            
            # Route to appropriate query method
            result = await self._execute_query(validated_params)
            
            execution_time = (time.time() - start_time) * 1000
            
            response = ToolResponse(
                success=True,
                data=result,
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            logger.info(
                f"MCP Query operation '{validated_params.operation}' completed successfully",
                extra={
                    "operation": validated_params.operation,
                    "execution_time_ms": execution_time,
                    "user_id": validated_params.user_id,
                    "entity_id": validated_params.entity_id
                }
            )
            
            return response.dict()
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(
                f"MCP Query operation failed: {str(e)}",
                extra={
                    "error": str(e),
                    "params": params,
                    "execution_time_ms": execution_time
                }
            )
            
            response = ToolResponse(
                success=False,
                error=f"MCP Query failed: {str(e)}",
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            return response.dict()
    
    async def _execute_query(self, params: DbQueryParams) -> Dict[str, Any]:
        """
        Execute the specific query operation using MCP.
        
        Args:
            params: Validated query parameters
            
        Returns:
            Query results as dictionary
        """
        # Asset operations
        if params.operation.startswith("get_user_assets"):
            return await self._get_user_assets(params)
        elif params.operation == "get_asset_by_id":
            return await self._get_asset_by_id(params)
        elif params.operation == "get_asset_summary":
            return await self._get_asset_summary(params)
        elif params.operation == "get_assets_by_category":
            return await self._get_assets_by_category(params)
        
        # Liability operations
        elif params.operation.startswith("get_user_liabilities"):
            return await self._get_user_liabilities(params)
        elif params.operation == "get_liability_by_id":
            return await self._get_liability_by_id(params)
        elif params.operation == "get_liability_summary":
            return await self._get_liability_summary(params)
        elif params.operation == "get_liabilities_by_type":
            return await self._get_liabilities_by_type(params)
        
        # Savings operations
        elif params.operation.startswith("get_user_savings"):
            return await self._get_user_savings(params)
        elif params.operation == "get_savings_by_id":
            return await self._get_savings_by_id(params)
        elif params.operation == "get_savings_summary":
            return await self._get_savings_summary(params)
        elif params.operation == "get_savings_by_type":
            return await self._get_savings_by_type(params)
        
        # Income operations
        elif params.operation.startswith("get_user_incomes"):
            return await self._get_user_incomes(params)
        elif params.operation == "get_income_by_id":
            return await self._get_income_by_id(params)
        elif params.operation == "get_income_summary":
            return await self._get_income_summary(params)
        elif params.operation == "get_incomes_by_category":
            return await self._get_incomes_by_category(params)
        elif params.operation == "get_incomes_by_frequency":
            return await self._get_incomes_by_frequency(params)
        
        # Expense operations
        elif params.operation.startswith("get_user_expenses"):
            return await self._get_user_expenses(params)
        elif params.operation == "get_expense_by_id":
            return await self._get_expense_by_id(params)
        elif params.operation == "get_expense_summary":
            return await self._get_expense_summary(params)
        elif params.operation == "get_expenses_by_category":
            return await self._get_expenses_by_category(params)
        elif params.operation == "get_expenses_by_subcategory":
            return await self._get_expenses_by_subcategory(params)
        elif params.operation == "get_expenses_by_payment_method":
            return await self._get_expenses_by_payment_method(params)
        
        # Stock operations
        elif params.operation.startswith("get_user_stocks"):
            return await self._get_user_stocks(params)
        elif params.operation == "get_stock_by_id":
            return await self._get_stock_by_id(params)
        elif params.operation == "get_stock_summary":
            return await self._get_stock_summary(params)
        elif params.operation == "get_stocks_by_type":
            return await self._get_stocks_by_type(params)
        
        # Insurance operations
        elif params.operation.startswith("get_user_insurances"):
            return await self._get_user_insurances(params)
        elif params.operation == "get_insurance_by_id":
            return await self._get_insurance_by_id(params)
        elif params.operation == "get_insurance_summary":
            return await self._get_insurance_summary(params)
        elif params.operation == "get_insurances_by_type":
            return await self._get_insurances_by_type(params)
        elif params.operation == "get_insurances_by_provider":
            return await self._get_insurances_by_provider(params)
        
        # Goal operations
        elif params.operation.startswith("get_user_goals"):
            return await self._get_user_goals(params)
        elif params.operation == "get_goal_by_id":
            return await self._get_goal_by_id(params)
        elif params.operation == "get_goals_by_category":
            return await self._get_goals_by_category(params)
        elif params.operation == "get_goals_by_priority":
            return await self._get_goals_by_priority(params)
        
        # Financial summary
        elif params.operation == "get_financial_summary":
            return await self._get_financial_summary(params)
        
        else:
            raise ValueError(f"Unsupported operation: {params.operation}")
    
    async def _execute_sql(self, sql: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query using direct database connection.
        
        Args:
            sql: SQL query string
            params: Query parameters for parameterized queries
            
        Returns:
            Query results as list of dictionaries
        """
        try:
            import asyncpg
            import os
            import asyncio
            from dotenv import load_dotenv
            
            # Load environment variables in a separate thread to avoid blocking
            await asyncio.to_thread(load_dotenv)
            
            # Get database URL from environment and add SSL bypass
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise Exception("DATABASE_URL not found in environment")
            
            # Add SSL bypass to the database URL to avoid certificate issues
            if '?sslmode=disable' not in database_url:
                database_url += '?sslmode=disable'
            
            # Connect to database and execute query
            conn = await asyncpg.connect(database_url)
            
            try:
                if params:
                    rows = await conn.fetch(sql, *params)
                else:
                    rows = await conn.fetch(sql)
                
                # Convert asyncpg.Record objects to dictionaries
                results = [dict(row) for row in rows]
                
                logger.info(f"SQL executed successfully: {len(results)} rows returned")
                return results
                
            finally:
                await conn.close()
            
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            raise Exception(f"Database query failed: {str(e)}")
    def _build_where_clause(self, params: DbQueryParams, table_name: str) -> tuple[str, List[Any]]:
        """
        Build WHERE clause and parameters for SQL query.
        
        Args:
            params: Query parameters
            table_name: Database table name
            
        Returns:
            Tuple of (WHERE clause, parameter list)
        """
        conditions = []
        query_params = []
        
        # Always filter by user_id if provided
        if params.user_id:
            conditions.append("\"userId\" = $%d" % (len(query_params) + 1))
            query_params.append(params.user_id)
        
        # Entity-specific filters
        if params.entity_id:
            conditions.append("id = $%d" % (len(query_params) + 1))
            query_params.append(params.entity_id)
        
        # Common filters
        if params.category:
            conditions.append("category = $%d" % (len(query_params) + 1))
            query_params.append(params.category)
        
        if params.subcategory:
            conditions.append("subcategory = $%d" % (len(query_params) + 1))
            query_params.append(params.subcategory)
        
        if params.type:
            conditions.append("type = $%d" % (len(query_params) + 1))
            query_params.append(params.type)
        
        if params.provider:
            conditions.append("provider = $%d" % (len(query_params) + 1))
            query_params.append(params.provider)
        
        if params.name:
            conditions.append("name ILIKE $%d" % (len(query_params) + 1))
            query_params.append(f"%{params.name}%")
        
        if params.source:
            conditions.append("source = $%d" % (len(query_params) + 1))
            query_params.append(params.source)
        
        if params.frequency:
            conditions.append("frequency = $%d" % (len(query_params) + 1))
            query_params.append(params.frequency)
        
        if params.payment_method:
            conditions.append("paymentMethod = $%d" % (len(query_params) + 1))
            query_params.append(params.payment_method)
        
        if params.priority:
            conditions.append("priority = $%d" % (len(query_params) + 1))
            query_params.append(params.priority)
        
        # Amount filters
        if params.min_amount is not None:
            conditions.append("amount >= $%d" % (len(query_params) + 1))
            query_params.append(params.min_amount)
        
        if params.max_amount is not None:
            conditions.append("amount <= $%d" % (len(query_params) + 1))
            query_params.append(params.max_amount)
        
        if params.min_premium is not None:
            conditions.append("premium >= $%d" % (len(query_params) + 1))
            query_params.append(params.min_premium)
        
        if params.max_premium is not None:
            conditions.append("premium <= $%d" % (len(query_params) + 1))
            query_params.append(params.max_premium)
        
        if params.min_coverage is not None:
            conditions.append("coverage >= $%d" % (len(query_params) + 1))
            query_params.append(params.min_coverage)
        
        if params.max_coverage is not None:
            conditions.append("coverage <= $%d" % (len(query_params) + 1))
            query_params.append(params.max_coverage)
        
        if params.min_current_value is not None:
            conditions.append("currentValue >= $%d" % (len(query_params) + 1))
            query_params.append(params.min_current_value)
        
        if params.max_current_value is not None:
            conditions.append("currentValue <= $%d" % (len(query_params) + 1))
            query_params.append(params.max_current_value)
        
        if params.min_target is not None:
            conditions.append("target >= $%d" % (len(query_params) + 1))
            query_params.append(params.min_target)
        
        if params.max_target is not None:
            conditions.append("target <= $%d" % (len(query_params) + 1))
            query_params.append(params.max_target)
        
        # Date filters
        if params.start_date:
            conditions.append("createdAt >= $%d" % (len(query_params) + 1))
            query_params.append(params.start_date.isoformat())
        
        if params.end_date:
            conditions.append("createdAt <= $%d" % (len(query_params) + 1))
            query_params.append(params.end_date.isoformat())
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, query_params
    
    def _build_order_clause(self, params: DbQueryParams) -> str:
        """Build ORDER BY clause for SQL query."""
        if params.order_by:
            direction = params.order_direction.upper() if params.order_direction else "DESC"
            return f"ORDER BY {params.order_by} {direction}"
        return "ORDER BY \"createdAt\" DESC"
    
    def _build_limit_clause(self, params: DbQueryParams) -> str:
        """Build LIMIT and OFFSET clause for SQL query."""
        limit = params.limit or 50
        offset = params.offset or 0
        return f"LIMIT {limit} OFFSET {offset}"
    
    # Asset methods
    async def _get_user_assets(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all assets for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "assets")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", name, category, subcategory, "currentValue", 
                   "purchaseValue", "purchaseDate", description, "createdAt", "updatedAt"
            FROM assets 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "asset_count": len(results),
            "assets": results
        }
    
    async def _get_asset_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific asset by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", name, category, subcategory, "currentValue", 
                   "purchaseValue", "purchaseDate", description, "createdAt", "updatedAt"
            FROM assets 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Asset with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_asset_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get asset summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_assets,
                SUM("currentValue") as total_value,
                SUM("purchaseValue") as total_purchase_value,
                AVG("currentValue") as avg_value,
                MIN("currentValue") as min_value,
                MAX("currentValue") as max_value
            FROM assets 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_assets_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get assets grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                category,
                COUNT(*) as count,
                SUM("currentValue") as total_value,
                AVG("currentValue") as avg_value
            FROM assets 
            WHERE "userId" = $1
            GROUP BY category
            ORDER BY total_value DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "categories": results
        }
    
    # Liability methods
    async def _get_user_liabilities(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all liabilities for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "liabilities")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", name, type, amount, emi, "interestRate", 
                   "startDate", "endDate", description, "createdAt", "updatedAt"
            FROM liabilities 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "liability_count": len(results),
            "liabilities": results
        }
    
    async def _get_liability_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific liability by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", name, type, amount, emi, "interestRate", 
                   "startDate", "endDate", description, "createdAt", "updatedAt"
            FROM liabilities 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Liability with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_liability_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get liability summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_liabilities,
                SUM(amount) as total_amount,
                SUM(emi) as total_emi,
                AVG("interestRate") as avg_interest_rate,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM liabilities 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_liabilities_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get liabilities grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                type,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                SUM(emi) as total_emi,
                AVG("interestRate") as avg_interest_rate
            FROM liabilities 
            WHERE "userId" = $1
            GROUP BY type
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "types": results
        }
    
    # Income methods
    async def _get_user_incomes(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all incomes for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "incomes")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", source, amount, frequency, category, date, 
                   description, "createdAt", "updatedAt"
            FROM incomes 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "income_count": len(results),
            "incomes": results
        }
    
    async def _get_income_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific income by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", source, amount, frequency, category, date, 
                   description, "createdAt", "updatedAt"
            FROM incomes 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Income with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_income_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get income summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_incomes,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM incomes 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_incomes_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get incomes grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                category,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM incomes 
            WHERE "userId" = $1
            GROUP BY category
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "categories": results
        }
    
    async def _get_incomes_by_frequency(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get incomes grouped by frequency."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                frequency,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM incomes 
            WHERE "userId" = $1
            GROUP BY frequency
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "frequencies": results
        }
    
    # Expense methods
    async def _get_user_expenses(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all expenses for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "expenses")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", category, subcategory, amount, date, 
                   description, "paymentMethod", "createdAt", "updatedAt"
            FROM expenses 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "expense_count": len(results),
            "expenses": results
        }
    
    async def _get_expense_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific expense by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", category, subcategory, amount, date, 
                   description, "paymentMethod", "createdAt", "updatedAt"
            FROM expenses 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Expense with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_expense_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expense summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_expenses,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM expenses 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_expenses_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                category,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM expenses 
            WHERE "userId" = $1
            GROUP BY category
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "categories": results
        }
    
    async def _get_expenses_by_subcategory(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by subcategory."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                subcategory,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM expenses 
            WHERE "userId" = $1
            GROUP BY subcategory
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "subcategories": results
        }
    
    async def _get_expenses_by_payment_method(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by payment method."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                "paymentMethod",
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM expenses 
            WHERE "userId" = $1
            GROUP BY "paymentMethod"
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "payment_methods": results
        }
    
    # Stock methods
    async def _get_user_stocks(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all stocks for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "stocks")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", name, type, amount, "currentValue", 
                   "purchaseDate", returns, description, "createdAt", "updatedAt"
            FROM stocks 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "stock_count": len(results),
            "stocks": results
        }
    
    async def _get_stock_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific stock by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", name, type, amount, "currentValue", 
                   "purchaseDate", returns, description, "createdAt", "updatedAt"
            FROM stocks 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Stock with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_stock_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get stock summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_stocks,
                SUM(amount) as total_amount,
                SUM("currentValue") as total_current_value,
                AVG(returns) as avg_returns,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM stocks 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_stocks_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get stocks grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                type,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                SUM("currentValue") as total_current_value,
                AVG(returns) as avg_returns
            FROM stocks 
            WHERE "userId" = $1
            GROUP BY type
            ORDER BY total_current_value DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "types": results
        }
    
    # Insurance methods
    async def _get_user_insurances(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all insurances for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "insurances")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", type, provider, "policyNumber", premium, 
                   coverage, "startDate", "endDate", description, "createdAt", "updatedAt"
            FROM insurances 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "insurance_count": len(results),
            "insurances": results
        }
    
    async def _get_insurance_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific insurance by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", type, provider, "policyNumber", premium, 
                   coverage, "startDate", "endDate", description, "createdAt", "updatedAt"
            FROM insurances 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Insurance with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_insurance_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurance summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_insurances,
                SUM(premium) as total_premium,
                SUM(coverage) as total_coverage,
                AVG(premium) as avg_premium,
                MIN(premium) as min_premium,
                MAX(premium) as max_premium
            FROM insurances 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_insurances_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurances grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                type,
                COUNT(*) as count,
                SUM(premium) as total_premium,
                SUM(coverage) as total_coverage,
                AVG(premium) as avg_premium
            FROM insurances 
            WHERE "userId" = $1
            GROUP BY type
            ORDER BY total_premium DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "types": results
        }
    
    async def _get_insurances_by_provider(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurances grouped by provider."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                provider,
                COUNT(*) as count,
                SUM(premium) as total_premium,
                SUM(coverage) as total_coverage,
                AVG(premium) as avg_premium
            FROM insurances 
            WHERE "userId" = $1
            GROUP BY provider
            ORDER BY total_premium DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "providers": results
        }
    
    # Goal methods
    async def _get_user_goals(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all goals for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "goals")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", name, target, current, "targetDate", 
                   category, priority, description, "createdAt", "updatedAt"
            FROM goals 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "goal_count": len(results),
            "goals": results
        }
    
    async def _get_goal_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific goal by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", name, target, current, "targetDate", 
                   category, priority, description, "createdAt", "updatedAt"
            FROM goals 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Goal with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_goals_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get goals grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                category,
                COUNT(*) as count,
                SUM(target) as total_target,
                SUM(current) as total_current,
                AVG(target) as avg_target
            FROM goals 
            WHERE "userId" = $1
            GROUP BY category
            ORDER BY total_target DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "categories": results
        }
    
    async def _get_goals_by_priority(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get goals grouped by priority."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                priority,
                COUNT(*) as count,
                SUM(target) as total_target,
                SUM(current) as total_current,
                AVG(target) as avg_target
            FROM goals 
            WHERE "userId" = $1
            GROUP BY priority
            ORDER BY total_target DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "priorities": results
        }
    
    # Financial summary
    async def _get_financial_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get comprehensive financial summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        # Get summaries from all entities
        asset_summary = await self._get_asset_summary(params)
        liability_summary = await self._get_liability_summary(params)
        savings_summary = await self._get_savings_summary(params)
        income_summary = await self._get_income_summary(params)
        expense_summary = await self._get_expense_summary(params)
        stock_summary = await self._get_stock_summary(params)
        insurance_summary = await self._get_insurance_summary(params)
        
        # Calculate net worth
        total_assets = (
            asset_summary.get("summary", {}).get("total_value", 0) +
            savings_summary.get("summary", {}).get("total_balance", 0) +
            stock_summary.get("summary", {}).get("total_current_value", 0)
        )
        total_liabilities = liability_summary.get("summary", {}).get("total_amount", 0)
        net_worth = total_assets - total_liabilities
        
        return {
            "user_id": params.user_id,
            "net_worth": net_worth,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "asset_summary": asset_summary.get("summary", {}),
            "liability_summary": liability_summary.get("summary", {}),
            "savings_summary": savings_summary.get("summary", {}),
            "income_summary": income_summary.get("summary", {}),
            "expense_summary": expense_summary.get("summary", {}),
            "stock_summary": stock_summary.get("summary", {}),
            "insurance_summary": insurance_summary.get("summary", {})
        }
    
    # Savings methods
    async def _get_user_savings(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all savings for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "savings")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", name, type, "currentBalance", "interestRate", 
                   "maturityDate", "monthlyContribution", "targetAmount", 
                   description, "createdAt", "updatedAt"
            FROM savings 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "savings_count": len(results),
            "savings": results
        }
    
    async def _get_savings_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific savings by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", name, type, "currentBalance", "interestRate", 
                   "maturityDate", "monthlyContribution", "targetAmount", 
                   description, "createdAt", "updatedAt"
            FROM savings 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Savings with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_savings_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get savings summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_savings,
                SUM("currentBalance") as total_balance,
                SUM("monthlyContribution") as total_monthly_contribution,
                AVG("interestRate") as avg_interest_rate,
                MIN("currentBalance") as min_balance,
                MAX("currentBalance") as max_balance
            FROM savings 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_savings_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get savings grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                type,
                COUNT(*) as count,
                SUM("currentBalance") as total_balance,
                SUM("monthlyContribution") as total_monthly_contribution,
                AVG("interestRate") as avg_interest_rate
            FROM savings 
            WHERE "userId" = $1
            GROUP BY type
            ORDER BY total_balance DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "types": results
        }
    
    # Income methods
    async def _get_user_incomes(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all incomes for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "incomes")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", source, amount, frequency, category, date, 
                   description, "createdAt", "updatedAt"
            FROM incomes 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "income_count": len(results),
            "incomes": results
        }
    
    async def _get_income_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific income by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", source, amount, frequency, category, date, 
                   description, "createdAt", "updatedAt"
            FROM incomes 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Income with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_income_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get income summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_incomes,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM incomes 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_incomes_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get incomes grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                category,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM incomes 
            WHERE "userId" = $1
            GROUP BY category
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "categories": results
        }
    
    async def _get_incomes_by_frequency(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get incomes grouped by frequency."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                frequency,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM incomes 
            WHERE "userId" = $1
            GROUP BY frequency
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "frequencies": results
        }
    
    # Expense methods
    async def _get_user_expenses(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all expenses for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "expenses")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", category, subcategory, amount, date, 
                   description, "paymentMethod", "createdAt", "updatedAt"
            FROM expenses 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "expense_count": len(results),
            "expenses": results
        }
    
    async def _get_expense_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific expense by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", category, subcategory, amount, date, 
                   description, "paymentMethod", "createdAt", "updatedAt"
            FROM expenses 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Expense with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_expense_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expense summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_expenses,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM expenses 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_expenses_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                category,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM expenses 
            WHERE "userId" = $1
            GROUP BY category
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "categories": results
        }
    
    async def _get_expenses_by_subcategory(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by subcategory."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                subcategory,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM expenses 
            WHERE "userId" = $1
            GROUP BY subcategory
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "subcategories": results
        }
    
    async def _get_expenses_by_payment_method(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by payment method."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                "paymentMethod",
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM expenses 
            WHERE "userId" = $1
            GROUP BY "paymentMethod"
            ORDER BY total_amount DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "payment_methods": results
        }
    
    # Stock methods
    async def _get_user_stocks(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all stocks for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "stocks")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", name, type, amount, "currentValue", 
                   "purchaseDate", returns, description, "createdAt", "updatedAt"
            FROM stocks 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "stock_count": len(results),
            "stocks": results
        }
    
    async def _get_stock_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific stock by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", name, type, amount, "currentValue", 
                   "purchaseDate", returns, description, "createdAt", "updatedAt"
            FROM stocks 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Stock with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_stock_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get stock summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_stocks,
                SUM(amount) as total_amount,
                SUM("currentValue") as total_current_value,
                AVG(returns) as avg_returns,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount
            FROM stocks 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_stocks_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get stocks grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                type,
                COUNT(*) as count,
                SUM(amount) as total_amount,
                SUM("currentValue") as total_current_value,
                AVG(returns) as avg_returns
            FROM stocks 
            WHERE "userId" = $1
            GROUP BY type
            ORDER BY total_current_value DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "types": results
        }
    
    # Insurance methods
    async def _get_user_insurances(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all insurances for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "insurances")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", type, provider, "policyNumber", premium, 
                   coverage, "startDate", "endDate", description, "createdAt", "updatedAt"
            FROM insurances 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "insurance_count": len(results),
            "insurances": results
        }
    
    async def _get_insurance_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific insurance by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", type, provider, "policyNumber", premium, 
                   coverage, "startDate", "endDate", description, "createdAt", "updatedAt"
            FROM insurances 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Insurance with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_insurance_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurance summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                COUNT(*) as total_insurances,
                SUM(premium) as total_premium,
                SUM(coverage) as total_coverage,
                AVG(premium) as avg_premium,
                MIN(premium) as min_premium,
                MAX(premium) as max_premium
            FROM insurances 
            WHERE "userId" = $1
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "summary": results[0] if results else {}
        }
    
    async def _get_insurances_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurances grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                type,
                COUNT(*) as count,
                SUM(premium) as total_premium,
                SUM(coverage) as total_coverage,
                AVG(premium) as avg_premium
            FROM insurances 
            WHERE "userId" = $1
            GROUP BY type
            ORDER BY total_premium DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "types": results
        }
    
    async def _get_insurances_by_provider(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurances grouped by provider."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                provider,
                COUNT(*) as count,
                SUM(premium) as total_premium,
                SUM(coverage) as total_coverage,
                AVG(premium) as avg_premium
            FROM insurances 
            WHERE "userId" = $1
            GROUP BY provider
            ORDER BY total_premium DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "providers": results
        }
    
    # Goal methods
    async def _get_user_goals(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all goals for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        where_clause, query_params = self._build_where_clause(params, "goals")
        order_clause = self._build_order_clause(params)
        limit_clause = self._build_limit_clause(params)
        
        sql = f"""
            SELECT id, "userId", name, target, current, "targetDate", 
                   category, priority, description, "createdAt", "updatedAt"
            FROM goals 
            WHERE {where_clause}
            {order_clause}
            {limit_clause}
        """
        
        results = await self._execute_sql(sql, query_params)
        
        return {
            "user_id": params.user_id,
            "goal_count": len(results),
            "goals": results
        }
    
    async def _get_goal_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific goal by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        sql = """
            SELECT id, "userId", name, target, current, "targetDate", 
                   category, priority, description, "createdAt", "updatedAt"
            FROM goals 
            WHERE id = $1
        """
        
        results = await self._execute_sql(sql, [params.entity_id])
        
        if not results:
            raise ValueError(f"Goal with ID {params.entity_id} not found")
        
        return results[0]
    
    async def _get_goals_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get goals grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                category,
                COUNT(*) as count,
                SUM(target) as total_target,
                SUM(current) as total_current,
                AVG(target) as avg_target
            FROM goals 
            WHERE "userId" = $1
            GROUP BY category
            ORDER BY total_target DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "categories": results
        }
    
    async def _get_goals_by_priority(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get goals grouped by priority."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        sql = """
            SELECT 
                priority,
                COUNT(*) as count,
                SUM(target) as total_target,
                SUM(current) as total_current,
                AVG(target) as avg_target
            FROM goals 
            WHERE "userId" = $1
            GROUP BY priority
            ORDER BY total_target DESC
        """
        
        results = await self._execute_sql(sql, [params.user_id])
        
        return {
            "user_id": params.user_id,
            "priorities": results
        }
    
    # Financial summary
    async def _get_financial_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get comprehensive financial summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        # Get summaries from all entities
        asset_summary = await self._get_asset_summary(params)
        liability_summary = await self._get_liability_summary(params)
        savings_summary = await self._get_savings_summary(params)
        income_summary = await self._get_income_summary(params)
        expense_summary = await self._get_expense_summary(params)
        stock_summary = await self._get_stock_summary(params)
        insurance_summary = await self._get_insurance_summary(params)
        
        # Calculate net worth
        total_assets = (
            asset_summary.get("summary", {}).get("total_value", 0) +
            savings_summary.get("summary", {}).get("total_balance", 0) +
            stock_summary.get("summary", {}).get("total_current_value", 0)
        )
        total_liabilities = liability_summary.get("summary", {}).get("total_amount", 0)
        net_worth = total_assets - total_liabilities
        
        return {
            "user_id": params.user_id,
            "net_worth": net_worth,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "asset_summary": asset_summary.get("summary", {}),
            "liability_summary": liability_summary.get("summary", {}),
            "savings_summary": savings_summary.get("summary", {}),
            "income_summary": income_summary.get("summary", {}),
            "expense_summary": expense_summary.get("summary", {}),
            "stock_summary": stock_summary.get("summary", {}),
            "insurance_summary": insurance_summary.get("summary", {})
        }

    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the database query tool with the given parameters.
        
        This method implements the ToolInterface protocol and handles
        Supabase JWT token authentication for MCP calls.
        
        Args:
            params: Dictionary of parameters including:
                - operation: The database operation to perform
                - user_id: User ID for the query
                - supabase_jwt_token: Optional Supabase JWT token for MCP authentication
                - Other operation-specific parameters
                
        Returns:
            Standardized ToolResponse dictionary
        """
        start_time = time.time()
        
        try:
            # Extract Supabase JWT token from params (passed from state metadata)
            supabase_jwt_token = params.pop("supabase_jwt_token", None)
            
            # Validate parameters
            validated_params = self.schema(**params)
            
            # Execute the operation
            result_data = await self._execute_operation(validated_params, supabase_jwt_token)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "success": True,
                "data": result_data,
                "error": None,
                "tool_name": self.name,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Database query tool execution failed: {e}")
            
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "tool_name": self.name,
                "execution_time_ms": execution_time
            }
    
    async def _execute_operation(self, params: DbQueryParams, supabase_jwt_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the specific database operation.
        
        Args:
            params: Validated parameters for the operation
            supabase_jwt_token: Optional Supabase JWT token for MCP authentication
            
        Returns:
            Operation result data
        """
        # If we have a Supabase JWT token, we can use MCP for the operation
        if supabase_jwt_token:
            return await self._execute_with_mcp(params, supabase_jwt_token)
        else:
            # Fall back to repository-based execution
            return await self._execute_with_repository(params)
    
    async def _execute_with_mcp(self, params: DbQueryParams, supabase_jwt_token: str) -> Dict[str, Any]:
        """
        Execute the operation using Supabase MCP with JWT authentication.
        
        Args:
            params: Validated parameters for the operation
            supabase_jwt_token: Supabase JWT token for authentication
            
        Returns:
            Operation result data
        """
        # TODO: Implement MCP-based execution with Supabase JWT token
        # This would involve calling the Supabase MCP with the JWT token in the Authorization header
        # For now, fall back to repository-based execution
        logger.info(f"Executing {params.operation} with Supabase JWT token for user {params.user_id}")
        return await self._execute_with_repository(params)
    
    async def _execute_with_repository(self, params: DbQueryParams) -> Dict[str, Any]:
        """
        Execute the operation using the repository layer (fallback).
        
        Args:
            params: Validated parameters for the operation
            
        Returns:
            Operation result data
        """
        # Route to the appropriate operation handler
        operation_handlers = {
            # Asset operations
            "get_user_assets": self._get_user_assets,
            "get_asset_by_id": self._get_asset_by_id,
            "get_asset_summary": self._get_asset_summary,
            "get_assets_by_category": self._get_assets_by_category,
            
            # Liability operations
            "get_user_liabilities": self._get_user_liabilities,
            "get_liability_by_id": self._get_liability_by_id,
            "get_liability_summary": self._get_liability_summary,
            "get_liabilities_by_type": self._get_liabilities_by_type,
            
            # Savings operations
            "get_user_savings": self._get_user_savings,
            "get_savings_by_id": self._get_savings_by_id,
            "get_savings_summary": self._get_savings_summary,
            "get_savings_by_type": self._get_savings_by_type,
            
            # Income operations
            "get_user_incomes": self._get_user_incomes,
            "get_income_by_id": self._get_income_by_id,
            "get_income_summary": self._get_income_summary,
            "get_incomes_by_category": self._get_incomes_by_category,
            "get_incomes_by_frequency": self._get_incomes_by_frequency,
            
            # Expense operations
            "get_user_expenses": self._get_user_expenses,
            "get_expense_by_id": self._get_expense_by_id,
            "get_expense_summary": self._get_expense_summary,
            "get_expenses_by_category": self._get_expenses_by_category,
            "get_expenses_by_subcategory": self._get_expenses_by_subcategory,
            "get_expenses_by_payment_method": self._get_expenses_by_payment_method,
            
            # Stock operations
            "get_user_stocks": self._get_user_stocks,
            "get_stock_by_id": self._get_stock_by_id,
            "get_stock_summary": self._get_stock_summary,
            "get_stocks_by_type": self._get_stocks_by_type,
            
            # Insurance operations
            "get_user_insurances": self._get_user_insurances,
            "get_insurance_by_id": self._get_insurance_by_id,
            "get_insurance_summary": self._get_insurance_summary,
            "get_insurances_by_type": self._get_insurances_by_type,
            "get_insurances_by_provider": self._get_insurances_by_provider,
            
            # Goal operations
            "get_user_goals": self._get_user_goals,
            "get_goal_by_id": self._get_goal_by_id,
            "get_goals_by_category": self._get_goals_by_category,
            "get_goals_by_priority": self._get_goals_by_priority,
            
            # Financial summary
            "get_financial_summary": self._get_financial_summary
        }
        
        handler = operation_handlers.get(params.operation)
        if not handler:
            raise ValueError(f"Unsupported operation: {params.operation}")
        
        return await handler(params)
