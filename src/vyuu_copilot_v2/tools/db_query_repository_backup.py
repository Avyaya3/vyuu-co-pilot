"""
Database query tool for read-only operations.

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

from ..repositories import (
    get_asset_repository, get_liability_repository, get_savings_repository,
    get_income_repository, get_expense_repository, get_stock_repository,
    get_insurance_repository, get_goal_repository
)
from ..schemas.database_models import (
    AssetFilters, LiabilityFilters, SavingsFilters, IncomeFilters,
    ExpenseFilters, StockFilters, InsuranceFilters, GoalFilters
)
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
    Database query tool for read-only operations.
    
    Provides standardized interface for querying financial data
    while maintaining separation between intent parameters and database operations.
    """
    
    name: Literal["db_query"] = "db_query"
    schema = DbQueryParams
    
    def __init__(self):
        # Initialize all repositories
        self.asset_repo = get_asset_repository()
        self.liability_repo = get_liability_repository()
        self.savings_repo = get_savings_repository()
        self.income_repo = get_income_repository()
        self.expense_repo = get_expense_repository()
        self.stock_repo = get_stock_repository()
        self.insurance_repo = get_insurance_repository()
        self.goal_repo = get_goal_repository()
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a database query operation.
        
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
                f"Query operation '{validated_params.operation}' completed successfully",
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
                f"Query operation failed: {str(e)}",
                extra={
                    "error": str(e),
                    "params": params,
                    "execution_time_ms": execution_time
                }
            )
            
            response = ToolResponse(
                success=False,
                error=f"Query failed: {str(e)}",
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            return response.dict()
    
    async def _execute_query(self, params: DbQueryParams) -> Dict[str, Any]:
        """
        Execute the specific query operation.
        
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
    
    def _build_filters(self, params: DbQueryParams, filter_class) -> filter_class:
        """Build filter object from parameters."""
        filter_data = {"userId": params.user_id}
        
        if params.category:
            filter_data["category"] = params.category
        if params.subcategory:
            filter_data["subcategory"] = params.subcategory
        if params.type:
            filter_data["type"] = params.type
        if params.provider:
            filter_data["provider"] = params.provider
        if params.name:
            filter_data["name"] = params.name
        if params.source:
            filter_data["source"] = params.source
        if params.frequency:
            filter_data["frequency"] = params.frequency
        if params.payment_method:
            filter_data["paymentMethod"] = params.payment_method
        if params.priority:
            filter_data["priority"] = params.priority
        
        # Amount filters
        if params.min_amount is not None:
            filter_data["minAmount"] = params.min_amount
        if params.max_amount is not None:
            filter_data["maxAmount"] = params.max_amount
        if params.min_premium is not None:
            filter_data["minPremium"] = params.min_premium
        if params.max_premium is not None:
            filter_data["maxPremium"] = params.max_premium
        if params.min_coverage is not None:
            filter_data["minCoverage"] = params.min_coverage
        if params.max_coverage is not None:
            filter_data["maxCoverage"] = params.max_coverage
        if params.min_current_value is not None:
            filter_data["minCurrentValue"] = params.min_current_value
        if params.max_current_value is not None:
            filter_data["maxCurrentValue"] = params.max_current_value
        if params.min_target is not None:
            filter_data["minTarget"] = params.min_target
        if params.max_target is not None:
            filter_data["maxTarget"] = params.max_target
        
        # Date filters
        if params.start_date:
            filter_data["startDate"] = params.start_date
        if params.end_date:
            filter_data["endDate"] = params.end_date
        
        # Pagination and ordering
        if params.limit:
            filter_data["limit"] = params.limit
        if params.offset:
            filter_data["offset"] = params.offset
        if params.order_by:
            filter_data["orderBy"] = params.order_by
        if params.order_direction:
            filter_data["orderDirection"] = params.order_direction
        
        return filter_class(**filter_data)
    
    # Asset methods
    async def _get_user_assets(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all assets for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, AssetFilters)
        assets = await self.asset_repo.get_assets_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "asset_count": len(assets),
            "assets": [
                {
                    "id": asset.id,
                    "name": asset.name,
                    "category": asset.category,
                    "subcategory": asset.subcategory,
                    "currentValue": asset.currentValue,
                    "purchaseValue": asset.purchaseValue,
                    "purchaseDate": asset.purchaseDate.isoformat(),
                    "description": asset.description,
                    "createdAt": asset.createdAt.isoformat()
                }
                for asset in assets
            ]
        }
    
    async def _get_asset_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific asset by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        asset = await self.asset_repo.get_asset_by_id(params.entity_id)
        
        return {
            "id": asset.id,
            "userId": asset.userId,
            "name": asset.name,
            "category": asset.category,
            "subcategory": asset.subcategory,
            "currentValue": asset.currentValue,
            "purchaseValue": asset.purchaseValue,
            "purchaseDate": asset.purchaseDate.isoformat(),
            "description": asset.description,
            "createdAt": asset.createdAt.isoformat(),
            "updatedAt": asset.updatedAt.isoformat()
        }
    
    async def _get_asset_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get asset summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        summary = await self.asset_repo.get_asset_summary(params.user_id)
        
        return {
            "user_id": params.user_id,
            "summary": summary
        }
    
    async def _get_assets_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get assets grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        categories = await self.asset_repo.get_assets_by_category(params.user_id)
        
        return {
            "user_id": params.user_id,
            "categories": categories
        }
    
    # Liability methods
    async def _get_user_liabilities(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all liabilities for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, LiabilityFilters)
        liabilities = await self.liability_repo.get_liabilities_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "liability_count": len(liabilities),
            "liabilities": [
                {
                    "id": liability.id,
                    "name": liability.name,
                    "type": liability.type,
                    "amount": liability.amount,
                    "emi": liability.emi,
                    "interestRate": liability.interestRate,
                    "startDate": liability.startDate.isoformat(),
                    "endDate": liability.endDate.isoformat(),
                    "description": liability.description,
                    "createdAt": liability.createdAt.isoformat()
                }
                for liability in liabilities
            ]
        }
    
    async def _get_liability_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific liability by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        liability = await self.liability_repo.get_liability_by_id(params.entity_id)
        
        return {
            "id": liability.id,
            "userId": liability.userId,
            "name": liability.name,
            "type": liability.type,
            "amount": liability.amount,
            "emi": liability.emi,
            "interestRate": liability.interestRate,
            "startDate": liability.startDate.isoformat(),
            "endDate": liability.endDate.isoformat(),
            "description": liability.description,
            "createdAt": liability.createdAt.isoformat(),
            "updatedAt": liability.updatedAt.isoformat()
        }
    
    async def _get_liability_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get liability summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        summary = await self.liability_repo.get_liability_summary(params.user_id)
        
        return {
            "user_id": params.user_id,
            "summary": summary
        }
    
    async def _get_liabilities_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get liabilities grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        types = await self.liability_repo.get_liabilities_by_type(params.user_id)
        
        return {
            "user_id": params.user_id,
            "types": types
        }
    
    # Savings methods
    async def _get_user_savings(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all savings for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, SavingsFilters)
        savings = await self.savings_repo.get_savings_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "savings_count": len(savings),
            "savings": [
                {
                    "id": saving.id,
                    "name": saving.name,
                    "type": saving.type,
                    "currentBalance": saving.currentBalance,
                    "interestRate": saving.interestRate,
                    "maturityDate": saving.maturityDate.isoformat() if saving.maturityDate else None,
                    "monthlyContribution": saving.monthlyContribution,
                    "targetAmount": saving.targetAmount,
                    "description": saving.description,
                    "createdAt": saving.createdAt.isoformat()
                }
                for saving in savings
            ]
        }
    
    async def _get_savings_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific savings by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        saving = await self.savings_repo.get_savings_by_id(params.entity_id)
        
        return {
            "id": saving.id,
            "userId": saving.userId,
            "name": saving.name,
            "type": saving.type,
            "currentBalance": saving.currentBalance,
            "interestRate": saving.interestRate,
            "maturityDate": saving.maturityDate.isoformat() if saving.maturityDate else None,
            "monthlyContribution": saving.monthlyContribution,
            "targetAmount": saving.targetAmount,
            "description": saving.description,
            "createdAt": saving.createdAt.isoformat(),
            "updatedAt": saving.updatedAt.isoformat()
        }
    
    async def _get_savings_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get savings summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        summary = await self.savings_repo.get_savings_summary(params.user_id)
        
        return {
            "user_id": params.user_id,
            "summary": summary
        }
    
    async def _get_savings_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get savings grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        types = await self.savings_repo.get_savings_by_type(params.user_id)
        
        return {
            "user_id": params.user_id,
            "types": types
        }
    
    # Income methods
    async def _get_user_incomes(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all incomes for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, IncomeFilters)
        incomes = await self.income_repo.get_incomes_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "income_count": len(incomes),
            "incomes": [
                {
                    "id": income.id,
                    "source": income.source,
                    "amount": income.amount,
                    "frequency": income.frequency,
                    "category": income.category,
                    "date": income.date.isoformat(),
                    "description": income.description,
                    "createdAt": income.createdAt.isoformat()
                }
                for income in incomes
            ]
        }
    
    async def _get_income_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific income by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        income = await self.income_repo.get_income_by_id(params.entity_id)
        
        return {
            "id": income.id,
            "userId": income.userId,
            "source": income.source,
            "amount": income.amount,
            "frequency": income.frequency,
            "category": income.category,
            "date": income.date.isoformat(),
            "description": income.description,
            "createdAt": income.createdAt.isoformat(),
            "updatedAt": income.updatedAt.isoformat()
        }
    
    async def _get_income_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get income summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        summary = await self.income_repo.get_income_summary(params.user_id)
        
        return {
            "user_id": params.user_id,
            "summary": summary
        }
    
    async def _get_incomes_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get incomes grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        categories = await self.income_repo.get_incomes_by_category(params.user_id)
        
        return {
            "user_id": params.user_id,
            "categories": categories
        }
    
    async def _get_incomes_by_frequency(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get incomes grouped by frequency."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        frequencies = await self.income_repo.get_incomes_by_frequency(params.user_id)
        
        return {
            "user_id": params.user_id,
            "frequencies": frequencies
        }
    
    # Expense methods
    async def _get_user_expenses(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all expenses for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, ExpenseFilters)
        expenses = await self.expense_repo.get_expenses_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "expense_count": len(expenses),
            "expenses": [
                {
                    "id": expense.id,
                    "category": expense.category,
                    "subcategory": expense.subcategory,
                    "amount": expense.amount,
                    "date": expense.date.isoformat(),
                    "description": expense.description,
                    "paymentMethod": expense.paymentMethod,
                    "createdAt": expense.createdAt.isoformat()
                }
                for expense in expenses
            ]
        }
    
    async def _get_expense_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific expense by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        expense = await self.expense_repo.get_expense_by_id(params.entity_id)
        
        return {
            "id": expense.id,
            "userId": expense.userId,
            "category": expense.category,
            "subcategory": expense.subcategory,
            "amount": expense.amount,
            "date": expense.date.isoformat(),
            "description": expense.description,
            "paymentMethod": expense.paymentMethod,
            "createdAt": expense.createdAt.isoformat(),
            "updatedAt": expense.updatedAt.isoformat()
        }
    
    async def _get_expense_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expense summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        summary = await self.expense_repo.get_expense_summary(params.user_id)
        
        return {
            "user_id": params.user_id,
            "summary": summary
        }
    
    async def _get_expenses_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        categories = await self.expense_repo.get_expenses_by_category(params.user_id)
        
        return {
            "user_id": params.user_id,
            "categories": categories
        }
    
    async def _get_expenses_by_subcategory(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by subcategory."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        subcategories = await self.expense_repo.get_expenses_by_subcategory(params.user_id)
        
        return {
            "user_id": params.user_id,
            "subcategories": subcategories
        }
    
    async def _get_expenses_by_payment_method(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get expenses grouped by payment method."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        payment_methods = await self.expense_repo.get_expenses_by_payment_method(params.user_id)
        
        return {
            "user_id": params.user_id,
            "payment_methods": payment_methods
        }
    
    # Stock methods
    async def _get_user_stocks(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all stocks for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, StockFilters)
        stocks = await self.stock_repo.get_stocks_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "stock_count": len(stocks),
            "stocks": [
                {
                    "id": stock.id,
                    "name": stock.name,
                    "type": stock.type,
                    "amount": stock.amount,
                    "currentValue": stock.currentValue,
                    "purchaseDate": stock.purchaseDate.isoformat(),
                    "returns": stock.returns,
                    "description": stock.description,
                    "createdAt": stock.createdAt.isoformat()
                }
                for stock in stocks
            ]
        }
    
    async def _get_stock_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific stock by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        stock = await self.stock_repo.get_stock_by_id(params.entity_id)
        
        return {
            "id": stock.id,
            "userId": stock.userId,
            "name": stock.name,
            "type": stock.type,
            "amount": stock.amount,
            "currentValue": stock.currentValue,
            "purchaseDate": stock.purchaseDate.isoformat(),
            "returns": stock.returns,
            "description": stock.description,
            "createdAt": stock.createdAt.isoformat(),
            "updatedAt": stock.updatedAt.isoformat()
        }
    
    async def _get_stock_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get stock summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        summary = await self.stock_repo.get_stock_summary(params.user_id)
        
        return {
            "user_id": params.user_id,
            "summary": summary
        }
    
    async def _get_stocks_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get stocks grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        types = await self.stock_repo.get_stocks_by_type(params.user_id)
        
        return {
            "user_id": params.user_id,
            "types": types
        }
    
    # Insurance methods
    async def _get_user_insurances(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all insurances for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, InsuranceFilters)
        insurances = await self.insurance_repo.get_insurances_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "insurance_count": len(insurances),
            "insurances": [
                {
                    "id": insurance.id,
                    "type": insurance.type,
                    "provider": insurance.provider,
                    "policyNumber": insurance.policyNumber,
                    "premium": insurance.premium,
                    "coverage": insurance.coverage,
                    "startDate": insurance.startDate.isoformat(),
                    "endDate": insurance.endDate.isoformat(),
                    "description": insurance.description,
                    "createdAt": insurance.createdAt.isoformat()
                }
                for insurance in insurances
            ]
        }
    
    async def _get_insurance_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific insurance by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        insurance = await self.insurance_repo.get_insurance_by_id(params.entity_id)
        
        return {
            "id": insurance.id,
            "userId": insurance.userId,
            "type": insurance.type,
            "provider": insurance.provider,
            "policyNumber": insurance.policyNumber,
            "premium": insurance.premium,
            "coverage": insurance.coverage,
            "startDate": insurance.startDate.isoformat(),
            "endDate": insurance.endDate.isoformat(),
            "description": insurance.description,
            "createdAt": insurance.createdAt.isoformat(),
            "updatedAt": insurance.updatedAt.isoformat()
        }
    
    async def _get_insurance_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurance summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        summary = await self.insurance_repo.get_insurance_summary(params.user_id)
        
        return {
            "user_id": params.user_id,
            "summary": summary
        }
    
    async def _get_insurances_by_type(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurances grouped by type."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        types = await self.insurance_repo.get_insurances_by_type(params.user_id)
        
        return {
            "user_id": params.user_id,
            "types": types
        }
    
    async def _get_insurances_by_provider(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get insurances grouped by provider."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        providers = await self.insurance_repo.get_insurances_by_provider(params.user_id)
        
        return {
            "user_id": params.user_id,
            "providers": providers
        }
    
    # Goal methods
    async def _get_user_goals(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all goals for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        filters = self._build_filters(params, GoalFilters)
        goals = await self.goal_repo.get_goals_by_user(params.user_id, filters)
        
        return {
            "user_id": params.user_id,
            "goal_count": len(goals),
            "goals": [
                {
                    "id": goal.id,
                    "name": goal.name,
                    "target": goal.target,
                    "current": goal.current,
                    "targetDate": goal.targetDate.isoformat(),
                    "category": goal.category,
                    "priority": goal.priority,
                    "description": goal.description,
                    "createdAt": goal.createdAt.isoformat()
                }
                for goal in goals
            ]
        }
    
    async def _get_goal_by_id(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get specific goal by ID."""
        if not params.entity_id:
            raise ValueError("entity_id is required")
        
        goal = await self.goal_repo.get_goal_by_id(params.entity_id)
        
        return {
            "id": goal.id,
            "userId": goal.userId,
            "name": goal.name,
            "target": goal.target,
            "current": goal.current,
            "targetDate": goal.targetDate.isoformat(),
            "category": goal.category,
            "priority": goal.priority,
            "description": goal.description,
            "createdAt": goal.createdAt.isoformat(),
            "updatedAt": goal.updatedAt.isoformat()
        }
    
    async def _get_goals_by_category(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get goals grouped by category."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        categories = await self.goal_repo.get_goals_by_category(params.user_id)
        
        return {
            "user_id": params.user_id,
            "categories": categories
        }
    
    async def _get_goals_by_priority(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get goals grouped by priority."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        priorities = await self.goal_repo.get_goals_by_priority(params.user_id)
        
        return {
            "user_id": params.user_id,
            "priorities": priorities
        }
    
    # Financial summary
    async def _get_financial_summary(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get comprehensive financial summary for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        # Get summaries from all repositories
        asset_summary = await self.asset_repo.get_asset_summary(params.user_id)
        liability_summary = await self.liability_repo.get_liability_summary(params.user_id)
        savings_summary = await self.savings_repo.get_savings_summary(params.user_id)
        income_summary = await self.income_repo.get_income_summary(params.user_id)
        expense_summary = await self.expense_repo.get_expense_summary(params.user_id)
        stock_summary = await self.stock_repo.get_stock_summary(params.user_id)
        insurance_summary = await self.insurance_repo.get_insurance_summary(params.user_id)
        
        # Calculate net worth
        total_assets = asset_summary.get("totalValue", 0) + savings_summary.get("totalBalance", 0) + stock_summary.get("totalCurrentValue", 0)
        total_liabilities = liability_summary.get("totalAmount", 0)
        net_worth = total_assets - total_liabilities
        
        return {
            "user_id": params.user_id,
            "net_worth": net_worth,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "asset_summary": asset_summary,
            "liability_summary": liability_summary,
            "savings_summary": savings_summary,
            "income_summary": income_summary,
            "expense_summary": expense_summary,
            "stock_summary": stock_summary,
            "insurance_summary": insurance_summary
        } 