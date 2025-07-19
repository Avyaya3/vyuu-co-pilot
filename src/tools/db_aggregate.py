"""
Database aggregation tool for analytics and summary operations.

Handles aggregations like:
- Spending analysis by category
- Monthly/weekly summaries
- Account totals and comparisons
- Budget tracking
"""

import time
import logging
from typing import Dict, Any, Literal, Optional, List
from decimal import Decimal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from ..services import get_financial_service
from ..schemas.database_models import TransactionType
from .base import ToolResponse

logger = logging.getLogger(__name__)


class DbAggregateParams(BaseModel):
    """
    Parameters for database aggregation operations.
    """
    
    operation: Literal[
        "spending_by_category",
        "monthly_summary",
        "account_totals",
        "budget_analysis",
        "transaction_trends",
        "income_vs_expense"
    ] = Field(description="Type of aggregation operation to perform")
    
    # Common parameters
    user_id: str = Field(description="User ID for user-specific aggregations")
    account_name: Optional[str] = Field(None, description="Account name filter")
    account_id: Optional[str] = Field(None, description="Specific account ID filter")
    
    # Time range parameters
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    days_back: Optional[int] = Field(30, ge=1, le=365, description="Number of days to analyze")
    
    # Filtering parameters
    category_name: Optional[str] = Field(None, description="Category filter")
    transaction_type: Optional[TransactionType] = Field(None, description="Transaction type filter")
    min_amount: Optional[float] = Field(None, description="Minimum amount filter")
    max_amount: Optional[float] = Field(None, description="Maximum amount filter")
    
    # Grouping parameters
    group_by: Optional[Literal["day", "week", "month"]] = Field("month", description="Time grouping")
    
    class Config:
        extra = "forbid"
        use_enum_values = True


class DbAggregateTool:
    """
    Database aggregation tool for analytics and summary operations.
    
    Provides insights into financial patterns, trends, and summaries
    while maintaining performance through optimized queries.
    """
    
    name: Literal["db_aggregate"] = "db_aggregate"
    schema = DbAggregateParams
    
    def __init__(self):
        self.financial_service = get_financial_service()
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a database aggregation operation.
        
        Args:
            params: Aggregation parameters matching DbAggregateParams schema
            
        Returns:
            ToolResponse with aggregation results or error information
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            validated_params = self.schema(**params)
            
            # Route to appropriate aggregation method
            result = await self._execute_aggregation(validated_params)
            
            execution_time = (time.time() - start_time) * 1000
            
            response = ToolResponse(
                success=True,
                data=result,
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            logger.info(
                f"Aggregation operation '{validated_params.operation}' completed successfully",
                extra={
                    "operation": validated_params.operation,
                    "execution_time_ms": execution_time,
                    "user_id": validated_params.user_id,
                    "days_back": validated_params.days_back
                }
            )
            
            return response.dict()
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(
                f"Aggregation operation failed: {str(e)}",
                extra={
                    "error": str(e),
                    "params": params,
                    "execution_time_ms": execution_time
                }
            )
            
            response = ToolResponse(
                success=False,
                error=f"Aggregation failed: {str(e)}",
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            return response.dict()
    
    async def _execute_aggregation(self, params: DbAggregateParams) -> Dict[str, Any]:
        """
        Execute the specific aggregation operation.
        
        Args:
            params: Validated aggregation parameters
            
        Returns:
            Aggregation results as dictionary
        """
        if params.operation == "spending_by_category":
            return await self._spending_by_category(params)
        
        elif params.operation == "monthly_summary":
            return await self._monthly_summary(params)
        
        elif params.operation == "account_totals":
            return await self._account_totals(params)
        
        elif params.operation == "budget_analysis":
            return await self._budget_analysis(params)
        
        elif params.operation == "transaction_trends":
            return await self._transaction_trends(params)
        
        elif params.operation == "income_vs_expense":
            return await self._income_vs_expense(params)
        
        else:
            raise ValueError(f"Unsupported operation: {params.operation}")
    
    async def _spending_by_category(self, params: DbAggregateParams) -> Dict[str, Any]:
        """Analyze spending by category."""
        # Get user accounts for filtering
        accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
        account_ids = [acc.id for acc in accounts]
        
        if params.account_name:
            # Filter to specific account
            target_account = next(
                (acc for acc in accounts if acc.account_name.lower() == params.account_name.lower()),
                None
            )
            if not target_account:
                raise ValueError(f"Account '{params.account_name}' not found")
            account_ids = [target_account.id]
        
        elif params.account_id:
            # Verify account ownership
            if params.account_id not in account_ids:
                raise ValueError("Account not found or access denied")
            account_ids = [params.account_id]
        
        # Get spending analysis
        analysis = await self.financial_service.get_spending_analysis(
            user_id=params.user_id,
            days_back=params.days_back or 30,
            account_ids=account_ids
        )
        
        return {
            "user_id": params.user_id,
            "analysis_period_days": params.days_back or 30,
            "account_filter": params.account_name or params.account_id,
            "total_spending": float(analysis.get("total_spending", 0)),
            "category_breakdown": analysis.get("by_category", {}),
            "top_categories": analysis.get("top_categories", []),
            "transaction_count": analysis.get("transaction_count", 0)
        }
    
    async def _monthly_summary(self, params: DbAggregateParams) -> Dict[str, Any]:
        """Generate monthly financial summary."""
        # Get all user accounts
        accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
        
        # Calculate total balances
        total_balance = sum(acc.balance for acc in accounts)
        
        # Get spending analysis for the month
        analysis = await self.financial_service.get_spending_analysis(
            user_id=params.user_id,
            days_back=30
        )
        
        # Calculate income (positive transactions)
        income_transactions = await self.financial_service.transaction_repo.get_transactions_by_type(
            user_id=params.user_id,
            transaction_type=TransactionType.RECEIVED,
            days_back=30
        )
        total_income = sum(txn.amount for txn in income_transactions)
        
        return {
            "user_id": params.user_id,
            "summary_period": "last_30_days",
            "account_summary": {
                "total_accounts": len(accounts),
                "total_balance": float(total_balance),
                "accounts": [
                    {
                        "name": acc.account_name,
                        "balance": float(acc.balance),
                        "type": acc.account_type.type_name if acc.account_type else "Unknown"
                    }
                    for acc in accounts
                ]
            },
            "income_summary": {
                "total_income": float(total_income),
                "income_transactions": len(income_transactions)
            },
            "spending_summary": {
                "total_spending": float(analysis.get("total_spending", 0)),
                "by_category": analysis.get("by_category", {}),
                "transaction_count": analysis.get("transaction_count", 0)
            },
            "net_change": float(total_income - analysis.get("total_spending", 0))
        }
    
    async def _account_totals(self, params: DbAggregateParams) -> Dict[str, Any]:
        """Get totals and comparisons across accounts."""
        accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
        
        # Calculate totals by account type
        totals_by_type = {}
        for account in accounts:
            account_type = account.account_type.type_name if account.account_type else "Unknown"
            if account_type not in totals_by_type:
                totals_by_type[account_type] = {
                    "count": 0,
                    "total_balance": 0,
                    "accounts": []
                }
            
            totals_by_type[account_type]["count"] += 1
            totals_by_type[account_type]["total_balance"] += float(account.balance)
            totals_by_type[account_type]["accounts"].append({
                "name": account.account_name,
                "balance": float(account.balance)
            })
        
        total_assets = sum(acc.balance for acc in accounts)
        
        return {
            "user_id": params.user_id,
            "total_accounts": len(accounts),
            "total_assets": float(total_assets),
            "by_account_type": totals_by_type,
            "largest_account": {
                "name": max(accounts, key=lambda x: x.balance).account_name,
                "balance": float(max(accounts, key=lambda x: x.balance).balance)
            } if accounts else None,
            "account_distribution": [
                {
                    "name": acc.account_name,
                    "balance": float(acc.balance),
                    "percentage": float((acc.balance / total_assets) * 100) if total_assets > 0 else 0
                }
                for acc in sorted(accounts, key=lambda x: x.balance, reverse=True)
            ]
        }
    
    async def _budget_analysis(self, params: DbAggregateParams) -> Dict[str, Any]:
        """Analyze spending against typical patterns (simplified budget analysis)."""
        # Get spending patterns for comparison
        current_analysis = await self.financial_service.get_spending_analysis(
            user_id=params.user_id,
            days_back=30
        )
        
        # Get previous period for comparison
        previous_analysis = await self.financial_service.get_spending_analysis(
            user_id=params.user_id,
            days_back=60  # Previous 30 days for comparison
        )
        
        current_spending = current_analysis.get("total_spending", 0)
        previous_spending = previous_analysis.get("total_spending", 0) / 2  # Approximate previous period
        
        spending_change = current_spending - previous_spending
        spending_change_percent = (spending_change / previous_spending * 100) if previous_spending > 0 else 0
        
        return {
            "user_id": params.user_id,
            "analysis_period": "last_30_days",
            "current_spending": float(current_spending),
            "previous_period_spending": float(previous_spending),
            "spending_change": float(spending_change),
            "spending_change_percent": float(spending_change_percent),
            "trend": "increasing" if spending_change > 0 else "decreasing" if spending_change < 0 else "stable",
            "category_changes": {
                cat: {
                    "current": float(amount),
                    "trend": "analysis_needed"  # Would need more complex logic for real trend analysis
                }
                for cat, amount in current_analysis.get("by_category", {}).items()
            }
        }
    
    async def _transaction_trends(self, params: DbAggregateParams) -> Dict[str, Any]:
        """Analyze transaction trends over time."""
        # Get recent transactions for trend analysis
        days_back = params.days_back or 30
        
        # Simple trend analysis - could be enhanced with more sophisticated algorithms
        analysis = await self.financial_service.get_spending_analysis(
            user_id=params.user_id,
            days_back=days_back
        )
        
        # Get transaction counts by type
        all_transactions = []
        accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
        
        for account in accounts:
            account_transactions = await self.financial_service.transaction_repo.get_account_transactions(
                account_id=account.id,
                limit=1000,  # Large limit for trend analysis
                days_back=days_back
            )
            all_transactions.extend(account_transactions)
        
        # Group by transaction type
        type_counts = {}
        type_amounts = {}
        
        for txn in all_transactions:
            txn_type = txn.transaction_type.value
            type_counts[txn_type] = type_counts.get(txn_type, 0) + 1
            type_amounts[txn_type] = type_amounts.get(txn_type, 0) + float(txn.amount)
        
        return {
            "user_id": params.user_id,
            "analysis_period_days": days_back,
            "total_transactions": len(all_transactions),
            "transaction_frequency": len(all_transactions) / days_back,  # transactions per day
            "by_transaction_type": {
                txn_type: {
                    "count": count,
                    "total_amount": float(type_amounts.get(txn_type, 0)),
                    "average_amount": float(type_amounts.get(txn_type, 0) / count) if count > 0 else 0
                }
                for txn_type, count in type_counts.items()
            },
            "spending_velocity": float(analysis.get("total_spending", 0) / days_back)  # spending per day
        }
    
    async def _income_vs_expense(self, params: DbAggregateParams) -> Dict[str, Any]:
        """Compare income vs expenses."""
        days_back = params.days_back or 30
        
        # Get all transactions
        all_transactions = []
        accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
        
        for account in accounts:
            account_transactions = await self.financial_service.transaction_repo.get_account_transactions(
                account_id=account.id,
                limit=1000,
                days_back=days_back
            )
            all_transactions.extend(account_transactions)
        
        # Separate income vs expenses
        income = 0
        expenses = 0
        
        for txn in all_transactions:
            if txn.transaction_type in [TransactionType.RECEIVED]:
                income += float(txn.amount)
            elif txn.transaction_type in [TransactionType.PAID]:
                expenses += float(txn.amount)
        
        net_income = income - expenses
        savings_rate = (net_income / income * 100) if income > 0 else 0
        
        return {
            "user_id": params.user_id,
            "analysis_period_days": days_back,
            "income": {
                "total": float(income),
                "daily_average": float(income / days_back),
                "transaction_count": len([t for t in all_transactions if t.transaction_type == TransactionType.RECEIVED])
            },
            "expenses": {
                "total": float(expenses),
                "daily_average": float(expenses / days_back),
                "transaction_count": len([t for t in all_transactions if t.transaction_type == TransactionType.PAID])
            },
            "net_income": float(net_income),
            "savings_rate_percent": float(savings_rate),
            "expense_ratio": float((expenses / income * 100)) if income > 0 else 0,
            "financial_health": "positive" if net_income > 0 else "negative" if net_income < 0 else "balanced"
        } 