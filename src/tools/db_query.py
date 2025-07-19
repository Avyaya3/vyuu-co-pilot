"""
Database query tool for read-only operations.

Handles queries like:
- Account balance retrieval
- Transaction history
- User account information
- Account verification
"""

import time
import logging
from typing import Dict, Any, Literal, Optional, List
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, Field

from ..services import get_financial_service
from ..schemas.database_models import Account, Transaction
from .base import ToolResponse

logger = logging.getLogger(__name__)


class DbQueryParams(BaseModel):
    """
    Parameters for database query operations.
    """
    
    operation: Literal[
        "get_account_balance",
        "get_transaction_history", 
        "get_user_accounts",
        "verify_account_exists",
        "get_account_details"
    ] = Field(description="Type of query operation to perform")
    
    # Common parameters
    user_id: Optional[str] = Field(None, description="User ID for user-specific queries")
    account_name: Optional[str] = Field(None, description="Account name (e.g., 'checking', 'savings')")
    account_id: Optional[str] = Field(None, description="Specific account ID")
    
    # Transaction history parameters
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of records to return")
    days_back: Optional[int] = Field(30, ge=1, le=365, description="Number of days to look back")
    
    class Config:
        extra = "forbid"


class DbQueryTool:
    """
    Database query tool for read-only operations.
    
    Provides standardized interface for querying financial data
    while maintaining separation between intent parameters and database operations.
    """
    
    name: Literal["db_query"] = "db_query"
    schema = DbQueryParams
    
    def __init__(self):
        self.financial_service = get_financial_service()
    
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
                    "account_name": validated_params.account_name
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
        if params.operation == "get_account_balance":
            return await self._get_account_balance(params)
        
        elif params.operation == "get_transaction_history":
            return await self._get_transaction_history(params)
        
        elif params.operation == "get_user_accounts":
            return await self._get_user_accounts(params)
        
        elif params.operation == "verify_account_exists":
            return await self._verify_account_exists(params)
        
        elif params.operation == "get_account_details":
            return await self._get_account_details(params)
        
        else:
            raise ValueError(f"Unsupported operation: {params.operation}")
    
    async def _get_account_balance(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get account balance by name or ID."""
        if not params.user_id:
            raise ValueError("user_id is required for account balance queries")
        
        if params.account_name:
            # Find account by name
            accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
            account = next(
                (acc for acc in accounts if acc.account_name.lower() == params.account_name.lower()),
                None
            )
            if not account:
                raise ValueError(f"Account '{params.account_name}' not found for user")
        
        elif params.account_id:
            account = await self.financial_service.account_repo.get_by_id(params.account_id)
            if not account or account.user_id != params.user_id:
                raise ValueError("Account not found or access denied")
        
        else:
            raise ValueError("Either account_name or account_id is required")
        
        return {
            "account_id": account.id,
            "account_name": account.account_name,
            "balance": float(account.balance),
            "currency": "USD",
            "last_updated": account.updated_at.isoformat() if account.updated_at else None
        }
    
    async def _get_transaction_history(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get transaction history for an account."""
        if not params.user_id:
            raise ValueError("user_id is required for transaction history")
        
        # Get account first
        if params.account_name:
            accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
            account = next(
                (acc for acc in accounts if acc.account_name.lower() == params.account_name.lower()),
                None
            )
            if not account:
                raise ValueError(f"Account '{params.account_name}' not found")
            account_id = account.id
        
        elif params.account_id:
            account = await self.financial_service.account_repo.get_by_id(params.account_id)
            if not account or account.user_id != params.user_id:
                raise ValueError("Account not found or access denied")
            account_id = params.account_id
        
        else:
            raise ValueError("Either account_name or account_id is required")
        
        # Get transactions
        transactions = await self.financial_service.transaction_repo.get_account_transactions(
            account_id=account_id,
            limit=params.limit or 10,
            days_back=params.days_back or 30
        )
        
        return {
            "account_id": account_id,
            "account_name": account.account_name,
            "transaction_count": len(transactions),
            "transactions": [
                {
                    "id": txn.id,
                    "amount": float(txn.amount),
                    "type": txn.transaction_type.value,
                    "counterparty": txn.counterparty,
                    "notes": txn.notes,
                    "date": txn.transaction_date.isoformat(),
                    "category": txn.category.name if txn.category else None
                }
                for txn in transactions
            ]
        }
    
    async def _get_user_accounts(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get all accounts for a user."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
        
        return {
            "user_id": params.user_id,
            "account_count": len(accounts),
            "accounts": [
                {
                    "id": acc.id,
                    "name": acc.account_name,
                    "type": acc.account_type.type_name if acc.account_type else "Unknown",
                    "balance": float(acc.balance),
                    "created": acc.created_at.isoformat() if acc.created_at else None
                }
                for acc in accounts
            ]
        }
    
    async def _verify_account_exists(self, params: DbQueryParams) -> Dict[str, Any]:
        """Verify if an account exists and is accessible."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        exists = False
        account_info = None
        
        if params.account_name:
            accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
            account = next(
                (acc for acc in accounts if acc.account_name.lower() == params.account_name.lower()),
                None
            )
            if account:
                exists = True
                account_info = {
                    "id": account.id,
                    "name": account.account_name,
                    "type": account.account_type.type_name if account.account_type else "Unknown"
                }
        
        elif params.account_id:
            try:
                account = await self.financial_service.account_repo.get_by_id(params.account_id)
                if account and account.user_id == params.user_id:
                    exists = True
                    account_info = {
                        "id": account.id,
                        "name": account.account_name,
                        "type": account.account_type.type_name if account.account_type else "Unknown"
                    }
            except Exception:
                exists = False
        
        return {
            "exists": exists,
            "account": account_info,
            "search_criteria": {
                "account_name": params.account_name,
                "account_id": params.account_id,
                "user_id": params.user_id
            }
        }
    
    async def _get_account_details(self, params: DbQueryParams) -> Dict[str, Any]:
        """Get detailed account information."""
        if not params.user_id:
            raise ValueError("user_id is required")
        
        if params.account_name:
            accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
            account = next(
                (acc for acc in accounts if acc.account_name.lower() == params.account_name.lower()),
                None
            )
            if not account:
                raise ValueError(f"Account '{params.account_name}' not found")
        
        elif params.account_id:
            account = await self.financial_service.account_repo.get_by_id(params.account_id)
            if not account or account.user_id != params.user_id:
                raise ValueError("Account not found or access denied")
        
        else:
            raise ValueError("Either account_name or account_id is required")
        
        return {
            "id": account.id,
            "name": account.account_name,
            "type": account.account_type.type_name if account.account_type else "Unknown",
            "balance": float(account.balance),
            "user_id": account.user_id,
            "created_at": account.created_at.isoformat() if account.created_at else None,
            "updated_at": account.updated_at.isoformat() if account.updated_at else None,
            "is_active": True  # Assuming all accounts are active for now
        } 