"""
Database action tool for create, update, and transfer operations.

Handles actions like:
- Creating transactions
- Transferring money between accounts
- Creating new accounts
- Updating transaction details
"""

import time
import logging
from typing import Dict, Any, Literal, Optional
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, Field, validator

from ..services import get_financial_service
from ..schemas.database_models import TransactionType, TransactionCreate, AccountCreate
from .base import ToolResponse

logger = logging.getLogger(__name__)


class DbActionParams(BaseModel):
    """
    Parameters for database action operations.
    """
    
    operation: Literal[
        "create_transaction",
        "transfer_money",
        "create_account",
        "update_transaction",
        "delete_transaction"
    ] = Field(description="Type of action operation to perform")
    
    # Common parameters
    user_id: str = Field(description="User ID for user-specific operations")
    
    # Transaction creation parameters
    account_name: Optional[str] = Field(None, description="Account name for transaction")
    account_id: Optional[str] = Field(None, description="Specific account ID")
    amount: Optional[float] = Field(None, description="Transaction amount")
    transaction_type: Optional[TransactionType] = Field(None, description="Type of transaction")
    counterparty: Optional[str] = Field(None, description="Transaction counterparty")
    category_name: Optional[str] = Field(None, description="Transaction category")
    notes: Optional[str] = Field(None, description="Transaction notes")
    
    # Transfer-specific parameters
    source_account: Optional[str] = Field(None, description="Source account for transfer")
    target_account: Optional[str] = Field(None, description="Target account for transfer")
    source_account_id: Optional[str] = Field(None, description="Source account ID")
    target_account_id: Optional[str] = Field(None, description="Target account ID")
    
    # Account creation parameters
    account_type: Optional[str] = Field(None, description="Type of account to create")
    initial_balance: Optional[float] = Field(0.0, description="Initial account balance")
    
    # Update/Delete parameters
    transaction_id: Optional[str] = Field(None, description="Transaction ID for updates/deletes")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Amount must be positive")
        return v
    
    @validator('initial_balance')
    def validate_initial_balance(cls, v):
        if v is not None and v < 0:
            raise ValueError("Initial balance cannot be negative")
        return v
    
    class Config:
        extra = "forbid"
        use_enum_values = True


class DbActionTool:
    """
    Database action tool for create, update, and transfer operations.
    
    Handles financial operations that modify database state,
    ensuring data integrity and business rule compliance.
    """
    
    name: Literal["db_action"] = "db_action"
    schema = DbActionParams
    
    def __init__(self):
        self.financial_service = get_financial_service()
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a database action operation.
        
        Args:
            params: Action parameters matching DbActionParams schema
            
        Returns:
            ToolResponse with action results or error information
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            validated_params = self.schema(**params)
            
            # Route to appropriate action method
            result = await self._execute_action(validated_params)
            
            execution_time = (time.time() - start_time) * 1000
            
            response = ToolResponse(
                success=True,
                data=result,
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            logger.info(
                f"Action operation '{validated_params.operation}' completed successfully",
                extra={
                    "operation": validated_params.operation,
                    "execution_time_ms": execution_time,
                    "user_id": validated_params.user_id,
                    "amount": validated_params.amount
                }
            )
            
            return response.dict()
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(
                f"Action operation failed: {str(e)}",
                extra={
                    "error": str(e),
                    "params": params,
                    "execution_time_ms": execution_time
                }
            )
            
            response = ToolResponse(
                success=False,
                error=f"Action failed: {str(e)}",
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            return response.dict()
    
    async def _execute_action(self, params: DbActionParams) -> Dict[str, Any]:
        """
        Execute the specific action operation.
        
        Args:
            params: Validated action parameters
            
        Returns:
            Action results as dictionary
        """
        if params.operation == "create_transaction":
            return await self._create_transaction(params)
        
        elif params.operation == "transfer_money":
            return await self._transfer_money(params)
        
        elif params.operation == "create_account":
            return await self._create_account(params)
        
        elif params.operation == "update_transaction":
            return await self._update_transaction(params)
        
        elif params.operation == "delete_transaction":
            return await self._delete_transaction(params)
        
        else:
            raise ValueError(f"Unsupported operation: {params.operation}")
    
    async def _create_transaction(self, params: DbActionParams) -> Dict[str, Any]:
        """Create a new transaction."""
        # Validate required parameters
        if not params.amount:
            raise ValueError("Amount is required for transaction creation")
        
        if not params.transaction_type:
            raise ValueError("Transaction type is required")
        
        # Get account
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
        
        # Get category (optional)
        category_id = None
        if params.category_name:
            categories = await self.financial_service.category_repo.get_all()
            category = next(
                (cat for cat in categories if cat.name.lower() == params.category_name.lower()),
                None
            )
            if category:
                category_id = category.id
        
        # Create transaction
        transaction = await self.financial_service.create_transaction(
            user_id=params.user_id,
            account_id=account_id,
            category_id=category_id,
            transaction_type=params.transaction_type,
            amount=Decimal(str(params.amount)),
            counterparty=params.counterparty or "Unknown",
            notes=params.notes
        )
        
        return {
            "transaction_id": transaction.id,
            "account_id": account_id,
            "account_name": account.account_name,
            "amount": float(transaction.amount),
            "transaction_type": transaction.transaction_type.value,
            "counterparty": transaction.counterparty,
            "category": transaction.category.name if transaction.category else None,
            "notes": transaction.notes,
            "created_at": transaction.created_at.isoformat() if transaction.created_at else None,
            "new_account_balance": float(account.balance)  # This would need to be refreshed
        }
    
    async def _transfer_money(self, params: DbActionParams) -> Dict[str, Any]:
        """Transfer money between accounts."""
        # Validate required parameters
        if not params.amount:
            raise ValueError("Amount is required for transfer")
        
        # Get source account
        if params.source_account:
            accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
            source_acc = next(
                (acc for acc in accounts if acc.account_name.lower() == params.source_account.lower()),
                None
            )
            if not source_acc:
                raise ValueError(f"Source account '{params.source_account}' not found")
            source_account_id = source_acc.id
        
        elif params.source_account_id:
            source_acc = await self.financial_service.account_repo.get_by_id(params.source_account_id)
            if not source_acc or source_acc.user_id != params.user_id:
                raise ValueError("Source account not found or access denied")
            source_account_id = params.source_account_id
        
        else:
            raise ValueError("Source account (name or ID) is required")
        
        # Get target account
        if params.target_account:
            accounts = await self.financial_service.account_repo.get_user_accounts(params.user_id)
            target_acc = next(
                (acc for acc in accounts if acc.account_name.lower() == params.target_account.lower()),
                None
            )
            if not target_acc:
                raise ValueError(f"Target account '{params.target_account}' not found")
            target_account_id = target_acc.id
        
        elif params.target_account_id:
            target_acc = await self.financial_service.account_repo.get_by_id(params.target_account_id)
            if not target_acc or target_acc.user_id != params.user_id:
                raise ValueError("Target account not found or access denied")
            target_account_id = params.target_account_id
        
        else:
            raise ValueError("Target account (name or ID) is required")
        
        # Check sufficient balance
        if source_acc.balance < Decimal(str(params.amount)):
            raise ValueError(f"Insufficient balance. Available: ${source_acc.balance}, Required: ${params.amount}")
        
        # Execute transfer
        transfer_result = await self.financial_service.transfer_between_accounts(
            user_id=params.user_id,
            source_account_id=source_account_id,
            target_account_id=target_account_id,
            amount=Decimal(str(params.amount)),
            notes=params.notes or f"Transfer from {source_acc.account_name} to {target_acc.account_name}"
        )
        
        return {
            "transfer_id": transfer_result.get("transfer_id"),
            "source_transaction_id": transfer_result.get("source_transaction_id"),
            "target_transaction_id": transfer_result.get("target_transaction_id"),
            "amount": float(params.amount),
            "source_account": {
                "id": source_account_id,
                "name": source_acc.account_name,
                "new_balance": float(transfer_result.get("source_new_balance", source_acc.balance))
            },
            "target_account": {
                "id": target_account_id,
                "name": target_acc.account_name,
                "new_balance": float(transfer_result.get("target_new_balance", target_acc.balance))
            },
            "notes": params.notes,
            "executed_at": datetime.now().isoformat()
        }
    
    async def _create_account(self, params: DbActionParams) -> Dict[str, Any]:
        """Create a new account for the user."""
        if not params.account_name:
            raise ValueError("Account name is required for account creation")
        
        if not params.account_type:
            raise ValueError("Account type is required")
        
        # Get account type ID
        account_types = await self.financial_service.account_type_repo.get_all_account_types()
        account_type = next(
            (at for at in account_types if at.type_name.lower() == params.account_type.lower()),
            None
        )
        
        if not account_type:
            # For now, raise an error if account type doesn't exist
            # In a real implementation, you might want to create it
            raise ValueError(f"Account type '{params.account_type}' not found. Available types: {[at.type_name for at in account_types]}")
        
        # Create account
        account = await self.financial_service.create_user_account(
            user_id=params.user_id,
            account_type_id=account_type.id,
            account_name=params.account_name,
            initial_balance=Decimal(str(params.initial_balance or 0))
        )
        
        return {
            "account_id": account.id,
            "account_name": account.account_name,
            "account_type": account_type.type_name,
            "initial_balance": float(params.initial_balance or 0),
            "user_id": params.user_id,
            "created_at": account.created_at.isoformat() if account.created_at else None
        }
    
    async def _update_transaction(self, params: DbActionParams) -> Dict[str, Any]:
        """Update an existing transaction."""
        if not params.transaction_id:
            raise ValueError("Transaction ID is required for updates")
        
        # Get existing transaction
        transaction = await self.financial_service.transaction_repo.get_by_id(params.transaction_id)
        if not transaction:
            raise ValueError("Transaction not found")
        
        # Verify ownership through account
        account = await self.financial_service.account_repo.get_by_id(transaction.account_id)
        if not account or account.user_id != params.user_id:
            raise ValueError("Transaction not found or access denied")
        
        # Prepare update data
        update_data = {}
        
        if params.amount is not None:
            update_data["amount"] = Decimal(str(params.amount))
        
        if params.counterparty is not None:
            update_data["counterparty"] = params.counterparty
        
        if params.notes is not None:
            update_data["notes"] = params.notes
        
        if params.category_name is not None:
            categories = await self.financial_service.category_repo.get_all()
            category = next(
                (cat for cat in categories if cat.name.lower() == params.category_name.lower()),
                None
            )
            if category:
                update_data["category_id"] = category.id
        
        if params.transaction_type is not None:
            update_data["transaction_type"] = params.transaction_type
        
        if not update_data:
            raise ValueError("No update data provided")
        
        # Update transaction
        updated_transaction = await self.financial_service.transaction_repo.update(
            params.transaction_id,
            update_data
        )
        
        return {
            "transaction_id": updated_transaction.id,
            "updated_fields": list(update_data.keys()),
            "amount": float(updated_transaction.amount),
            "counterparty": updated_transaction.counterparty,
            "notes": updated_transaction.notes,
            "category": updated_transaction.category.name if updated_transaction.category else None,
            "transaction_type": updated_transaction.transaction_type.value,
            "updated_at": updated_transaction.updated_at.isoformat() if updated_transaction.updated_at else None
        }
    
    async def _delete_transaction(self, params: DbActionParams) -> Dict[str, Any]:
        """Delete a transaction."""
        if not params.transaction_id:
            raise ValueError("Transaction ID is required for deletion")
        
        # Get existing transaction
        transaction = await self.financial_service.transaction_repo.get_by_id(params.transaction_id)
        if not transaction:
            raise ValueError("Transaction not found")
        
        # Verify ownership through account
        account = await self.financial_service.account_repo.get_by_id(transaction.account_id)
        if not account or account.user_id != params.user_id:
            raise ValueError("Transaction not found or access denied")
        
        # Store transaction details before deletion
        deleted_transaction_info = {
            "transaction_id": transaction.id,
            "amount": float(transaction.amount),
            "counterparty": transaction.counterparty,
            "transaction_type": transaction.transaction_type.value,
            "account_name": account.account_name,
            "deleted_at": datetime.now().isoformat()
        }
        
        # Delete transaction
        await self.financial_service.transaction_repo.delete(params.transaction_id)
        
        return {
            "success": True,
            "deleted_transaction": deleted_transaction_info,
            "message": f"Transaction {params.transaction_id} deleted successfully"
        } 