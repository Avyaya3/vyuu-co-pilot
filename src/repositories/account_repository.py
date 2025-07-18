"""
Account Repository for Financial Account Management.

This module provides domain-specific repository operations for account entities,
including account creation, balance management, account type handling, and
financial account business logic.

Features:
- Account CRUD operations with validation
- Balance tracking and history
- Account type management
- User account aggregations
- Account status operations
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID

from ..repositories.base_repository import BaseRepository, DatabaseOperationError, EntityValidationError
from ..schemas.database_models import (
    Account, AccountCreate, AccountUpdate, AccountType, 
    AccountWithType, AccountFilters, AccountSummary
)

logger = logging.getLogger(__name__)


class AccountRepository(BaseRepository[Account, AccountCreate, AccountUpdate, UUID]):
    """
    Repository for account entity operations.
    
    Provides specialized account management operations including balance tracking,
    account type management, and user account aggregations.
    """
    
    def __init__(self):
        """Initialize the account repository."""
        super().__init__(Account, "accounts")
    
    async def create(self, account_data: AccountCreate) -> Account:
        """
        Create a new account with validation.
        
        Args:
            account_data: Account creation data
            
        Returns:
            Created account entity
            
        Raises:
            EntityValidationError: If validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Creating account for user {account_data.user_id}")
        
        try:
            # Validate user exists (would need UserRepository for this)
            # For now, assume user validation is handled at service layer
            
            # Validate account type exists
            if not await self._account_type_exists(account_data.account_type_id):
                raise EntityValidationError(f"Account type {account_data.account_type_id} does not exist")
            
            account_dict = account_data.model_dump()
            
            query = """
                INSERT INTO accounts (user_id, account_type_id, name, current_balance)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            """
            
            result = await self._execute_query(
                query,
                account_dict['user_id'],
                account_dict['account_type_id'],
                account_dict['name'],
                account_dict['current_balance'],
                fetch_one=True
            )
            
            created_account = self._row_to_model(result)
            self._logger.info(f"Account created successfully with ID: {created_account.id}")
            return created_account
            
        except Exception as e:
            self._logger.error(f"Failed to create account: {e}")
            raise DatabaseOperationError(f"Account creation failed: {e}")
    
    async def get_by_id(self, account_id: UUID) -> Optional[Account]:
        """
        Get an account by its ID.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Account if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = "SELECT * FROM accounts WHERE id = $1"
            result = await self._execute_query(query, account_id, fetch_one=True)
            return self._row_to_model(result)
        except Exception as e:
            self._logger.error(f"Failed to get account by ID {account_id}: {e}")
            raise DatabaseOperationError(f"Get account by ID failed: {e}")
    
    async def update(self, account_id: UUID, account_update: AccountUpdate) -> Optional[Account]:
        """
        Update an account's information.
        
        Args:
            account_id: Account identifier
            account_update: Update data
            
        Returns:
            Updated account if found, None otherwise
            
        Raises:
            EntityValidationError: If validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Updating account {account_id}")
        
        try:
            # Check if account exists
            if not await self.exists(account_id):
                return None
            
            # Build dynamic update query
            update_fields = []
            params = []
            param_idx = 1
            
            update_dict = account_update.model_dump(exclude_unset=True)
            
            for field, value in update_dict.items():
                if value is not None:
                    update_fields.append(f"{field} = ${param_idx}")
                    params.append(value)
                    param_idx += 1
            
            if not update_fields:
                return await self.get_by_id(account_id)
            
            params.append(account_id)
            
            query = f"""
                UPDATE accounts 
                SET {', '.join(update_fields)}
                WHERE id = ${param_idx}
                RETURNING *
            """
            
            result = await self._execute_query(query, *params, fetch_one=True)
            updated_account = self._row_to_model(result)
            
            if updated_account:
                self._logger.info(f"Account {account_id} updated successfully")
            
            return updated_account
            
        except Exception as e:
            self._logger.error(f"Failed to update account {account_id}: {e}")
            raise DatabaseOperationError(f"Account update failed: {e}")
    
    async def delete(self, account_id: UUID) -> bool:
        """
        Delete an account by its ID.
        
        Args:
            account_id: Account identifier
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Deleting account {account_id}")
        
        try:
            # Check if account has transactions
            transaction_count = await self._count_account_transactions(account_id)
            if transaction_count > 0:
                self._logger.warning(f"Account {account_id} has {transaction_count} transactions")
                # Depending on business rules, might want to prevent deletion
                # For now, we'll allow it but log the warning
            
            query = "DELETE FROM accounts WHERE id = $1"
            result = await self._execute_query(query, account_id, fetch_one=False, fetch_all=False)
            
            deleted = result == "DELETE 1"
            
            if deleted:
                self._logger.info(f"Account {account_id} deleted successfully")
            else:
                self._logger.warning(f"Account {account_id} not found for deletion")
            
            return deleted
            
        except Exception as e:
            self._logger.error(f"Failed to delete account {account_id}: {e}")
            raise DatabaseOperationError(f"Account deletion failed: {e}")
    
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Account]:
        """
        List all accounts with optional pagination.
        
        Args:
            limit: Maximum number of accounts to return
            offset: Number of accounts to skip
            
        Returns:
            List of accounts
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            if limit is not None and offset is not None:
                query = "SELECT * FROM accounts ORDER BY created_at DESC LIMIT $1 OFFSET $2"
                result = await self._execute_query(query, limit, offset, fetch_all=True)
            elif limit is not None:
                query = "SELECT * FROM accounts ORDER BY created_at DESC LIMIT $1"
                result = await self._execute_query(query, limit, fetch_all=True)
            else:
                query = "SELECT * FROM accounts ORDER BY created_at DESC"
                result = await self._execute_query(query, fetch_all=True)
            
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to list accounts: {e}")
            raise DatabaseOperationError(f"List accounts failed: {e}")
    
    # Account-specific domain methods
    
    async def get_user_accounts(self, user_id: UUID) -> List[Account]:
        """
        Get all accounts for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user's accounts
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT * FROM accounts 
                WHERE user_id = $1 
                ORDER BY created_at DESC
            """
            
            result = await self._execute_query(query, user_id, fetch_all=True)
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to get accounts for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get user accounts failed: {e}")
    
    async def get_user_accounts_with_type(self, user_id: UUID) -> List[AccountWithType]:
        """
        Get all accounts for a user with account type information.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of accounts with type information
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT 
                    a.id,
                    a.user_id,
                    a.account_type_id,
                    at.name as account_type_name,
                    a.name,
                    a.current_balance,
                    a.created_at
                FROM accounts a
                INNER JOIN account_types at ON a.account_type_id = at.id
                WHERE a.user_id = $1
                ORDER BY a.created_at DESC
            """
            
            result = await self._execute_query(query, user_id, fetch_all=True)
            
            accounts = []
            for row in result or []:
                account = AccountWithType.model_validate(dict(row))
                accounts.append(account)
            
            return accounts
            
        except Exception as e:
            self._logger.error(f"Failed to get accounts with type for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get user accounts with type failed: {e}")
    
    async def update_balance(self, account_id: UUID, new_balance: Decimal) -> Optional[Account]:
        """
        Update an account's balance.
        
        Args:
            account_id: Account identifier
            new_balance: New balance amount
            
        Returns:
            Updated account if found, None otherwise
            
        Raises:
            EntityValidationError: If balance validation fails
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                UPDATE accounts 
                SET current_balance = $1
                WHERE id = $2
                RETURNING *
            """
            
            result = await self._execute_query(query, new_balance, account_id, fetch_one=True)
            updated_account = self._row_to_model(result)
            
            if updated_account:
                self._logger.info(f"Account {account_id} balance updated to {new_balance}")
            
            return updated_account
            
        except Exception as e:
            self._logger.error(f"Failed to update balance for account {account_id}: {e}")
            raise DatabaseOperationError(f"Balance update failed: {e}")
    
    async def adjust_balance(self, account_id: UUID, amount: Decimal) -> Optional[Account]:
        """
        Adjust an account's balance by a specific amount.
        
        Args:
            account_id: Account identifier
            amount: Amount to add (positive) or subtract (negative)
            
        Returns:
            Updated account if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                UPDATE accounts 
                SET current_balance = current_balance + $1
                WHERE id = $2
                RETURNING *
            """
            
            result = await self._execute_query(query, amount, account_id, fetch_one=True)
            updated_account = self._row_to_model(result)
            
            if updated_account:
                self._logger.info(f"Account {account_id} balance adjusted by {amount}")
            
            return updated_account
            
        except Exception as e:
            self._logger.error(f"Failed to adjust balance for account {account_id}: {e}")
            raise DatabaseOperationError(f"Balance adjustment failed: {e}")
    
    async def get_accounts_by_type(self, user_id: UUID, account_type_id: int) -> List[Account]:
        """
        Get accounts by user and account type.
        
        Args:
            user_id: User identifier
            account_type_id: Account type identifier
            
        Returns:
            List of matching accounts
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT * FROM accounts 
                WHERE user_id = $1 AND account_type_id = $2
                ORDER BY created_at DESC
            """
            
            result = await self._execute_query(query, user_id, account_type_id, fetch_all=True)
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to get accounts by type for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get accounts by type failed: {e}")
    
    async def get_user_total_balance(self, user_id: UUID) -> Decimal:
        """
        Get the total balance across all user accounts.
        
        Args:
            user_id: User identifier
            
        Returns:
            Total balance across all accounts
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT COALESCE(SUM(current_balance), 0) as total_balance
                FROM accounts 
                WHERE user_id = $1
            """
            
            result = await self._execute_query(query, user_id, fetch_one=True)
            return Decimal(str(result['total_balance'])) if result else Decimal('0.00')
            
        except Exception as e:
            self._logger.error(f"Failed to get total balance for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get total balance failed: {e}")
    
    async def get_account_summary(self, user_id: UUID) -> AccountSummary:
        """
        Get account summary statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Account summary with statistics
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT 
                    COUNT(*) as total_accounts,
                    COALESCE(SUM(a.current_balance), 0) as total_balance,
                    at.name as account_type_name,
                    COUNT(a.id) as type_count,
                    COALESCE(SUM(a.current_balance), 0) as type_balance
                FROM accounts a
                INNER JOIN account_types at ON a.account_type_id = at.id
                WHERE a.user_id = $1
                GROUP BY at.id, at.name
            """
            
            result = await self._execute_query(query, user_id, fetch_all=True)
            
            if not result:
                return AccountSummary(
                    total_accounts=0,
                    total_balance=Decimal('0.00'),
                    accounts_by_type={},
                    balance_by_type={}
                )
            
            total_accounts = sum(row['type_count'] for row in result)
            total_balance = sum(Decimal(str(row['type_balance'])) for row in result)
            
            accounts_by_type = {row['account_type_name']: row['type_count'] for row in result}
            balance_by_type = {
                row['account_type_name']: Decimal(str(row['type_balance'])) 
                for row in result
            }
            
            return AccountSummary(
                total_accounts=total_accounts,
                total_balance=total_balance,
                accounts_by_type=accounts_by_type,
                balance_by_type=balance_by_type
            )
            
        except Exception as e:
            self._logger.error(f"Failed to get account summary for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get account summary failed: {e}")
    
    async def search_accounts(self, filters: AccountFilters) -> List[Account]:
        """
        Search accounts with flexible filters.
        
        Args:
            filters: Search filters
            
        Returns:
            List of matching accounts
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            where_conditions = []
            params = []
            param_idx = 1
            
            if filters.user_id:
                where_conditions.append(f"user_id = ${param_idx}")
                params.append(filters.user_id)
                param_idx += 1
            
            if filters.account_type_ids:
                placeholders = ','.join(f'${param_idx + i}' for i in range(len(filters.account_type_ids)))
                where_conditions.append(f"account_type_id IN ({placeholders})")
                params.extend(filters.account_type_ids)
                param_idx += len(filters.account_type_ids)
            
            if filters.min_balance is not None:
                where_conditions.append(f"current_balance >= ${param_idx}")
                params.append(filters.min_balance)
                param_idx += 1
            
            if filters.max_balance is not None:
                where_conditions.append(f"current_balance <= ${param_idx}")
                params.append(filters.max_balance)
                param_idx += 1
            
            if filters.created_after:
                where_conditions.append(f"created_at > ${param_idx}")
                params.append(filters.created_after)
                param_idx += 1
            
            if filters.created_before:
                where_conditions.append(f"created_at < ${param_idx}")
                params.append(filters.created_before)
                param_idx += 1
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f"""
                SELECT * FROM accounts 
                WHERE {where_clause}
                ORDER BY created_at DESC
            """
            
            result = await self._execute_query(query, *params, fetch_all=True)
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to search accounts: {e}")
            raise DatabaseOperationError(f"Account search failed: {e}")
    
    # Account type management methods
    
    async def get_all_account_types(self) -> List[AccountType]:
        """
        Get all available account types.
        
        Returns:
            List of account types
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = "SELECT * FROM account_types ORDER BY name"
            result = await self._execute_query(query, fetch_all=True)
            
            account_types = []
            for row in result or []:
                account_type = AccountType.model_validate(dict(row))
                account_types.append(account_type)
            
            return account_types
            
        except Exception as e:
            self._logger.error(f"Failed to get account types: {e}")
            raise DatabaseOperationError(f"Get account types failed: {e}")
    
    async def get_account_type_by_name(self, type_name: str) -> Optional[AccountType]:
        """
        Get an account type by name.
        
        Args:
            type_name: Account type name
            
        Returns:
            Account type if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = "SELECT * FROM account_types WHERE name = $1"
            result = await self._execute_query(query, type_name.lower(), fetch_one=True)
            
            if result:
                return AccountType.model_validate(dict(result))
            return None
            
        except Exception as e:
            self._logger.error(f"Failed to get account type by name {type_name}: {e}")
            raise DatabaseOperationError(f"Get account type by name failed: {e}")
    
    # Helper methods
    
    async def _account_type_exists(self, account_type_id: int) -> bool:
        """Check if an account type exists."""
        try:
            query = "SELECT 1 FROM account_types WHERE id = $1"
            result = await self._execute_query(query, account_type_id, fetch_one=True)
            return result is not None
        except Exception:
            return False
    
    async def _count_account_transactions(self, account_id: UUID) -> int:
        """Count transactions for an account."""
        try:
            query = "SELECT COUNT(*) FROM transactions WHERE account_id = $1"
            result = await self._execute_query(query, account_id, fetch_one=True)
            return result['count'] if result else 0
        except Exception:
            return 0 