"""
Transaction Repository for Financial Transaction Management.

This module provides domain-specific repository operations for transaction entities,
including complex filtering, aggregations, financial analysis, and transaction
history management.

Features:
- Transaction CRUD operations with validation
- Complex filtering and search capabilities
- Financial aggregations and analytics
- Transaction history and reporting
- Balance impact calculations
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from ..repositories.base_repository import BaseRepository, DatabaseOperationError, EntityValidationError
from ..schemas.database_models import (
    Transaction, TransactionCreate, TransactionUpdate, TransactionType,
    TransactionWithDetails, TransactionFilters, SpendingByCategory, SpendingByMonth
)

logger = logging.getLogger(__name__)


class TransactionRepository(BaseRepository[Transaction, TransactionCreate, TransactionUpdate, UUID]):
    """
    Repository for transaction entity operations.
    
    Provides specialized transaction management operations including complex filtering,
    financial analytics, aggregations, and reporting capabilities.
    """
    
    def __init__(self):
        """Initialize the transaction repository."""
        super().__init__(Transaction, "transactions")
    
    async def create(self, transaction_data: TransactionCreate) -> Transaction:
        """
        Create a new transaction with validation.
        
        Args:
            transaction_data: Transaction creation data
            
        Returns:
            Created transaction entity
            
        Raises:
            EntityValidationError: If validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Creating transaction for user {transaction_data.user_id}")
        
        try:
            # Validate account exists and belongs to user
            if not await self._validate_account_ownership(
                transaction_data.account_id, 
                transaction_data.user_id
            ):
                raise EntityValidationError(
                    f"Account {transaction_data.account_id} does not exist or belong to user {transaction_data.user_id}"
                )
            
            # Validate category exists
            if not await self._category_exists(transaction_data.category_id):
                raise EntityValidationError(f"Category {transaction_data.category_id} does not exist")
            
            # Validate group if provided
            if transaction_data.group_id and not await self._validate_group_ownership(
                transaction_data.group_id, 
                transaction_data.user_id
            ):
                raise EntityValidationError(
                    f"Group {transaction_data.group_id} does not exist or belong to user {transaction_data.user_id}"
                )
            
            transaction_dict = transaction_data.model_dump()
            
            query = """
                INSERT INTO transactions 
                (user_id, account_id, category_id, group_id, type, amount, 
                 counterparty, occurred_at, notes)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING *
            """
            
            result = await self._execute_query(
                query,
                transaction_dict['user_id'],
                transaction_dict['account_id'],
                transaction_dict['category_id'],
                transaction_dict.get('group_id'),
                transaction_dict['type'],
                transaction_dict['amount'],
                transaction_dict.get('counterparty'),
                transaction_dict['occurred_at'],
                transaction_dict.get('notes'),
                fetch_one=True
            )
            
            created_transaction = self._row_to_model(result)
            self._logger.info(f"Transaction created successfully with ID: {created_transaction.id}")
            return created_transaction
            
        except Exception as e:
            self._logger.error(f"Failed to create transaction: {e}")
            raise DatabaseOperationError(f"Transaction creation failed: {e}")
    
    async def get_by_id(self, transaction_id: UUID) -> Optional[Transaction]:
        """
        Get a transaction by its ID.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Transaction if found, None otherwise
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = "SELECT * FROM transactions WHERE id = $1"
            result = await self._execute_query(query, transaction_id, fetch_one=True)
            return self._row_to_model(result)
        except Exception as e:
            self._logger.error(f"Failed to get transaction by ID {transaction_id}: {e}")
            raise DatabaseOperationError(f"Get transaction by ID failed: {e}")
    
    async def update(self, transaction_id: UUID, transaction_update: TransactionUpdate) -> Optional[Transaction]:
        """
        Update a transaction's information.
        
        Args:
            transaction_id: Transaction identifier
            transaction_update: Update data
            
        Returns:
            Updated transaction if found, None otherwise
            
        Raises:
            EntityValidationError: If validation fails
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Updating transaction {transaction_id}")
        
        try:
            # Check if transaction exists
            if not await self.exists(transaction_id):
                return None
            
            # Build dynamic update query
            update_fields = []
            params = []
            param_idx = 1
            
            update_dict = transaction_update.model_dump(exclude_unset=True)
            
            for field, value in update_dict.items():
                if value is not None:
                    update_fields.append(f"{field} = ${param_idx}")
                    params.append(value)
                    param_idx += 1
            
            if not update_fields:
                return await self.get_by_id(transaction_id)
            
            params.append(transaction_id)
            
            query = f"""
                UPDATE transactions 
                SET {', '.join(update_fields)}
                WHERE id = ${param_idx}
                RETURNING *
            """
            
            result = await self._execute_query(query, *params, fetch_one=True)
            updated_transaction = self._row_to_model(result)
            
            if updated_transaction:
                self._logger.info(f"Transaction {transaction_id} updated successfully")
            
            return updated_transaction
            
        except Exception as e:
            self._logger.error(f"Failed to update transaction {transaction_id}: {e}")
            raise DatabaseOperationError(f"Transaction update failed: {e}")
    
    async def delete(self, transaction_id: UUID) -> bool:
        """
        Delete a transaction by its ID.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        self._logger.info(f"Deleting transaction {transaction_id}")
        
        try:
            query = "DELETE FROM transactions WHERE id = $1"
            result = await self._execute_query(query, transaction_id, fetch_one=False, fetch_all=False)
            
            deleted = result == "DELETE 1"
            
            if deleted:
                self._logger.info(f"Transaction {transaction_id} deleted successfully")
            else:
                self._logger.warning(f"Transaction {transaction_id} not found for deletion")
            
            return deleted
            
        except Exception as e:
            self._logger.error(f"Failed to delete transaction {transaction_id}: {e}")
            raise DatabaseOperationError(f"Transaction deletion failed: {e}")
    
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Transaction]:
        """
        List all transactions with optional pagination.
        
        Args:
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip
            
        Returns:
            List of transactions
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            if limit is not None and offset is not None:
                query = "SELECT * FROM transactions ORDER BY occurred_at DESC LIMIT $1 OFFSET $2"
                result = await self._execute_query(query, limit, offset, fetch_all=True)
            elif limit is not None:
                query = "SELECT * FROM transactions ORDER BY occurred_at DESC LIMIT $1"
                result = await self._execute_query(query, limit, fetch_all=True)
            else:
                query = "SELECT * FROM transactions ORDER BY occurred_at DESC"
                result = await self._execute_query(query, fetch_all=True)
            
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to list transactions: {e}")
            raise DatabaseOperationError(f"List transactions failed: {e}")
    
    # Transaction-specific domain methods
    
    async def get_user_transactions(
        self, 
        user_id: UUID, 
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Transaction]:
        """
        Get all transactions for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of transactions to return
            offset: Number of transactions to skip
            
        Returns:
            List of user's transactions
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            if limit is not None and offset is not None:
                query = """
                    SELECT * FROM transactions 
                    WHERE user_id = $1 
                    ORDER BY occurred_at DESC 
                    LIMIT $2 OFFSET $3
                """
                result = await self._execute_query(query, user_id, limit, offset, fetch_all=True)
            elif limit is not None:
                query = """
                    SELECT * FROM transactions 
                    WHERE user_id = $1 
                    ORDER BY occurred_at DESC 
                    LIMIT $2
                """
                result = await self._execute_query(query, user_id, limit, fetch_all=True)
            else:
                query = """
                    SELECT * FROM transactions 
                    WHERE user_id = $1 
                    ORDER BY occurred_at DESC
                """
                result = await self._execute_query(query, user_id, fetch_all=True)
            
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to get transactions for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get user transactions failed: {e}")
    
    async def get_transactions_with_details(
        self, 
        user_id: UUID, 
        limit: Optional[int] = 100
    ) -> List[TransactionWithDetails]:
        """
        Get transactions with related entity details (account, category, group names).
        
        Args:
            user_id: User identifier
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions with details
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            query = """
                SELECT 
                    t.id,
                    t.user_id,
                    t.account_id,
                    a.name as account_name,
                    t.category_id,
                    c.name as category_name,
                    t.group_id,
                    g.name as group_name,
                    t.type,
                    t.amount,
                    t.counterparty,
                    t.occurred_at,
                    t.notes,
                    t.created_at
                FROM transactions t
                INNER JOIN accounts a ON t.account_id = a.id
                INNER JOIN categories c ON t.category_id = c.id
                LEFT JOIN groups g ON t.group_id = g.id
                WHERE t.user_id = $1
                ORDER BY t.occurred_at DESC
                LIMIT $2
            """
            
            result = await self._execute_query(query, user_id, limit, fetch_all=True)
            
            transactions = []
            for row in result or []:
                transaction = TransactionWithDetails.model_validate(dict(row))
                transactions.append(transaction)
            
            return transactions
            
        except Exception as e:
            self._logger.error(f"Failed to get transactions with details for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get transactions with details failed: {e}")
    
    async def search_transactions(self, filters: TransactionFilters) -> List[Transaction]:
        """
        Search transactions with complex filters.
        
        Args:
            filters: Transaction search filters
            
        Returns:
            List of matching transactions
            
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
            
            if filters.account_ids:
                placeholders = ','.join(f'${param_idx + i}' for i in range(len(filters.account_ids)))
                where_conditions.append(f"account_id IN ({placeholders})")
                params.extend(filters.account_ids)
                param_idx += len(filters.account_ids)
            
            if filters.category_ids:
                placeholders = ','.join(f'${param_idx + i}' for i in range(len(filters.category_ids)))
                where_conditions.append(f"category_id IN ({placeholders})")
                params.extend(filters.category_ids)
                param_idx += len(filters.category_ids)
            
            if filters.group_ids:
                placeholders = ','.join(f'${param_idx + i}' for i in range(len(filters.group_ids)))
                where_conditions.append(f"group_id IN ({placeholders})")
                params.extend(filters.group_ids)
                param_idx += len(filters.group_ids)
            
            if filters.transaction_type:
                where_conditions.append(f"type = ${param_idx}")
                params.append(filters.transaction_type)
                param_idx += 1
            
            if filters.min_amount is not None:
                where_conditions.append(f"amount >= ${param_idx}")
                params.append(filters.min_amount)
                param_idx += 1
            
            if filters.max_amount is not None:
                where_conditions.append(f"amount <= ${param_idx}")
                params.append(filters.max_amount)
                param_idx += 1
            
            if filters.start_date:
                where_conditions.append(f"occurred_at >= ${param_idx}")
                params.append(filters.start_date)
                param_idx += 1
            
            if filters.end_date:
                where_conditions.append(f"occurred_at <= ${param_idx}")
                params.append(filters.end_date)
                param_idx += 1
            
            if filters.counterparty:
                where_conditions.append(f"counterparty ILIKE ${param_idx}")
                params.append(f"%{filters.counterparty}%")
                param_idx += 1
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Build ORDER BY clause
            order_by = filters.order_by or "occurred_at"
            order_direction = filters.order_direction or "desc"
            
            query = f"""
                SELECT * FROM transactions 
                WHERE {where_clause}
                ORDER BY {order_by} {order_direction.upper()}
            """
            
            # Add LIMIT and OFFSET
            if filters.limit:
                query += f" LIMIT ${param_idx}"
                params.append(filters.limit)
                param_idx += 1
            
            if filters.offset:
                query += f" OFFSET ${param_idx}"
                params.append(filters.offset)
                param_idx += 1
            
            result = await self._execute_query(query, *params, fetch_all=True)
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to search transactions: {e}")
            raise DatabaseOperationError(f"Transaction search failed: {e}")
    
    async def get_account_transactions(
        self, 
        account_id: UUID, 
        limit: Optional[int] = None
    ) -> List[Transaction]:
        """
        Get all transactions for a specific account.
        
        Args:
            account_id: Account identifier
            limit: Maximum number of transactions to return
            
        Returns:
            List of account transactions
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            if limit:
                query = """
                    SELECT * FROM transactions 
                    WHERE account_id = $1 
                    ORDER BY occurred_at DESC 
                    LIMIT $2
                """
                result = await self._execute_query(query, account_id, limit, fetch_all=True)
            else:
                query = """
                    SELECT * FROM transactions 
                    WHERE account_id = $1 
                    ORDER BY occurred_at DESC
                """
                result = await self._execute_query(query, account_id, fetch_all=True)
            
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to get transactions for account {account_id}: {e}")
            raise DatabaseOperationError(f"Get account transactions failed: {e}")
    
    # Financial Analytics Methods
    
    async def get_spending_by_category(
        self, 
        user_id: UUID, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[SpendingByCategory]:
        """
        Get spending breakdown by category.
        
        Args:
            user_id: User identifier
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of spending by category
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            where_conditions = ["t.user_id = $1", "t.type = 'paid'"]
            params = [user_id]
            param_idx = 2
            
            if start_date:
                where_conditions.append(f"t.occurred_at >= ${param_idx}")
                params.append(start_date)
                param_idx += 1
            
            if end_date:
                where_conditions.append(f"t.occurred_at <= ${param_idx}")
                params.append(end_date)
                param_idx += 1
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT 
                    c.id as category_id,
                    c.name as category_name,
                    SUM(t.amount) as total_amount,
                    COUNT(t.id) as transaction_count
                FROM transactions t
                INNER JOIN categories c ON t.category_id = c.id
                WHERE {where_clause}
                GROUP BY c.id, c.name
                ORDER BY total_amount DESC
            """
            
            result = await self._execute_query(query, *params, fetch_all=True)
            
            spending = []
            for row in result or []:
                spending_item = SpendingByCategory.model_validate(dict(row))
                spending.append(spending_item)
            
            return spending
            
        except Exception as e:
            self._logger.error(f"Failed to get spending by category for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get spending by category failed: {e}")
    
    async def get_spending_by_month(
        self, 
        user_id: UUID, 
        year: Optional[int] = None
    ) -> List[SpendingByMonth]:
        """
        Get spending breakdown by month.
        
        Args:
            user_id: User identifier
            year: Year to analyze (current year if not specified)
            
        Returns:
            List of spending by month
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            if year is None:
                year = datetime.now().year
            
            query = """
                SELECT 
                    EXTRACT(YEAR FROM occurred_at) as year,
                    EXTRACT(MONTH FROM occurred_at) as month,
                    SUM(amount) as total_amount,
                    COUNT(id) as transaction_count
                FROM transactions
                WHERE user_id = $1 
                    AND type = 'paid'
                    AND EXTRACT(YEAR FROM occurred_at) = $2
                GROUP BY EXTRACT(YEAR FROM occurred_at), EXTRACT(MONTH FROM occurred_at)
                ORDER BY year, month
            """
            
            result = await self._execute_query(query, user_id, year, fetch_all=True)
            
            spending = []
            for row in result or []:
                spending_item = SpendingByMonth(
                    year=int(row['year']),
                    month=int(row['month']),
                    total_amount=Decimal(str(row['total_amount'])),
                    transaction_count=row['transaction_count']
                )
                spending.append(spending_item)
            
            return spending
            
        except Exception as e:
            self._logger.error(f"Failed to get spending by month for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get spending by month failed: {e}")
    
    async def get_total_spending(
        self, 
        user_id: UUID, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Decimal:
        """
        Get total spending for a user in a date range.
        
        Args:
            user_id: User identifier
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            Total spending amount
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            where_conditions = ["user_id = $1", "type = 'paid'"]
            params = [user_id]
            param_idx = 2
            
            if start_date:
                where_conditions.append(f"occurred_at >= ${param_idx}")
                params.append(start_date)
                param_idx += 1
            
            if end_date:
                where_conditions.append(f"occurred_at <= ${param_idx}")
                params.append(end_date)
                param_idx += 1
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT COALESCE(SUM(amount), 0) as total_spending
                FROM transactions
                WHERE {where_clause}
            """
            
            result = await self._execute_query(query, *params, fetch_one=True)
            return Decimal(str(result['total_spending'])) if result else Decimal('0.00')
            
        except Exception as e:
            self._logger.error(f"Failed to get total spending for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get total spending failed: {e}")
    
    async def get_recent_transactions(
        self, 
        user_id: UUID, 
        days: int = 30,
        limit: int = 50
    ) -> List[Transaction]:
        """
        Get recent transactions for a user.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            limit: Maximum number of transactions
            
        Returns:
            List of recent transactions
            
        Raises:
            DatabaseOperationError: If database operation fails
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT * FROM transactions
                WHERE user_id = $1 AND occurred_at >= $2
                ORDER BY occurred_at DESC
                LIMIT $3
            """
            
            result = await self._execute_query(query, user_id, cutoff_date, limit, fetch_all=True)
            return self._rows_to_models(result or [])
            
        except Exception as e:
            self._logger.error(f"Failed to get recent transactions for user {user_id}: {e}")
            raise DatabaseOperationError(f"Get recent transactions failed: {e}")
    
    # Helper methods
    
    async def _validate_account_ownership(self, account_id: UUID, user_id: UUID) -> bool:
        """Validate that an account belongs to a user."""
        try:
            query = "SELECT 1 FROM accounts WHERE id = $1 AND user_id = $2"
            result = await self._execute_query(query, account_id, user_id, fetch_one=True)
            return result is not None
        except Exception:
            return False
    
    async def _category_exists(self, category_id: int) -> bool:
        """Check if a category exists."""
        try:
            query = "SELECT 1 FROM categories WHERE id = $1"
            result = await self._execute_query(query, category_id, fetch_one=True)
            return result is not None
        except Exception:
            return False
    
    async def _validate_group_ownership(self, group_id: UUID, user_id: UUID) -> bool:
        """Validate that a group belongs to a user."""
        try:
            query = "SELECT 1 FROM groups WHERE id = $1 AND user_id = $2"
            result = await self._execute_query(query, group_id, user_id, fetch_one=True)
            return result is not None
        except Exception:
            return False 