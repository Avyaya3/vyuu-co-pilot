"""
Financial Service for Business Logic Operations.

This module provides high-level financial management operations combining multiple
repositories and implementing complex business logic for financial operations.

Features:
- Account and transaction management
- Financial analytics and reporting
- Balance calculations and validations
- Transfer operations with validation
- Financial planning and goal tracking
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from ..repositories import (
    get_user_repository, get_account_repository, 
    get_transaction_repository, get_goal_repository,
    EntityNotFoundError, EntityValidationError, DatabaseOperationError
)
from ..schemas.database_models import (
    User, Account, Transaction, Goal,
    AccountCreate, TransactionCreate, GoalCreate,
    TransactionType, AccountSummary, SpendingByCategory, SpendingByMonth
)

logger = logging.getLogger(__name__)


class FinancialServiceError(Exception):
    """Base exception for financial service operations."""
    pass


class InsufficientFundsError(FinancialServiceError):
    """Raised when account has insufficient funds for operation."""
    pass


class TransferValidationError(FinancialServiceError):
    """Raised when transfer validation fails."""
    pass


class FinancialService:
    """
    High-level financial service providing business logic operations.
    
    Combines multiple repositories to implement complex financial operations
    with proper validation, error handling, and business rule enforcement.
    """
    
    def __init__(self):
        """Initialize the financial service."""
        self.user_repo = get_user_repository()
        self.account_repo = get_account_repository()
        self.transaction_repo = get_transaction_repository()
        self.goal_repo = get_goal_repository()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    # Account Management Operations
    
    async def create_user_account(
        self, 
        user_id: UUID, 
        account_type_id: int,
        account_name: str,
        initial_balance: Decimal = Decimal('0.00')
    ) -> Account:
        """
        Create a new account for a user with validation.
        
        Args:
            user_id: User identifier
            account_type_id: Account type identifier
            account_name: Name for the account
            initial_balance: Starting balance
            
        Returns:
            Created account
            
        Raises:
            EntityNotFoundError: If user doesn't exist
            EntityValidationError: If validation fails
            FinancialServiceError: If business logic validation fails
        """
        self._logger.info(f"Creating account '{account_name}' for user {user_id}")
        
        try:
            # Validate user exists
            user = await self.user_repo.get_by_id(user_id)
            if not user:
                raise EntityNotFoundError(f"User {user_id} not found")
            
            # Validate initial balance
            if initial_balance < 0:
                raise EntityValidationError("Initial balance cannot be negative")
            
            # Create account
            account_data = AccountCreate(
                user_id=user_id,
                account_type_id=account_type_id,
                name=account_name,
                current_balance=initial_balance
            )
            
            account = await self.account_repo.create(account_data)
            
            # If initial balance > 0, create an initial deposit transaction
            if initial_balance > 0:
                await self._create_initial_deposit_transaction(account, initial_balance)
            
            self._logger.info(f"Account created successfully: {account.id}")
            return account
            
        except Exception as e:
            self._logger.error(f"Failed to create account: {e}")
            if isinstance(e, (EntityNotFoundError, EntityValidationError)):
                raise
            raise FinancialServiceError(f"Account creation failed: {e}")
    
    async def get_user_financial_overview(self, user_id: UUID) -> Dict[str, any]:
        """
        Get comprehensive financial overview for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Financial overview with accounts, transactions, goals, and analytics
            
        Raises:
            EntityNotFoundError: If user doesn't exist
            FinancialServiceError: If operation fails
        """
        try:
            # Validate user exists
            user = await self.user_repo.get_by_id(user_id)
            if not user:
                raise EntityNotFoundError(f"User {user_id} not found")
            
            # Get user data in parallel
            accounts = await self.account_repo.get_user_accounts_with_type(user_id)
            account_summary = await self.account_repo.get_account_summary(user_id)
            recent_transactions = await self.transaction_repo.get_recent_transactions(user_id, days=30)
            goals = await self.goal_repo.get_user_goals(user_id, active_only=True)
            goal_summary = await self.goal_repo.get_user_goal_summary(user_id)
            
            # Get spending analytics
            spending_by_category = await self.transaction_repo.get_spending_by_category(
                user_id, 
                start_date=datetime.now() - timedelta(days=30)
            )
            
            return {
                'user': {
                    'id': user.id,
                    'name': user.name,
                    'email': user.email
                },
                'accounts': {
                    'summary': account_summary,
                    'accounts': accounts
                },
                'transactions': {
                    'recent': recent_transactions[:10],  # Last 10 transactions
                    'total_count': len(recent_transactions)
                },
                'goals': {
                    'summary': goal_summary,
                    'active_goals': goals
                },
                'analytics': {
                    'spending_by_category': spending_by_category
                }
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get financial overview for user {user_id}: {e}")
            if isinstance(e, EntityNotFoundError):
                raise
            raise FinancialServiceError(f"Financial overview failed: {e}")
    
    # Transaction Operations
    
    async def create_transaction(
        self,
        user_id: UUID,
        account_id: UUID,
        category_id: int,
        transaction_type: TransactionType,
        amount: Decimal,
        counterparty: Optional[str] = None,
        notes: Optional[str] = None,
        occurred_at: Optional[datetime] = None,
        group_id: Optional[UUID] = None
    ) -> Transaction:
        """
        Create a new transaction with balance updates.
        
        Args:
            user_id: User identifier
            account_id: Account identifier
            category_id: Category identifier
            transaction_type: Type of transaction (paid/received)
            amount: Transaction amount
            counterparty: Optional counterparty
            notes: Optional notes
            occurred_at: Transaction date (defaults to now)
            group_id: Optional group identifier
            
        Returns:
            Created transaction
            
        Raises:
            EntityNotFoundError: If entities don't exist
            InsufficientFundsError: If account has insufficient funds
            FinancialServiceError: If operation fails
        """
        self._logger.info(f"Creating {transaction_type} transaction of {amount} for account {account_id}")
        
        try:
            # Validate account exists and belongs to user
            account = await self.account_repo.get_by_id(account_id)
            if not account:
                raise EntityNotFoundError(f"Account {account_id} not found")
            
            if account.user_id != user_id:
                raise EntityValidationError(f"Account {account_id} does not belong to user {user_id}")
            
            # Check balance for outgoing transactions
            if transaction_type == TransactionType.PAID:
                if account.current_balance < amount:
                    raise InsufficientFundsError(
                        f"Insufficient funds. Balance: {account.current_balance}, Required: {amount}"
                    )
            
            # Create transaction
            transaction_data = TransactionCreate(
                user_id=user_id,
                account_id=account_id,
                category_id=category_id,
                group_id=group_id,
                type=transaction_type,
                amount=amount,
                counterparty=counterparty,
                occurred_at=occurred_at or datetime.now(),
                notes=notes
            )
            
            # Use transaction to ensure atomicity
            async with self.account_repo.transaction() as conn:
                # Create transaction
                transaction = await self.transaction_repo.create(transaction_data)
                
                # Update account balance
                balance_change = amount if transaction_type == TransactionType.RECEIVED else -amount
                await self.account_repo.adjust_balance(account_id, balance_change)
            
            self._logger.info(f"Transaction created successfully: {transaction.id}")
            return transaction
            
        except Exception as e:
            self._logger.error(f"Failed to create transaction: {e}")
            if isinstance(e, (EntityNotFoundError, EntityValidationError, InsufficientFundsError)):
                raise
            raise FinancialServiceError(f"Transaction creation failed: {e}")
    
    async def transfer_between_accounts(
        self,
        user_id: UUID,
        source_account_id: UUID,
        target_account_id: UUID,
        amount: Decimal,
        notes: Optional[str] = None,
        category_id: Optional[int] = None
    ) -> Tuple[Transaction, Transaction]:
        """
        Transfer money between user accounts.
        
        Args:
            user_id: User identifier
            source_account_id: Source account identifier
            target_account_id: Target account identifier
            amount: Transfer amount
            notes: Optional transfer notes
            category_id: Optional category for the transaction
            
        Returns:
            Tuple of (outgoing_transaction, incoming_transaction)
            
        Raises:
            EntityNotFoundError: If accounts don't exist
            TransferValidationError: If transfer validation fails
            InsufficientFundsError: If insufficient funds
            FinancialServiceError: If operation fails
        """
        self._logger.info(f"Transferring {amount} from {source_account_id} to {target_account_id}")
        
        try:
            # Validate both accounts exist and belong to user
            source_account = await self.account_repo.get_by_id(source_account_id)
            target_account = await self.account_repo.get_by_id(target_account_id)
            
            if not source_account:
                raise EntityNotFoundError(f"Source account {source_account_id} not found")
            if not target_account:
                raise EntityNotFoundError(f"Target account {target_account_id} not found")
            
            if source_account.user_id != user_id:
                raise TransferValidationError("Source account does not belong to user")
            if target_account.user_id != user_id:
                raise TransferValidationError("Target account does not belong to user")
            
            if source_account_id == target_account_id:
                raise TransferValidationError("Cannot transfer to the same account")
            
            # Check sufficient funds
            if source_account.current_balance < amount:
                raise InsufficientFundsError(
                    f"Insufficient funds in source account. Balance: {source_account.current_balance}, Required: {amount}"
                )
            
            # Default category for transfers (could be configurable)
            if category_id is None:
                category_id = 1  # Assuming 1 is "Transfer" category
            
            transfer_notes = notes or f"Transfer to {target_account.name}"
            
            # Execute transfer atomically
            async with self.account_repo.transaction() as conn:
                # Create outgoing transaction
                outgoing_transaction = await self.create_transaction(
                    user_id=user_id,
                    account_id=source_account_id,
                    category_id=category_id,
                    transaction_type=TransactionType.PAID,
                    amount=amount,
                    counterparty=f"Transfer to {target_account.name}",
                    notes=transfer_notes
                )
                
                # Create incoming transaction
                incoming_transaction = await self.create_transaction(
                    user_id=user_id,
                    account_id=target_account_id,
                    category_id=category_id,
                    transaction_type=TransactionType.RECEIVED,
                    amount=amount,
                    counterparty=f"Transfer from {source_account.name}",
                    notes=transfer_notes
                )
            
            self._logger.info(f"Transfer completed successfully")
            return outgoing_transaction, incoming_transaction
            
        except Exception as e:
            self._logger.error(f"Failed to transfer between accounts: {e}")
            if isinstance(e, (EntityNotFoundError, TransferValidationError, InsufficientFundsError)):
                raise
            raise FinancialServiceError(f"Transfer failed: {e}")
    
    # Financial Analytics Operations
    
    async def get_spending_analysis(
        self, 
        user_id: UUID, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, any]:
        """
        Get comprehensive spending analysis for a user.
        
        Args:
            user_id: User identifier
            start_date: Analysis start date (defaults to 30 days ago)
            end_date: Analysis end date (defaults to now)
            
        Returns:
            Spending analysis with breakdowns and trends
            
        Raises:
            EntityNotFoundError: If user doesn't exist
            FinancialServiceError: If operation fails
        """
        try:
            # Validate user exists
            user = await self.user_repo.get_by_id(user_id)
            if not user:
                raise EntityNotFoundError(f"User {user_id} not found")
            
            # Default date range
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Get spending data
            spending_by_category = await self.transaction_repo.get_spending_by_category(
                user_id, start_date, end_date
            )
            spending_by_month = await self.transaction_repo.get_spending_by_month(user_id)
            total_spending = await self.transaction_repo.get_total_spending(
                user_id, start_date, end_date
            )
            
            # Calculate insights
            total_income = await self._calculate_total_income(user_id, start_date, end_date)
            net_flow = total_income - total_spending
            
            return {
                'period': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'days': (end_date - start_date).days
                },
                'summary': {
                    'total_spending': total_spending,
                    'total_income': total_income,
                    'net_flow': net_flow,
                    'average_daily_spending': total_spending / max((end_date - start_date).days, 1)
                },
                'breakdowns': {
                    'by_category': spending_by_category,
                    'by_month': spending_by_month
                },
                'insights': await self._generate_spending_insights(spending_by_category, total_spending)
            }
            
        except Exception as e:
            self._logger.error(f"Failed to get spending analysis for user {user_id}: {e}")
            if isinstance(e, EntityNotFoundError):
                raise
            raise FinancialServiceError(f"Spending analysis failed: {e}")
    
    # Goal Management Operations
    
    async def create_financial_goal(
        self,
        user_id: UUID,
        title: str,
        target_amount: Decimal,
        current_amount: Decimal = Decimal('0.00')
    ) -> Goal:
        """
        Create a new financial goal for a user.
        
        Args:
            user_id: User identifier
            title: Goal title
            target_amount: Target amount to achieve
            current_amount: Current progress amount
            
        Returns:
            Created goal
            
        Raises:
            EntityNotFoundError: If user doesn't exist
            FinancialServiceError: If operation fails
        """
        try:
            # Validate user exists
            user = await self.user_repo.get_by_id(user_id)
            if not user:
                raise EntityNotFoundError(f"User {user_id} not found")
            
            goal_data = GoalCreate(
                user_id=user_id,
                title=title,
                target_amount=target_amount,
                current_amount=current_amount
            )
            
            goal = await self.goal_repo.create(goal_data)
            self._logger.info(f"Goal created successfully: {goal.id}")
            return goal
            
        except Exception as e:
            self._logger.error(f"Failed to create goal: {e}")
            if isinstance(e, EntityNotFoundError):
                raise
            raise FinancialServiceError(f"Goal creation failed: {e}")
    
    async def update_goal_progress(self, goal_id: UUID, amount: Decimal) -> Goal:
        """
        Update progress on a financial goal.
        
        Args:
            goal_id: Goal identifier
            amount: New current amount
            
        Returns:
            Updated goal
            
        Raises:
            EntityNotFoundError: If goal doesn't exist
            FinancialServiceError: If operation fails
        """
        try:
            goal = await self.goal_repo.update_goal_progress(goal_id, amount)
            if not goal:
                raise EntityNotFoundError(f"Goal {goal_id} not found")
            
            # Check if goal is achieved
            if goal.current_amount >= goal.target_amount:
                self._logger.info(f"Goal {goal_id} achieved!")
                # Could trigger notifications or other business logic here
            
            return goal
            
        except Exception as e:
            self._logger.error(f"Failed to update goal progress: {e}")
            if isinstance(e, EntityNotFoundError):
                raise
            raise FinancialServiceError(f"Goal progress update failed: {e}")
    
    # Helper Methods
    
    async def _create_initial_deposit_transaction(self, account: Account, amount: Decimal) -> Transaction:
        """Create an initial deposit transaction for a new account."""
        transaction_data = TransactionCreate(
            user_id=account.user_id,
            account_id=account.id,
            category_id=1,  # Assuming 1 is "Initial Deposit" category
            type=TransactionType.RECEIVED,
            amount=amount,
            counterparty="Initial Deposit",
            occurred_at=datetime.now(),
            notes="Initial account deposit"
        )
        
        return await self.transaction_repo.create(transaction_data)
    
    async def _calculate_total_income(
        self, 
        user_id: UUID, 
        start_date: datetime, 
        end_date: datetime
    ) -> Decimal:
        """Calculate total income for a user in a date range."""
        try:
            # This would need a method in transaction repository to filter by type
            # For now, simplified implementation
            query = """
                SELECT COALESCE(SUM(amount), 0) as total_income
                FROM transactions
                WHERE user_id = $1 
                    AND type = 'received'
                    AND occurred_at >= $2 
                    AND occurred_at <= $3
            """
            
            # Would need to add this to transaction repo or use existing search method
            # Simplified for demo
            return Decimal('0.00')
            
        except Exception:
            return Decimal('0.00')
    
    def _generate_spending_insights(
        self, 
        spending_by_category: List[SpendingByCategory], 
        total_spending: Decimal
    ) -> List[str]:
        """Generate insights from spending data."""
        insights = []
        
        if not spending_by_category or total_spending == 0:
            return ["No spending data available for this period."]
        
        # Find top spending category
        top_category = max(spending_by_category, key=lambda x: x.total_amount)
        top_percentage = (top_category.total_amount / total_spending) * 100
        
        insights.append(
            f"Your top spending category is {top_category.category_name}, "
            f"accounting for {top_percentage:.1f}% of total spending."
        )
        
        # Check for high spending categories
        high_spending_categories = [
            cat for cat in spending_by_category 
            if (cat.total_amount / total_spending) > 0.3
        ]
        
        if high_spending_categories:
            insights.append(
                f"Consider reviewing spending in categories where you spend more than 30% of your budget."
            )
        
        return insights 