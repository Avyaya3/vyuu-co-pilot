"""
Database Schema Models for Financial Management System.

This module contains Pydantic models that represent the database schema,
providing type safety, validation, and serialization for all financial entities.

Features:
- Comprehensive validation with custom validators
- Proper UUID and datetime handling
- Decimal precision for financial amounts
- Relationship modeling with foreign keys
- Enum definitions for constrained fields
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    PAID = "paid"
    RECEIVED = "received"


class BaseEntity(BaseModel):
    """Base model for all database entities."""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            UUID: lambda v: str(v),
        }
    )


# User Models
class User(BaseEntity):
    """User profile model."""
    id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    profile_pic_url: Optional[str] = Field(None, max_length=500)
    settings: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email format')
        return v.lower().strip()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize name."""
        return v.strip()


class UserCreate(BaseEntity):
    """User creation model."""
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    profile_pic_url: Optional[str] = Field(None, max_length=500)
    settings: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email format')
        return v.lower().strip()


class UserUpdate(BaseEntity):
    """User update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    profile_pic_url: Optional[str] = Field(None, max_length=500)
    settings: Optional[Dict[str, Any]] = None


# Account Type Models
class AccountType(BaseEntity):
    """Account type model."""
    id: int
    name: str = Field(..., min_length=1, max_length=100)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize account type name."""
        return v.strip().lower()


# Account Models
class Account(BaseEntity):
    """Financial account model."""
    id: UUID
    user_id: UUID
    account_type_id: int
    name: str = Field(..., min_length=1, max_length=255)
    current_balance: Decimal = Field(..., decimal_places=2)
    created_at: Optional[datetime] = None

    @field_validator('current_balance')
    @classmethod
    def validate_balance(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert balance to Decimal with 2 decimal places."""
        try:
            balance = Decimal(str(v)).quantize(Decimal('0.01'))
            return balance
        except Exception:
            raise ValueError('Invalid balance format')

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize account name."""
        return v.strip()


class AccountCreate(BaseEntity):
    """Account creation model."""
    user_id: UUID
    account_type_id: int
    name: str = Field(..., min_length=1, max_length=255)
    current_balance: Decimal = Field(default=Decimal('0.00'), decimal_places=2)

    @field_validator('current_balance')
    @classmethod
    def validate_balance(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert balance to Decimal with 2 decimal places."""
        try:
            balance = Decimal(str(v)).quantize(Decimal('0.01'))
            return balance
        except Exception:
            raise ValueError('Invalid balance format')


class AccountUpdate(BaseEntity):
    """Account update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    current_balance: Optional[Decimal] = None

    @field_validator('current_balance')
    @classmethod
    def validate_balance(cls, v: Optional[Union[Decimal, float, str]]) -> Optional[Decimal]:
        """Validate and convert balance to Decimal with 2 decimal places."""
        if v is None:
            return None
        try:
            balance = Decimal(str(v)).quantize(Decimal('0.01'))
            return balance
        except Exception:
            raise ValueError('Invalid balance format')


# Category Models
class Category(BaseEntity):
    """Transaction category model."""
    id: int
    name: str = Field(..., min_length=1, max_length=100)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize category name."""
        return v.strip().lower()


# Group Models
class Group(BaseEntity):
    """Transaction group model."""
    id: UUID
    user_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    created_at: Optional[datetime] = None

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize group name."""
        return v.strip()


class GroupCreate(BaseEntity):
    """Group creation model."""
    user_id: UUID
    name: str = Field(..., min_length=1, max_length=255)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize group name."""
        return v.strip()


# Transaction Models
class Transaction(BaseEntity):
    """Financial transaction model."""
    id: UUID
    user_id: UUID
    account_id: UUID
    category_id: int
    group_id: Optional[UUID] = None
    type: TransactionType
    amount: Decimal = Field(..., decimal_places=2)
    counterparty: Optional[str] = Field(None, max_length=255)
    occurred_at: datetime
    notes: Optional[str] = Field(None, max_length=1000)
    created_at: Optional[datetime] = None

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert amount to Decimal with 2 decimal places."""
        try:
            amount = Decimal(str(v)).quantize(Decimal('0.01'))
            if amount <= 0:
                raise ValueError('Amount must be positive')
            return amount
        except Exception as e:
            if 'Amount must be positive' in str(e):
                raise e
            raise ValueError('Invalid amount format')

    @field_validator('counterparty')
    @classmethod
    def validate_counterparty(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize counterparty."""
        if v is None:
            return None
        return v.strip()

    @field_validator('notes')
    @classmethod
    def validate_notes(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize notes."""
        if v is None:
            return None
        return v.strip()


class TransactionCreate(BaseEntity):
    """Transaction creation model."""
    user_id: UUID
    account_id: UUID
    category_id: int
    group_id: Optional[UUID] = None
    type: TransactionType
    amount: Decimal = Field(..., decimal_places=2)
    counterparty: Optional[str] = Field(None, max_length=255)
    occurred_at: datetime
    notes: Optional[str] = Field(None, max_length=1000)

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert amount to Decimal with 2 decimal places."""
        try:
            amount = Decimal(str(v)).quantize(Decimal('0.01'))
            if amount <= 0:
                raise ValueError('Amount must be positive')
            return amount
        except Exception as e:
            if 'Amount must be positive' in str(e):
                raise e
            raise ValueError('Invalid amount format')


class TransactionUpdate(BaseEntity):
    """Transaction update model."""
    category_id: Optional[int] = None
    group_id: Optional[UUID] = None
    type: Optional[TransactionType] = None
    amount: Optional[Decimal] = None
    counterparty: Optional[str] = Field(None, max_length=255)
    occurred_at: Optional[datetime] = None
    notes: Optional[str] = Field(None, max_length=1000)

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: Optional[Union[Decimal, float, str]]) -> Optional[Decimal]:
        """Validate and convert amount to Decimal with 2 decimal places."""
        if v is None:
            return None
        try:
            amount = Decimal(str(v)).quantize(Decimal('0.01'))
            if amount <= 0:
                raise ValueError('Amount must be positive')
            return amount
        except Exception as e:
            if 'Amount must be positive' in str(e):
                raise e
            raise ValueError('Invalid amount format')


# Goal Models
class Goal(BaseEntity):
    """Financial goal model."""
    id: UUID
    user_id: UUID
    title: str = Field(..., min_length=1, max_length=255)
    target_amount: Decimal = Field(..., decimal_places=2)
    current_amount: Decimal = Field(default=Decimal('0.00'), decimal_places=2)
    is_active: bool = Field(default=True)
    created_at: Optional[datetime] = None

    @field_validator('target_amount')
    @classmethod
    def validate_target_amount(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert target amount to Decimal with 2 decimal places."""
        try:
            amount = Decimal(str(v)).quantize(Decimal('0.01'))
            if amount <= 0:
                raise ValueError('Target amount must be positive')
            return amount
        except Exception as e:
            if 'Target amount must be positive' in str(e):
                raise e
            raise ValueError('Invalid target amount format')

    @field_validator('current_amount')
    @classmethod
    def validate_current_amount(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert current amount to Decimal with 2 decimal places."""
        try:
            amount = Decimal(str(v)).quantize(Decimal('0.01'))
            if amount < 0:
                raise ValueError('Current amount cannot be negative')
            return amount
        except Exception as e:
            if 'Current amount cannot be negative' in str(e):
                raise e
            raise ValueError('Invalid current amount format')

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and normalize title."""
        return v.strip()


class GoalCreate(BaseEntity):
    """Goal creation model."""
    user_id: UUID
    title: str = Field(..., min_length=1, max_length=255)
    target_amount: Decimal = Field(..., decimal_places=2)
    current_amount: Decimal = Field(default=Decimal('0.00'), decimal_places=2)
    is_active: bool = Field(default=True)

    @field_validator('target_amount')
    @classmethod
    def validate_target_amount(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert target amount to Decimal with 2 decimal places."""
        try:
            amount = Decimal(str(v)).quantize(Decimal('0.01'))
            if amount <= 0:
                raise ValueError('Target amount must be positive')
            return amount
        except Exception as e:
            if 'Target amount must be positive' in str(e):
                raise e
            raise ValueError('Invalid target amount format')


class GoalUpdate(BaseEntity):
    """Goal update model."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    target_amount: Optional[Decimal] = None
    current_amount: Optional[Decimal] = None
    is_active: Optional[bool] = None


# Net Worth Snapshot Models
class NetWorthSnapshot(BaseEntity):
    """Net worth snapshot model."""
    id: UUID
    user_id: UUID
    total_value: Decimal = Field(..., decimal_places=2)
    measured_at: datetime

    @field_validator('total_value')
    @classmethod
    def validate_total_value(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert total value to Decimal with 2 decimal places."""
        try:
            value = Decimal(str(v)).quantize(Decimal('0.01'))
            return value  # Can be negative for debt
        except Exception:
            raise ValueError('Invalid total value format')


class AccountSnapshot(BaseEntity):
    """Account snapshot model."""
    id: UUID
    snapshot_id: UUID
    account_id: UUID
    balance: Decimal = Field(..., decimal_places=2)

    @field_validator('balance')
    @classmethod
    def validate_balance(cls, v: Union[Decimal, float, str]) -> Decimal:
        """Validate and convert balance to Decimal with 2 decimal places."""
        try:
            balance = Decimal(str(v)).quantize(Decimal('0.01'))
            return balance  # Can be negative for credit/debt accounts
        except Exception:
            raise ValueError('Invalid balance format')


# Aggregate Models for Complex Queries
class AccountWithType(BaseEntity):
    """Account with type information."""
    id: UUID
    user_id: UUID
    account_type_id: int
    account_type_name: str
    name: str
    current_balance: Decimal
    created_at: Optional[datetime] = None


class TransactionWithDetails(BaseEntity):
    """Transaction with related entity details."""
    id: UUID
    user_id: UUID
    account_id: UUID
    account_name: str
    category_id: int
    category_name: str
    group_id: Optional[UUID] = None
    group_name: Optional[str] = None
    type: TransactionType
    amount: Decimal
    counterparty: Optional[str] = None
    occurred_at: datetime
    notes: Optional[str] = None
    created_at: Optional[datetime] = None


# Query Filter Models
class TransactionFilters(BaseEntity):
    """Filters for transaction queries."""
    user_id: Optional[UUID] = None
    account_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[int]] = None
    group_ids: Optional[List[UUID]] = None
    transaction_type: Optional[TransactionType] = None
    min_amount: Optional[Decimal] = None
    max_amount: Optional[Decimal] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    counterparty: Optional[str] = None
    limit: Optional[int] = Field(None, ge=1, le=10000)
    offset: Optional[int] = Field(None, ge=0)
    order_by: Optional[str] = Field(None, pattern=r'^(occurred_at|amount|created_at)$')
    order_direction: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class AccountFilters(BaseEntity):
    """Filters for account queries."""
    user_id: Optional[UUID] = None
    account_type_ids: Optional[List[int]] = None
    min_balance: Optional[Decimal] = None
    max_balance: Optional[Decimal] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


# Aggregation Models
class SpendingByCategory(BaseEntity):
    """Spending aggregation by category."""
    category_id: int
    category_name: str
    total_amount: Decimal
    transaction_count: int


class SpendingByMonth(BaseEntity):
    """Spending aggregation by month."""
    year: int
    month: int
    total_amount: Decimal
    transaction_count: int


class AccountSummary(BaseEntity):
    """Account summary statistics."""
    total_accounts: int
    total_balance: Decimal
    accounts_by_type: Dict[str, int]
    balance_by_type: Dict[str, Decimal] 