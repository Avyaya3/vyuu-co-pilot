"""
Database Schema Models for Financial Management System.

This module contains Pydantic models that represent the new database schema,
providing type safety, validation, and serialization for all financial entities.

Features:
- Comprehensive validation with custom validators
- Proper datetime handling
- Integer precision for financial amounts (cents)
- Relationship modeling with foreign keys
- Enum definitions for constrained fields
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


class BaseEntity(BaseModel):
    """Base model for all database entities."""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )


# User Models
class User(BaseEntity):
    """User profile model."""
    id: str  # cuid format
    name: Optional[str] = None
    email: Optional[str] = None
    emailVerified: Optional[datetime] = None
    image: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    monthly_income: Optional[int] = None
    monthly_expenses: Optional[int] = None
    risk_profile: Optional[str] = None
    deleted: bool = False
    createdAt: datetime
    updatedAt: datetime

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Basic email validation."""
        if v is None:
            return None
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email format')
        return v.lower().strip()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize name."""
        if v is None:
            return None
        return v.strip()


class UserCreate(BaseEntity):
    """User creation model."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    monthly_income: Optional[int] = None
    monthly_expenses: Optional[int] = None
    risk_profile: Optional[str] = None


class UserUpdate(BaseEntity):
    """User update model."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    monthly_income: Optional[int] = None
    monthly_expenses: Optional[int] = None
    risk_profile: Optional[str] = None


# Asset Models
class Asset(BaseEntity):
    """Asset model."""
    id: str  # cuid format
    userId: str  # cuid format
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: str = Field(..., min_length=1, max_length=100)
    currentValue: int  # Amount in cents
    purchaseValue: int  # Amount in cents
    purchaseDate: datetime
    description: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    @field_validator('currentValue', 'purchaseValue')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is positive."""
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

    @field_validator('name', 'category', 'subcategory')
    @classmethod
    def validate_string_fields(cls, v: str) -> str:
        """Validate and normalize string fields."""
        return v.strip()


class AssetCreate(BaseEntity):
    """Asset creation model."""
    userId: str
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: str = Field(..., min_length=1, max_length=100)
    currentValue: int
    purchaseValue: int
    purchaseDate: datetime
    description: Optional[str] = None


class AssetUpdate(BaseEntity):
    """Asset update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    subcategory: Optional[str] = Field(None, min_length=1, max_length=100)
    currentValue: Optional[int] = None
    purchaseValue: Optional[int] = None
    purchaseDate: Optional[datetime] = None
    description: Optional[str] = None


# Liability Models
class Liability(BaseEntity):
    """Liability model."""
    id: str  # cuid format
    userId: str  # cuid format
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=100)
    amount: int  # Amount in cents
    emi: int  # Amount in cents
    interestRate: float
    startDate: datetime
    endDate: datetime
    description: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    @field_validator('amount', 'emi')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is positive."""
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

    @field_validator('interestRate')
    @classmethod
    def validate_interest_rate(cls, v: float) -> float:
        """Validate interest rate is reasonable."""
        if v < 0 or v > 100:
            raise ValueError('Interest rate must be between 0 and 100')
        return v


class LiabilityCreate(BaseEntity):
    """Liability creation model."""
    userId: str
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=100)
    amount: int
    emi: int
    interestRate: float
    startDate: datetime
    endDate: datetime
    description: Optional[str] = None


class LiabilityUpdate(BaseEntity):
    """Liability update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    type: Optional[str] = Field(None, min_length=1, max_length=100)
    amount: Optional[int] = None
    emi: Optional[int] = None
    interestRate: Optional[float] = None
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    description: Optional[str] = None


# Savings Models
class Savings(BaseEntity):
    """Savings model."""
    id: str  # cuid format
    userId: str  # cuid format
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=100)
    currentBalance: int  # Amount in cents
    interestRate: float
    maturityDate: Optional[datetime] = None
    monthlyContribution: int  # Amount in cents
    targetAmount: Optional[int] = None  # Amount in cents
    description: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    @field_validator('currentBalance', 'monthlyContribution', 'targetAmount')
    @classmethod
    def validate_amount(cls, v: Optional[int]) -> Optional[int]:
        """Validate amount is non-negative."""
        if v is not None and v < 0:
            raise ValueError('Amount must be non-negative')
        return v

    @field_validator('interestRate')
    @classmethod
    def validate_interest_rate(cls, v: float) -> float:
        """Validate interest rate is reasonable."""
        if v < 0 or v > 100:
            raise ValueError('Interest rate must be between 0 and 100')
        return v


class SavingsCreate(BaseEntity):
    """Savings creation model."""
    userId: str
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=100)
    currentBalance: int
    interestRate: float
    maturityDate: Optional[datetime] = None
    monthlyContribution: int
    targetAmount: Optional[int] = None
    description: Optional[str] = None


class SavingsUpdate(BaseEntity):
    """Savings update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    type: Optional[str] = Field(None, min_length=1, max_length=100)
    currentBalance: Optional[int] = None
    interestRate: Optional[float] = None
    maturityDate: Optional[datetime] = None
    monthlyContribution: Optional[int] = None
    targetAmount: Optional[int] = None
    description: Optional[str] = None


# Income Models
class Income(BaseEntity):
    """Income model."""
    id: str  # cuid format
    userId: str  # cuid format
    source: str = Field(..., min_length=1, max_length=255)
    amount: int  # Amount in cents
    frequency: str = Field(..., pattern=r'^(monthly|quarterly|yearly)$')
    category: str = Field(..., min_length=1, max_length=100)
    date: datetime
    description: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is positive."""
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class IncomeCreate(BaseEntity):
    """Income creation model."""
    userId: str
    source: str = Field(..., min_length=1, max_length=255)
    amount: int
    frequency: str = Field(..., pattern=r'^(monthly|quarterly|yearly)$')
    category: str = Field(..., min_length=1, max_length=100)
    date: datetime
    description: Optional[str] = None


class IncomeUpdate(BaseEntity):
    """Income update model."""
    source: Optional[str] = Field(None, min_length=1, max_length=255)
    amount: Optional[int] = None
    frequency: Optional[str] = Field(None, pattern=r'^(monthly|quarterly|yearly)$')
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    date: Optional[datetime] = None
    description: Optional[str] = None


# Expense Models
class Expense(BaseEntity):
    """Expense model."""
    id: str  # cuid format
    userId: str  # cuid format
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: str = Field(..., min_length=1, max_length=100)
    amount: int  # Amount in cents
    date: datetime
    description: Optional[str] = None
    paymentMethod: str = Field(..., min_length=1, max_length=100)
    createdAt: datetime
    updatedAt: datetime

    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is positive."""
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class ExpenseCreate(BaseEntity):
    """Expense creation model."""
    userId: str
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: str = Field(..., min_length=1, max_length=100)
    amount: int
    date: datetime
    description: Optional[str] = None
    paymentMethod: str = Field(..., min_length=1, max_length=100)


class ExpenseUpdate(BaseEntity):
    """Expense update model."""
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    subcategory: Optional[str] = Field(None, min_length=1, max_length=100)
    amount: Optional[int] = None
    date: Optional[datetime] = None
    description: Optional[str] = None
    paymentMethod: Optional[str] = Field(None, min_length=1, max_length=100)


# Goal Models
class Goal(BaseEntity):
    """Financial goal model."""
    id: str  # cuid format
    userId: str  # cuid format
    name: str = Field(..., min_length=1, max_length=255)
    target: int  # Amount in cents
    current: int  # Amount in cents
    targetDate: datetime
    category: str = Field(..., min_length=1, max_length=100)
    priority: str = Field(..., pattern=r'^(low|medium|high)$')
    description: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    @field_validator('target', 'current')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is non-negative."""
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v


class GoalCreate(BaseEntity):
    """Goal creation model."""
    userId: str
    name: str = Field(..., min_length=1, max_length=255)
    target: int
    current: int
    targetDate: datetime
    category: str = Field(..., min_length=1, max_length=100)
    priority: str = Field(..., pattern=r'^(low|medium|high)$')
    description: Optional[str] = None


class GoalUpdate(BaseEntity):
    """Goal update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    target: Optional[int] = None
    current: Optional[int] = None
    targetDate: Optional[datetime] = None
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    priority: Optional[str] = Field(None, pattern=r'^(low|medium|high)$')
    description: Optional[str] = None


# Stock Models
class Stock(BaseEntity):
    """Stock model."""
    id: str  # cuid format
    userId: str  # cuid format
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=100)
    amount: int  # Amount in cents
    currentValue: int  # Amount in cents
    purchaseDate: datetime
    returns: float
    description: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    @field_validator('amount', 'currentValue')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is positive."""
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

    @field_validator('returns')
    @classmethod
    def validate_returns(cls, v: float) -> float:
        """Validate returns is reasonable."""
        if v < -100 or v > 1000:
            raise ValueError('Returns must be between -100 and 1000')
        return v


class StockCreate(BaseEntity):
    """Stock creation model."""
    userId: str
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=100)
    amount: int
    currentValue: int
    purchaseDate: datetime
    returns: float
    description: Optional[str] = None


class StockUpdate(BaseEntity):
    """Stock update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    type: Optional[str] = Field(None, min_length=1, max_length=100)
    amount: Optional[int] = None
    currentValue: Optional[int] = None
    purchaseDate: Optional[datetime] = None
    returns: Optional[float] = None
    description: Optional[str] = None


# Insurance Models
class Insurance(BaseEntity):
    """Insurance model."""
    id: str  # cuid format
    userId: str  # cuid format
    type: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., min_length=1, max_length=255)
    policyNumber: str = Field(..., min_length=1, max_length=255)
    premium: int  # Amount in cents
    coverage: int  # Amount in cents
    startDate: datetime
    endDate: datetime
    description: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    @field_validator('premium', 'coverage')
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is positive."""
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v


class InsuranceCreate(BaseEntity):
    """Insurance creation model."""
    userId: str
    type: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., min_length=1, max_length=255)
    policyNumber: str = Field(..., min_length=1, max_length=255)
    premium: int
    coverage: int
    startDate: datetime
    endDate: datetime
    description: Optional[str] = None


class InsuranceUpdate(BaseEntity):
    """Insurance update model."""
    type: Optional[str] = Field(None, min_length=1, max_length=100)
    provider: Optional[str] = Field(None, min_length=1, max_length=255)
    policyNumber: Optional[str] = Field(None, min_length=1, max_length=255)
    premium: Optional[int] = None
    coverage: Optional[int] = None
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    description: Optional[str] = None


# Query Filter Models
class AssetFilters(BaseEntity):
    """Filters for asset queries."""
    userId: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    minValue: Optional[int] = None
    maxValue: Optional[int] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(name|currentValue|purchaseDate|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class LiabilityFilters(BaseEntity):
    """Filters for liability queries."""
    userId: Optional[str] = None
    type: Optional[str] = None
    minAmount: Optional[int] = None
    maxAmount: Optional[int] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(name|amount|startDate|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class SavingsFilters(BaseEntity):
    """Filters for savings queries."""
    userId: Optional[str] = None
    type: Optional[str] = None
    minBalance: Optional[int] = None
    maxBalance: Optional[int] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(name|currentBalance|maturityDate|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class IncomeFilters(BaseEntity):
    """Filters for income queries."""
    userId: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    frequency: Optional[str] = Field(None, pattern=r'^(monthly|quarterly|yearly)$')
    minAmount: Optional[int] = None
    maxAmount: Optional[int] = None
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(source|amount|date|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class ExpenseFilters(BaseEntity):
    """Filters for expense queries."""
    userId: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    paymentMethod: Optional[str] = None
    minAmount: Optional[int] = None
    maxAmount: Optional[int] = None
    startDate: Optional[datetime] = None
    endDate: Optional[datetime] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(category|amount|date|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class GoalFilters(BaseEntity):
    """Filters for goal queries."""
    userId: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = Field(None, pattern=r'^(low|medium|high)$')
    minTarget: Optional[int] = None
    maxTarget: Optional[int] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(name|target|targetDate|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class StockFilters(BaseEntity):
    """Filters for stock queries."""
    userId: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    minAmount: Optional[int] = None
    maxAmount: Optional[int] = None
    minCurrentValue: Optional[int] = None
    maxCurrentValue: Optional[int] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(name|amount|currentValue|purchaseDate|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


class InsuranceFilters(BaseEntity):
    """Filters for insurance queries."""
    userId: Optional[str] = None
    type: Optional[str] = None
    provider: Optional[str] = None
    minPremium: Optional[int] = None
    maxPremium: Optional[int] = None
    minCoverage: Optional[int] = None
    maxCoverage: Optional[int] = None
    limit: Optional[int] = Field(None, ge=1, le=1000)
    offset: Optional[int] = Field(None, ge=0)
    orderBy: Optional[str] = Field(None, pattern=r'^(type|provider|premium|coverage|startDate|createdAt)$')
    orderDirection: Optional[str] = Field(None, pattern=r'^(asc|desc)$')


# Aggregation Models
class SpendingByCategory(BaseEntity):
    """Spending aggregation by category."""
    category: str
    subcategory: Optional[str] = None
    totalAmount: int  # Amount in cents
    transactionCount: int


class SpendingByMonth(BaseEntity):
    """Spending aggregation by month."""
    year: int
    month: int
    totalAmount: int  # Amount in cents
    transactionCount: int


# Conversation Session Models
class ConversationSession(BaseEntity):
    """Conversation session model for LangGraph state persistence."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    state_data: Dict[str, Any] = Field(..., description="Serialized LangGraph state")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Session expiration timestamp")
    is_active: bool = Field(True, description="Whether the session is active")
    message_count: int = Field(0, description="Number of messages in the session")
    last_intent: Optional[str] = Field(None, description="Last classified intent")
    last_confidence: Optional[float] = Field(None, description="Last intent confidence score")
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        if not v or len(v) < 8:
            raise ValueError("Session ID must be at least 8 characters long")
        return v
    
    @field_validator("state_data")
    @classmethod
    def validate_state_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state data is a dictionary."""
        if not isinstance(v, dict):
            raise ValueError("State data must be a dictionary")
        return v


class ConversationSessionCreate(BaseEntity):
    """Model for creating a new conversation session."""
    session_id: str
    user_id: Optional[str] = None
    state_data: Dict[str, Any]
    expires_at: Optional[datetime] = None


class ConversationSessionUpdate(BaseEntity):
    """Model for updating an existing conversation session."""
    state_data: Optional[Dict[str, Any]] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None
    message_count: Optional[int] = None
    last_intent: Optional[str] = None
    last_confidence: Optional[float] = None 