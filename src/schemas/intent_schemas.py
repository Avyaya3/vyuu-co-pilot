"""
Intent Classification Schema Definitions.

This module provides Pydantic models for structured LLM output during intent classification,
including confidence scoring, parameter extraction, and fallback handling.

Features:
- Structured intent classification responses
- Confidence scoring with validation
- Parameter extraction schemas
- Fallback intent handling
- Comprehensive validation
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class IntentCategory(str, Enum):
    """Supported intent categories for user requests."""
    DATA_FETCH = "data_fetch"
    AGGREGATE = "aggregate"
    ACTION = "action"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level categories for intent classification."""
    HIGH = "high"      # 0.8 - 1.0
    MEDIUM = "medium"  # 0.5 - 0.79
    LOW = "low"        # 0.0 - 0.49


class DataFetchParams(BaseModel):
    """Parameters specific to data fetch intents."""
    
    entity_type: Optional[str] = Field(
        None,
        description="Type of data entity (transactions, accounts, balances, etc.)"
    )
    time_period: Optional[str] = Field(
        None,
        description="Time range for data (last_month, last_3_months, specific_date, etc.)"
    )
    account_types: Optional[List[str]] = Field(
        None,
        description="Types of accounts to include (checking, savings, credit, etc.)"
    )
    limit: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Maximum number of records to fetch"
    )
    sort_by: Optional[str] = Field(
        None,
        description="Field to sort by (amount, date, description, etc.)"
    )
    order: Optional[str] = Field(
        None,
        description="Sort order (asc, desc)"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional filters for data query"
    )


class AggregateParams(BaseModel):
    """Parameters specific to aggregate intents."""
    
    metric_type: Optional[str] = Field(
        None,
        description="Type of aggregation (sum, average, count, min, max, etc.)"
    )
    group_by: Optional[List[str]] = Field(
        None,
        description="Fields to group aggregation by (category, account, date, etc.)"
    )
    time_period: Optional[str] = Field(
        None,
        description="Time range for aggregation"
    )
    category_filter: Optional[List[str]] = Field(
        None,
        description="Categories to include in aggregation"
    )
    account_filter: Optional[List[str]] = Field(
        None,
        description="Accounts to include in aggregation"
    )
    comparison_period: Optional[str] = Field(
        None,
        description="Period to compare against (previous_month, last_year, etc.)"
    )


class ActionParams(BaseModel):
    """Parameters specific to action intents."""
    
    action_type: Optional[str] = Field(
        None,
        description="Type of action (transfer, payment, categorization, etc.)"
    )
    amount: Optional[float] = Field(
        None,
        description="Monetary amount for the action"
    )
    source_account: Optional[str] = Field(
        None,
        description="Source account for transfers/payments"
    )
    target_account: Optional[str] = Field(
        None,
        description="Target account or recipient"
    )
    description: Optional[str] = Field(
        None,
        description="Description or memo for the action"
    )
    schedule_date: Optional[str] = Field(
        None,
        description="When to execute the action (now, specific_date, recurring)"
    )
    confirmation_required: Optional[bool] = Field(
        True,
        description="Whether user confirmation is required before execution"
    )


class IntentClassificationResult(BaseModel):
    """
    Structured result from intent classification LLM call.
    
    This model represents the complete output from LLM intent classification,
    including the classified intent, confidence scoring, and extracted parameters.
    """
    
    intent: IntentCategory = Field(
        ...,
        description="Classified intent category"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for intent classification (0.0-1.0)"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief explanation of why this intent was chosen"
    )
    data_fetch_params: Optional[DataFetchParams] = Field(
        None,
        description="Parameters extracted for data fetch intents"
    )
    aggregate_params: Optional[AggregateParams] = Field(
        None,
        description="Parameters extracted for aggregate intents"
    )
    action_params: Optional[ActionParams] = Field(
        None,
        description="Parameters extracted for action intents"
    )
    missing_params: List[str] = Field(
        default_factory=list,
        description="List of required parameters that could not be extracted"
    )
    clarification_needed: bool = Field(
        False,
        description="Whether clarification is needed for missing parameters"
    )
    user_input_analysis: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Brief analysis of the user input"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v):
        """Validate confidence is within expected range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='after')
    def validate_intent_params_consistency(self):
        """Ensure intent-specific parameters are consistent with intent type."""
        if self.intent == IntentCategory.DATA_FETCH:
            if self.data_fetch_params is None:
                # Allow None but require reasoning if no params extracted
                if not any(phrase in self.reasoning.lower() for phrase in [
                    "no parameters", "parameters missing", "insufficient information", 
                    "unspecified", "unclear", "vague"
                ]):
                    raise ValueError("DataFetch intent should have data_fetch_params or explain why none extracted")
        
        elif self.intent == IntentCategory.AGGREGATE:
            if self.aggregate_params is None:
                if not any(phrase in self.reasoning.lower() for phrase in [
                    "no parameters", "parameters missing", "insufficient information",
                    "unspecified", "unclear", "vague"
                ]):
                    raise ValueError("Aggregate intent should have aggregate_params or explain why none extracted")
        
        elif self.intent == IntentCategory.ACTION:
            if self.action_params is None:
                if not any(phrase in self.reasoning.lower() for phrase in [
                    "no parameters", "parameters missing", "insufficient information",
                    "unspecified", "unclear", "vague"
                ]):
                    raise ValueError("Action intent should have action_params or explain why none extracted")
        
        return self
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    @property
    def requires_clarification(self) -> bool:
        """Check if this classification requires clarification."""
        return (
            self.clarification_needed or 
            len(self.missing_params) > 0 or 
            self.confidence < 0.7 or
            self.intent == IntentCategory.UNKNOWN
        )
    
    @property
    def extracted_parameters(self) -> Dict[str, Any]:
        """Get all extracted parameters as a single dictionary."""
        params = {}
        
        if self.data_fetch_params:
            params.update(self.data_fetch_params.model_dump(exclude_none=True))
        
        if self.aggregate_params:
            params.update(self.aggregate_params.model_dump(exclude_none=True))
        
        if self.action_params:
            params.update(self.action_params.model_dump(exclude_none=True))
        
        return params


class IntentClassificationError(Exception):
    """Custom exception for intent classification errors."""
    
    def __init__(self, message: str, error_type: str = "classification_error", user_input: str = ""):
        self.message = message
        self.error_type = error_type
        self.user_input = user_input
        super().__init__(message)


class FallbackIntentResult(BaseModel):
    """Fallback result when intent classification fails."""
    
    intent: IntentCategory = Field(
        default=IntentCategory.UNKNOWN,
        description="Fallback intent type"
    )
    confidence: float = Field(
        default=0.0,
        description="Low confidence for fallback"
    )
    reasoning: str = Field(
        default="Unable to classify intent from user input",
        description="Explanation for fallback"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error that caused fallback"
    )
    user_input_analysis: str = Field(
        default="Could not analyze user input",
        description="Fallback analysis"
    )
    missing_params: List[str] = Field(
        default_factory=lambda: ["intent_clarification"],
        description="Requires intent clarification"
    )
    clarification_needed: bool = Field(
        default=True,
        description="Always needs clarification for unknown intents"
    )
    
    @classmethod
    def from_error(cls, error: Exception, user_input: str = "") -> "FallbackIntentResult":
        """Create fallback result from an error."""
        return cls(
            error_message=str(error),
            user_input_analysis=f"Failed to analyze: '{user_input[:50]}...'" if user_input else "No input provided",
            reasoning=f"Classification failed due to: {type(error).__name__}"
        )
    
    def to_classification_result(self) -> IntentClassificationResult:
        """Convert to standard IntentClassificationResult."""
        return IntentClassificationResult(
            intent=self.intent,
            confidence=self.confidence,
            reasoning=self.reasoning,
            user_input_analysis=self.user_input_analysis,
            missing_params=self.missing_params,
            clarification_needed=self.clarification_needed
        ) 