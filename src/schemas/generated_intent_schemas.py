"""
Auto-generated Intent Schemas from YAML Configuration.

Generated on: 2025-07-17T17:28:46.831910
Source: /Users/manjiripathak/Avyaya/vyuu-copilot-v2/config/intent_parameters.yaml

DO NOT EDIT MANUALLY - Run scripts/generate_intent_schemas.py to regenerate.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class IntentCategory(str, Enum):
    """Supported intent categories for user requests."""
    DATA_FETCH = "data_fetch"
    AGGREGATE = "aggregate"
    ACTION = "action"
    UNKNOWN = "unknown"
    CLARIFICATION = "clarification"


class ConfidenceLevel(str, Enum):
    """Confidence level categories for intent classification."""
    HIGH = "high"      # 0.8 - 1.0
    MEDIUM = "medium"  # 0.5 - 0.79
    LOW = "low"        # 0.0 - 0.49


class DataFetchParams(BaseModel):
    """
    Retrieve and display financial data
    
    Auto-generated from intent_parameters.yaml
    """
    
    # Critical parameters
    entity_type: Optional[str] = Field(None, description="Type of data entity (transactions, accounts, balances, etc.)")

    # Optional parameters
    time_period: Optional[str] = Field(None, description="Time range for data (last_month, last_3_months, specific_date, etc.)")
    account_types: Optional[List[str]] = Field(None, description="Types of accounts to include (checking, savings, credit, etc.)")
    limit: Optional[int] = Field(None, description="Maximum number of records to return", ge=1, le=1000)
    sort_by: Optional[str] = Field(None, description="Field to sort results by")
    order: Optional[str] = Field(None, description="Sort order (asc, desc)")


class AggregateParams(BaseModel):
    """
    Analyze and summarize financial data
    
    Auto-generated from intent_parameters.yaml
    """
    
    # Critical parameters
    metric_type: Optional[str] = Field(None, description="Type of metric to calculate (sum, average, count, etc.)")

    # Optional parameters
    group_by: Optional[str] = Field(None, description="Field to group results by")
    time_period: Optional[str] = Field(None, description="Time range for aggregation")
    category_filter: Optional[List[str]] = Field(None, description="Categories to include in aggregation")
    account_filter: Optional[List[str]] = Field(None, description="Accounts to include in aggregation")


class ActionParams(BaseModel):
    """
    Execute financial operations and transactions
    
    Auto-generated from intent_parameters.yaml
    """
    
    # Critical parameters
    action_type: Optional[str] = Field(None, description="Type of action to perform (transfer, payment, etc.)")

    # Optional parameters
    amount: Optional[float] = Field(None, description="Amount for the action", ge=0)
    source_account: Optional[str] = Field(None, description="Source account for the action")
    target_account: Optional[str] = Field(None, description="Target account for the action")
    description: Optional[str] = Field(None, description="Description of the action")


class UnknownParams(BaseModel):
    """
    Intent not clearly determined
    
    Auto-generated from intent_parameters.yaml
    """
    
    pass  # No parameters defined


class ClarificationParams(BaseModel):
    """
    Request for additional information or clarification
    
    Auto-generated from intent_parameters.yaml
    """
    
    pass  # No parameters defined


# Exception and Utility Classes

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
    
    def to_classification_result(self) -> "IntentClassificationResult":
        """Convert to standard IntentClassificationResult."""
        return IntentClassificationResult(
            intent=self.intent,
            confidence=self.confidence,
            reasoning=self.reasoning,
            user_input_analysis=self.user_input_analysis,
            missing_params=self.missing_params,
            clarification_needed=self.clarification_needed
        )


class IntentClassificationResult(BaseModel):
    """
    Structured result from intent classification LLM call.
    
    This model represents the complete output from LLM intent classification,
    including the classified intent, confidence scoring, and extracted parameters.
    
    Auto-generated from intent_parameters.yaml
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
        description="Parameters extracted for data_fetch intents"
    )
    aggregate_params: Optional[AggregateParams] = Field(
        None,
        description="Parameters extracted for aggregate intents"
    )
    action_params: Optional[ActionParams] = Field(
        None,
        description="Parameters extracted for action intents"
    )
    unknown_params: Optional[UnknownParams] = Field(
        None,
        description="Parameters extracted for unknown intents"
    )
    clarification_params: Optional[ClarificationParams] = Field(
        None,
        description="Parameters extracted for clarification intents"
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
        if self.unknown_params:
            params.update(self.unknown_params.model_dump(exclude_none=True))
        if self.clarification_params:
            params.update(self.clarification_params.model_dump(exclude_none=True))
        
        return params




# Auto-generated mappings
INTENT_PARAM_MODELS = {
    IntentCategory.DATA_FETCH: DataFetchParams,
    IntentCategory.AGGREGATE: AggregateParams,
    IntentCategory.ACTION: ActionParams,
    IntentCategory.UNKNOWN: UnknownParams,
    IntentCategory.CLARIFICATION: ClarificationParams,
}

# Intent type validation
SUPPORTED_INTENTS = ['data_fetch', 'aggregate', 'action', 'unknown', 'clarification']
