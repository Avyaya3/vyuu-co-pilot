"""
Auto-generated Intent Schemas from YAML Configuration.

Generated on: 2025-09-20T18:07:52.190618
Source: /Users/manjiripathak/Avyaya/vyuu-copilot-v2/config/intent_parameters.yaml

DO NOT EDIT MANUALLY - Run scripts/generate_intent_schemas.py to regenerate.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class IntentCategory(str, Enum):
    """Supported intent categories for user requests."""
    READ = "read"
    DATABASE_OPERATIONS = "database_operations"
    ADVICE = "advice"
    UNKNOWN = "unknown"
    CLARIFICATION = "clarification"


class ConfidenceLevel(str, Enum):
    """Confidence level categories for intent classification."""
    HIGH = "high"      # 0.8 - 1.0
    MEDIUM = "medium"  # 0.5 - 0.79
    LOW = "low"        # 0.0 - 0.49


class ReadParams(BaseModel):
    """
    Retrieve and display financial data from various entity types
    
    Auto-generated from intent_parameters.yaml
    """
    
    # Critical parameters
    entity_type: Optional[str] = Field(None, description="Type of financial entity (assets, liabilities, savings, income, expenses, stocks, insurance, goals, financial_summary)")

    # Optional parameters
    time_period: Optional[str] = Field(None, description="Time range for data (last_month, last_3_months, this_year, specific_date, etc.)")
    category: Optional[str] = Field(None, description="Category filter (real_estate, vehicles, loans, credit_cards, etc.)")
    subcategory: Optional[str] = Field(None, description="Subcategory filter (groceries, entertainment, transportation, etc.)")
    type: Optional[str] = Field(None, description="Type filter (checking, savings, mortgage, health_insurance, etc.)")
    provider: Optional[str] = Field(None, description="Provider filter (bank names, insurance companies, etc.)")
    name: Optional[str] = Field(None, description="Name filter for specific entities")
    source: Optional[str] = Field(None, description="Income source filter (salary, freelance, investment, etc.)")
    frequency: Optional[str] = Field(None, description="Frequency filter (monthly, quarterly, yearly)")
    payment_method: Optional[str] = Field(None, description="Payment method filter (credit_card, cash, bank_transfer, etc.)")
    priority: Optional[str] = Field(None, description="Priority filter (low, medium, high)")
    min_amount: Optional[int] = Field(None, description="Minimum amount filter (in cents)", ge=0)
    max_amount: Optional[int] = Field(None, description="Maximum amount filter (in cents)", ge=0)
    min_premium: Optional[int] = Field(None, description="Minimum premium filter for insurance (in cents)", ge=0)
    max_premium: Optional[int] = Field(None, description="Maximum premium filter for insurance (in cents)", ge=0)
    min_coverage: Optional[int] = Field(None, description="Minimum coverage filter for insurance (in cents)", ge=0)
    max_coverage: Optional[int] = Field(None, description="Maximum coverage filter for insurance (in cents)", ge=0)
    min_current_value: Optional[int] = Field(None, description="Minimum current value filter for assets/stocks (in cents)", ge=0)
    max_current_value: Optional[int] = Field(None, description="Maximum current value filter for assets/stocks (in cents)", ge=0)
    min_target: Optional[int] = Field(None, description="Minimum target filter for goals (in cents)", ge=0)
    max_target: Optional[int] = Field(None, description="Maximum target filter for goals (in cents)", ge=0)
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    limit: Optional[int] = Field(None, description="Maximum number of records to return", ge=1, le=1000)
    offset: Optional[int] = Field(None, description="Number of records to skip for pagination", ge=0)
    order_by: Optional[str] = Field(None, description="Field to sort results by (name, amount, date, etc.)")
    order_direction: Optional[str] = Field(None, description="Sort order (asc, desc)")
    entity_id: Optional[str] = Field(None, description="Specific entity ID for single record queries")


class DatabaseOperationsParams(BaseModel):
    """
    Execute database operations (create, update, delete) on financial entities
    
    Auto-generated from intent_parameters.yaml
    """
    
    # Critical parameters
    action_type: Optional[str] = Field(None, description="Type of action to perform (create, update, delete, transfer, etc.)")

    # Optional parameters
    amount: Optional[int] = Field(None, description="Amount for the action (in cents)", ge=0)
    entity_type: Optional[str] = Field(None, description="Type of entity for the action (asset, liability, savings, income, expense, stock, insurance, goal)")
    entity_id: Optional[str] = Field(None, description="Specific entity ID for update/delete operations")
    name: Optional[str] = Field(None, description="Name for the entity")
    category: Optional[str] = Field(None, description="Category for the entity")
    subcategory: Optional[str] = Field(None, description="Subcategory for the entity")
    type: Optional[str] = Field(None, description="Type for the entity")
    provider: Optional[str] = Field(None, description="Provider for the entity")
    source: Optional[str] = Field(None, description="Source for income")
    frequency: Optional[str] = Field(None, description="Frequency for income")
    payment_method: Optional[str] = Field(None, description="Payment method for expenses")
    priority: Optional[str] = Field(None, description="Priority for goals")
    description: Optional[str] = Field(None, description="Description of the entity or action")
    date: Optional[datetime] = Field(None, description="Date for the entity")
    start_date: Optional[datetime] = Field(None, description="Start date for the entity")
    end_date: Optional[datetime] = Field(None, description="End date for the entity")
    target_date: Optional[datetime] = Field(None, description="Target date for goals")
    current_value: Optional[int] = Field(None, description="Current value for assets/stocks (in cents)")
    purchase_value: Optional[int] = Field(None, description="Purchase value for assets (in cents)")
    purchase_date: Optional[datetime] = Field(None, description="Purchase date for assets/stocks")
    returns: Optional[float] = Field(None, description="Returns percentage for stocks")
    premium: Optional[int] = Field(None, description="Premium for insurance (in cents)")
    coverage: Optional[int] = Field(None, description="Coverage for insurance (in cents)")
    policy_number: Optional[str] = Field(None, description="Policy number for insurance")
    current_balance: Optional[int] = Field(None, description="Current balance for savings (in cents)")
    interest_rate: Optional[float] = Field(None, description="Interest rate for savings/liabilities")
    maturity_date: Optional[datetime] = Field(None, description="Maturity date for savings")
    monthly_contribution: Optional[int] = Field(None, description="Monthly contribution for savings (in cents)")
    target_amount: Optional[int] = Field(None, description="Target amount for savings/goals (in cents)")
    emi: Optional[int] = Field(None, description="EMI for liabilities (in cents)")
    target: Optional[int] = Field(None, description="Target amount for goals (in cents)")
    current: Optional[int] = Field(None, description="Current amount for goals (in cents)")


class AdviceParams(BaseModel):
    """
    Provide personalized financial advice based on user query and financial data
    
    Auto-generated from intent_parameters.yaml
    """
    
    # Optional parameters
    user_query: Optional[str] = Field(None, description="The user's question or request for advice")
    context_data: Optional[str] = Field(None, description="Financial data context provided from frontend")


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
    
    @property
    def extracted_parameters(self) -> Dict[str, Any]:
        """Get extracted parameters (empty for fallback)."""
        return {}
    
    @property
    def requires_clarification(self) -> bool:
        """Check if clarification is needed (always true for fallback)."""
        return self.clarification_needed
    
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
    read_params: Optional[ReadParams] = Field(
        None,
        description="Parameters extracted for read intents"
    )
    database_operations_params: Optional[DatabaseOperationsParams] = Field(
        None,
        description="Parameters extracted for database_operations intents"
    )
    advice_params: Optional[AdviceParams] = Field(
        None,
        description="Parameters extracted for advice intents"
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
        
        if self.read_params:
            params.update(self.read_params.model_dump(exclude_none=True))
        if self.database_operations_params:
            params.update(self.database_operations_params.model_dump(exclude_none=True))
        if self.advice_params:
            params.update(self.advice_params.model_dump(exclude_none=True))
        if self.unknown_params:
            params.update(self.unknown_params.model_dump(exclude_none=True))
        if self.clarification_params:
            params.update(self.clarification_params.model_dump(exclude_none=True))
        
        return params




# Auto-generated mappings
INTENT_PARAM_MODELS = {
    IntentCategory.READ: ReadParams,
    IntentCategory.DATABASE_OPERATIONS: DatabaseOperationsParams,
    IntentCategory.ADVICE: AdviceParams,
    IntentCategory.UNKNOWN: UnknownParams,
    IntentCategory.CLARIFICATION: ClarificationParams,
}

# Intent type validation
SUPPORTED_INTENTS = ['read', 'database_operations', 'advice', 'unknown', 'clarification']
