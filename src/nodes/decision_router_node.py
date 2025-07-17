"""
Decision Router Node for LangGraph Intent Orchestration.

This module implements routing logic to determine the next processing step based on
intent classification results. It routes requests to either clarification or direct
orchestrator subgraphs based on confidence thresholds and parameter completeness.
"""

import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

from src.schemas.state_schemas import MainState, MessageManager, IntentType
from src.schemas.generated_intent_schemas import IntentCategory, ConfidenceLevel
from src.utils.parameter_config import get_parameter_config

logger = logging.getLogger(__name__)

# Routing Decision Types
RoutingDecision = Literal["clarification", "direct_orchestrator"]


class RouterConfig(BaseModel):
    """Configuration for decision router thresholds and settings."""
    
    # Confidence thresholds
    high_confidence_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for direct orchestrator routing"
    )
    medium_confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for attempting parameter completion"
    )
    low_confidence_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for attempting clarification"
    )
    
    # Parameter completeness settings
    require_critical_params: bool = Field(
        True,
        description="Whether to require critical parameters for direct routing"
    )
    max_missing_params: int = Field(
        2,
        ge=0,
        description="Maximum number of missing parameters allowed for direct routing"
    )
    
    # Intent-specific routing settings
    intent_specific_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "data_fetch": 0.7,  # Data fetch can proceed with moderate confidence
            "aggregate": 0.8,   # Aggregation needs high confidence
            "action": 0.9,      # Actions need very high confidence
            "unknown": 0.0      # Unknown intents always need clarification
        },
        description="Intent-specific confidence thresholds"
    )
    
    # Routing behavior flags
    enable_fallback_routing: bool = Field(
        True,
        description="Enable fallback routing for edge cases"
    )
    log_routing_decisions: bool = Field(
        True,
        description="Enable detailed logging of routing decisions"
    )
    
    @model_validator(mode='after')
    def validate_thresholds(self):
        """Validate that thresholds are properly ordered."""
        if self.high_confidence_threshold <= self.medium_confidence_threshold:
            raise ValueError("High confidence threshold must be greater than medium threshold")
        if self.medium_confidence_threshold <= self.low_confidence_threshold:
            raise ValueError("Medium confidence threshold must be greater than low threshold")
        return self


class RoutingReason(str, Enum):
    """Reasons for routing decisions."""
    HIGH_CONFIDENCE_COMPLETE = "high_confidence_complete"
    HIGH_CONFIDENCE_INCOMPLETE = "high_confidence_incomplete"
    MEDIUM_CONFIDENCE_COMPLETE = "medium_confidence_complete"
    MEDIUM_CONFIDENCE_INCOMPLETE = "medium_confidence_incomplete"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN_INTENT = "unknown_intent"
    MISSING_CRITICAL_PARAMS = "missing_critical_params"
    EXCESSIVE_MISSING_PARAMS = "excessive_missing_params"
    INTENT_SPECIFIC_THRESHOLD = "intent_specific_threshold"
    FALLBACK_ROUTING = "fallback_routing"
    ERROR_CONDITION = "error_condition"


class RoutingResult(BaseModel):
    """Result of routing decision with detailed reasoning."""
    
    decision: RoutingDecision = Field(
        ...,
        description="The routing decision made"
    )
    reason: RoutingReason = Field(
        ...,
        description="Primary reason for the routing decision"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score that influenced the decision"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Categorical confidence level"
    )
    parameters_complete: bool = Field(
        ...,
        description="Whether parameters are considered complete"
    )
    missing_params: List[str] = Field(
        default_factory=list,
        description="List of missing parameters"
    )
    critical_params_missing: List[str] = Field(
        default_factory=list,
        description="List of missing critical parameters"
    )
    intent_type: IntentType = Field(
        ...,
        description="The intent type being routed"
    )
    routing_explanation: str = Field(
        ...,
        description="Human-readable explanation of the routing decision"
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to make routing decision in milliseconds"
    )
    config_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration settings that influenced the decision"
    )


class DecisionRouter:
    """
    Decision router for intent classification results.
    
    Routes classified intents to appropriate subgraphs based on confidence
    thresholds and parameter completeness analysis.
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self.param_config = get_parameter_config()
        logger.info(f"Decision router initialized with config: {self.config.model_dump()}")
        logger.info(f"Parameter configuration loaded with intents: {self.param_config.get_available_intents()}")
    
    def route_intent(self, state: MainState) -> RoutingResult:
        """
        Route intent based on classification results and parameter completeness.
        
        Args:
            state: MainState with intent classification results
            
        Returns:
            RoutingResult with decision and detailed reasoning
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Extract routing inputs
            intent = state.intent
            confidence = state.confidence or 0.0
            parameters = state.parameters or {}
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(confidence)
            
            # Analyze parameter completeness
            param_analysis = self._analyze_parameter_completeness(state)
            
            # Apply routing logic
            routing_decision = self._apply_routing_logic(
                intent=intent,
                confidence=confidence,
                confidence_level=confidence_level,
                param_analysis=param_analysis
            )
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Create routing result
            result = RoutingResult(
                decision=routing_decision["decision"],
                reason=routing_decision["reason"],
                confidence_score=confidence,
                confidence_level=confidence_level,
                parameters_complete=param_analysis["complete"],
                missing_params=param_analysis["missing"],
                critical_params_missing=param_analysis["critical_missing"],
                intent_type=intent,
                routing_explanation=routing_decision["explanation"],
                processing_time_ms=processing_time,
                config_applied={
                    "threshold_used": routing_decision["threshold_used"],
                    "require_critical_params": self.config.require_critical_params,
                    "max_missing_params": self.config.max_missing_params
                }
            )
            
            # Log routing decision
            if self.config.log_routing_decisions:
                self._log_routing_decision(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in routing decision: {e}")
            
            # Return fallback routing decision
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Handle None state gracefully
            fallback_intent = IntentType.UNKNOWN
            if state is not None and hasattr(state, 'intent') and state.intent is not None:
                fallback_intent = state.intent
            
            return RoutingResult(
                decision="clarification",
                reason=RoutingReason.ERROR_CONDITION,
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.LOW,
                parameters_complete=False,
                missing_params=["error_recovery"],
                critical_params_missing=["error_recovery"],
                intent_type=fallback_intent,
                routing_explanation=f"Routing failed due to error: {str(e)}. Defaulting to clarification.",
                processing_time_ms=processing_time,
                config_applied={}
            )
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level based on configured thresholds."""
        if confidence >= self.config.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.config.medium_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _analyze_parameter_completeness(self, state: MainState) -> Dict[str, Any]:
        """
        Analyze parameter completeness for the given intent.
        
        Args:
            state: MainState with intent and parameters
            
        Returns:
            Dictionary with completeness analysis
        """
        intent = state.intent
        parameters = state.parameters or {}
        
        # Get parameters from configuration
        intent_key = intent.value if hasattr(intent, 'value') else str(intent).lower()
        expected_critical = list(self.param_config.get_critical_parameters(intent_key))
        expected_all = list(self.param_config.get_all_parameters(intent_key))
        
        # Find missing parameters
        provided_params = set(parameters.keys())
        missing_params = [p for p in expected_all if p not in provided_params]
        critical_missing = [p for p in expected_critical if p not in provided_params]
        
        # Determine completeness - be more permissive
        has_critical = len(critical_missing) == 0
        within_missing_limit = len(missing_params) <= self.config.max_missing_params
        
        # Parameters are considered complete if:
        # 1. We have critical parameters AND
        # 2. Either we're within missing limit OR we don't require all critical params
        complete = has_critical and (within_missing_limit or not self.config.require_critical_params)
        
        # For cases where we have some parameters and critical ones, be more lenient
        if has_critical and len(provided_params) > 0:
            complete = True  # If we have critical params and any other params, consider complete enough
        
        return {
            "complete": complete,
            "missing": missing_params,
            "critical_missing": critical_missing,
            "has_critical": has_critical,
            "within_missing_limit": within_missing_limit,
            "provided_count": len(provided_params),
            "expected_count": len(expected_all)
        }
    
    def _apply_routing_logic(
        self, 
        intent: IntentType, 
        confidence: float, 
        confidence_level: ConfidenceLevel,
        param_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply routing logic based on confidence and parameter analysis.
        
        Args:
            intent: The classified intent
            confidence: Confidence score
            confidence_level: Categorical confidence level
            param_analysis: Parameter completeness analysis
            
        Returns:
            Dictionary with routing decision and reasoning
        """
        # Handle unknown intents
        if intent == IntentType.UNKNOWN:
            return {
                "decision": "clarification",
                "reason": RoutingReason.UNKNOWN_INTENT,
                "explanation": "Unknown intent requires clarification to determine user needs.",
                "threshold_used": "unknown_intent_rule"
            }
        
        # Check intent-specific thresholds
        intent_key = intent.value if hasattr(intent, 'value') else str(intent)
        intent_threshold = self.config.intent_specific_thresholds.get(intent_key)
        
        if intent_threshold is not None and confidence < intent_threshold:
            return {
                "decision": "clarification",
                "reason": RoutingReason.INTENT_SPECIFIC_THRESHOLD,
                "explanation": f"Confidence {confidence:.2f} below intent-specific threshold {intent_threshold:.2f} for {intent_key}.",
                "threshold_used": intent_threshold
            }
        
        # Check for critical missing parameters
        if param_analysis["critical_missing"] and self.config.require_critical_params:
            return {
                "decision": "clarification",
                "reason": RoutingReason.MISSING_CRITICAL_PARAMS,
                "explanation": f"Critical parameters missing: {param_analysis['critical_missing']}",
                "threshold_used": "critical_params_rule"
            }
        
        # Apply confidence-based routing
        if confidence_level == ConfidenceLevel.HIGH:
            if param_analysis["complete"]:
                return {
                    "decision": "direct_orchestrator",
                    "reason": RoutingReason.HIGH_CONFIDENCE_COMPLETE,
                    "explanation": f"High confidence ({confidence:.2f}) with complete parameters. Routing to direct orchestrator.",
                    "threshold_used": self.config.high_confidence_threshold
                }
            else:
                return {
                    "decision": "clarification",
                    "reason": RoutingReason.HIGH_CONFIDENCE_INCOMPLETE,
                    "explanation": f"High confidence ({confidence:.2f}) but missing parameters: {param_analysis['missing']}",
                    "threshold_used": self.config.high_confidence_threshold
                }
        
        elif confidence_level == ConfidenceLevel.MEDIUM:
            if param_analysis["complete"]:
                return {
                    "decision": "direct_orchestrator",
                    "reason": RoutingReason.MEDIUM_CONFIDENCE_COMPLETE,
                    "explanation": f"Medium confidence ({confidence:.2f}) with complete parameters. Routing to direct orchestrator.",
                    "threshold_used": self.config.medium_confidence_threshold
                }
            else:
                return {
                    "decision": "clarification",
                    "reason": RoutingReason.MEDIUM_CONFIDENCE_INCOMPLETE,
                    "explanation": f"Medium confidence ({confidence:.2f}) with missing parameters: {param_analysis['missing']}",
                    "threshold_used": self.config.medium_confidence_threshold
                }
        
        else:  # LOW confidence
            return {
                "decision": "clarification",
                "reason": RoutingReason.LOW_CONFIDENCE,
                "explanation": f"Low confidence ({confidence:.2f}) requires clarification.",
                "threshold_used": self.config.low_confidence_threshold
            }
    
    def _log_routing_decision(self, result: RoutingResult):
        """Log detailed routing decision information."""
        logger.info(
            f"Routing Decision: {result.decision} | "
            f"Reason: {result.reason.value} | "
            f"Intent: {result.intent_type.value} | "
            f"Confidence: {result.confidence_score:.2f} ({result.confidence_level.value}) | "
            f"Parameters Complete: {result.parameters_complete} | "
            f"Missing: {len(result.missing_params)} params | "
            f"Critical Missing: {len(result.critical_params_missing)} params | "
            f"Processing Time: {result.processing_time_ms:.2f}ms"
        )
        
        logger.debug(
            f"Routing Details: {result.routing_explanation} | "
            f"Missing Params: {result.missing_params} | "
            f"Critical Missing: {result.critical_params_missing} | "
            f"Config Applied: {result.config_applied}"
        )


# LangGraph Node Implementation
async def decision_router_node(state: MainState) -> MainState:
    """
    LangGraph node for routing decisions based on intent classification.
    
    This node analyzes the intent classification results and determines
    whether to route to clarification or direct orchestrator subgraphs.
    
    Args:
        state: MainState with intent classification results
        
    Returns:
        MainState with routing decision and metadata
    """
    node_name = "decision_router_node"
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Decision router processing session: {state.session_id[:8]}...")
        
        # Add system message for tracking
        state = MessageManager.add_system_message(
            state,
            f"Starting routing decision for intent: {state.intent.value if state.intent else 'unknown'}",
            node_name
        )
        
        # Initialize router with default config
        router = DecisionRouter()
        
        # Make routing decision
        routing_result = router.route_intent(state)
        
        # Update state with routing decision using model_copy
        updated_state = state.model_copy(update={
            "metadata": {
                **state.metadata,
                "routing_decision": routing_result.decision,
                "routing_reason": routing_result.reason.value,
                "routing_explanation": routing_result.routing_explanation,
                "routing_result": routing_result.model_dump(),
                "node_processing_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "router_config": router.config.model_dump()
            }
        })
        
        # Add assistant message with routing decision
        routing_message = (
            f"Based on your request classification (intent: {state.intent.value if state.intent else 'unknown'}, "
            f"confidence: {state.confidence:.0%}), I've determined the next step: {routing_result.decision}. "
            f"{routing_result.routing_explanation}"
        )
        
        updated_state = MessageManager.add_assistant_message(
            updated_state,
            routing_message,
            node_name
        )
        
        # Log success
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Decision router completed successfully: "
            f"decision={routing_result.decision}, reason={routing_result.reason.value}, "
            f"processing_time={processing_time:.3f}s"
        )
        
        return updated_state
        
    except Exception as e:
        # Log error and add to state metadata
        error_message = f"Decision router failed: {str(e)}"
        logger.error(error_message)
        
        # Add error tracking to state using model_copy
        error_state = state.model_copy(update={
            "metadata": {
                **state.metadata,
                "routing_decision": "clarification",  # Fallback to clarification
                "routing_reason": "error_condition",
                "routing_explanation": f"Router failed with error: {str(e)}. Defaulting to clarification.",
                "error": {
                    "node": node_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        })
        
        # Add error message
        error_state = MessageManager.add_system_message(
            error_state,
            f"Error in decision router: {str(e)}. Defaulting to clarification.",
            node_name
        )
        
        return error_state


def get_routing_decision(state: MainState) -> str:
    """
    Extract routing decision from state metadata for LangGraph conditional routing.
    
    This function is used by LangGraph to determine which edge to take based on
    the routing decision made by the decision router node.
    
    Args:
        state: MainState with routing decision in metadata
        
    Returns:
        Routing decision string: "clarification" or "direct_orchestrator"
    """
    try:
        routing_decision = state.metadata.get("routing_decision")
        if routing_decision in ["clarification", "direct_orchestrator"]:
            return routing_decision
        else:
            logger.warning(f"Invalid routing decision: {routing_decision}. Defaulting to clarification.")
            return "clarification"
    except Exception as e:
        logger.error(f"Error extracting routing decision: {e}. Defaulting to clarification.")
        return "clarification" 