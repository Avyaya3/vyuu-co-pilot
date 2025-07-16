"""
Completeness Validator Node for LangGraph Clarification Subgraph.

This module provides rule-based validation to determine whether enough parameter data
has been collected to proceed to the Direct Orchestrator, whether clarification should
continue, or whether max attempts have been reached requiring exit with partial data.
"""

import logging
from typing import Dict, Any, Optional, Set, Literal, Tuple, Union
from decimal import Decimal, InvalidOperation

from src.schemas.state_schemas import ClarificationState, IntentType
from src.utils.parameter_config import get_parameter_config
from src.schemas.intent_schemas import DataFetchParams, AggregateParams, ActionParams

logger = logging.getLogger(__name__)

# Status type for completeness validation
CompletenessStatus = Literal["complete", "incomplete", "max_attempts_reached"]
ValidationResult = Tuple[CompletenessStatus, ClarificationState]


class CompletenessValidator:
    """
    Rule-based completeness validator for parameter collection.
    
    Determines whether clarification should continue, complete, or exit
    based on missing critical parameters and attempt limits.
    """
    
    def __init__(self):
        """Initialize the completeness validator."""
        self.parameter_config = get_parameter_config()
    
    def validate_completeness(self, state: ClarificationState) -> ValidationResult:
        """
        Validate parameter completeness and determine next action.
        
        Args:
            state: Current ClarificationState
            
        Returns:
            Tuple of (status, updated_state) where:
            - "complete": All critical parameters present and valid
            - "incomplete": Missing critical parameters, can continue clarification
            - "max_attempts_reached": Hit max attempts limit, must exit
        """
        session_prefix = state.session_id[:8]
        logger.info(f"[CompletenessValidator] Starting validation for session {session_prefix}")
        
        try:
            # Check max attempts first using the state's helper property
            if not state.can_attempt_clarification:
                if state.missing_critical_params:
                    logger.warning(
                        f"[CompletenessValidator] Max attempts reached ({state.clarification_attempts}/{state.max_attempts}) "
                        f"with {len(state.missing_critical_params)} critical params still missing: {state.missing_critical_params}"
                    )
                    return self._handle_max_attempts_reached(state)
                else:
                    # Max attempts reached but no critical params missing - treat as complete
                    logger.info(f"[CompletenessValidator] Max attempts reached but no critical params missing - treating as complete")
                    return self._handle_completion(state)
            
            # Check if critical parameters are missing
            if state.missing_critical_params:
                logger.info(
                    f"[CompletenessValidator] Critical parameters still missing: {state.missing_critical_params}. "
                    f"Attempts: {state.clarification_attempts}/{state.max_attempts}"
                )
                return self._handle_incomplete(state)
            
            # No critical parameters missing - perform quality validation
            validation_passed = self._perform_quality_validation(state)
            if not validation_passed:
                logger.warning(f"[CompletenessValidator] Quality validation failed - treating as incomplete")
                return self._handle_incomplete(state)
            
            # All validations passed
            logger.info(f"[CompletenessValidator] All critical parameters present and valid - completion achieved")
            return self._handle_completion(state)
            
        except Exception as e:
            logger.error(f"[CompletenessValidator] Unexpected error during validation: {e}")
            # On error, default to incomplete if we can still attempt clarification
            if state.can_attempt_clarification:
                return self._handle_incomplete(state, error_message=str(e))
            else:
                return self._handle_max_attempts_reached(state, error_message=str(e))
    
    def _handle_completion(self, state: ClarificationState) -> ValidationResult:
        """
        Handle successful completion of parameter collection.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of ("complete", updated_state)
        """
        updated_state = state.model_copy(update={
            "metadata": {
                **state.metadata,
                "clarification_status": "complete",
                "completion_reason": "all_critical_parameters_collected",
                "validation_timestamp": state.timestamp.isoformat(),
                "total_attempts_used": state.clarification_attempts
            }
        })
        
        logger.info(f"[CompletenessValidator] Parameter collection complete after {state.clarification_attempts} attempts")
        return "complete", updated_state
    
    def _handle_incomplete(self, state: ClarificationState, error_message: Optional[str] = None) -> ValidationResult:
        """
        Handle incomplete parameter collection requiring continued clarification.
        
        Args:
            state: Current state
            error_message: Optional error message if validation failed
            
        Returns:
            Tuple of ("incomplete", updated_state)
        """
        metadata_update = {
            **state.metadata,
            "clarification_status": "incomplete",
            "remaining_critical_params": state.missing_critical_params,
            "attempts_remaining": state.max_attempts - state.clarification_attempts
        }
        
        if error_message:
            metadata_update["validation_error"] = error_message
        
        updated_state = state.model_copy(update={
            "metadata": metadata_update
        })
        
        logger.info(
            f"[CompletenessValidator] Parameter collection incomplete. "
            f"Missing critical: {state.missing_critical_params}. "
            f"Attempts remaining: {state.max_attempts - state.clarification_attempts}"
        )
        return "incomplete", updated_state
    
    def _handle_max_attempts_reached(self, state: ClarificationState, error_message: Optional[str] = None) -> ValidationResult:
        """
        Handle max attempts reached requiring exit with partial data.
        
        Args:
            state: Current state
            error_message: Optional error message
            
        Returns:
            Tuple of ("max_attempts_reached", updated_state)
        """
        exit_message = (
            f"I've asked {state.max_attempts} clarifying questions but still need some information. "
            f"I'll proceed with what we have and may make reasonable assumptions where needed."
        )
        
        metadata_update = {
            **state.metadata,
            "clarification_status": "max_attempts_reached",
            "exit_message": exit_message,
            "missing_critical_params_at_exit": state.missing_critical_params.copy(),
            "total_attempts_used": state.clarification_attempts,
            "partial_data_available": bool(state.extracted_parameters)
        }
        
        if error_message:
            metadata_update["exit_error"] = error_message
        
        updated_state = state.model_copy(update={
            "metadata": metadata_update
        })
        
        logger.warning(
            f"[CompletenessValidator] Max attempts reached. "
            f"Exiting with partial data. Missing critical: {state.missing_critical_params}"
        )
        return "max_attempts_reached", updated_state
    
    def _perform_quality_validation(self, state: ClarificationState) -> bool:
        """
        Perform lightweight quality checks on extracted parameters.
        
        Args:
            state: Current state with extracted parameters
            
        Returns:
            True if all quality checks pass, False otherwise
        """
        if not state.extracted_parameters:
            logger.warning("[CompletenessValidator] No extracted parameters available for quality validation")
            return False
        
        try:
            # Get critical parameters for this intent
            intent_str = state.intent.value if state.intent else "unknown"
            critical_params = self.parameter_config.get_critical_parameters(intent_str)
            
            # Check each critical parameter for quality issues
            for param_name in critical_params:
                param_value = state.extracted_parameters.get(param_name)
                
                if param_value is None:
                    logger.warning(f"[CompletenessValidator] Critical parameter '{param_name}' is None")
                    return False
                
                if not self._validate_parameter_quality(param_name, param_value, state.intent):
                    logger.warning(f"[CompletenessValidator] Quality validation failed for parameter '{param_name}': {param_value}")
                    return False
            
            logger.info("[CompletenessValidator] All quality validations passed")
            return True
            
        except Exception as e:
            logger.error(f"[CompletenessValidator] Error during quality validation: {e}")
            return False
    
    def _validate_parameter_quality(self, param_name: str, param_value: Any, intent: Optional[IntentType]) -> bool:
        """
        Validate individual parameter quality.
        
        Args:
            param_name: Name of the parameter
            param_value: Value to validate
            intent: Current intent type for context
            
        Returns:
            True if parameter passes quality checks
        """
        # Null/empty checks
        if param_value is None:
            return False
        
        # String parameters should not be empty
        if isinstance(param_value, str):
            if not param_value.strip():
                return False
            
            # Check for obvious placeholder values
            placeholder_values = {
                "null", "none", "n/a", "na", "unknown", "todo", "tbd", 
                "placeholder", "temp", "test", "example"
            }
            if param_value.lower().strip() in placeholder_values:
                return False
        
        # Numeric parameters should be positive for amounts
        if param_name in ["amount", "limit"] and isinstance(param_value, (int, float, Decimal)):
            if param_value <= 0:
                return False
        
        # List parameters should not be empty
        if isinstance(param_value, list) and len(param_value) == 0:
            return False
        
        # Intent-specific validations
        if intent == IntentType.ACTION:
            return self._validate_action_param_quality(param_name, param_value)
        elif intent == IntentType.DATA_FETCH:
            return self._validate_data_fetch_param_quality(param_name, param_value)
        elif intent == IntentType.AGGREGATE:
            return self._validate_aggregate_param_quality(param_name, param_value)
        
        return True
    
    def _validate_action_param_quality(self, param_name: str, param_value: Any) -> bool:
        """Validate ACTION intent parameter quality."""
        if param_name == "amount":
            try:
                amount = float(param_value)
                # Reasonable amount range check
                return 0.01 <= amount <= 1_000_000
            except (ValueError, TypeError):
                return False
        
        if param_name == "action_type":
            # Valid action types
            valid_actions = {"transfer", "payment", "categorization", "budget", "goal"}
            return isinstance(param_value, str) and param_value.lower() in valid_actions
        
        return True
    
    def _validate_data_fetch_param_quality(self, param_name: str, param_value: Any) -> bool:
        """Validate DATA_FETCH intent parameter quality."""
        if param_name == "limit":
            try:
                limit = int(param_value)
                # Reasonable limit range
                return 1 <= limit <= 1000
            except (ValueError, TypeError):
                return False
        
        if param_name == "entity_type":
            # Valid entity types
            valid_entities = {"transactions", "accounts", "balances", "categories", "budgets"}
            return isinstance(param_value, str) and param_value.lower() in valid_entities
        
        return True
    
    def _validate_aggregate_param_quality(self, param_name: str, param_value: Any) -> bool:
        """Validate AGGREGATE intent parameter quality."""
        if param_name == "metric_type":
            # Valid aggregation types
            valid_metrics = {"sum", "average", "count", "min", "max", "total"}
            return isinstance(param_value, str) and param_value.lower() in valid_metrics
        
        return True


async def completeness_validator_node(state: ClarificationState) -> ValidationResult:
    """
    Validate parameter completeness and determine clarification flow.
    
    This node is part of the clarification subgraph flow:
    1. Entry State → Missing Parameter Analysis → Clarification Question Generator
    2. Question to User → User Response Processor → **Completeness Validator**
    3. [Complete?] → Yes → Exit to Direct Orchestrator
    4. [Complete?] → No → Loop back to Missing Parameter Analysis (if attempts remaining)
    5. Max Attempts → Exit with Partial Data
    
    Args:
        state: Current ClarificationState
        
    Returns:
        Tuple of (status, updated_state) where status is:
        - "complete": All critical parameters collected, proceed to orchestrator
        - "incomplete": Missing parameters, continue clarification loop
        - "max_attempts_reached": Hit max attempts, exit with partial data
    """
    logger.info(f"[CompletenessValidator] Starting validation for session {state.session_id[:8]}")
    
    validator = CompletenessValidator()
    return validator.validate_completeness(state) 