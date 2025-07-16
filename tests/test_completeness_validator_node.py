"""
Tests for Completeness Validator Node.

This module provides comprehensive test coverage for the completeness validator node,
including rule-based completeness checking, quality validation, max attempts handling,
and state transitions.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from src.nodes.completeness_validator_node import (
    completeness_validator_node,
    CompletenessValidator,
    CompletenessStatus,
    ValidationResult,
)
from src.schemas.state_schemas import ClarificationState, IntentType


@pytest.fixture
def base_clarification_state():
    """Base clarification state for testing."""
    return ClarificationState(
        session_id=str(uuid4()),
        user_input="Transfer $500 to my savings account",
        intent=IntentType.ACTION,
        extracted_parameters={
            "action_type": "transfer",
            "amount": 500.0,
            "source_account": "checking",
            "target_account": "savings"
        },
        missing_params=[],
        missing_critical_params=[],
        parameter_priorities=[],
        normalization_suggestions={},
        ambiguity_flags={},
        clarification_history=[
            {
                "question": "What amount would you like to transfer?",
                "user_response": "Five hundred dollars",
                "targeted_param": "amount",
                "attempt": 1
            }
        ],
        clarification_attempts=1,
        max_attempts=3
    )


class TestCompletenessValidator:
    """Test CompletenessValidator class functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return CompletenessValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator.parameter_config is not None
    
    def test_validate_completeness_complete(self, validator, base_clarification_state):
        """Test validation when all critical parameters are present."""
        # No missing critical parameters
        state = base_clarification_state
        state.missing_critical_params = []
        
        with patch.object(validator, '_perform_quality_validation', return_value=True):
            status, updated_state = validator.validate_completeness(state)
        
        assert status == "complete"
        assert updated_state.metadata["clarification_status"] == "complete"
        assert updated_state.metadata["completion_reason"] == "all_critical_parameters_collected"
        assert updated_state.metadata["total_attempts_used"] == 1
    
    def test_validate_completeness_incomplete(self, validator, base_clarification_state):
        """Test validation when critical parameters are missing."""
        state = base_clarification_state
        state.missing_critical_params = ["source_account"]
        state.clarification_attempts = 1
        
        status, updated_state = validator.validate_completeness(state)
        
        assert status == "incomplete"
        assert updated_state.metadata["clarification_status"] == "incomplete"
        assert updated_state.metadata["remaining_critical_params"] == ["source_account"]
        assert updated_state.metadata["attempts_remaining"] == 2  # 3 - 1
    
    def test_validate_completeness_max_attempts_with_missing(self, validator, base_clarification_state):
        """Test validation when max attempts reached with missing critical params."""
        state = base_clarification_state
        state.missing_critical_params = ["source_account"]
        state.clarification_attempts = 3
        state.max_attempts = 3
        
        status, updated_state = validator.validate_completeness(state)
        
        assert status == "max_attempts_reached"
        assert updated_state.metadata["clarification_status"] == "max_attempts_reached"
        assert "I've asked 3 clarifying questions" in updated_state.metadata["exit_message"]
        assert updated_state.metadata["missing_critical_params_at_exit"] == ["source_account"]
        assert updated_state.metadata["total_attempts_used"] == 3
    
    def test_validate_completeness_max_attempts_without_missing(self, validator, base_clarification_state):
        """Test validation when max attempts reached but no missing critical params."""
        state = base_clarification_state
        state.missing_critical_params = []
        state.clarification_attempts = 3
        state.max_attempts = 3
        
        with patch.object(validator, '_perform_quality_validation', return_value=True):
            status, updated_state = validator.validate_completeness(state)
        
        # Should treat as complete even if max attempts reached
        assert status == "complete"
        assert updated_state.metadata["clarification_status"] == "complete"
    
    def test_validate_completeness_quality_validation_failed(self, validator, base_clarification_state):
        """Test validation when quality validation fails."""
        state = base_clarification_state
        state.missing_critical_params = []
        
        with patch.object(validator, '_perform_quality_validation', return_value=False):
            status, updated_state = validator.validate_completeness(state)
        
        assert status == "incomplete"
        assert updated_state.metadata["clarification_status"] == "incomplete"
    
    def test_validate_completeness_exception_handling(self, validator, base_clarification_state):
        """Test exception handling during validation."""
        state = base_clarification_state
        state.clarification_attempts = 1
        
        with patch.object(validator, '_perform_quality_validation', side_effect=Exception("Test error")):
            status, updated_state = validator.validate_completeness(state)
        
        # Should default to incomplete if can still attempt clarification
        assert status == "incomplete"
        assert "Test error" in updated_state.metadata["validation_error"]
    
    def test_validate_completeness_exception_handling_quality_validation(self, validator, base_clarification_state):
        """Test exception handling during quality validation when not at max attempts."""
        state = base_clarification_state
        state.clarification_attempts = 1  # Not at max attempts
        state.max_attempts = 3
        state.missing_critical_params = []  # No missing params so quality validation runs
        
        with patch.object(validator, '_perform_quality_validation', side_effect=Exception("Test error")):
            status, updated_state = validator.validate_completeness(state)
        
        # Should default to incomplete since can still attempt clarification
        assert status == "incomplete"
        assert "Test error" in updated_state.metadata["validation_error"]
    



class TestQualityValidation:
    """Test quality validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return CompletenessValidator()
    
    def test_perform_quality_validation_success(self, validator, base_clarification_state):
        """Test successful quality validation."""
        state = base_clarification_state
        
        with patch.object(validator, 'parameter_config') as mock_config:
            mock_config.get_critical_parameters.return_value = {"amount", "action_type"}
            
            with patch.object(validator, '_validate_parameter_quality', return_value=True):
                result = validator._perform_quality_validation(state)
        
        assert result is True
    
    def test_perform_quality_validation_no_parameters(self, validator, base_clarification_state):
        """Test quality validation with no extracted parameters."""
        state = base_clarification_state
        state.extracted_parameters = {}
        
        result = validator._perform_quality_validation(state)
        assert result is False
    
    def test_perform_quality_validation_missing_critical_param(self, validator, base_clarification_state):
        """Test quality validation when critical parameter is None."""
        state = base_clarification_state
        state.extracted_parameters["amount"] = None
        
        with patch.object(validator, 'parameter_config') as mock_config:
            mock_config.get_critical_parameters.return_value = {"amount"}
            
            result = validator._perform_quality_validation(state)
        
        assert result is False
    
    def test_perform_quality_validation_parameter_fails(self, validator, base_clarification_state):
        """Test quality validation when parameter quality check fails."""
        state = base_clarification_state
        
        with patch.object(validator, 'parameter_config') as mock_config:
            mock_config.get_critical_parameters.return_value = {"amount"}
            
            with patch.object(validator, '_validate_parameter_quality', return_value=False):
                result = validator._perform_quality_validation(state)
        
        assert result is False
    
    def test_perform_quality_validation_exception(self, validator, base_clarification_state):
        """Test quality validation exception handling."""
        state = base_clarification_state
        
        with patch.object(validator, 'parameter_config') as mock_config:
            mock_config.get_critical_parameters.side_effect = Exception("Config error")
            
            result = validator._perform_quality_validation(state)
        
        assert result is False


class TestParameterQualityValidation:
    """Test individual parameter quality validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return CompletenessValidator()
    
    def test_validate_parameter_quality_none_value(self, validator):
        """Test parameter quality validation with None value."""
        result = validator._validate_parameter_quality("amount", None, IntentType.ACTION)
        assert result is False
    
    def test_validate_parameter_quality_empty_string(self, validator):
        """Test parameter quality validation with empty string."""
        result = validator._validate_parameter_quality("description", "", IntentType.ACTION)
        assert result is False
        
        result = validator._validate_parameter_quality("description", "   ", IntentType.ACTION)
        assert result is False
    
    def test_validate_parameter_quality_placeholder_values(self, validator):
        """Test parameter quality validation with placeholder values."""
        placeholder_values = ["null", "none", "n/a", "unknown", "todo", "placeholder"]
        
        for placeholder in placeholder_values:
            result = validator._validate_parameter_quality("description", placeholder, IntentType.ACTION)
            assert result is False
            
            # Test case insensitive
            result = validator._validate_parameter_quality("description", placeholder.upper(), IntentType.ACTION)
            assert result is False
    
    def test_validate_parameter_quality_negative_amounts(self, validator):
        """Test parameter quality validation with negative amounts."""
        result = validator._validate_parameter_quality("amount", -100.0, IntentType.ACTION)
        assert result is False
        
        result = validator._validate_parameter_quality("limit", 0, IntentType.DATA_FETCH)
        assert result is False
    
    def test_validate_parameter_quality_empty_list(self, validator):
        """Test parameter quality validation with empty list."""
        result = validator._validate_parameter_quality("account_types", [], IntentType.DATA_FETCH)
        assert result is False
    
    def test_validate_parameter_quality_valid_values(self, validator):
        """Test parameter quality validation with valid values."""
        result = validator._validate_parameter_quality("description", "Transfer to savings", IntentType.ACTION)
        assert result is True
        
        result = validator._validate_parameter_quality("amount", 100.0, IntentType.ACTION)
        assert result is True
        
        result = validator._validate_parameter_quality("account_types", ["checking", "savings"], IntentType.DATA_FETCH)
        assert result is True


class TestIntentSpecificValidation:
    """Test intent-specific parameter validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return CompletenessValidator()
    
    def test_validate_action_param_quality_amount(self, validator):
        """Test ACTION intent amount validation."""
        # Valid amounts
        assert validator._validate_action_param_quality("amount", 100.0) is True
        assert validator._validate_action_param_quality("amount", 0.01) is True
        assert validator._validate_action_param_quality("amount", 1000000) is True
        
        # Invalid amounts
        assert validator._validate_action_param_quality("amount", 0) is False
        assert validator._validate_action_param_quality("amount", -100) is False
        assert validator._validate_action_param_quality("amount", 1000001) is False
        assert validator._validate_action_param_quality("amount", "invalid") is False
    
    def test_validate_action_param_quality_action_type(self, validator):
        """Test ACTION intent action_type validation."""
        # Valid action types
        valid_actions = ["transfer", "payment", "categorization", "budget", "goal"]
        for action in valid_actions:
            assert validator._validate_action_param_quality("action_type", action) is True
            assert validator._validate_action_param_quality("action_type", action.upper()) is True
        
        # Invalid action types
        assert validator._validate_action_param_quality("action_type", "invalid_action") is False
        assert validator._validate_action_param_quality("action_type", 123) is False
    
    def test_validate_data_fetch_param_quality_limit(self, validator):
        """Test DATA_FETCH intent limit validation."""
        # Valid limits
        assert validator._validate_data_fetch_param_quality("limit", 1) is True
        assert validator._validate_data_fetch_param_quality("limit", 100) is True
        assert validator._validate_data_fetch_param_quality("limit", 1000) is True
        
        # Invalid limits
        assert validator._validate_data_fetch_param_quality("limit", 0) is False
        assert validator._validate_data_fetch_param_quality("limit", 1001) is False
        assert validator._validate_data_fetch_param_quality("limit", "invalid") is False
    
    def test_validate_data_fetch_param_quality_entity_type(self, validator):
        """Test DATA_FETCH intent entity_type validation."""
        # Valid entity types
        valid_entities = ["transactions", "accounts", "balances", "categories", "budgets"]
        for entity in valid_entities:
            assert validator._validate_data_fetch_param_quality("entity_type", entity) is True
            assert validator._validate_data_fetch_param_quality("entity_type", entity.upper()) is True
        
        # Invalid entity types
        assert validator._validate_data_fetch_param_quality("entity_type", "invalid_entity") is False
    
    def test_validate_aggregate_param_quality_metric_type(self, validator):
        """Test AGGREGATE intent metric_type validation."""
        # Valid metric types
        valid_metrics = ["sum", "average", "count", "min", "max", "total"]
        for metric in valid_metrics:
            assert validator._validate_aggregate_param_quality("metric_type", metric) is True
            assert validator._validate_aggregate_param_quality("metric_type", metric.upper()) is True
        
        # Invalid metric types
        assert validator._validate_aggregate_param_quality("metric_type", "invalid_metric") is False


class TestCompletenessValidatorNode:
    """Test the main completeness validator node function."""
    
    @pytest.mark.asyncio
    async def test_completeness_validator_node_complete(self, base_clarification_state):
        """Test node function with complete parameters."""
        state = base_clarification_state
        state.missing_critical_params = []
        
        with patch('src.nodes.completeness_validator_node.CompletenessValidator') as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_completeness.return_value = ("complete", state)
            
            status, updated_state = await completeness_validator_node(state)
        
        assert status == "complete"
        assert updated_state == state
        mock_validator.validate_completeness.assert_called_once_with(state)
    
    @pytest.mark.asyncio
    async def test_completeness_validator_node_incomplete(self, base_clarification_state):
        """Test node function with incomplete parameters."""
        state = base_clarification_state
        state.missing_critical_params = ["source_account"]
        
        with patch('src.nodes.completeness_validator_node.CompletenessValidator') as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_completeness.return_value = ("incomplete", state)
            
            status, updated_state = await completeness_validator_node(state)
        
        assert status == "incomplete"
        assert updated_state == state
    
    @pytest.mark.asyncio
    async def test_completeness_validator_node_max_attempts(self, base_clarification_state):
        """Test node function with max attempts reached."""
        state = base_clarification_state
        state.clarification_attempts = 3
        state.max_attempts = 3
        state.missing_critical_params = ["source_account"]
        
        with patch('src.nodes.completeness_validator_node.CompletenessValidator') as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_completeness.return_value = ("max_attempts_reached", state)
            
            status, updated_state = await completeness_validator_node(state)
        
        assert status == "max_attempts_reached"
        assert updated_state == state


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return CompletenessValidator()
    
    def test_empty_extracted_parameters(self, validator, base_clarification_state):
        """Test validation with empty extracted parameters."""
        state = base_clarification_state
        state.extracted_parameters = {}
        state.missing_critical_params = ["amount"]
        
        status, updated_state = validator.validate_completeness(state)
        
        assert status == "incomplete"
        assert updated_state.metadata["clarification_status"] == "incomplete"
    
    def test_unknown_intent(self, validator, base_clarification_state):
        """Test validation with unknown intent."""
        state = base_clarification_state
        state.intent = IntentType.UNKNOWN
        state.missing_critical_params = []
        
        # Mock the parameter config to return empty set for unknown intent
        with patch.object(validator, 'parameter_config') as mock_config:
            mock_config.get_critical_parameters.return_value = set()
            
            status, updated_state = validator.validate_completeness(state)
        
        assert status == "complete"  # No critical params for unknown intent
    
    def test_mixed_valid_invalid_parameters(self, validator, base_clarification_state):
        """Test validation with mix of valid and invalid parameters."""
        state = base_clarification_state
        state.extracted_parameters = {
            "amount": 100.0,  # Valid
            "description": "none",  # Invalid (placeholder)
            "action_type": "transfer"  # Valid
        }
        state.missing_critical_params = []
        
        with patch.object(validator, 'parameter_config') as mock_config:
            mock_config.get_critical_parameters.return_value = {"amount", "description", "action_type"}
            
            status, updated_state = validator.validate_completeness(state)
        
        # Should fail due to invalid description
        assert status == "incomplete"
    
    def test_zero_max_attempts_edge_case(self, validator, base_clarification_state):
        """Test edge case with zero max attempts."""
        state = base_clarification_state
        state.max_attempts = 0
        state.clarification_attempts = 0
        state.missing_critical_params = ["amount"]
        
        status, updated_state = validator.validate_completeness(state)
        
        # Should immediately go to max_attempts_reached
        assert status == "max_attempts_reached"
        assert updated_state.metadata["clarification_status"] == "max_attempts_reached" 