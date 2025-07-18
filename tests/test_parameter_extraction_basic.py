"""
Basic tests for Parameter Extraction Node - Core functionality verification.

This module provides simplified testing to verify the core parameter extraction,
validation, and normalization features work correctly.
"""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from src.nodes.parameter_extraction_node import (
    ParameterExtractor,
    ParameterNormalizer,
    ParameterValidator
)
from src.schemas.state_schemas import OrchestratorState, IntentType, StateTransitions, MainState
from src.schemas.generated_intent_schemas import DataFetchParams, ActionParams


class TestBasicParameterExtraction:
    """Test basic parameter extraction functionality."""
    
    def test_parameter_normalizer_currency(self):
        """Test currency normalization works correctly."""
        normalizer = ParameterNormalizer()
        
        # Test various currency formats
        assert normalizer.normalize_numeric_value("$100.50") == 100.50
        assert normalizer.normalize_numeric_value("€250") == 250.0
        assert normalizer.normalize_numeric_value("1,500.75") == 1500.75
        assert normalizer.normalize_numeric_value("25%") == 0.25
    
    def test_parameter_normalizer_lists(self):
        """Test list normalization works correctly."""
        normalizer = ParameterNormalizer()
        
        # Test various list formats
        result = normalizer.normalize_list_value("checking, savings, credit")
        assert result == ["checking", "savings", "credit"]
        
        result = normalizer.normalize_list_value("checking and savings")
        assert result == ["checking", "savings"]
        
        result = normalizer.normalize_list_value(["already", "a", "list"])
        assert result == ["already", "a", "list"]
    
    def test_parameter_normalizer_dates(self):
        """Test date normalization works correctly."""
        normalizer = ParameterNormalizer()
        
        # Test date alias normalization
        result = normalizer.normalize_date_aliases("last 7 days")
        assert "start_date" in result
        assert "end_date" in result
        assert "original_value" in result
        
        # Test ISO date passthrough
        iso_date = "2024-01-15T10:30:00Z"
        result = normalizer.normalize_date_aliases(iso_date)
        assert result["start_date"] == "2024-01-15T10:30:00+00:00"
        assert result["end_date"] == "2024-01-15T10:30:00+00:00"
    
    def test_parameter_validator_amounts(self):
        """Test amount validation works correctly."""
        validator = ParameterValidator()
        
        # Valid amounts should pass
        errors = validator.validate_numeric_ranges({"amount": 100.50})
        assert len(errors) == 0
        
        # Invalid amounts should fail
        errors = validator.validate_numeric_ranges({"amount": -50})
        assert len(errors) == 1
        assert "amount must be greater than 0" in errors[0]
    
    def test_parameter_validator_date_ranges(self):
        """Test date range validation works correctly."""
        validator = ParameterValidator()
        
        # Valid date range should pass
        valid_params = {
            'start_date': '2024-01-01T00:00:00Z',
            'end_date': '2024-01-31T23:59:59Z'
        }
        errors = validator.validate_date_range(valid_params)
        assert len(errors) == 0
        
        # Invalid date range should fail
        invalid_params = {
            'start_date': '2024-01-31T00:00:00Z',
            'end_date': '2024-01-01T23:59:59Z'
        }
        errors = validator.validate_date_range(invalid_params)
        assert len(errors) == 1
        assert "start_date must be before or equal to end_date" in errors[0]
    
    def test_parameter_validator_accounts(self):
        """Test account relationship validation works correctly."""
        validator = ParameterValidator()
        
        # Different accounts should pass
        valid_params = {
            'source_account': 'checking',
            'target_account': 'savings'
        }
        errors = validator.validate_account_relationships(valid_params)
        assert len(errors) == 0
        
        # Same source and target should fail
        invalid_params = {
            'source_account': 'checking',
            'target_account': 'checking'
        }
        errors = validator.validate_account_relationships(invalid_params)
        assert len(errors) == 1
        assert "source_account and target_account must be different" in errors[0]
    
    def test_parameter_extractor_schema_prompt(self):
        """Test schema prompt generation works correctly."""
        extractor = ParameterExtractor()
        
        prompt = extractor.build_schema_prompt(IntentType.DATA_FETCH, DataFetchParams)
        
        assert "Schema for data_fetch intent:" in prompt
        assert "entity_type" in prompt
        assert "limit" in prompt
        assert "time_period" in prompt
        assert "account_types" in prompt
    
    def test_parameter_extractor_extraction_prompt(self):
        """Test extraction prompt creation works correctly."""
        extractor = ParameterExtractor()
        
        user_input = "Show me my last 10 transactions"
        coarse_params = {"entity_type": "transactions"}
        
        prompt = extractor.create_extraction_prompt(
            IntentType.DATA_FETCH,
            user_input,
            coarse_params,
            DataFetchParams
        )
        
        assert user_input in prompt
        assert "transactions" in prompt
        assert "Schema for data_fetch intent" in prompt
        assert "Return a JSON object" in prompt
        assert "parameters" in prompt
        assert "confidence" in prompt
    
    def test_parameter_extractor_normalization(self):
        """Test parameter normalization pipeline works correctly."""
        extractor = ParameterExtractor()
        
        raw_parameters = {
            "entity_type": "transactions",
            "limit": "10",
            "time_period": "last month",
            "account_types": "checking, savings"
        }
        
        normalized = extractor.normalize_parameters(raw_parameters, DataFetchParams)
        
        assert normalized["entity_type"] == "transactions"
        assert normalized["limit"] == 10.0  # String converted to float
        assert isinstance(normalized["account_types"], list)
        assert "checking" in normalized["account_types"]
        assert "savings" in normalized["account_types"]
    
    def test_parameter_extractor_validation_success(self):
        """Test successful parameter validation."""
        extractor = ParameterExtractor()
        
        valid_parameters = {
            "entity_type": "transactions",
            "limit": 10,
            "time_period": "last_month",
            "account_types": ["checking"]
        }
        
        is_valid, errors = extractor.validate_parameters(valid_parameters, DataFetchParams)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_parameter_extractor_validation_errors(self):
        """Test parameter validation with errors."""
        extractor = ParameterExtractor()
        
        invalid_parameters = {
            "entity_type": "transactions",
            "limit": -5,  # Invalid limit
            "start_date": "2024-12-31",
            "end_date": "2024-01-01"  # Invalid date range
        }
        
        is_valid, errors = extractor.validate_parameters(invalid_parameters, DataFetchParams)
        
        assert not is_valid
        assert len(errors) > 0
        # Should have errors for both limit and date range
        error_text = " ".join(errors)
        assert "limit must be at least 1" in error_text
        assert "start_date must be before or equal to end_date" in error_text


class TestStateCreation:
    """Test state creation and transitions for parameter extraction."""
    
    def test_orchestrator_state_creation(self):
        """Test creating OrchestratorState for parameter extraction."""
        # Create a MainState (this is what parameter_extraction_node receives)
        main_state = MainState(
            user_input="Show me my last 10 transactions from checking account",
            intent=IntentType.DATA_FETCH,
            confidence=0.85,
            session_id=str(uuid4()),
            parameters={
                "entity_type": "transactions",
                "account_types": "checking"
            }
        )
        
        # The parameter extraction node converts MainState to OrchestratorState internally
        orchestrator_state = StateTransitions.to_orchestrator_state(main_state)
        
        # Verify state conversion
        assert orchestrator_state.user_input == main_state.user_input
        assert orchestrator_state.intent == main_state.intent
        assert orchestrator_state.confidence == main_state.confidence
        assert orchestrator_state.session_id == main_state.session_id
        
        # Verify parameters copied to extracted_params
        assert orchestrator_state.extracted_params == main_state.parameters
        
        # Verify orchestrator-specific fields initialized
        assert orchestrator_state.execution_plan is None
        assert orchestrator_state.tool_results is None
        assert orchestrator_state.final_response is None
    
    def test_orchestrator_state_update(self):
        """Test updating OrchestratorState with extracted parameters."""
        main_state = MainState(
            user_input="Transfer $500 from checking to savings",
            intent=IntentType.ACTION,
            confidence=0.9,
            parameters={"action_type": "transfer"}
        )
        
        orchestrator_state = StateTransitions.to_orchestrator_state(main_state)
        
        # Simulate parameter extraction results
        extracted_params = {
            "action_type": "transfer",
            "amount": 500.0,
            "source_account": "checking",
            "target_account": "savings",
            "description": None
        }
        
        # Update state with extracted parameters
        updated_state = orchestrator_state.model_copy(update={
            "extracted_params": extracted_params,
            "metadata": {
                **orchestrator_state.metadata,
                "extraction_status": "success",
                "extraction_confidence": 0.95,
                "extraction_errors": []
            }
        })
        
        # Verify updates
        assert updated_state.extracted_params == extracted_params
        assert updated_state.metadata["extraction_status"] == "success"
        assert updated_state.metadata["extraction_confidence"] == 0.95
        assert len(updated_state.metadata["extraction_errors"]) == 0
    
    def test_state_conversion_back_to_main(self):
        """Test converting OrchestratorState back to MainState."""
        main_state = MainState(
            user_input="Test input",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,
            parameters={"original": "param"}
        )
        
        # Convert to orchestrator state
        orchestrator_state = StateTransitions.to_orchestrator_state(main_state)
        
        # Update with extracted parameters
        orchestrator_state.extracted_params = {
            "original": "param",
            "new": "extracted_param"
        }
        orchestrator_state.tool_results = {"tool1": {"result": "success"}}
        orchestrator_state.final_response = "Execution complete"
        
        # Convert back to MainState
        result_state = StateTransitions.from_orchestrator_state(orchestrator_state)
        
        # Verify conversion
        assert result_state.user_input == orchestrator_state.user_input
        assert result_state.intent == orchestrator_state.intent
        assert result_state.confidence == orchestrator_state.confidence
        assert result_state.session_id == orchestrator_state.session_id
        
        # Verify merged results
        assert result_state.parameters == orchestrator_state.extracted_params
        assert result_state.execution_results == orchestrator_state.tool_results
        assert result_state.response == orchestrator_state.final_response


class TestSchemaIntegration:
    """Test integration with auto-generated schemas."""
    
    def test_data_fetch_schema_integration(self):
        """Test parameter extraction with DataFetchParams schema."""
        from src.nodes.missing_param_analysis_node import get_pydantic_model_for_intent
        
        # Get schema for DATA_FETCH intent
        schema = get_pydantic_model_for_intent(IntentType.DATA_FETCH)
        assert schema == DataFetchParams
        
        # Test field access
        fields = schema.model_fields
        assert "entity_type" in fields
        assert "limit" in fields
        assert "time_period" in fields
        assert "account_types" in fields
        assert "sort_by" in fields
        assert "order" in fields
    
    def test_action_schema_integration(self):
        """Test parameter extraction with ActionParams schema."""
        from src.nodes.missing_param_analysis_node import get_pydantic_model_for_intent
        
        # Get schema for ACTION intent
        schema = get_pydantic_model_for_intent(IntentType.ACTION)
        assert schema == ActionParams
        
        # Test field access
        fields = schema.model_fields
        assert "action_type" in fields
        assert "amount" in fields
        assert "source_account" in fields
        assert "target_account" in fields
        assert "description" in fields
    
    def test_schema_validation_integration(self):
        """Test parameter validation using auto-generated schemas."""
        extractor = ParameterExtractor()
        
        # Test valid ACTION parameters
        valid_action_params = {
            "action_type": "transfer",
            "amount": 100.0,
            "source_account": "checking",
            "target_account": "savings"
        }
        
        is_valid, errors = extractor.validate_parameters(valid_action_params, ActionParams)
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid ACTION parameters
        invalid_action_params = {
            "action_type": "transfer",
            "amount": -100.0,  # Invalid amount
            "source_account": "checking",
            "target_account": "checking"  # Same as source
        }
        
        is_valid, errors = extractor.validate_parameters(invalid_action_params, ActionParams)
        assert not is_valid
        assert len(errors) >= 2  # Should have multiple validation errors 