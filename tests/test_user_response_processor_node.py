"""
Tests for User Response Processor Node.

This module provides comprehensive test coverage for the user response processor node,
including LLM-based parsing, validation, normalization, state updates, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List
from uuid import uuid4

from vyuu_copilot_v2.nodes.user_response_processor_node import (
    user_response_processor_node,
    UserResponseProcessor,
    ValueValidator,
    ResponseParsingResult,
    ValidationResult,
)
from vyuu_copilot_v2.schemas.state_schemas import ClarificationState, IntentType


@pytest.fixture
def sample_clarification_state():
    """Sample clarification state for testing."""
    return ClarificationState(
        session_id=str(uuid4()),
        user_id=str(uuid4()), 
        user_input="Show me sales data for Q1",
        intent=IntentType.DATA_FETCH,
        extracted_parameters={
            "entity_type": "sales_data",
            "time_period": None,
            "region": None
        },
        missing_params=["time_period", "region"],
        missing_critical_params=["time_period"],
        parameter_priorities=["time_period", "region"],
        normalization_suggestions={
            "Q1": "2024-Q1",
            "first quarter": "2024-Q1"
        },
        ambiguity_flags={},
        clarification_history=[
            {
                "question": "What time period would you like to see sales data for?",
                "targeted_param": "time_period",
                "timestamp": "2024-01-15T10:00:00",
                "attempt_number": 1
            }
        ],
        clarification_attempts=1,
        max_attempts=3
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock()
    client.model = "gpt-4"
    client.temperature = 0.2
    client.chat_completion = AsyncMock()
    return client


class TestValueValidator:
    """Test cases for ValueValidator utility class."""
    
    def test_validate_numeric_valid_integer(self):
        """Test numeric validation with valid integer."""
        result = ValueValidator.validate_numeric("42", "amount")
        assert result.is_valid
        assert result.normalized_value == 42
        assert result.error_reason is None
    
    def test_validate_numeric_valid_float(self):
        """Test numeric validation with valid float."""
        result = ValueValidator.validate_numeric("42.50", "amount")
        assert result.is_valid
        assert result.normalized_value == 42.50
        assert result.error_reason is None
    
    def test_validate_numeric_with_currency(self):
        """Test numeric validation with currency symbols."""
        result = ValueValidator.validate_numeric("$1,000.50", "amount")
        assert result.is_valid
        assert result.normalized_value == 1000.50
        assert result.error_reason is None
    
    def test_validate_numeric_percentage(self):
        """Test numeric validation with percentage."""
        result = ValueValidator.validate_numeric("15%", "rate")
        assert result.is_valid
        assert result.normalized_value == 0.15
        assert result.error_reason is None
    
    def test_validate_numeric_invalid_string(self):
        """Test numeric validation with invalid string."""
        result = ValueValidator.validate_numeric("not a number", "amount")
        assert not result.is_valid
        assert result.normalized_value is None
        assert "Cannot convert" in result.error_reason
    
    def test_validate_numeric_none_value(self):
        """Test numeric validation with None value."""
        result = ValueValidator.validate_numeric(None, "amount")
        assert result.is_valid
        assert result.normalized_value is None
        assert result.error_reason is None
    
    def test_validate_date_valid_iso_format(self):
        """Test date validation with valid ISO format."""
        result = ValueValidator.validate_date("2024-01-15", "date")
        assert result.is_valid
        assert result.normalized_value == "2024-01-15"
        assert result.error_reason is None
    
    def test_validate_date_valid_us_format(self):
        """Test date validation with valid US format."""
        result = ValueValidator.validate_date("01/15/2024", "date")
        assert result.is_valid
        assert result.normalized_value == "01/15/2024"
        assert result.error_reason is None
    
    def test_validate_date_invalid_format(self):
        """Test date validation with invalid format."""
        result = ValueValidator.validate_date("15-Jan-2024", "date")
        assert not result.is_valid
        assert result.normalized_value is None
        assert "Invalid date format" in result.error_reason
    
    def test_validate_enum_exact_match(self):
        """Test enum validation with exact match."""
        valid_options = ["pending", "completed", "cancelled"]
        result = ValueValidator.validate_enum("pending", "status", valid_options)
        assert result.is_valid
        assert result.normalized_value == "pending"
        assert result.error_reason is None
    
    def test_validate_enum_case_insensitive(self):
        """Test enum validation with case insensitive match."""
        valid_options = ["pending", "completed", "cancelled"]
        result = ValueValidator.validate_enum("PENDING", "status", valid_options)
        assert result.is_valid
        assert result.normalized_value == "pending"
        assert result.error_reason is None
    
    def test_validate_enum_partial_match(self):
        """Test enum validation with partial match."""
        valid_options = ["in_progress", "completed", "cancelled"]
        result = ValueValidator.validate_enum("progress", "status", valid_options)
        assert result.is_valid
        assert result.normalized_value == "in_progress"
        assert result.error_reason is None
    
    def test_validate_enum_invalid_option(self):
        """Test enum validation with invalid option."""
        valid_options = ["pending", "completed", "cancelled"]
        result = ValueValidator.validate_enum("invalid", "status", valid_options)
        assert not result.is_valid
        assert result.normalized_value is None
        assert "not a valid option" in result.error_reason
    
    def test_validate_string_valid(self):
        """Test string validation with valid input."""
        result = ValueValidator.validate_string("  Hello World  ", "name")
        assert result.is_valid
        assert result.normalized_value == "Hello World"
        assert result.error_reason is None
    
    def test_validate_string_empty(self):
        """Test string validation with empty input."""
        result = ValueValidator.validate_string("   ", "name")
        assert not result.is_valid
        assert result.normalized_value is None
        assert "cannot be empty" in result.error_reason


class TestUserResponseProcessor:
    """Test cases for UserResponseProcessor class."""
    
    def test_identify_target_slots_from_targeted_param(self):
        """Test identifying target slots from targeted_param field."""
        processor = UserResponseProcessor()
        history = [
            {
                "question": "What time period?",
                "targeted_param": "time_period",
                "timestamp": "2024-01-15T10:00:00"
            }
        ]
        
        result = processor.identify_target_slots(history)
        assert result == ["time_period"]
    
    def test_identify_target_slots_from_list(self):
        """Test identifying target slots from list of parameters."""
        processor = UserResponseProcessor()
        history = [
            {
                "question": "What are the start and end dates?",
                "targeted_param": ["start_date", "end_date"],
                "timestamp": "2024-01-15T10:00:00"
            }
        ]
        
        result = processor.identify_target_slots(history)
        assert result == ["start_date", "end_date"]
    
    def test_identify_target_slots_from_question_text(self):
        """Test identifying target slots from question text as fallback."""
        processor = UserResponseProcessor()
        history = [
            {
                "question": "What \"region\" would you like to see data for?",
                "timestamp": "2024-01-15T10:00:00"
            }
        ]
        
        result = processor.identify_target_slots(history)
        assert result == ["region"]
    
    def test_identify_target_slots_empty_history(self):
        """Test identifying target slots with empty history."""
        processor = UserResponseProcessor()
        result = processor.identify_target_slots([])
        assert result == []
    
    def test_create_llm_prompt(self):
        """Test LLM prompt creation."""
        processor = UserResponseProcessor()
        
        system_prompt, user_prompt = processor.create_llm_prompt(
            "Q1 2024",
            ["time_period"],
            {"Q1": "2024-Q1"},
            {"time_period": "Needs specific quarter"},
            {"intent": IntentType.DATA_FETCH, "extracted_parameters": {"entity_type": "sales"}}
        )
        
        assert "parameter extraction assistant" in system_prompt
        assert "USER RESPONSE: \"Q1 2024\"" in user_prompt
        assert "TARGET SLOTS: ['time_period']" in user_prompt
        assert "NORMALIZATION HINTS:" in user_prompt
        assert "PREVIOUS AMBIGUITIES:" in user_prompt
        assert "INTENT CONTEXT: IntentType.DATA_FETCH" in user_prompt
    
    @pytest.mark.asyncio
    async def test_parse_user_response_success(self, mock_llm_client):
        """Test successful user response parsing with LLM."""
        # Mock LLM response
        mock_llm_client.chat_completion.return_value = """{
            "slot_values": {"time_period": "2024-Q1"},
            "ambiguity_flags": {},
            "extraction_confidence": 0.95,
            "parsing_notes": "Successfully extracted quarter"
        }"""
        
        processor = UserResponseProcessor(mock_llm_client)
        
        result = await processor.parse_user_response(
            "Q1 2024",
            ["time_period"],
            {"Q1": "2024-Q1"},
            {},
            {"intent": IntentType.DATA_FETCH}
        )
        
        assert result.slot_values == {"time_period": "2024-Q1"}
        assert result.ambiguity_flags == {}
        assert result.extraction_confidence == 0.95
        assert result.parsing_notes == "Successfully extracted quarter"
    
    @pytest.mark.asyncio
    async def test_parse_user_response_json_error(self, mock_llm_client):
        """Test user response parsing with JSON decode error."""
        # Mock invalid JSON response
        mock_llm_client.chat_completion.return_value = "Invalid JSON"
        
        processor = UserResponseProcessor(mock_llm_client)
        
        result = await processor.parse_user_response(
            "I don't know",
            ["time_period"],
            {},
            {},
            {}
        )
        
        # Should fall back to heuristic parsing
        assert result.slot_values == {"time_period": None}
        assert "don't know this information" in result.ambiguity_flags["time_period"]
        assert result.extraction_confidence == 0.3
    
    @pytest.mark.asyncio
    async def test_parse_user_response_llm_error(self, mock_llm_client):
        """Test user response parsing with LLM call error."""
        # Mock LLM error
        mock_llm_client.chat_completion.side_effect = Exception("API Error")
        
        processor = UserResponseProcessor(mock_llm_client)
        
        result = await processor.parse_user_response(
            "Some value",
            ["time_period"],
            {},
            {},
            {}
        )
        
        # Should fall back to heuristic parsing
        assert result.slot_values == {"time_period": "Some value"}
        assert "manual review recommended" in result.ambiguity_flags["time_period"]
        assert result.extraction_confidence == 0.1  # Updated for unexpected_error type
    
    def test_validate_and_normalize_values(self):
        """Test value validation and normalization."""
        processor = UserResponseProcessor()
        
        slot_values = {
            "amount": "1,000.50",
            "date": "2024-01-15",
            "description": "  Sales Report  "
        }
        
        slot_types = {
            "amount": "numeric",
            "date": "date",
            "description": "string"
        }
        
        normalized, errors = processor.validate_and_normalize_values(
            slot_values, IntentType.ACTION, slot_types
        )
        
        assert normalized["amount"] == 1000.50
        assert normalized["date"] == "2024-01-15"
        assert normalized["description"] == "Sales Report"
        assert len(errors) == 0
    
    def test_validate_and_normalize_values_with_errors(self):
        """Test value validation with validation errors."""
        processor = UserResponseProcessor()
        
        slot_values = {
            "amount": "not a number",
            "date": "invalid date"
        }
        
        slot_types = {
            "amount": "numeric",
            "date": "date"
        }
        
        normalized, errors = processor.validate_and_normalize_values(
            slot_values, IntentType.ACTION, slot_types
        )
        
        assert normalized["amount"] is None
        assert normalized["date"] is None
        assert "Cannot convert" in errors["amount"]
        assert "Invalid date format" in errors["date"]
    
    def test_recompute_missing_parameters(self):
        """Test recomputing missing parameters."""
        processor = UserResponseProcessor()
        
        extracted_params = {
            "entity_type": "sales_data",
            "time_period": "2024-Q1",
            "region": None,
            "filters": None
        }
        
        missing, missing_critical = processor.recompute_missing_parameters(
            extracted_params,
            IntentType.DATA_FETCH
        )
        
        # Check that some parameters are missing but entity_type is provided
        assert len(missing) > 0  # Some parameters should be missing
        assert len(missing_critical) == 0  # entity_type is provided (critical parameter)
    
    def test_update_clarification_history(self):
        """Test updating clarification history."""
        processor = UserResponseProcessor()
        
        history = [
            {
                "question": "What time period?",
                "targeted_param": "time_period",
                "timestamp": "2024-01-15T10:00:00"
            }
        ]
        
        parsing_result = ResponseParsingResult(
            slot_values={"time_period": "2024-Q1"},
            ambiguity_flags={},
            extraction_confidence=0.95,
            parsing_notes="Extracted successfully"
        )
        
        updated = processor.update_clarification_history(
            history,
            "Q1 2024",
            parsing_result
        )
        
        assert len(updated) == 1
        assert updated[0]["user_response"] == "Q1 2024"
        assert updated[0]["extracted_values"] == {"time_period": "2024-Q1"}
        assert updated[0]["extraction_confidence"] == 0.95
        assert "response_timestamp" in updated[0]
    
    def test_get_slot_types_from_schema(self):
        """Test extracting slot types from Pydantic schema."""
        processor = UserResponseProcessor()
        
        # Test ACTION intent schema
        action_types = processor.get_slot_types_from_schema(IntentType.ACTION)
        assert "amount" in action_types
        assert action_types["amount"] == "numeric"  # Should be detected as numeric
        
        # Test DATA_FETCH intent schema
        data_fetch_types = processor.get_slot_types_from_schema(IntentType.DATA_FETCH)
        assert "limit" in data_fetch_types
        assert data_fetch_types["limit"] == "numeric"


class TestUserResponseProcessorNode:
    """Test cases for the main user response processor node function."""
    
    @pytest.mark.asyncio
    async def test_user_response_processor_node_success(self, sample_clarification_state):
        """Test successful user response processing."""
        with patch('src.nodes.user_response_processor_node.UserResponseProcessor') as mock_processor_class:
            # Setup mock processor
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            # Mock processor methods
            mock_processor.identify_target_slots.return_value = ["time_period"]
            mock_processor.parse_user_response = AsyncMock(return_value=ResponseParsingResult(
                slot_values={"time_period": "2024-Q1"},
                ambiguity_flags={},
                extraction_confidence=0.95
            ))
            mock_processor.validate_and_normalize_values.return_value = (
                {"time_period": "2024-Q1"}, {}
            )
            mock_processor.recompute_missing_parameters.return_value = (
                ["region"], []
            )
            mock_processor.update_clarification_history.return_value = [
                {
                    "question": "What time period would you like to see sales data for?",
                    "targeted_param": "time_period",
                    "timestamp": "2024-01-15T10:00:00",
                    "attempt_number": 1,
                    "user_response": "Q1 2024",
                    "extracted_values": {"time_period": "2024-Q1"}
                }
            ]
            
            # Call the node
            result = await user_response_processor_node(
                sample_clarification_state,
                "Q1 2024"
            )
            
            # Verify results
            assert result.extracted_parameters["time_period"] == "2024-Q1"
            assert result.missing_params == ["region"]
            assert result.missing_critical_params == []
            assert len(result.clarification_history) == 1
            assert result.clarification_history[0]["user_response"] == "Q1 2024"
    
    @pytest.mark.asyncio
    async def test_user_response_processor_node_no_target_slots(self, sample_clarification_state):
        """Test user response processing when no target slots are identified."""
        with patch('src.nodes.user_response_processor_node.UserResponseProcessor') as mock_processor_class:
            # Setup mock processor
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            # Mock no target slots found
            mock_processor.identify_target_slots.return_value = []
            
            # Call the node
            result = await user_response_processor_node(
                sample_clarification_state,
                "Some response"
            )
            
            # Verify error handling
            assert "error" in result.metadata
            assert "Could not identify which parameters" in result.metadata["error"]
    
    @pytest.mark.asyncio
    async def test_user_response_processor_node_with_validation_errors(self, sample_clarification_state):
        """Test user response processing with validation errors."""
        with patch('src.nodes.user_response_processor_node.UserResponseProcessor') as mock_processor_class:
            # Setup mock processor
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            # Mock processor methods with validation errors
            mock_processor.identify_target_slots.return_value = ["amount"]
            mock_processor.parse_user_response = AsyncMock(return_value=ResponseParsingResult(
                slot_values={"amount": "not a number"},
                ambiguity_flags={},
                extraction_confidence=0.8
            ))
            mock_processor.validate_and_normalize_values.return_value = (
                {"amount": None},
                {"amount": "Cannot convert 'not a number' to number"}
            )
            mock_processor.recompute_missing_parameters.return_value = (
                ["amount", "region"], ["amount"]
            )
            mock_processor.update_clarification_history.return_value = sample_clarification_state.clarification_history
            
            # Call the node
            result = await user_response_processor_node(
                sample_clarification_state,
                "not a number"
            )
            
            # Verify validation error handling
            assert result.extracted_parameters["amount"] is None
            assert "Validation error" in result.ambiguity_flags["amount"]
            assert "amount" in result.missing_params
    
    @pytest.mark.asyncio
    async def test_user_response_processor_node_exception_handling(self, sample_clarification_state):
        """Test user response processing with unexpected exceptions."""
        with patch('src.nodes.user_response_processor_node.UserResponseProcessor') as mock_processor_class:
            # Mock processor to raise exception
            mock_processor_class.side_effect = Exception("Unexpected error")
            
            # Call the node
            result = await user_response_processor_node(
                sample_clarification_state,
                "Some response"
            )
            
            # Verify error handling
            assert "error" in result.metadata
            assert "System error" in result.metadata["error"]
            assert "Unexpected error" in result.metadata["error"]
    
    @pytest.mark.asyncio
    async def test_user_response_processor_node_dont_know_response(self, sample_clarification_state):
        """Test user response processing with 'don't know' type responses."""
        with patch('src.nodes.user_response_processor_node.UserResponseProcessor') as mock_processor_class:
            # Setup mock processor
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            
            # Mock processor methods for "don't know" response
            mock_processor.identify_target_slots.return_value = ["time_period"]
            mock_processor.parse_user_response = AsyncMock(return_value=ResponseParsingResult(
                slot_values={"time_period": None},
                ambiguity_flags={"time_period": "User indicated they don't know this information"},
                extraction_confidence=0.9
            ))
            mock_processor.validate_and_normalize_values.return_value = (
                {"time_period": None}, {}
            )
            mock_processor.recompute_missing_parameters.return_value = (
                ["time_period", "region"], ["time_period"]
            )
            mock_processor.update_clarification_history.return_value = sample_clarification_state.clarification_history
            
            # Call the node
            result = await user_response_processor_node(
                sample_clarification_state,
                "I don't know"
            )
            
            # Verify "don't know" handling
            assert result.extracted_parameters["time_period"] is None
            assert "don't know this information" in result.ambiguity_flags["time_period"]
            assert "time_period" in result.missing_params
            assert "time_period" in result.missing_critical_params


if __name__ == "__main__":
    pytest.main([__file__]) 