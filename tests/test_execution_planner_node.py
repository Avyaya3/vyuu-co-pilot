"""
Tests for the Hybrid Execution Planner Node.

Tests both LLM-based planning and rule-based validation functionality,
including error handling and fallback mechanisms.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from vyuu_copilot_v2.nodes.execution_planner_node import (
    execution_planner_node,
    _generate_llm_draft_plan,
    _validate_and_sanitize_plan,
    _merge_parameters,
    _create_fallback_plan,
    INTENT_OPERATION_MAPPING
)
from vyuu_copilot_v2.schemas.state_schemas import OrchestratorState, PlanStep, ExecutionPlan
from vyuu_copilot_v2.schemas.generated_intent_schemas import IntentCategory


@pytest.fixture
def sample_orchestrator_state():
    """Create sample orchestrator state for testing."""
    return OrchestratorState(
        user_input="Show me my account balance",
        intent=IntentCategory.DATA_FETCH,
        confidence=0.95,
        messages=[],
        session_id="12345678-1234-1234-1234-123456789012",
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={},
        execution_results={},
        response="",
        extracted_params={
            "user_id": "test_user_123",
            "account_name": "checking"
        }
    )


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing."""
    return '''[
        {
            "tool_name": "db_query",
            "operation": "get_account_balance",
            "params": {
                "user_id": "test_user_123",
                "account_name": "checking"
            }
        }
    ]'''


class TestExecutionPlannerNode:
    """Test the main execution planner node function."""

    @pytest.mark.asyncio
    async def test_successful_planning(self, sample_orchestrator_state):
        """Test successful execution planning with valid LLM response."""
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = '''[
            {
                "tool_name": "db_query",
                "operation": "get_account_balance",
                "params": {
                    "user_id": "test_user_123",
                    "account_name": "checking"
                }
            }
        ]'''

        with patch('src.utils.llm_client.LLMClient') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = mock_llm_response
            mock_get_client.return_value = mock_client

            result = await execution_planner_node(sample_orchestrator_state)

            # Verify successful planning
            assert result.metadata["planning_status"] == "success"
            assert "planning_errors" in result.metadata
            assert result.execution_plan is not None
            
            execution_plan = ExecutionPlan(**result.execution_plan)
            assert len(execution_plan.steps) == 1
            assert execution_plan.steps[0].tool_name == "db_query"
            assert execution_plan.steps[0].operation == "get_account_balance"

    @pytest.mark.asyncio
    async def test_llm_failure_with_fallback(self, sample_orchestrator_state):
        """Test fallback planning when LLM fails."""
        with patch('src.nodes.execution_planner_node._generate_llm_draft_plan') as mock_generate:
            # Mock LLM to fail (return None)
            mock_generate.return_value = None

            result = await execution_planner_node(sample_orchestrator_state)

            # Should use fallback plan
            assert result.metadata["planning_status"] == "success"
            assert "LLM failed to generate valid plan" in result.metadata["planning_errors"]
            
            # Should have a fallback plan
            execution_plan = ExecutionPlan(**result.execution_plan)
            assert len(execution_plan.steps) > 0

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, sample_orchestrator_state):
        """Test handling of invalid JSON from LLM."""
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "This is not valid JSON"

        with patch('src.utils.llm_client.LLMClient') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = mock_llm_response
            mock_get_client.return_value = mock_client

            result = await execution_planner_node(sample_orchestrator_state)

            # Should use fallback plan
            assert result.metadata["planning_status"] == "success"
            assert result.execution_plan is not None

    @pytest.mark.asyncio
    async def test_action_intent_no_fallback(self):
        """Test that action intents don't get fallback plans when LLM fails."""
        state = OrchestratorState(
            user_input="Transfer $100 to savings",
            intent=IntentCategory.ACTION,
            confidence=0.95,
            messages=[],
            session_id="12345678-1234-1234-1234-123456789012",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            extracted_params={
                "user_id": "test_user_123",
                "amount": 100.0,
                "source_account": "checking",
                "target_account": "savings"
            }
        )

        with patch('src.nodes.execution_planner_node._generate_llm_draft_plan') as mock_generate:
            # Mock LLM to fail (return None)
            mock_generate.return_value = None

            result = await execution_planner_node(state)

            # Should have no valid steps for action fallback
            execution_plan = ExecutionPlan(**result.execution_plan)
            assert len(execution_plan.steps) == 0
            assert result.metadata["planning_status"] == "error"  # Error because no fallback plan for actions


class TestLLMDraftGeneration:
    """Test LLM draft plan generation."""

    @pytest.mark.asyncio
    async def test_successful_llm_generation(self, sample_orchestrator_state, sample_llm_response):
        """Test successful LLM plan generation."""
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = sample_llm_response

        with patch('src.utils.llm_client.LLMClient') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = mock_llm_response
            mock_get_client.return_value = mock_client

            draft_plan = await _generate_llm_draft_plan(sample_orchestrator_state)

            assert draft_plan is not None
            assert len(draft_plan) == 1
            assert draft_plan[0]["tool_name"] == "db_query"
            assert draft_plan[0]["operation"] == "get_account_balance"

    @pytest.mark.asyncio
    async def test_llm_json_extraction(self, sample_orchestrator_state):
        """Test JSON extraction from LLM response with extra text."""
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = '''
        Here's the execution plan for your request:
        
        [
            {
                "tool_name": "db_query",
                "operation": "get_account_balance",
                "params": {
                    "user_id": "test_user_123",
                    "account_name": "checking"
                }
            }
        ]
        
        This plan will retrieve your account balance.
        '''

        with patch('src.utils.llm_client.LLMClient') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = mock_llm_response
            mock_get_client.return_value = mock_client

            draft_plan = await _generate_llm_draft_plan(sample_orchestrator_state)

            assert draft_plan is not None
            assert len(draft_plan) == 1
            assert draft_plan[0]["tool_name"] == "db_query"

    @pytest.mark.asyncio
    async def test_llm_api_error(self, sample_orchestrator_state):
        """Test handling of LLM API errors."""
        with patch('src.nodes.execution_planner_node.LLMClient') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.chat_completion.side_effect = Exception("API Error")
            mock_llm_class.return_value = mock_llm

            draft_plan = await _generate_llm_draft_plan(sample_orchestrator_state)

            assert draft_plan is None


class TestPlanValidation:
    """Test plan validation and sanitization."""

    @pytest.mark.asyncio
    async def test_valid_plan_validation(self):
        """Test validation of a valid plan."""
        draft_plan = [
            {
                "tool_name": "db_query",
                "operation": "get_account_balance",
                "params": {
                    "user_id": "test_user_123",
                    "account_name": "checking"
                }
            }
        ]
        
        extracted_params = {"user_id": "test_user_123", "account_name": "checking"}

        with patch('src.nodes.execution_planner_node.get_tool_schema') as mock_get_schema, \
             patch('src.nodes.execution_planner_node.get_tool_info') as mock_get_info:
            
            # Mock tool schema validation
            mock_schema = MagicMock()
            mock_schema.return_value.dict.return_value = {
                "operation": "get_account_balance",
                "user_id": "test_user_123",
                "account_name": "checking"
            }
            mock_get_schema.return_value = mock_schema

            # Mock tool info
            mock_get_info.return_value = {
                "db_query": {
                    "operations": ["get_account_balance", "get_transaction_history"]
                }
            }

            validated_steps, errors = await _validate_and_sanitize_plan(draft_plan, extracted_params)

            assert len(validated_steps) == 1
            assert len(errors) == 0
            assert validated_steps[0].tool_name == "db_query"
            assert validated_steps[0].operation == "get_account_balance"

    @pytest.mark.asyncio
    async def test_invalid_tool_name(self):
        """Test validation with invalid tool name."""
        draft_plan = [
            {
                "tool_name": "invalid_tool",
                "operation": "some_operation",
                "params": {}
            }
        ]
        
        extracted_params = {}

        validated_steps, errors = await _validate_and_sanitize_plan(draft_plan, extracted_params)

        assert len(validated_steps) == 0
        assert len(errors) == 1
        assert "Unknown tool 'invalid_tool'" in errors[0]

    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test validation with invalid operation."""
        draft_plan = [
            {
                "tool_name": "db_query",
                "operation": "invalid_operation",
                "params": {}
            }
        ]
        
        extracted_params = {}

        with patch('src.nodes.execution_planner_node.get_tool_info') as mock_get_info:
            mock_get_info.return_value = {
                "db_query": {
                    "operations": ["get_account_balance", "get_transaction_history"]
                }
            }

            validated_steps, errors = await _validate_and_sanitize_plan(draft_plan, extracted_params)

            assert len(validated_steps) == 0
            assert len(errors) == 1
            assert "Invalid operation 'invalid_operation'" in errors[0]


class TestParameterMerging:
    """Test parameter merging logic (no longer does mapping - parameters come pre-normalized)."""

    def test_basic_parameter_merging(self):
        """Test merging of LLM params with extracted params."""
        raw_params = {
            "operation": "get_account_balance",
            "additional_field": "extra_value"
        }
        
        extracted_params = {
            "user_id": "test_user_123",
            "account_name": "checking"
        }

        merged = _merge_parameters(raw_params, extracted_params)

        # Extracted params should be preserved
        assert merged["user_id"] == "test_user_123"
        assert merged["account_name"] == "checking"
        # LLM params should be added if not in extracted
        assert merged["additional_field"] == "extra_value"

    def test_extracted_params_priority(self):
        """Test that extracted parameters take priority over LLM params."""
        raw_params = {
            "user_id": "llm_user",
            "account_name": "llm_account"
        }
        
        extracted_params = {
            "user_id": "extracted_user_123",
            "account_name": "checking",
            "limit": 10
        }

        merged = _merge_parameters(raw_params, extracted_params)

        # Extracted params should take priority
        assert merged["user_id"] == "extracted_user_123"
        assert merged["account_name"] == "checking"
        assert merged["limit"] == 10

    def test_user_id_preservation(self):
        """Test that user_id is always preserved from extracted params."""
        raw_params = {"operation": "get_balance"}
        extracted_params = {"user_id": "test_user_123"}

        merged = _merge_parameters(raw_params, extracted_params)

        assert merged["user_id"] == "test_user_123"
        assert merged["operation"] == "get_balance"


class TestFallbackPlanning:
    """Test fallback plan generation."""

    def test_question_intent_fallback(self):
        """Test fallback for question intent."""
        state = OrchestratorState(
            user_input="What are my accounts?",
            intent=IntentCategory.DATA_FETCH,
            confidence=0.95,
            messages=[],
            session_id="12345678-1234-1234-1234-123456789012",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            extracted_params={"user_id": "test_user_123"}
        )

        fallback_plan = _create_fallback_plan(state)

        assert len(fallback_plan) == 1
        assert fallback_plan[0]["tool_name"] == "db_query"
        assert fallback_plan[0]["operation"] == "get_user_accounts"
        assert fallback_plan[0]["params"]["user_id"] == "test_user_123"

    def test_analytics_intent_fallback(self):
        """Test fallback for analytics intent."""
        state = OrchestratorState(
            user_input="Show me my spending",
            intent=IntentCategory.AGGREGATE,
            confidence=0.95,
            messages=[],
            session_id="12345678-1234-1234-1234-123456789012",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            extracted_params={"user_id": "test_user_123"}
        )

        fallback_plan = _create_fallback_plan(state)

        assert len(fallback_plan) == 1
        assert fallback_plan[0]["tool_name"] == "db_aggregate"
        assert fallback_plan[0]["operation"] == "spending_by_category"
        assert fallback_plan[0]["params"]["user_id"] == "test_user_123"
        assert fallback_plan[0]["params"]["days_back"] == 30

    def test_action_intent_no_fallback(self):
        """Test that action intents don't get fallback plans."""
        state = OrchestratorState(
            user_input="Transfer money",
            intent=IntentCategory.ACTION,
            confidence=0.95,
            messages=[],
            session_id="12345678-1234-1234-1234-123456789012",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            extracted_params={"user_id": "test_user_123"}
        )

        fallback_plan = _create_fallback_plan(state)

        assert len(fallback_plan) == 0

    def test_no_intent_fallback(self):
        """Test fallback when no intent is provided."""
        state = OrchestratorState(
            user_input="Something unclear",
            intent=None,
            confidence=0.95,
            messages=[],
            session_id="12345678-1234-1234-1234-123456789012",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            extracted_params={"user_id": "test_user_123"}
        )

        fallback_plan = _create_fallback_plan(state)

        assert len(fallback_plan) == 0 