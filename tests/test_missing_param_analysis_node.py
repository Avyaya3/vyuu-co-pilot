import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from src.nodes.missing_param_analysis_node import missing_param_analysis_node
from src.schemas.state_schemas import ClarificationState, IntentType

@pytest.mark.asyncio
class TestMissingParamAnalysisNode:
    """Test suite for the Missing Parameter Analysis node."""

    @pytest.fixture
    def base_state(self):
        return ClarificationState(
            user_input="Transfer $500 from checking to savings",
            intent=IntentType.ACTION,
            extracted_parameters={"action_type": "transfer"},
            missing_params=["amount", "source_account", "target_account"],
            missing_critical_params=["amount"],
            clarification_history=[],
            clarification_attempts=0,
            max_attempts=3
        )

    @patch("src.nodes.missing_param_analysis_node.LLMClient")
    async def test_successful_llm_enrichment(self, mock_llm, base_state):
        """Test that the node enriches ClarificationState with LLM output."""
        # Mock LLM response
        mock_llm.return_value.client.chat.completions.create.return_value.choices = [
            type("obj", (), {"message": type("obj", (), {"content": """
            {"extracted_parameters": {"action_type": "transfer", "amount": 500.0, "source_account": "checking", "target_account": "savings"},
              "missing_params": [],
              "missing_critical_params": [],
              "parameter_priorities": ["amount", "source_account", "target_account"],
              "normalization_suggestions": {"checking acct": "checking"},
              "ambiguity_flags": {}}
            """})})
        ]
        result = await missing_param_analysis_node(base_state)
        assert result.extracted_parameters["amount"] == 500.0
        assert result.missing_params == []
        assert result.parameter_priorities == ["amount", "source_account", "target_account"]
        assert result.normalization_suggestions["checking acct"] == "checking"
        assert result.ambiguity_flags == {}
        # Original fields preserved
        assert result.clarification_attempts == base_state.clarification_attempts
        assert result.max_attempts == base_state.max_attempts

    @patch("src.nodes.missing_param_analysis_node.LLMClient")
    async def test_llm_malformed_json(self, mock_llm, base_state):
        """Test that malformed LLM output is handled gracefully."""
        # LLM returns invalid JSON
        mock_llm.return_value.client.chat.completions.create.return_value.choices = [
            type("obj", (), {"message": type("obj", (), {"content": "not a json"})})
        ]
        result = await missing_param_analysis_node(base_state)
        # Should return a copy of the prior state
        assert result.extracted_parameters == base_state.extracted_parameters
        assert result.missing_params == base_state.missing_params
        assert result.missing_critical_params == base_state.missing_critical_params

    @patch("src.nodes.missing_param_analysis_node.LLMClient")
    async def test_type_validation_and_ambiguity(self, mock_llm, base_state):
        """Test that invalid slot values are set to null and ambiguity_flags updated."""
        # LLM returns invalid type for amount (should be float)
        mock_llm.return_value.client.chat.completions.create.return_value.choices = [
            type("obj", (), {"message": type("obj", (), {"content": """
            {"extracted_parameters": {"action_type": "transfer", "amount": "five hundred", "source_account": "checking", "target_account": "savings"},
              "missing_params": ["amount"],
              "missing_critical_params": ["amount"],
              "parameter_priorities": ["amount"],
              "normalization_suggestions": {},
              "ambiguity_flags": {}}
            """})})
        ]
        result = await missing_param_analysis_node(base_state)
        # amount should be set to None due to type error
        assert result.extracted_parameters["amount"] is None
        # ambiguity_flags should mention the validation error
        assert "amount" in result.ambiguity_flags
        assert result.missing_params == ["amount"]
        assert result.missing_critical_params == ["amount"]

    @patch("src.nodes.missing_param_analysis_node.LLMClient")
    async def test_all_required_fields_present(self, mock_llm, base_state):
        """Test that all required fields are present in the output state."""
        mock_llm.return_value.client.chat.completions.create.return_value.choices = [
            type("obj", (), {"message": type("obj", (), {"content": """
            {"extracted_parameters": {"action_type": "transfer", "amount": 100.0, "source_account": "checking", "target_account": "savings"},
              "missing_params": [],
              "missing_critical_params": [],
              "parameter_priorities": ["amount"],
              "normalization_suggestions": {"checking acct": "checking"},
              "ambiguity_flags": {}}
            """})})
        ]
        result = await missing_param_analysis_node(base_state)
        # All required fields should be present
        assert hasattr(result, "extracted_parameters")
        assert hasattr(result, "missing_params")
        assert hasattr(result, "missing_critical_params")
        assert hasattr(result, "parameter_priorities")
        assert hasattr(result, "normalization_suggestions")
        assert hasattr(result, "ambiguity_flags") 