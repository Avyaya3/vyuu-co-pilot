"""
Tests for Response Synthesis Node.

This module tests the response synthesis functionality including:
- DataFormatter utility methods
- ResponseSynthesizer internal methods
- Main node function with various scenarios
- Error handling and edge cases
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from uuid import uuid4

from src.nodes.response_synthesis_node import (
    ResponseSynthesizer,
    response_synthesis_node,
    BASE_SYSTEM_PROMPT,
    INTENT_GUIDELINES,
)
from src.schemas.state_schemas import OrchestratorState, IntentType
from src.schemas.generated_intent_schemas import IntentCategory
from src.utils.data_formatters import DataFormatter


class TestDataFormatter:
    """Test DataFormatter utility methods."""

    def test_format_currency(self):
        """Test currency formatting."""
        formatter = DataFormatter()
        
        assert formatter.format_currency(1234.56) == "$1,234.56"
        assert formatter.format_currency(-1234.56) == "-$1,234.56"
        assert formatter.format_currency(0) == "$0.00"
        assert formatter.format_currency(None) == "$0.00"

    def test_format_percentage(self):
        """Test percentage formatting."""
        formatter = DataFormatter()
        
        assert formatter.format_percentage(25.5) == "25.5%"
        assert formatter.format_percentage(0) == "0.0%"
        assert formatter.format_percentage(None) == "0%"

    def test_format_date(self):
        """Test date formatting."""
        formatter = DataFormatter()
        
        # Test with datetime object
        dt = datetime(2024, 1, 15)
        assert formatter.format_date(dt) == "January 15, 2024"
        assert formatter.format_date(dt, "short") == "01/15/2024"
        
        # Test with string
        assert formatter.format_date("2024-01-15") == "January 15, 2024"
        assert formatter.format_date("2024-01-15", "short") == "01/15/2024"

    def test_format_data_for_llm(self):
        """Test LLM data formatting."""
        formatter = DataFormatter()
        
        data = {
            "balances": {"checking": 1500.00, "savings": 5000.00},
            "transactions": [
                {"amount": 100.00, "description": "Grocery Store", "date": "2024-01-15"}
            ]
        }
        
        formatted = formatter.format_data_for_llm(data)
        assert "Account Balances:" in formatted
        assert "checking: $1,500.00" in formatted
        assert "Recent Transactions" in formatted


class TestResponseSynthesizer:
    """Test ResponseSynthesizer internal methods."""

    @pytest.fixture
    def synthesizer(self):
        """Create a ResponseSynthesizer instance."""
        return ResponseSynthesizer()

    @pytest.fixture
    def sample_state(self):
        """Create a sample OrchestratorState."""
        return OrchestratorState(
            user_input="Show me my account balances",
            intent=IntentType.DATA_FETCH,
            confidence=0.9,
            session_id=str(uuid4()),
            extracted_params={"account_name": "checking"}
        )

    def test_process_tool_results_empty(self, synthesizer):
        """Test processing empty tool results."""
        result = synthesizer._process_tool_results({})
        
        assert result["execution_summary"]["total_steps"] == 0
        assert result["execution_summary"]["successful_steps"] == 0
        assert result["execution_summary"]["failed_steps"] == 0
        assert not result["execution_summary"]["has_data"]

    def test_process_tool_results_success(self, synthesizer):
        """Test processing successful tool results."""
        tool_results = {
            "step_0": {
                "success": True,
                "data": {"balance": {"checking": 1500.00}},
                "step_index": 0,
                "tool_name": "db_query",
                "operation": "get_account_balance"
            }
        }
        
        result = synthesizer._process_tool_results(tool_results)
        
        assert result["execution_summary"]["total_steps"] == 1
        assert result["execution_summary"]["successful_steps"] == 1
        assert result["execution_summary"]["failed_steps"] == 0
        assert result["execution_summary"]["has_data"]
        assert "balance" in result["aggregated_data"]

    def test_process_tool_results_failure(self, synthesizer):
        """Test processing failed tool results."""
        tool_results = {
            "step_0": {
                "success": False,
                "error": "Account not found",
                "step_index": 0,
                "tool_name": "db_query",
                "operation": "get_account_balance"
            }
        }
        
        result = synthesizer._process_tool_results(tool_results)
        
        assert result["execution_summary"]["total_steps"] == 1
        assert result["execution_summary"]["successful_steps"] == 0
        assert result["execution_summary"]["failed_steps"] == 1
        assert not result["execution_summary"]["has_data"]
        assert len(result["failed_steps"]) == 1

    def test_aggregate_step_data(self, synthesizer):
        """Test data aggregation from steps."""
        aggregated_data = {}
        
        # Test balance data
        step_data = {"balance": {"checking": 1500.00}}
        synthesizer._aggregate_step_data(aggregated_data, step_data)
        
        assert "balance" in aggregated_data
        assert "balances" in aggregated_data
        assert aggregated_data["balance"]["checking"] == 1500.00

    @pytest.mark.asyncio
    async def test_generate_data_fetch_response_llm_success(self, synthesizer, sample_state):
        """Test successful LLM-based data fetch response generation."""
        processed_data = {
            "aggregated_data": {"balance": {"checking": 1500.00}},
            "execution_summary": {"has_data": True}
        }
        
        with patch.object(synthesizer.llm_client, 'generate_response') as mock_generate:
            mock_generate.return_value = "Here are your account balances: checking account has $1,500.00"
            
            response = await synthesizer._generate_data_fetch_response(sample_state, processed_data)
            
            assert "Here are your account balances" in response
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_data_fetch_response_llm_failure(self, synthesizer, sample_state):
        """Test fallback to template when LLM fails."""
        processed_data = {
            "aggregated_data": {"balance": {"checking": 1500.00}},
            "execution_summary": {"has_data": True}
        }
        
        with patch.object(synthesizer.llm_client, 'generate_response') as mock_generate:
            mock_generate.side_effect = Exception("LLM API error")
            
            response = await synthesizer._generate_data_fetch_response(sample_state, processed_data)
            
            assert "Here's the information you requested:" in response
            assert "checking: $1,500.00" in response

    @pytest.mark.asyncio
    async def test_generate_generic_response_llm_success(self, synthesizer, sample_state):
        """Test successful LLM-based generic response generation."""
        processed_data = {
            "aggregated_data": {"balance": {"checking": 1500.00}},
            "execution_summary": {"has_data": True}
        }
        
        with patch.object(synthesizer.llm_client, 'generate_response') as mock_generate:
            mock_generate.return_value = "I found your account information. Your checking account has $1,500.00."
            
            response = await synthesizer._generate_generic_response(sample_state, processed_data)
            
            assert "I found your account information" in response
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_generic_response_llm_failure(self, synthesizer, sample_state):
        """Test fallback when LLM fails for generic response."""
        processed_data = {
            "aggregated_data": {"balance": {"checking": 1500.00}},
            "execution_summary": {"has_data": True}
        }
        
        with patch.object(synthesizer.llm_client, 'generate_response') as mock_generate:
            mock_generate.side_effect = Exception("LLM API error")
            
            response = await synthesizer._generate_generic_response(sample_state, processed_data)
            
            assert "I've processed your request" in response
            assert "checking" in response

    def test_generate_action_error_response(self, synthesizer):
        """Test action error response generation."""
        processed_data = {
            "failed_steps": [
                {"step": "step_0", "error": "Insufficient funds"}
            ]
        }
        
        response = synthesizer._generate_action_error_response(processed_data)
        
        assert "requested action couldn't be completed" in response
        assert "Insufficient funds" in response

    def test_generate_fallback_response(self, synthesizer, sample_state):
        """Test fallback response generation."""
        # Test with no tool results
        response = synthesizer._generate_fallback_response(sample_state, {})
        assert "couldn't process your request" in response
        
        # Test with all failed steps
        tool_results = {
            "step_0": {"success": False},
            "step_1": {"success": False}
        }
        response = synthesizer._generate_fallback_response(sample_state, tool_results)
        assert "All operations failed" in response
        
        # Test with partial success
        tool_results = {
            "step_0": {"success": True},
            "step_1": {"success": False}
        }
        response = synthesizer._generate_fallback_response(sample_state, tool_results)
        assert "partially completed" in response


class TestResponseSynthesisNode:
    """Test the main response synthesis node function."""

    @pytest.fixture
    def sample_state(self):
        """Create a sample OrchestratorState."""
        return OrchestratorState(
            user_input="Show me my account balances",
            intent=IntentType.DATA_FETCH,
            confidence=0.9,
            session_id=str(uuid4()),
            extracted_params={"account_name": "checking"},
            tool_results={
                "step_0": {
                    "success": True,
                    "data": {"balance": {"checking": 1500.00}},
                    "step_index": 0,
                    "tool_name": "db_query",
                    "operation": "get_account_balance"
                }
            }
        )

    @pytest.mark.asyncio
    async def test_response_synthesis_node_success(self, sample_state):
        """Test successful response synthesis."""
        with patch('src.nodes.response_synthesis_node.ResponseSynthesizer') as mock_synthesizer_class:
            mock_synthesizer = AsyncMock()
            mock_synthesizer_class.return_value = mock_synthesizer
            mock_synthesizer.synthesize_response.return_value = "Here are your account balances: checking account has $1,500.00"
            
            result = await response_synthesis_node(sample_state)
            
            assert result.final_response == "Here are your account balances: checking account has $1,500.00"
            assert result.metadata["synthesis_status"] == "success"
            assert "synthesis_time_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_response_synthesis_node_no_tool_results(self, sample_state):
        """Test response synthesis with no tool results."""
        sample_state.tool_results = None
        
        result = await response_synthesis_node(sample_state)
        
        assert "couldn't process your request" in result.final_response
        assert result.metadata["synthesis_status"] == "error"

    @pytest.mark.asyncio
    async def test_response_synthesis_node_synthesis_failure(self, sample_state):
        """Test response synthesis when synthesis fails."""
        with patch('src.nodes.response_synthesis_node.ResponseSynthesizer') as mock_synthesizer_class:
            mock_synthesizer = MagicMock()
            mock_synthesizer_class.return_value = mock_synthesizer
            mock_synthesizer.synthesize_response.side_effect = Exception("Synthesis error")
            
            result = await response_synthesis_node(sample_state)
            
            assert "encountered an error" in result.final_response
            assert result.metadata["synthesis_status"] == "error"
            assert "Synthesis error" in result.metadata["synthesis_errors"][0]


class TestConstants:
    """Test the constants used in response synthesis."""

    def test_base_system_prompt(self):
        """Test base system prompt format."""
        assert "You are a helpful financial assistant" in BASE_SYSTEM_PROMPT
        assert "{intent_specific_guidelines}" in BASE_SYSTEM_PROMPT
        assert "Be conversational and helpful" in BASE_SYSTEM_PROMPT

    def test_intent_guidelines(self):
        """Test intent-specific guidelines."""
        assert IntentCategory.DATA_FETCH in INTENT_GUIDELINES
        assert IntentCategory.AGGREGATE in INTENT_GUIDELINES
        assert IntentCategory.ACTION in INTENT_GUIDELINES
        
        # Test that guidelines are descriptive
        assert "presenting data clearly" in INTENT_GUIDELINES[IntentCategory.DATA_FETCH]
        assert "insights, trends" in INTENT_GUIDELINES[IntentCategory.AGGREGATE]
        assert "confirming actions" in INTENT_GUIDELINES[IntentCategory.ACTION]
