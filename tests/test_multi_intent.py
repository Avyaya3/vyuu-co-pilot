"""
Comprehensive tests for multi-intent functionality.

Tests cover:
- Multi-intent classification (read + advice)
- Multi-intent routing decisions
- Parallel execution (read + advice)
- Sequential execution (database_operations + read)
- Mixed execution (all three intent types)
- Error handling in multi-intent scenarios
- Backward compatibility with single intents
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from src.vyuu_copilot_v2.schemas.state_schemas import MainState, OrchestratorState
from src.vyuu_copilot_v2.schemas.generated_intent_schemas import (
    IntentEntry, IntentCategory, IntentClassificationResult
)
from src.vyuu_copilot_v2.nodes.intent_classification_node import IntentClassifier
from src.vyuu_copilot_v2.nodes.decision_router_node import DecisionRouter
from src.vyuu_copilot_v2.nodes.execution_planner_node import (
    can_execute_parallel, execute_multiple_intents, combine_intent_results
)
from src.vyuu_copilot_v2.nodes.response_synthesis_node import synthesize_multi_intent_response


class TestMultiIntentClassification:
    """Test multi-intent classification functionality."""
    
    @pytest.fixture
    def sample_multi_intent_llm_response(self):
        """Sample LLM response with multiple intents."""
        return {
            "multiple_intents": [
                {
                    "intent": "read",
                    "confidence": 0.9,
                    "reasoning": "User wants to view financial data",
                    "params": {
                        "entity_type": "expenses",
                        "time_period": "last_month"
                    }
                },
                {
                    "intent": "advice",
                    "confidence": 0.85,
                    "reasoning": "User wants budgeting advice",
                    "params": {
                        "user_query": "budgeting advice"
                    }
                }
            ]
        }
    
    @pytest.fixture
    def sample_single_intent_llm_response(self):
        """Sample LLM response with single intent."""
        return {
            "intent": "read",
            "confidence": 0.9,
            "reasoning": "User wants to view financial data",
            "read_params": {
                "entity_type": "expenses",
                "time_period": "last_month"
            }
        }
    
    @pytest.mark.asyncio
    async def test_parse_multiple_intents(self, sample_multi_intent_llm_response):
        """Test parsing multiple intents from LLM response."""
        classifier = IntentClassifier()
        
        result = classifier._parse_multiple_intents(sample_multi_intent_llm_response, "test input")
        
        assert result.intent == IntentCategory.READ  # Primary intent (highest confidence)
        assert result.confidence == 0.9
        assert hasattr(result, 'multiple_intents')
        assert len(result.multiple_intents) == 2
        
        # Check individual intents
        intents = result.multiple_intents
        assert intents[0].intent == "read"
        assert intents[0].confidence == 0.9
        assert intents[1].intent == "advice"
        assert intents[1].confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_parse_single_intent(self, sample_single_intent_llm_response):
        """Test parsing single intent from LLM response."""
        classifier = IntentClassifier()
        
        result = classifier._parse_single_intent(sample_single_intent_llm_response, "test input")
        
        assert result.intent == IntentCategory.READ
        assert result.confidence == 0.9
        assert not hasattr(result, 'multiple_intents') or result.multiple_intents is None
    
    @pytest.mark.asyncio
    async def test_classify_intent_with_mock_llm(self):
        """Test intent classification with mocked LLM response."""
        classifier = IntentClassifier()
        
        # Mock the LLM client
        with patch.object(classifier.llm_client, 'chat_completion') as mock_llm:
            mock_llm.return_value = '{"multiple_intents": [{"intent": "read", "confidence": 0.9, "reasoning": "test", "params": {}}, {"intent": "advice", "confidence": 0.8, "reasoning": "test", "params": {}}]}'
            
            result = await classifier.classify_intent("Show me my expenses and give me advice")
            
            assert result.intent == IntentCategory.READ
            assert result.confidence == 0.9
            assert hasattr(result, 'multiple_intents')
            assert len(result.multiple_intents) == 2


class TestMultiIntentRouting:
    """Test multi-intent routing functionality."""
    
    @pytest.fixture
    def sample_multiple_intents(self):
        """Sample multiple intents for testing."""
        return [
            IntentEntry(intent="read", confidence=0.9, reasoning="test", params={}),
            IntentEntry(intent="advice", confidence=0.85, reasoning="test", params={})
        ]
    
    @pytest.fixture
    def sample_low_confidence_intents(self):
        """Sample intents with low confidence."""
        return [
            IntentEntry(intent="read", confidence=0.6, reasoning="test", params={}),
            IntentEntry(intent="advice", confidence=0.5, reasoning="test", params={})
        ]
    
    def test_route_multiple_intents_high_confidence(self, sample_multiple_intents):
        """Test routing multiple intents with high confidence."""
        router = DecisionRouter()
        
        result = router.route_multiple_intents(sample_multiple_intents)
        
        assert result["decision"] == "direct_orchestrator"
        assert result["reason"].value == "high_confidence_complete"
        assert "All 2 intents are clear" in result["explanation"]
    
    def test_route_multiple_intents_low_confidence(self, sample_low_confidence_intents):
        """Test routing multiple intents with low confidence."""
        router = DecisionRouter()
        
        result = router.route_multiple_intents(sample_low_confidence_intents)
        
        assert result["decision"] == "clarification"
        assert result["reason"].value == "low_confidence"
        assert "low confidence" in result["explanation"]
    
    def test_route_multiple_intents_with_unknown(self):
        """Test routing multiple intents with unknown intent."""
        router = DecisionRouter()
        intents = [
            IntentEntry(intent="read", confidence=0.9, reasoning="test", params={}),
            IntentEntry(intent="unknown", confidence=0.5, reasoning="test", params={})
        ]
        
        result = router.route_multiple_intents(intents)
        
        assert result["decision"] == "clarification"
        assert result["reason"].value == "unknown_intent"
        assert "unknown" in result["explanation"]


class TestMultiIntentExecution:
    """Test multi-intent execution functionality."""
    
    @pytest.fixture
    def sample_orchestrator_state(self):
        """Sample orchestrator state with multiple intents."""
        return OrchestratorState(
            user_input="Show expenses and give advice",
            intent="read",
            confidence=0.9,
            messages=[],
            session_id="test-session",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results=None,
            response=None,
            extracted_params={},
            execution_plan=None,
            tool_results=None,
            final_response=None,
            multiple_intents=[
                IntentEntry(intent="read", confidence=0.9, reasoning="test", params={"entity_type": "expenses"}),
                IntentEntry(intent="advice", confidence=0.85, reasoning="test", params={"user_query": "advice"})
            ]
        )
    
    def test_can_execute_parallel(self):
        """Test parallel execution capability detection."""
        read_intent = IntentEntry(intent="read", confidence=0.9, reasoning="test", params={})
        advice_intent = IntentEntry(intent="advice", confidence=0.8, reasoning="test", params={})
        db_intent = IntentEntry(intent="database_operations", confidence=0.9, reasoning="test", params={})
        
        assert can_execute_parallel(read_intent) == True
        assert can_execute_parallel(advice_intent) == True
        assert can_execute_parallel(db_intent) == False
    
    @pytest.mark.asyncio
    async def test_execute_multiple_intents_parallel(self, sample_orchestrator_state):
        """Test executing multiple intents in parallel."""
        with patch('src.vyuu_copilot_v2.nodes.execution_planner_node.execute_single_intent_from_classification') as mock_execute:
            mock_execute.return_value = {
                "intent": "read",
                "confidence": 0.9,
                "success": True,
                "summary": "Data retrieved successfully",
                "data": {}
            }
            
            result_state = await execute_multiple_intents(sample_orchestrator_state)
            
            assert result_state.execution_results is not None
            assert result_state.execution_results["successful_count"] == 2
            assert result_state.execution_results["failed_count"] == 0
            assert "Successfully completed" in result_state.response
    
    @pytest.mark.asyncio
    async def test_execute_multiple_intents_mixed(self):
        """Test executing mixed parallel and sequential intents."""
        state = OrchestratorState(
            user_input="Show expenses, transfer money, and give advice",
            intent="read",
            confidence=0.9,
            messages=[],
            session_id="test-session",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results=None,
            response=None,
            extracted_params={},
            execution_plan=None,
            tool_results=None,
            final_response=None,
            multiple_intents=[
                IntentEntry(intent="read", confidence=0.9, reasoning="test", params={}),
                IntentEntry(intent="database_operations", confidence=0.9, reasoning="test", params={}),
                IntentEntry(intent="advice", confidence=0.8, reasoning="test", params={})
            ]
        )
        
        with patch('src.vyuu_copilot_v2.nodes.execution_planner_node.execute_single_intent_from_classification') as mock_execute:
            mock_execute.return_value = {
                "intent": "read",
                "confidence": 0.9,
                "success": True,
                "summary": "Executed successfully",
                "data": {}
            }
            
            result_state = await execute_multiple_intents(state)
            
            assert result_state.execution_results is not None
            assert result_state.execution_results["successful_count"] == 3
            assert result_state.execution_results["failed_count"] == 0
    
    def test_combine_intent_results(self, sample_orchestrator_state):
        """Test combining results from multiple intent executions."""
        results = [
            {"intent": "read", "success": True, "summary": "Data retrieved"},
            {"intent": "advice", "success": True, "summary": "Advice provided"},
            {"intent": "database_operations", "success": False, "error": "Insufficient funds"}
        ]
        
        result_state = combine_intent_results(sample_orchestrator_state, results)
        
        assert result_state.execution_results["successful_count"] == 2
        assert result_state.execution_results["failed_count"] == 1
        assert "Successfully completed" in result_state.response
        assert "Issues encountered" in result_state.response


class TestMultiIntentResponseSynthesis:
    """Test multi-intent response synthesis functionality."""
    
    @pytest.fixture
    def sample_orchestrator_state_with_results(self):
        """Sample orchestrator state with execution results."""
        return OrchestratorState(
            user_input="Show expenses and give advice",
            intent="read",
            confidence=0.9,
            messages=[],
            session_id="test-session",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={
                "multiple_intent_results": [
                    {"intent": "read", "success": True, "summary": "Retrieved expense data"},
                    {"intent": "advice", "success": True, "summary": "Provided budgeting advice"}
                ],
                "successful_count": 2,
                "failed_count": 0,
                "execution_strategy": "success"
            },
            response=None,
            extracted_params={},
            execution_plan=None,
            tool_results=None,
            final_response=None,
            multiple_intents=[
                IntentEntry(intent="read", confidence=0.9, reasoning="test", params={}),
                IntentEntry(intent="advice", confidence=0.85, reasoning="test", params={})
            ]
        )
    
    @pytest.mark.asyncio
    async def test_synthesize_multi_intent_response_success(self, sample_orchestrator_state_with_results):
        """Test synthesizing response for successful multi-intent execution."""
        result_state = await synthesize_multi_intent_response(sample_orchestrator_state_with_results)
        
        assert result_state.final_response is not None
        assert "✅ I've successfully completed all 2 requested actions" in result_state.final_response
        assert "📊 **Data Retrieved:**" in result_state.final_response
        assert "💡 **Financial Advice:**" in result_state.final_response
        assert "Retrieved expense data" in result_state.final_response
        assert "Provided budgeting advice" in result_state.final_response
    
    @pytest.mark.asyncio
    async def test_synthesize_multi_intent_response_mixed(self):
        """Test synthesizing response for mixed success/failure results."""
        state = OrchestratorState(
            user_input="Show expenses, transfer money, and give advice",
            intent="read",
            confidence=0.9,
            messages=[],
            session_id="test-session",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={
                "multiple_intent_results": [
                    {"intent": "read", "success": True, "summary": "Retrieved expense data"},
                    {"intent": "database_operations", "success": False, "error": "Insufficient funds"},
                    {"intent": "advice", "success": True, "summary": "Provided budgeting advice"}
                ],
                "successful_count": 2,
                "failed_count": 1,
                "execution_strategy": "mixed"
            },
            response=None,
            extracted_params={},
            execution_plan=None,
            tool_results=None,
            final_response=None,
            multiple_intents=[
                IntentEntry(intent="read", confidence=0.9, reasoning="test", params={}),
                IntentEntry(intent="database_operations", confidence=0.9, reasoning="test", params={}),
                IntentEntry(intent="advice", confidence=0.8, reasoning="test", params={})
            ]
        )
        
        result_state = await synthesize_multi_intent_response(state)
        
        assert result_state.final_response is not None
        assert "I've processed your request with 3 actions" in result_state.final_response
        assert "📊 **Data Retrieved:**" in result_state.final_response
        assert "💾 **Database Operations:**" in result_state.final_response
        assert "💡 **Financial Advice:**" in result_state.final_response
        assert "❌ Insufficient funds" in result_state.final_response


class TestBackwardCompatibility:
    """Test backward compatibility with single intent flows."""
    
    def test_mainstate_has_multiple_intents_property(self):
        """Test MainState has_multiple_intents property."""
        # Single intent
        state = MainState(
            user_input="Show my expenses",
            intent="read",
            confidence=0.9,
            messages=[],
            session_id="test-session",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results=None,
            response=None,
            multiple_intents=None
        )
        assert not state.has_multiple_intents
        
        # Multiple intents
        state.multiple_intents = [
            IntentEntry(intent="read", confidence=0.9, reasoning="test", params={}),
            IntentEntry(intent="advice", confidence=0.8, reasoning="test", params={})
        ]
        assert state.has_multiple_intents
    
    def test_mainstate_primary_intent_property(self):
        """Test MainState primary_intent property."""
        # Single intent
        state = MainState(
            user_input="Show my expenses",
            intent="read",
            confidence=0.9,
            messages=[],
            session_id="test-session",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results=None,
            response=None,
            multiple_intents=None
        )
        assert state.primary_intent == "read"
        
        # Multiple intents (should return highest confidence)
        state.multiple_intents = [
            IntentEntry(intent="advice", confidence=0.8, reasoning="test", params={}),
            IntentEntry(intent="read", confidence=0.9, reasoning="test", params={})
        ]
        assert state.primary_intent == "read"  # Higher confidence
    
    def test_mainstate_primary_confidence_property(self):
        """Test MainState primary_confidence property."""
        # Single intent
        state = MainState(
            user_input="Show my expenses",
            intent="read",
            confidence=0.9,
            messages=[],
            session_id="test-session",
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results=None,
            response=None,
            multiple_intents=None
        )
        assert state.primary_confidence == 0.9
        
        # Multiple intents (should return highest confidence)
        state.multiple_intents = [
            IntentEntry(intent="advice", confidence=0.8, reasoning="test", params={}),
            IntentEntry(intent="read", confidence=0.9, reasoning="test", params={})
        ]
        assert state.primary_confidence == 0.9  # Higher confidence


if __name__ == "__main__":
    pytest.main([__file__])
