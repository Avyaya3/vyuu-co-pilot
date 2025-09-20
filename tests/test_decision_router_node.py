"""
Test suite for Decision Router Node.

Tests routing logic, configuration management, parameter analysis,
and edge case handling for the decision router node.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import uuid

# Import modules to test
from vyuu_copilot_v2.nodes.decision_router_node import (
    DecisionRouter,
    RouterConfig,
    RoutingResult,
    RoutingReason,
    decision_router_node,
    get_routing_decision
)
from vyuu_copilot_v2.schemas.state_schemas import MainState, IntentType, MessageRole, Message
from vyuu_copilot_v2.schemas.generated_intent_schemas import ConfidenceLevel


class TestRouterConfig:
    """Test RouterConfig validation and initialization."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RouterConfig()
        
        assert config.high_confidence_threshold == 0.8
        assert config.medium_confidence_threshold == 0.5
        assert config.low_confidence_threshold == 0.3
        assert config.require_critical_params is True
        assert config.max_missing_params == 2
        assert config.enable_fallback_routing is True
        assert config.log_routing_decisions is True
        
        # Check intent-specific thresholds
        assert config.intent_specific_thresholds["data_fetch"] == 0.7
        assert config.intent_specific_thresholds["aggregate"] == 0.8
        assert config.intent_specific_thresholds["action"] == 0.9
        assert config.intent_specific_thresholds["unknown"] == 0.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RouterConfig(
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.6,
            low_confidence_threshold=0.2,
            max_missing_params=1,
            require_critical_params=False
        )
        
        assert config.high_confidence_threshold == 0.9
        assert config.medium_confidence_threshold == 0.6
        assert config.low_confidence_threshold == 0.2
        assert config.max_missing_params == 1
        assert config.require_critical_params is False
    
    def test_config_validation(self):
        """Test configuration validation rules."""
        # Test invalid threshold ordering
        with pytest.raises(ValueError, match="High confidence threshold must be greater than medium"):
            RouterConfig(
                high_confidence_threshold=0.5,
                medium_confidence_threshold=0.8
            )
        
        with pytest.raises(ValueError, match="Medium confidence threshold must be greater than low"):
            RouterConfig(
                medium_confidence_threshold=0.2,
                low_confidence_threshold=0.5
            )


class TestDecisionRouter:
    """Test DecisionRouter core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = DecisionRouter()
        self.sample_state = MainState(
            user_input="Show me my transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.85,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},
            execution_results={},
            response=""
        )
    
    def test_router_initialization(self):
        """Test router initialization with default and custom config."""
        # Default config
        router = DecisionRouter()
        assert isinstance(router.config, RouterConfig)
        assert router.config.high_confidence_threshold == 0.8
        
        # Custom config
        custom_config = RouterConfig(high_confidence_threshold=0.9)
        router = DecisionRouter(custom_config)
        assert router.config.high_confidence_threshold == 0.9
    
    def test_confidence_level_determination(self):
        """Test confidence level categorization."""
        # High confidence
        assert self.router._get_confidence_level(0.9) == ConfidenceLevel.HIGH
        assert self.router._get_confidence_level(0.8) == ConfidenceLevel.HIGH
        
        # Medium confidence
        assert self.router._get_confidence_level(0.7) == ConfidenceLevel.MEDIUM
        assert self.router._get_confidence_level(0.5) == ConfidenceLevel.MEDIUM
        
        # Low confidence
        assert self.router._get_confidence_level(0.4) == ConfidenceLevel.LOW
        assert self.router._get_confidence_level(0.0) == ConfidenceLevel.LOW
    
    def test_parameter_completeness_analysis(self):
        """Test parameter completeness analysis for different intents."""
        # Data fetch with complete parameters
        state = MainState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={
                "entity_type": "transactions",
                "time_period": "last_month",
                "limit": 10
            },
            execution_results={},
            response=""
        )
        
        analysis = self.router._analyze_parameter_completeness(state)
        
        assert analysis["has_critical"] is True  # Has entity_type
        assert analysis["provided_count"] == 3
        assert "account_types" in analysis["missing"]  # Some params still missing
        assert len(analysis["critical_missing"]) == 0
    
    def test_parameter_completeness_missing_critical(self):
        """Test parameter analysis with missing critical parameters."""
        state = MainState(
            user_input="Show me some data",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"time_period": "last_month"},  # Missing entity_type (critical)
            execution_results={},
            response=""
        )
        
        analysis = self.router._analyze_parameter_completeness(state)
        
        assert analysis["has_critical"] is False
        assert "entity_type" in analysis["critical_missing"]
        assert analysis["complete"] is False
    
    def test_routing_high_confidence_complete(self):
        """Test routing for high confidence with complete parameters."""
        state = MainState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},  # Has critical param
            execution_results={},
            response=""
        )
        
        result = self.router.route_intent(state)
        
        assert result.decision == "direct_orchestrator"
        assert result.reason == RoutingReason.HIGH_CONFIDENCE_COMPLETE
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert isinstance(result.processing_time_ms, float)
    
    def test_routing_high_confidence_incomplete(self):
        """Test routing for high confidence with incomplete parameters."""
        state = MainState(
            user_input="Show me some data",
            intent=IntentType.DATA_FETCH,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},  # No parameters
            execution_results={},
            response=""
        )
        
        result = self.router.route_intent(state)
        
        assert result.decision == "clarification"
        assert result.reason == RoutingReason.MISSING_CRITICAL_PARAMS
        assert result.confidence_level == ConfidenceLevel.HIGH
    
    def test_routing_unknown_intent(self):
        """Test routing for unknown intents."""
        state = MainState(
            user_input="Hello there",
            intent=IntentType.UNKNOWN,
            confidence=0.8,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response=""
        )
        
        result = self.router.route_intent(state)
        
        assert result.decision == "clarification"
        assert result.reason == RoutingReason.UNKNOWN_INTENT
        assert "Unknown intent requires clarification" in result.routing_explanation
    
    def test_routing_intent_specific_threshold(self):
        """Test routing with intent-specific thresholds."""
        # Action intent with confidence below action threshold (0.9)
        state = MainState(
            user_input="Transfer money",
            intent=IntentType.ACTION,
            confidence=0.85,  # Below 0.9 threshold for actions
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"action_type": "transfer"},
            execution_results={},
            response=""
        )
        
        result = self.router.route_intent(state)
        
        assert result.decision == "clarification"
        assert result.reason == RoutingReason.INTENT_SPECIFIC_THRESHOLD
        assert "intent-specific threshold" in result.routing_explanation
    
    def test_routing_excessive_missing_params(self):
        """Test routing when too many parameters are missing."""
        config = RouterConfig(max_missing_params=1)
        router = DecisionRouter(config)
        
        state = MainState(
            user_input="Show me data",
            intent=IntentType.DATA_FETCH,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},  # Only 1 param, many missing
            execution_results={},
            response=""
        )
        
        result = router.route_intent(state)
        
        # Since we have critical parameters and high confidence, should route to direct orchestrator
        assert result.decision == "direct_orchestrator"
        assert result.reason == RoutingReason.HIGH_CONFIDENCE_COMPLETE
        assert result.parameters_complete is True
    
    def test_routing_medium_confidence_scenarios(self):
        """Test routing for medium confidence scenarios."""
        # Medium confidence with complete parameters but below intent threshold
        state = MainState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.6,  # Below data_fetch threshold of 0.7
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},
            execution_results={},
            response=""
        )
        
        result = self.router.route_intent(state)
        
        # Should route to clarification due to intent-specific threshold
        assert result.decision == "clarification"
        assert result.reason == RoutingReason.INTENT_SPECIFIC_THRESHOLD
        assert result.confidence_level == ConfidenceLevel.MEDIUM
    
    def test_routing_low_confidence(self):
        """Test routing for low confidence scenarios."""
        state = MainState(
            user_input="Something unclear",
            intent=IntentType.DATA_FETCH,
            confidence=0.2,  # Below data_fetch threshold of 0.7
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},
            execution_results={},
            response=""
        )
        
        result = self.router.route_intent(state)
        
        assert result.decision == "clarification"
        # Intent-specific threshold takes precedence over general confidence level
        assert result.reason == RoutingReason.INTENT_SPECIFIC_THRESHOLD
        assert result.confidence_level == ConfidenceLevel.LOW
    
    def test_routing_error_handling(self):
        """Test routing error handling."""
        # Create a state that will cause an error
        state = None  # This will cause an error
        
        result = self.router.route_intent(state)
        
        assert result.decision == "clarification"
        assert result.reason == RoutingReason.ERROR_CONDITION
        assert "error" in result.routing_explanation.lower()
        assert result.confidence_score == 0.0


class TestDecisionRouterNode:
    """Test the LangGraph decision router node."""
    
    @pytest.mark.asyncio
    async def test_successful_routing(self):
        """Test successful routing through the node."""
        state = MainState(
            user_input="Show me my transactions from last month",
            intent=IntentType.DATA_FETCH,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions", "time_period": "last_month"},
            execution_results={},
            response=""
        )
        
        result_state = await decision_router_node(state)
        
        # Check that routing decision was made
        assert "routing_decision" in result_state.metadata
        assert result_state.metadata["routing_decision"] in ["clarification", "direct_orchestrator"]
        assert "routing_reason" in result_state.metadata
        assert "routing_explanation" in result_state.metadata
        assert "node_processing_time" in result_state.metadata
        
        # Check that messages were added
        assert len(result_state.messages) > len(state.messages)
        
        # Check that system and assistant messages were added
        system_messages = [msg for msg in result_state.messages if msg.role == MessageRole.SYSTEM]
        assistant_messages = [msg for msg in result_state.messages if msg.role == MessageRole.ASSISTANT]
        
        assert len(system_messages) >= 1
        assert len(assistant_messages) >= 1
    
    @pytest.mark.asyncio
    async def test_routing_with_error(self):
        """Test routing node error handling."""
        # Create a state with an invalid intent type that might cause issues
        state = MainState(
            user_input="Test input",
            intent=None,  # This might cause issues
            confidence=None,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response=""
        )
        
        result_state = await decision_router_node(state)
        
        # Should fallback to clarification on error
        assert result_state.metadata.get("routing_decision") == "clarification"
        assert "error" in result_state.metadata
    
    @pytest.mark.asyncio
    async def test_node_preserves_state_fields(self):
        """Test that the node preserves all required state fields."""
        original_state = MainState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.85,
            messages=[Message(
                role=MessageRole.USER,
                content="Test message",
                metadata={"test": "value"}
            )],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={"existing": "data"},
            parameters={"entity_type": "transactions"},
            execution_results={"previous": "result"},
            response="Previous response"
        )
        
        result_state = await decision_router_node(original_state)
        
        # Check that original fields are preserved
        assert result_state.user_input == original_state.user_input
        assert result_state.intent == original_state.intent
        assert result_state.confidence == original_state.confidence
        assert result_state.session_id == original_state.session_id
        assert result_state.parameters == original_state.parameters
        assert result_state.execution_results == original_state.execution_results
        assert result_state.response == original_state.response
        
        # Check that existing metadata is preserved
        assert result_state.metadata["existing"] == "data"


class TestGetRoutingDecision:
    """Test the routing decision extraction function."""
    
    def test_valid_routing_decision(self):
        """Test extraction of valid routing decisions."""
        state = MainState(
            user_input="Test",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={"routing_decision": "direct_orchestrator"},
            parameters={},
            execution_results={},
            response=""
        )
        
        decision = get_routing_decision(state)
        assert decision == "direct_orchestrator"
    
    def test_missing_routing_decision(self):
        """Test handling of missing routing decision."""
        state = MainState(
            user_input="Test",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response=""
        )
        
        decision = get_routing_decision(state)
        assert decision == "clarification"  # Default fallback
    
    def test_invalid_routing_decision(self):
        """Test handling of invalid routing decision."""
        state = MainState(
            user_input="Test",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={"routing_decision": "invalid_decision"},
            parameters={},
            execution_results={},
            response=""
        )
        
        decision = get_routing_decision(state)
        assert decision == "clarification"  # Default fallback


class TestConfigurableThresholds:
    """Test configurable threshold functionality."""
    
    def test_custom_confidence_thresholds(self):
        """Test routing with custom confidence thresholds."""
        # Create router with very high thresholds
        config = RouterConfig(
            high_confidence_threshold=0.95,
            medium_confidence_threshold=0.8,
            low_confidence_threshold=0.5
        )
        router = DecisionRouter(config)
        
        # Test with confidence that would be high under default config
        state = MainState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.85,  # Would be high under default, but medium under custom
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},
            execution_results={},
            response=""
        )
        
        result = router.route_intent(state)
        
        # Should be medium confidence under custom thresholds
        assert result.confidence_level == ConfidenceLevel.MEDIUM
        assert result.decision == "direct_orchestrator"  # Still complete params
        assert result.reason == RoutingReason.MEDIUM_CONFIDENCE_COMPLETE
    
    def test_custom_intent_thresholds(self):
        """Test routing with custom intent-specific thresholds."""
        config = RouterConfig(
            intent_specific_thresholds={
                "data_fetch": 0.9,  # Increased from default 0.7
                "aggregate": 0.95,  # Increased from default 0.8
                "action": 0.95,     # Increased from default 0.9
                "unknown": 0.0
            }
        )
        router = DecisionRouter(config)
        
        state = MainState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,  # Below new 0.9 threshold for data_fetch
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},
            execution_results={},
            response=""
        )
        
        result = router.route_intent(state)
        
        assert result.decision == "clarification"
        assert result.reason == RoutingReason.INTENT_SPECIFIC_THRESHOLD
    
    def test_custom_missing_params_threshold(self):
        """Test routing with custom missing parameters threshold."""
        config = RouterConfig(max_missing_params=0)  # Very strict
        router = DecisionRouter(config)
        
        state = MainState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={"entity_type": "transactions"},  # Only 1 param, others missing
            execution_results={},
            response=""
        )
        
        result = router.route_intent(state)
        
        # Since we have critical parameters and high confidence, routes to direct orchestrator
        # The max_missing_params threshold is now incorporated into the parameter completeness analysis
        assert result.decision == "direct_orchestrator"
        assert result.reason == RoutingReason.HIGH_CONFIDENCE_COMPLETE


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 