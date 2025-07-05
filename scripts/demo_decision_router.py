"""
Demonstration of Decision Router Node.

This script demonstrates the decision router's ability to route classified intents
to appropriate subgraphs based on confidence thresholds and parameter completeness.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nodes.decision_router_node import (
    DecisionRouter,
    RouterConfig,
    decision_router_node,
    get_routing_decision
)
from schemas.state_schemas import MainState, IntentType
from schemas.intent_schemas import ConfidenceLevel


async def demo_basic_routing_scenarios():
    """Demonstrate basic routing scenarios with different confidence levels."""
    
    print("🎯 Basic Routing Scenarios Demo")
    print("=" * 50)
    
    router = DecisionRouter()
    
    test_cases = [
        {
            "name": "High Confidence + Complete Parameters",
            "intent": IntentType.DATA_FETCH,
            "confidence": 0.95,
            "parameters": {
                "entity_type": "transactions",
                "time_period": "last_month",
                "limit": 10
            },
            "expected": "direct_orchestrator"
        },
        {
            "name": "High Confidence + Missing Critical Parameters",
            "intent": IntentType.DATA_FETCH,
            "confidence": 0.95,
            "parameters": {
                "time_period": "last_month"
                # Missing entity_type (critical)
            },
            "expected": "clarification"
        },
        {
            "name": "Medium Confidence + Complete Parameters",
            "intent": IntentType.AGGREGATE,
            "confidence": 0.65,
            "parameters": {
                "metric_type": "sum",
                "group_by": ["category"],
                "time_period": "last_year"
            },
            "expected": "direct_orchestrator"
        },
        {
            "name": "Medium Confidence + Incomplete Parameters",
            "intent": IntentType.AGGREGATE,
            "confidence": 0.65,
            "parameters": {
                "group_by": ["category"]
                # Missing metric_type (critical)
            },
            "expected": "clarification"
        },
        {
            "name": "Low Confidence",
            "intent": IntentType.ACTION,
            "confidence": 0.3,
            "parameters": {
                "action_type": "transfer",
                "amount": 100.0
            },
            "expected": "clarification"
        },
        {
            "name": "Unknown Intent",
            "intent": IntentType.UNKNOWN,
            "confidence": 0.8,
            "parameters": {},
            "expected": "clarification"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Intent: {test_case['intent'].value}")
        print(f"Confidence: {test_case['confidence']:.0%}")
        print(f"Parameters: {len(test_case['parameters'])} provided")
        print("-" * 30)
        
        # Create state for testing
        state = MainState(
            user_input=f"Test case {i}",
            intent=test_case['intent'],
            confidence=test_case['confidence'],
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters=test_case['parameters'],
            execution_results={},
            response=""
        )
        
        # Make routing decision
        result = router.route_intent(state)
        
        # Display results
        expected_symbol = "✅" if result.decision == test_case['expected'] else "❌"
        print(f"{expected_symbol} Decision: {result.decision}")
        print(f"   Reason: {result.reason.value}")
        print(f"   Confidence Level: {result.confidence_level.value}")
        print(f"   Parameters Complete: {result.parameters_complete}")
        print(f"   Explanation: {result.routing_explanation}")
        
        if result.missing_params:
            print(f"   Missing Parameters: {result.missing_params}")
        
        if result.critical_params_missing:
            print(f"   Critical Missing: {result.critical_params_missing}")
        
        print(f"   Processing Time: {result.processing_time_ms:.2f}ms")


async def demo_configurable_thresholds():
    """Demonstrate configurable threshold functionality."""
    
    print("\n\n⚙️ Configurable Thresholds Demo")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {
            "name": "Default Configuration",
            "config": RouterConfig()
        },
        {
            "name": "Strict Configuration",
            "config": RouterConfig(
                high_confidence_threshold=0.95,
                medium_confidence_threshold=0.8,
                max_missing_params=0,
                intent_specific_thresholds={
                    "data_fetch": 0.9,
                    "aggregate": 0.95,
                    "action": 0.98,
                    "unknown": 0.0
                }
            )
        },
        {
            "name": "Permissive Configuration",
            "config": RouterConfig(
                high_confidence_threshold=0.6,
                medium_confidence_threshold=0.4,
                low_confidence_threshold=0.2,
                max_missing_params=5,
                require_critical_params=False,
                intent_specific_thresholds={
                    "data_fetch": 0.3,
                    "aggregate": 0.4,
                    "action": 0.5,
                    "unknown": 0.0
                }
            )
        }
    ]
    
    # Test state that will behave differently under different configs
    test_state = MainState(
        user_input="Transfer $500 to savings",
        intent=IntentType.ACTION,
        confidence=0.85,
        messages=[],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={
            "action_type": "transfer",
            "amount": 500.0
            # Missing source_account, target_account
        },
        execution_results={},
        response=""
    )
    
    print(f"\n🧪 Test Scenario:")
    print(f"Intent: {test_state.intent.value}")
    print(f"Confidence: {test_state.confidence:.0%}")
    print(f"Parameters: {list(test_state.parameters.keys())}")
    
    for config_info in configs:
        print(f"\n📊 {config_info['name']}:")
        print("-" * 30)
        
        router = DecisionRouter(config_info['config'])
        result = router.route_intent(test_state)
        
        print(f"Decision: {result.decision}")
        print(f"Reason: {result.reason.value}")
        print(f"Explanation: {result.routing_explanation}")
        
        # Show key config settings that influenced decision
        config = config_info['config']
        if result.reason.value == "intent_specific_threshold":
            action_threshold = config.intent_specific_thresholds.get("action", "N/A")
            print(f"Action Threshold Used: {action_threshold}")
        
        print(f"High Confidence Threshold: {config.high_confidence_threshold}")
        print(f"Max Missing Params: {config.max_missing_params}")
        print(f"Require Critical Params: {config.require_critical_params}")


async def demo_intent_specific_routing():
    """Demonstrate intent-specific routing behavior."""
    
    print("\n\n🎭 Intent-Specific Routing Demo")
    print("=" * 50)
    
    router = DecisionRouter()
    
    # Same confidence, different intents
    base_confidence = 0.85
    
    intent_scenarios = [
        {
            "intent": IntentType.DATA_FETCH,
            "parameters": {"entity_type": "transactions"},
            "description": "Data fetch with moderate confidence"
        },
        {
            "intent": IntentType.AGGREGATE,
            "parameters": {"metric_type": "sum"},
            "description": "Aggregate with moderate confidence"
        },
        {
            "intent": IntentType.ACTION,
            "parameters": {"action_type": "transfer"},
            "description": "Action with moderate confidence"
        }
    ]
    
    print(f"Testing all intents with {base_confidence:.0%} confidence:")
    
    for scenario in intent_scenarios:
        print(f"\n🔍 {scenario['description']}")
        print(f"Intent: {scenario['intent'].value}")
        print("-" * 25)
        
        state = MainState(
            user_input="Test intent-specific routing",
            intent=scenario['intent'],
            confidence=base_confidence,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters=scenario['parameters'],
            execution_results={},
            response=""
        )
        
        result = router.route_intent(state)
        
        print(f"Decision: {result.decision}")
        print(f"Reason: {result.reason.value}")
        
        # Show intent-specific threshold
        intent_key = scenario['intent'].value
        intent_threshold = router.config.intent_specific_thresholds.get(intent_key, "N/A")
        print(f"Intent Threshold: {intent_threshold}")
        print(f"Confidence vs Threshold: {base_confidence:.2f} vs {intent_threshold}")
        
        if result.reason.value == "intent_specific_threshold":
            print("❌ Below intent-specific threshold")
        else:
            print("✅ Above intent-specific threshold")


async def demo_parameter_analysis():
    """Demonstrate parameter completeness analysis."""
    
    print("\n\n📊 Parameter Analysis Demo")
    print("=" * 50)
    
    router = DecisionRouter()
    
    parameter_scenarios = [
        {
            "name": "Complete Parameters",
            "intent": IntentType.DATA_FETCH,
            "parameters": {
                "entity_type": "transactions",
                "time_period": "last_month",
                "account_types": ["checking"],
                "limit": 10,
                "sort_by": "date",
                "order": "desc"
            }
        },
        {
            "name": "Critical Parameters Only",
            "intent": IntentType.DATA_FETCH,
            "parameters": {
                "entity_type": "transactions"
            }
        },
        {
            "name": "Missing Critical Parameters",
            "intent": IntentType.DATA_FETCH,
            "parameters": {
                "time_period": "last_month",
                "limit": 10
            }
        },
        {
            "name": "No Parameters",
            "intent": IntentType.DATA_FETCH,
            "parameters": {}
        }
    ]
    
    for scenario in parameter_scenarios:
        print(f"\n📋 {scenario['name']}")
        print(f"Intent: {scenario['intent'].value}")
        print(f"Parameters Provided: {list(scenario['parameters'].keys())}")
        print("-" * 30)
        
        state = MainState(
            user_input="Parameter analysis test",
            intent=scenario['intent'],
            confidence=0.8,  # Consistent confidence
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters=scenario['parameters'],
            execution_results={},
            response=""
        )
        
        # Analyze parameters
        param_analysis = router._analyze_parameter_completeness(state)
        result = router.route_intent(state)
        
        print(f"Complete: {param_analysis['complete']}")
        print(f"Has Critical: {param_analysis['has_critical']}")
        print(f"Provided Count: {param_analysis['provided_count']}")
        print(f"Expected Count: {param_analysis['expected_count']}")
        print(f"Missing: {param_analysis['missing']}")
        print(f"Critical Missing: {param_analysis['critical_missing']}")
        print(f"Routing Decision: {result.decision}")
        print(f"Routing Reason: {result.reason.value}")


async def demo_langgraph_node():
    """Demonstrate the LangGraph node implementation."""
    
    print("\n\n🕸️ LangGraph Node Demo")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Successful High-Confidence Routing",
            "intent": IntentType.DATA_FETCH,
            "confidence": 0.9,
            "parameters": {"entity_type": "transactions", "time_period": "last_month"}
        },
        {
            "name": "Low-Confidence Routing",
            "intent": IntentType.UNKNOWN,
            "confidence": 0.3,
            "parameters": {}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")
        print("-" * 30)
        
        # Create initial state
        initial_state = MainState(
            user_input="Test LangGraph node",
            intent=test_case['intent'],
            confidence=test_case['confidence'],
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={"initial": "data"},
            parameters=test_case['parameters'],
            execution_results={},
            response=""
        )
        
        # Add user message using MessageManager
        from schemas.state_schemas import MessageManager
        initial_state = MessageManager.add_user_message(
            initial_state, 
            "Test LangGraph node"
        )
        
        print(f"Initial Messages: {len(initial_state.messages)}")
        print(f"Initial Metadata Keys: {list(initial_state.metadata.keys())}")
        
        # Process through node
        result_state = await decision_router_node(initial_state)
        
        print(f"Final Messages: {len(result_state.messages)}")
        print(f"Routing Decision: {result_state.metadata.get('routing_decision')}")
        print(f"Routing Reason: {result_state.metadata.get('routing_reason')}")
        print(f"Processing Time: {result_state.metadata.get('node_processing_time'):.4f}s")
        
        # Test routing decision extraction
        extracted_decision = get_routing_decision(result_state)
        print(f"Extracted Decision: {extracted_decision}")
        
        # Show new messages
        new_messages = result_state.messages[len(initial_state.messages):]
        print(f"New Messages Added: {len(new_messages)}")
        for msg in new_messages:
            print(f"  - {msg.role.value}: {msg.content[:50]}...")


async def demo_error_handling():
    """Demonstrate error handling and edge cases."""
    
    print("\n\n🚨 Error Handling Demo")
    print("=" * 50)
    
    router = DecisionRouter()
    
    error_scenarios = [
        {
            "name": "None State",
            "state": None,
            "description": "Passing None as state"
        },
        {
            "name": "Missing Intent",
            "state": MainState(
                user_input="Test",
                intent=None,
                confidence=0.8,
                messages=[],
                session_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                metadata={},
                parameters={},
                execution_results={},
                response=""
            ),
            "description": "State with None intent"
        },
        {
            "name": "Missing Confidence",
            "state": MainState(
                user_input="Test",
                intent=IntentType.DATA_FETCH,
                confidence=None,
                messages=[],
                session_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                metadata={},
                parameters={},
                execution_results={},
                response=""
            ),
            "description": "State with None confidence"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n🔍 {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 25)
        
        try:
            result = router.route_intent(scenario['state'])
            print(f"✅ Handled gracefully")
            print(f"Decision: {result.decision}")
            print(f"Reason: {result.reason.value}")
            print(f"Explanation: {result.routing_explanation}")
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {str(e)}")


async def main():
    """Run all demonstration scenarios."""
    await demo_basic_routing_scenarios()
    await demo_configurable_thresholds()
    await demo_intent_specific_routing()
    await demo_parameter_analysis()
    await demo_langgraph_node()
    await demo_error_handling()
    
    print("\n" + "=" * 50)
    print("✅ Decision Router Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Confidence-based routing with configurable thresholds")
    print("• Parameter completeness analysis")
    print("• Intent-specific routing rules")
    print("• Comprehensive error handling")
    print("• LangGraph node integration")
    print("• Detailed routing decision logging")
    print("• Flexible configuration management")


if __name__ == "__main__":
    asyncio.run(main()) 