"""
Demonstration of OpenAI Intent Classification.

This script demonstrates how to use the real OpenAI integration for intent classification.
Requires a valid OpenAI API key in the OPENAI_API_KEY environment variable.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schemas.state_schemas import MainState, MessageManager, IntentType, MessageRole, Message
from schemas.intent_schemas import IntentCategory
from nodes.intent_classification_node import intent_classification_node
from datetime import datetime, timezone
import uuid


async def demo_classification_examples():
    """Demonstrate classification with various example inputs."""
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found.")
        print("Please set your OPENAI_API_KEY environment variable to run this demo.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🚀 OpenAI Intent Classification Demo")
    print("=" * 50)
    
    # Test cases for different intent types
    test_cases = [
        {
            "input": "Show me my transactions from last month",
            "description": "Data fetch - transactions with time period"
        },
        {
            "input": "What's my total spending this year by category?",
            "description": "Aggregate - spending analysis with grouping"
        },
        {
            "input": "Transfer $500 from checking to savings",
            "description": "Action - money transfer with amount"
        },
        {
            "input": "List my top 10 largest expenses",
            "description": "Data fetch - with sorting and limiting"
        },
        {
            "input": "How much did I spend on groceries last month?",
            "description": "Aggregate - category-specific spending"
        },
        {
            "input": "Pay my electricity bill",
            "description": "Action - bill payment (needs clarification)"
        },
        {
            "input": "Hello, how are you?",
            "description": "Unknown - greeting/casual conversation"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"Input: \"{test_case['input']}\"")
        print("-" * 30)
        
        try:
            # Create a MainState for this test
            state = MainState(
                user_input=test_case['input'],
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                messages=[],  # Start with empty messages
                session_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                metadata={},
                parameters={},
                execution_results={},
                response=""
            )
            
            # Add user message using MessageManager
            state = MessageManager.add_user_message(state, test_case['input'])
            
            # Classify the intent
            result_state = await intent_classification_node(state)
            
            # Display results
            print(f"✅ Classification successful!")
            print(f"   Intent: {result_state.intent.value}")
            print(f"   Confidence: {result_state.confidence:.1%}")
            
            if result_state.parameters:
                print(f"   Parameters extracted: {len(result_state.parameters)}")
                for key, value in result_state.parameters.items():
                    print(f"     - {key}: {value}")
            else:
                print(f"   No parameters extracted")
            
            # Show classification details from metadata
            if "classification_result" in result_state.metadata:
                classification = result_state.metadata["classification_result"]
                print(f"   Reasoning: {classification.get('reasoning', 'N/A')}")
                
                if classification.get('missing_params'):
                    print(f"   Missing parameters: {classification['missing_params']}")
                
                if classification.get('clarification_needed'):
                    print(f"   ⚠️ Clarification needed")
            
            # Show processing time
            if "node_processing_time" in result_state.metadata:
                processing_time = result_state.metadata["node_processing_time"]
                print(f"   Processing time: {processing_time:.3f}s")
                
        except Exception as e:
            print(f"❌ Classification failed: {e}")


async def demo_conversation_context():
    """Demonstrate how conversation context improves classification."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found for conversation context demo.")
        return
    
    print("\n\n🗣️ Conversation Context Demo")
    print("=" * 50)
    
    # Simulate a conversation with context
    conversation_steps = [
        "I want to analyze my spending patterns",
        "Show me by category for last 3 months",
        "What about just restaurants?",
        "Transfer $200 to savings"
    ]
    
    # Create a state that accumulates conversation history
    session_id = str(uuid.uuid4())
    state = MainState(
        user_input="I want to analyze my spending patterns",  # Fix: provide non-empty initial input
        intent=IntentType.UNKNOWN,
        confidence=0.0,
        messages=[],  # Start with empty messages
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={},
        execution_results={},
        response=""
    )
    
    for i, user_input in enumerate(conversation_steps, 1):
        print(f"\n💬 Step {i}: \"{user_input}\"")
        print("-" * 30)
        
        # Add user message using MessageManager
        state = MessageManager.add_user_message(state, user_input)
        state.user_input = user_input
        
        try:
            # Classify with conversation context
            result_state = await intent_classification_node(state)
            
            print(f"   Intent: {result_state.intent.value}")
            print(f"   Confidence: {result_state.confidence:.1%}")
            
            if result_state.parameters:
                print(f"   Parameters: {result_state.parameters}")
            
            # Update state for next iteration
            state = result_state
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def demo_parameter_extraction():
    """Demonstrate intelligent parameter extraction capabilities."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found for parameter extraction demo.")
        return
    
    print("\n\n🔍 Parameter Extraction Demo")
    print("=" * 50)
    
    complex_examples = [
        {
            "input": "Show me the 5 largest transactions from my checking account in the last 30 days sorted by amount descending",
            "expected_params": ["entity_type", "limit", "account_types", "time_period", "sort_by", "order"]
        },
        {
            "input": "What's my average monthly spending on food and entertainment compared to last year?",
            "expected_params": ["metric_type", "group_by", "category_filter", "comparison_period"]
        },
        {
            "input": "Transfer $1,250.50 from my primary checking to emergency savings on Friday",
            "expected_params": ["action_type", "amount", "source_account", "target_account", "schedule_date"]
        }
    ]
    
    for example in complex_examples:
        print(f"\n📊 Complex Input: \"{example['input']}\"")
        print(f"Expected parameters: {example['expected_params']}")
        print("-" * 30)
        
        try:
            state = MainState(
                user_input=example['input'],
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                messages=[],  # Start with empty messages
                session_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                metadata={},
                parameters={},
                execution_results={},
                response=""
            )
            
            # Add user message using MessageManager
            state = MessageManager.add_user_message(state, example['input'])

            result_state = await intent_classification_node(state)
            
            print(f"   Intent: {result_state.intent.value}")
            print(f"   Confidence: {result_state.confidence:.1%}")
            print(f"   Extracted {len(result_state.parameters)} parameters:")
            
            for key, value in result_state.parameters.items():
                if key in example['expected_params']:
                    print(f"     ✅ {key}: {value}")
                else:
                    print(f"     📝 {key}: {value}")
            
            missing = set(example['expected_params']) - set(result_state.parameters.keys())
            if missing:
                print(f"   ⚠️ Missing expected parameters: {list(missing)}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def main():
    """Run all demonstrations."""
    await demo_classification_examples()
    await demo_conversation_context()
    await demo_parameter_extraction()
    
    print("\n" + "=" * 50)
    print("✅ OpenAI Intent Classification Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Real OpenAI API integration with structured output")
    print("• Intelligent intent classification (data_fetch, aggregate, action)")
    print("• Automatic parameter extraction from natural language")
    print("• Conversation context awareness")
    print("• Confidence scoring and clarification detection")
    print("• Comprehensive error handling and fallbacks")


if __name__ == "__main__":
    asyncio.run(main()) 