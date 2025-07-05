"""
Test script for OpenAI integration in intent classification.

This script tests the structure and import of the OpenAI integration
without requiring an actual API key.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schemas.intent_schemas import IntentClassificationError


async def test_llm_client_initialization():
    """Test LLMClient initialization with and without API key."""
    print("Testing LLMClient initialization...")
    
    # Import first
    from nodes.intent_classification_node import LLMClient
    
    # Save original environment state
    original_key = os.environ.pop("OPENAI_API_KEY", None)
    
    # Test without API key (should raise error)
    try:
        client = LLMClient()
        print("❌ ERROR: Should have failed without API key")
    except IntentClassificationError as e:
        if "API key not found" in str(e):
            print("✅ Correctly failed without API key")
        else:
            print(f"❌ Unexpected error: {e}")
    except Exception as e:
        print(f"❌ Unexpected exception type: {type(e).__name__}: {e}")
    
    # Restore API key for next test
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        os.environ["OPENAI_API_KEY"] = "test-key-for-structure-test"
    
    # Test with API key (should succeed in creating client)
    try:
        client = LLMClient(model="gpt-3.5-turbo", temperature=0.2)
        print("✅ Successfully created LLMClient with API key")
        print(f"   Model: {client.model}")
        print(f"   Temperature: {client.temperature}")
    except Exception as e:
        print(f"❌ Failed to create LLMClient: {e}")
    
    # Clean up test key
    if not original_key:
        os.environ.pop("OPENAI_API_KEY", None)


async def test_intent_classifier_structure():
    """Test IntentClassifier structure without API calls."""
    print("\nTesting IntentClassifier structure...")
    
    try:
        from nodes.intent_classification_node import IntentClassifier
        
        # Set temporary API key for structure test
        os.environ["OPENAI_API_KEY"] = "test-key-for-structure-test"
        
        classifier = IntentClassifier()
        print("✅ Successfully created IntentClassifier")
        
        # Test structure of methods
        assert hasattr(classifier, 'classify_intent'), "Missing classify_intent method"
        assert hasattr(classifier, '_parse_llm_result'), "Missing _parse_llm_result method"
        assert hasattr(classifier, '_create_fallback_result'), "Missing _create_fallback_result method"
        print("✅ All required methods present")
        
        # Clean up test key
        os.environ.pop("OPENAI_API_KEY", None)
        
    except Exception as e:
        print(f"❌ Error testing IntentClassifier: {e}")


async def test_schema_integration():
    """Test integration with intent schemas."""
    print("\nTesting schema integration...")
    
    try:
        from schemas.intent_schemas import (
            IntentCategory, 
            IntentClassificationResult,
            DataFetchParams,
            AggregateParams,
            ActionParams
        )
        
        # Test creating classification result
        result = IntentClassificationResult(
            intent=IntentCategory.DATA_FETCH,
            confidence=0.85,
            reasoning="Test data fetch intent",
            user_input_analysis="Test analysis",
            data_fetch_params=DataFetchParams(entity_type="transactions"),
            missing_params=[],
            clarification_needed=False
        )
        
        print("✅ Successfully created IntentClassificationResult")
        print(f"   Intent: {result.intent}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Requires clarification: {result.requires_clarification}")
        print(f"   Extracted parameters: {result.extracted_parameters}")
        
    except Exception as e:
        print(f"❌ Error testing schema integration: {e}")


async def test_prompt_generation():
    """Test prompt generation methods."""
    print("\nTesting prompt generation...")
    
    try:
        from nodes.intent_classification_node import LLMClient
        
        # Set temporary API key for structure test
        os.environ["OPENAI_API_KEY"] = "test-key-for-structure-test"
        
        client = LLMClient()
        
        # Test system prompt
        system_prompt = client._create_system_prompt()
        assert len(system_prompt) > 100, "System prompt too short"
        assert "data_fetch" in system_prompt, "Missing data_fetch in system prompt"
        assert "aggregate" in system_prompt, "Missing aggregate in system prompt"
        assert "action" in system_prompt, "Missing action in system prompt"
        print("✅ System prompt generated correctly")
        
        # Test user prompt
        user_prompt = client._create_user_prompt(
            "Show me my transactions", 
            "Previous context here"
        )
        assert "Show me my transactions" in user_prompt, "User input missing from prompt"
        assert "Previous context here" in user_prompt, "Context missing from prompt"
        print("✅ User prompt generated correctly")
        
        # Clean up test key
        os.environ.pop("OPENAI_API_KEY", None)
        
    except Exception as e:
        print(f"❌ Error testing prompt generation: {e}")


async def test_response_validation():
    """Test response validation logic."""
    print("\nTesting response validation...")
    
    try:
        from nodes.intent_classification_node import LLMClient
        
        # Set temporary API key for structure test
        os.environ["OPENAI_API_KEY"] = "test-key-for-structure-test"
        
        client = LLMClient()
        
        # Test valid response
        valid_response = {
            "intent": "data_fetch",
            "confidence": 0.85,
            "reasoning": "User wants to see transaction data",
            "user_input_analysis": "Request for transaction viewing",
            "missing_params": [],
            "clarification_needed": False,
            "data_fetch_params": {
                "entity_type": "transactions",
                "time_period": "last_month"
            }
        }
        
        normalized = client._validate_and_normalize_response(valid_response)
        print("✅ Valid response normalized correctly")
        print(f"   Intent: {normalized['intent']}")
        print(f"   Confidence: {normalized['confidence']}")
        
        # Test invalid response (missing required field)
        invalid_response = {
            "intent": "data_fetch",
            "confidence": 0.85
            # Missing required fields
        }
        
        try:
            client._validate_and_normalize_response(invalid_response)
            print("❌ Should have failed with invalid response")
        except ValueError:
            print("✅ Correctly rejected invalid response")
        
        # Clean up test key
        os.environ.pop("OPENAI_API_KEY", None)
        
    except Exception as e:
        print(f"❌ Error testing response validation: {e}")


async def main():
    """Run all tests."""
    print("🧪 Testing OpenAI Integration Structure")
    print("=" * 50)
    
    await test_llm_client_initialization()
    await test_intent_classifier_structure()
    await test_schema_integration()
    await test_prompt_generation()
    await test_response_validation()
    
    print("\n" + "=" * 50)
    print("✅ All structure tests completed!")
    print("\nTo test with real OpenAI API:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Run the intent classification node with actual user input")


if __name__ == "__main__":
    asyncio.run(main()) 