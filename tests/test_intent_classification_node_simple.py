"""
Simple tests for Intent Classification Node with OpenAI Integration.

This module provides basic tests for the intent classification functionality
without complex mocking to verify the system works correctly.
"""

import pytest
import json
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from src.schemas.state_schemas import (
    MainState,
    IntentType,
    MessageRole,
    Message,
)
from src.schemas.generated_intent_schemas import (
    IntentCategory,
    IntentClassificationResult,
    IntentClassificationError,
    DataFetchParams,
)
from src.utils.llm_client import LLMClient

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def create_test_message(content: str) -> Message:
    """Helper function to create test messages."""
    return Message(
        role=MessageRole.USER,
        content=content,
        metadata={"source": "test"}
    )


def create_test_state(user_input: str) -> MainState:
    """Helper function to create test states."""
    return MainState(
        user_input=user_input,
        intent=IntentType.UNKNOWN,
        confidence=0.0,
        messages=[create_test_message(user_input)],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={},
        execution_results={},
        response=""
    )


class TestLLMClientBasics:
    """Test basic LLMClient functionality."""
    
    def test_initialization_without_api_key(self):
        """Test that LLMClient requires API key."""
        
        # Clear API key even if it was loaded from .env file
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}, clear=False):
            with pytest.raises(ValueError) as exc_info:
                LLMClient()
            
            assert "OPENAI_API_KEY environment variable not set" in str(exc_info.value)
    
    def test_initialization_with_api_key(self):
        """Test that LLMClient initializes properly with API key."""
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('openai.OpenAI'):
                client = LLMClient()
                assert client.model == "gpt-4-turbo"  # Default model
                assert client.temperature == 0.2  # Default temperature
    
    def test_chat_completion_method(self):
        """Test that chat_completion method exists and works."""
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('openai.OpenAI'):
                client = LLMClient()
                
                # Test that chat_completion method exists
                assert hasattr(client, 'chat_completion')
                assert callable(client.chat_completion)


class TestIntentClassifierBasics:
    """Test basic IntentClassifier functionality."""
    
    @pytest.mark.asyncio
    async def test_empty_input_validation(self):
        """Test that empty input is properly validated."""
        
        from src.nodes.intent_classification_node import IntentClassifier
        
        with patch('src.nodes.intent_classification_node.LLMClient'):
            classifier = IntentClassifier()
            
            with pytest.raises(IntentClassificationError) as exc_info:
                await classifier.classify_intent("")
            
            assert "Empty user input provided" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_successful_classification_mock(self):
        """Test successful classification with mocked LLM."""
        
        from src.nodes.intent_classification_node import IntentClassifier
        
        with patch('src.utils.llm_client.LLMClient') as mock_llm_class:
            # Create mock LLM client
            mock_llm = MagicMock()
            mock_llm.chat_completion = AsyncMock(return_value=json.dumps({
                "intent": "data_fetch",
                "confidence": 0.85,
                "reasoning": "Test reasoning",
                "user_input_analysis": "Test analysis",
                "missing_params": [],
                "clarification_needed": False,
                "data_fetch_params": {"entity_type": "transactions"}
            }))
            mock_llm_class.return_value = mock_llm
            
            classifier = IntentClassifier(llm_client=mock_llm)
            result = await classifier.classify_intent("Show me my transactions")
            
            assert isinstance(result, IntentClassificationResult)
            assert result.intent == IntentCategory.DATA_FETCH
            assert result.confidence == 0.85
            assert result.data_fetch_params is not None
            assert result.data_fetch_params.entity_type == "transactions"


class TestIntentClassificationNode:
    """Test the main LangGraph node."""
    
    @pytest.mark.asyncio
    async def test_node_with_mock_classifier(self):
        """Test the node with mocked classifier."""
        from src.nodes.intent_classification_node import intent_classification_node
        
        # Create test state
        state = create_test_state("Show me my transactions")
        
        with patch('src.nodes.intent_classification_node.IntentClassifier') as mock_classifier_class:
            # Mock successful classification
            mock_result = IntentClassificationResult(
                intent=IntentCategory.DATA_FETCH,
                confidence=0.85,
                reasoning="Test reasoning",
                user_input_analysis="Test analysis",
                data_fetch_params=DataFetchParams(entity_type="transactions"),
                missing_params=[],
                clarification_needed=False
            )
            
            mock_classifier = MagicMock()
            mock_classifier.classify_intent = AsyncMock(return_value=mock_result)
            mock_classifier_class.return_value = mock_classifier
            
            # Execute node
            result_state = await intent_classification_node(state)
            
            # Verify results
            assert result_state.intent == IntentType.DATA_FETCH
            assert result_state.confidence == 0.85
            assert result_state.parameters == {"entity_type": "transactions"}
            assert "classification_result" in result_state.metadata
            assert "llm_provider" in result_state.metadata
            assert result_state.metadata["llm_provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_node_error_handling(self):
        """Test node error handling."""
        from src.nodes.intent_classification_node import intent_classification_node
        
        # Create test state
        state = create_test_state("Test input")
        
        with patch('src.nodes.intent_classification_node.IntentClassifier') as mock_classifier_class:
            # Mock classifier that raises an error
            mock_classifier = MagicMock()
            mock_classifier.classify_intent = AsyncMock(side_effect=Exception("Test error"))
            mock_classifier_class.return_value = mock_classifier
            
            # Execute node
            result_state = await intent_classification_node(state)
            
            # Verify error handling
            assert result_state.intent == IntentType.UNKNOWN
            assert result_state.confidence == 0.0
            assert "error" in result_state.metadata
            assert "Test error" in result_state.metadata["error"]["error_message"]


class TestResponseValidation:
    """Test response validation functionality."""
    
    def test_valid_response_parsing(self):
        """Test parsing of valid responses."""
        
        from src.nodes.intent_classification_node import IntentClassifier
        
        classifier = IntentClassifier()
        
        valid_response = {
            "intent": "data_fetch",
            "confidence": 0.85,
            "reasoning": "Test reasoning",
            "user_input_analysis": "Test analysis",
            "missing_params": [],
            "clarification_needed": False,
            "data_fetch_params": {"entity_type": "transactions"}
        }
        
        result = classifier._parse_llm_result(valid_response, "test input")
        
        assert result.intent == IntentCategory.DATA_FETCH
        assert result.confidence == 0.85
        assert result.data_fetch_params is not None
        assert result.data_fetch_params.entity_type == "transactions"
    
    def test_invalid_intent_handling(self):
        """Test handling of invalid intent values."""
        
        from src.nodes.intent_classification_node import IntentClassifier
        from src.schemas.generated_intent_schemas import IntentClassificationError
        
        classifier = IntentClassifier()
        
        invalid_response = {
            "intent": "invalid_intent",
            "confidence": 0.85,
            "reasoning": "Test reasoning",
            "user_input_analysis": "Test analysis"
        }
        
        # Should raise IntentClassificationError for invalid intent
        with pytest.raises(IntentClassificationError) as exc_info:
            classifier._parse_llm_result(invalid_response, "test input")
        
        assert "Invalid LLM response format" in str(exc_info.value)
        assert "invalid_intent" in str(exc_info.value)


# Optional integration test
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_openai_integration():
    """
    Integration test with real OpenAI API.
    
    Only runs if OPENAI_API_KEY is set and -m integration is used.
    """
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No OpenAI API key available for integration test")
    
    from src.nodes.intent_classification_node import intent_classification_node
    
    # Create test state
    state = create_test_state("Show me my transactions from last month")
    
    try:
        # Execute with real OpenAI
        result_state = await intent_classification_node(state)
        
        # Verify structure
        assert result_state.intent in [IntentType.DATA_FETCH, IntentType.AGGREGATE, IntentType.ACTION, IntentType.UNKNOWN]
        assert 0.0 <= result_state.confidence <= 1.0
        assert "classification_result" in result_state.metadata
        assert "llm_provider" in result_state.metadata
        assert result_state.metadata["llm_provider"] == "openai"
        
        print(f"✅ Integration test successful: {result_state.intent} with {result_state.confidence:.2f} confidence")
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 