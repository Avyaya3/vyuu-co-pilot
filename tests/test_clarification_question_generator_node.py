import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from vyuu_copilot_v2.schemas.state_schemas import ClarificationState, IntentType
from vyuu_copilot_v2.nodes.clarification_question_generator_node import (
    clarification_question_generator_node,
    ClarificationQuestionGenerator,
    QuestionTemplate
)


@pytest.fixture
def sample_clarification_state():
    """Create a sample ClarificationState for testing."""
    return ClarificationState(
        user_input="Transfer money",
        session_id=str(uuid4()),
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["amount", "account"],
        missing_critical_params=["amount"],
        parameter_priorities=["amount", "account"],
        clarification_attempts=0,
        max_attempts=3
    )


@pytest.mark.asyncio
async def test_clarification_question_generator_success(sample_clarification_state):
    """Test successful question generation."""
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "How much would you like to transfer?"
        mock_llm_class.return_value = mock_llm
        
        # Call the node
        updated_state = await clarification_question_generator_node(sample_clarification_state)
        
        # Verify the state was updated correctly
        assert isinstance(updated_state, ClarificationState)
        assert updated_state.pending_question == "How much would you like to transfer?"
        assert updated_state.waiting_for_response is True
        assert updated_state.clarification_phase == "waiting"
        assert updated_state.clarification_attempts == 1
        assert updated_state.metadata["clarification_status"] == "waiting_for_user_response"
        
        # Verify LLM was called
        mock_llm.chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_clarification_question_generator_llm_failure(sample_clarification_state):
    """Test question generation when LLM fails."""
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        # Mock LLM failure
        mock_llm = AsyncMock()
        mock_llm.chat_completion.side_effect = Exception("LLM API error")
        mock_llm_class.return_value = mock_llm
        
        # Call the node
        updated_state = await clarification_question_generator_node(sample_clarification_state)
        
        # Verify fallback behavior - LLM failure is handled internally and returns fallback question
        assert isinstance(updated_state, ClarificationState)
        assert updated_state.pending_question is not None
        assert updated_state.waiting_for_response is True
        assert updated_state.clarification_phase == "waiting"
        assert updated_state.clarification_attempts == 1
        # The fallback is handled internally, so status is normal
        assert updated_state.metadata["clarification_status"] == "waiting_for_user_response"


@pytest.mark.asyncio
async def test_clarification_question_generator_max_attempts_reached():
    """Test question generation when max attempts are reached."""
    # Create state with max attempts reached
    state = ClarificationState(
        user_input="Transfer money",
        session_id=str(uuid4()),
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["amount"],
        missing_critical_params=["amount"],
        clarification_attempts=3,  # Max attempts reached
        max_attempts=3
    )
    
    # Call the node
    updated_state = await clarification_question_generator_node(state)
    
    # Verify exit behavior
    assert isinstance(updated_state, ClarificationState)
    assert updated_state.metadata["clarification_status"] == "max_attempts_reached"
    assert "exit_message" in updated_state.metadata


@pytest.mark.asyncio
async def test_clarification_question_generator_with_history(sample_clarification_state):
    """Test question generation with existing clarification history."""
    # Add some history
    sample_clarification_state.clarification_history = [
        {
            "question": "How much would you like to transfer?",
            "user_response": "$500",
            "targeted_param": "amount",
            "attempt": 1
        }
    ]
    sample_clarification_state.clarification_attempts = 1
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "Which account would you like to transfer from?"
        mock_llm_class.return_value = mock_llm
        
        updated_state = await clarification_question_generator_node(sample_clarification_state)
        
        # Verify state updates
        assert updated_state.clarification_attempts == 2
        assert len(updated_state.clarification_history) == 2
        assert updated_state.pending_question == "Which account would you like to transfer from?"


@pytest.mark.asyncio
async def test_clarification_question_generator_with_extracted_parameters():
    """Test question generation with some already extracted parameters."""
    state = ClarificationState(
        user_input="Transfer $500",
        session_id=str(uuid4()),
        intent=IntentType.ACTION,
        confidence=0.8,
        messages=[],
        metadata={},
        missing_params=["account"],
        missing_critical_params=["account"],
        extracted_parameters={"amount": "$500"},
        clarification_attempts=0,
        max_attempts=3
    )
    
    with patch('src.nodes.clarification_question_generator_node.LLMClient') as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.chat_completion.return_value = "Which account would you like to transfer from?"
        mock_llm_class.return_value = mock_llm
        
        updated_state = await clarification_question_generator_node(state)
        
        # Verify the question was generated
        assert updated_state.pending_question == "Which account would you like to transfer from?"
        assert updated_state.waiting_for_response is True


class TestClarificationQuestionGenerator:
    """Test the ClarificationQuestionGenerator class."""
    
    def test_select_next_slots(self, sample_clarification_state):
        """Test slot selection logic."""
        generator = ClarificationQuestionGenerator()
        
        # Test with priorities
        slots = generator.select_next_slots(sample_clarification_state)
        assert slots == ["amount", "account"]
        
        # Test with history (should skip already asked slots)
        sample_clarification_state.clarification_history = [
            {"targeted_param": "amount"}
        ]
        slots = generator.select_next_slots(sample_clarification_state)
        assert slots == ["account"]
    
    def test_parse_priorities_list(self):
        """Test priority parsing with list input."""
        generator = ClarificationQuestionGenerator()
        priorities = ["amount", "account", "description"]
        result = generator._parse_priorities(priorities)
        assert result == ["amount", "account", "description"]
    
    def test_parse_priorities_dict(self):
        """Test priority parsing with dict input."""
        generator = ClarificationQuestionGenerator()
        priorities = {
            "critical": ["amount"],
            "high": ["account"],
            "optional": ["description"]
        }
        result = generator._parse_priorities(priorities)
        assert "amount" in result
        assert "account" in result
        assert "description" in result


class TestQuestionTemplate:
    """Test the QuestionTemplate class."""
    
    def test_build_context_prompt(self, sample_clarification_state):
        """Test context prompt building."""
        prompt = QuestionTemplate.build_context_prompt(sample_clarification_state)
        assert "Intent: action" in prompt
        assert "User input: \"Transfer money\"" in prompt
    
    def test_build_context_prompt_with_extracted_parameters(self, sample_clarification_state):
        """Test context prompt with extracted parameters."""
        sample_clarification_state.extracted_parameters = {"amount": "$500"}
        prompt = QuestionTemplate.build_context_prompt(sample_clarification_state)
        assert "Already provided" in prompt
        assert "$500" in prompt
    
    def test_build_context_prompt_with_history(self, sample_clarification_state):
        """Test context prompt with clarification history."""
        sample_clarification_state.clarification_history = [
            {"question": "How much?"}
        ]
        prompt = QuestionTemplate.build_context_prompt(sample_clarification_state)
        assert "Previously asked" in prompt
    
    def test_build_instruction_prompt_single_slot(self):
        """Test instruction prompt for single slot."""
        prompt = QuestionTemplate.build_instruction_prompt(
            ["amount"],
            {},
            {},
            IntentType.ACTION
        )
        assert "amount" in prompt.lower()
        # The instruction prompt doesn't end with a question mark, it's instructions for generating a question
        assert "question" in prompt.lower()
    
    def test_build_instruction_prompt_multiple_slots(self):
        """Test instruction prompt for multiple slots."""
        prompt = QuestionTemplate.build_instruction_prompt(
            ["amount", "account"],
            {},
            {},
            IntentType.ACTION
        )
        assert "amount" in prompt.lower()
        assert "account" in prompt.lower()
    
    def test_get_intent_guidance(self):
        """Test intent-specific guidance."""
        guidance = QuestionTemplate._get_intent_guidance(IntentType.ACTION, ["amount"])
        assert guidance is not None
        assert "dollar amount" in guidance.lower() 