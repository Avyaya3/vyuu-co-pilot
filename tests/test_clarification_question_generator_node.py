import pytest
import asyncio
import uuid
from unittest.mock import patch, AsyncMock
from datetime import datetime, timezone

from src.nodes.clarification_question_generator_node import (
    clarification_question_generator_node,
    ClarificationQuestionGenerator,
    QuestionTemplate,
    ClarificationResult
)
from src.schemas.state_schemas import ClarificationState, IntentType, Message, MessageRole


@pytest.mark.asyncio
class TestClarificationQuestionGeneratorNode:
    """Test Clarification Question Generator Node functionality."""
    
    def test_question_template_context_building(self):
        """Test context prompt building."""
        state = ClarificationState(
            user_input="Transfer $500 to savings",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            extracted_parameters={
                "action_type": "transfer",
                "amount": 500.0,
                "source_account": None
            },
            clarification_history=[
                {"question": "Which account would you like to transfer from?", "user_response": None}
            ]
        )
        
        context = QuestionTemplate.build_context_prompt(state)
        
        assert "Intent: action" in context
        assert "Transfer $500 to savings" in context
        assert "action_type" in context
        assert "amount" in context
        assert "Which account would you like to transfer from?" in context
    
    def test_question_template_instruction_building(self):
        """Test instruction prompt building with different scenarios."""
        # Single slot with normalization
        instructions = QuestionTemplate.build_instruction_prompt(
            slot_names=["amount"],
            normalization_suggestions={"amount": "Clarify if exact or approximate"},
            ambiguity_flags={"amount": "approximation"},
            intent=IntentType.ACTION
        )
        
        assert "amount" in instructions
        assert "Clarification note" in instructions
        assert "Ambiguity note" in instructions
        assert "specific dollar amount" in instructions  # Intent guidance
        
        # Multiple slots
        instructions = QuestionTemplate.build_instruction_prompt(
            slot_names=["source_account", "target_account"],
            normalization_suggestions={},
            ambiguity_flags={},
            intent=IntentType.ACTION
        )
        
        assert "source_account, target_account" in instructions
        assert "multiple related pieces" in instructions
    
    def test_slot_selection_with_priorities(self):
        """Test slot selection based on parameter priorities."""
        generator = ClarificationQuestionGenerator()
        
        # Test with list priorities
        state = ClarificationState(
            user_input="Test",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            parameter_priorities=["amount", "source_account", "target_account"],
            clarification_history=[]
        )
        
        slots = generator.select_next_slots(state, max_slots=2)
        assert slots == ["amount", "source_account"]
        
        # Test with dict priorities
        state.parameter_priorities = {
            "high": ["amount"],
            "medium": ["source_account", "target_account"],
            "low": ["description"]
        }
        
        slots = generator.select_next_slots(state, max_slots=2)
        assert slots == ["amount", "source_account"]
    
    def test_slot_selection_excludes_asked_slots(self):
        """Test that already asked slots are excluded."""
        generator = ClarificationQuestionGenerator()
        
        state = ClarificationState(
            user_input="Test",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            parameter_priorities=["amount", "source_account", "target_account"],
            clarification_history=[
                {"question": "What amount?", "targeted_param": "amount", "attempt": 1}
            ]
        )
        
        slots = generator.select_next_slots(state, max_slots=2)
        assert "amount" not in slots
        assert slots == ["source_account", "target_account"]
    
    def test_slot_selection_fallbacks(self):
        """Test fallback behavior when no priorities available."""
        generator = ClarificationQuestionGenerator()
        
        state = ClarificationState(
            user_input="Test",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            parameter_priorities=[],  # Empty priorities
            missing_critical_params=["amount", "source_account"],
            missing_params=["amount", "source_account", "description"],
            clarification_history=[]
        )
        
        slots = generator.select_next_slots(state, max_slots=2)
        assert slots == ["amount", "source_account"]  # Falls back to critical params
        
        # Test final fallback to missing_params
        state.missing_critical_params = []
        slots = generator.select_next_slots(state, max_slots=2)
        assert slots == ["amount", "source_account"]  # Falls back to missing_params
    
    async def test_successful_question_generation(self):
        """Test successful question generation with LLM."""
        state = ClarificationState(
            user_input="Transfer money to savings",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            missing_params=["amount", "source_account"],
            missing_critical_params=["amount", "source_account"],
            parameter_priorities=["amount", "source_account"],
            clarification_attempts=0,
            max_attempts=3,
            extracted_parameters={"action_type": "transfer", "target_account": "savings"},
            normalization_suggestions={},
            ambiguity_flags={},
            clarification_history=[]
        )
        
        # Mock LLM response
        with patch('src.nodes.clarification_question_generator_node.asyncio.to_thread') as mock_to_thread:
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.content = "What amount would you like to transfer?"
            mock_to_thread.return_value = mock_response
            
            question, updated_state = await clarification_question_generator_node(state)
        
        assert question == "What amount would you like to transfer?"
        assert updated_state.clarification_attempts == 1
        assert len(updated_state.clarification_history) == 1
        assert updated_state.clarification_history[0]["question"] == question
        assert updated_state.clarification_history[0]["targeted_param"] == "amount"
        assert updated_state.clarification_history[0]["exit_condition"] == False
        
        # Check continued clarification metadata
        assert updated_state.metadata["clarification_status"] == "awaiting_user_response"
        assert updated_state.metadata["current_question"] == question
        assert updated_state.clarification_history[0]["user_response"] is None
    
    async def test_llm_failure_fallback(self):
        """Test fallback behavior when LLM call fails."""
        state = ClarificationState(
            user_input="Transfer money",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            missing_params=["amount"],
            missing_critical_params=["amount"],
            parameter_priorities=["amount"],
            clarification_attempts=0,
            max_attempts=3,
            extracted_parameters={},
            normalization_suggestions={},
            ambiguity_flags={},
            clarification_history=[]
        )
        
        # Mock LLM failure
        with patch('src.nodes.clarification_question_generator_node.asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = Exception("API Error")
            
            question, updated_state = await clarification_question_generator_node(state)
        
        # Should get fallback question specific to the missing slot
        assert "amount" in question.lower()
        assert updated_state.clarification_attempts == 1
        assert len(updated_state.clarification_history) == 1
        assert updated_state.clarification_history[0]["exit_condition"] == False
        # Note: LLM failure is handled internally in generate_question, so node sees it as successful
        assert updated_state.metadata["clarification_status"] == "awaiting_user_response"
    
    async def test_max_attempts_reached(self):
        """Test exit behavior when max attempts are reached."""
        state = ClarificationState(
            user_input="Transfer money",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            missing_params=["amount"],
            missing_critical_params=["amount"],
            parameter_priorities=["amount"],
            clarification_attempts=3,  # At max
            max_attempts=3,
            extracted_parameters={},
            normalization_suggestions={},
            ambiguity_flags={},
            clarification_history=[]
        )
        
        exit_signal, updated_state = await clarification_question_generator_node(state)
        
        # Should exit with partial data instead of asking another question
        assert exit_signal == "EXIT_WITH_PARTIAL_DATA"
        assert updated_state.clarification_attempts == 3  # Not incremented further
        assert len(updated_state.clarification_history) == 1
        assert updated_state.clarification_history[0]["targeted_param"] == "max_attempts_exit"
        assert updated_state.clarification_history[0]["exit_condition"] == True
        
        # Check exit metadata
        assert updated_state.metadata["clarification_status"] == "max_attempts_reached"
        assert "amount" in updated_state.metadata["exit_message"]
        assert "before I can proceed" in updated_state.metadata["exit_message"]
        assert updated_state.metadata["remaining_missing_params"] == ["amount"]
        assert updated_state.metadata["remaining_critical_params"] == ["amount"]
    
    async def test_no_slots_to_ask(self):
        """Test behavior when no slots need to be asked about."""
        state = ClarificationState(
            user_input="Transfer money",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            missing_params=[],  # No missing params
            missing_critical_params=[],
            parameter_priorities=[],
            clarification_attempts=0,
            max_attempts=3,
            extracted_parameters={"action_type": "transfer", "amount": 500.0},
            normalization_suggestions={},
            ambiguity_flags={},
            clarification_history=[]
        )
        
        question, updated_state = await clarification_question_generator_node(state)
        
        assert "additional information" in question
        assert updated_state.clarification_attempts == 1
        assert len(updated_state.clarification_history) == 1
        assert updated_state.clarification_history[0]["exit_condition"] == False
    
    def test_question_post_processing(self):
        """Test question post-processing functionality."""
        generator = ClarificationQuestionGenerator()
        
        # Test adding question mark
        processed = generator._post_process_question("What is your amount")
        assert processed == "What is your amount?"
        
        # Test removing quotes
        processed = generator._post_process_question('"What is your amount?"')
        assert processed == "What is your amount?"
        
        # Test capitalizing first letter
        processed = generator._post_process_question("what is your amount?")
        assert processed == "What is your amount?"
        
        # Test removing extra spaces
        processed = generator._post_process_question("What   is    your amount?")
        assert processed == "What is your amount?"
        
        # Test stripping whitespace
        processed = generator._post_process_question("  What is your amount?  ")
        assert processed == "What is your amount?"
    
    def test_fallback_question_generation(self):
        """Test fallback question generation."""
        generator = ClarificationQuestionGenerator()
        
        # Single slot
        question = generator._generate_fallback_question(["amount"], IntentType.ACTION)
        assert "amount" in question.lower()
        assert question.endswith("?")
        
        # Multiple slots
        question = generator._generate_fallback_question(
            ["source_account", "target_account"], 
            IntentType.ACTION
        )
        assert "source account" in question.lower()
        assert "target account" in question.lower()
        
        # Unknown slot
        question = generator._generate_fallback_question(["unknown_slot"], IntentType.ACTION)
        assert "unknown slot" in question.lower()
    
    async def test_different_intent_types(self):
        """Test question generation for different intent types."""
        # DATA_FETCH intent
        state = ClarificationState(
            user_input="Show me transactions",
            intent=IntentType.DATA_FETCH,
            confidence=0.8,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            missing_params=["entity_type"],
            missing_critical_params=["entity_type"],
            parameter_priorities=["entity_type"],
            clarification_attempts=0,
            max_attempts=3,
            extracted_parameters={},
            normalization_suggestions={},
            ambiguity_flags={},
            clarification_history=[]
        )
        
        with patch('src.nodes.clarification_question_generator_node.asyncio.to_thread') as mock_to_thread:
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.content = "What type of data would you like to see?"
            mock_to_thread.return_value = mock_response
            
            question, updated_state = await clarification_question_generator_node(state)
        
        assert "data" in question.lower() or "type" in question.lower()
        assert updated_state.clarification_attempts == 1
        assert updated_state.clarification_history[0]["exit_condition"] == False
    
    async def test_normalization_and_ambiguity_integration(self):
        """Test integration of normalization suggestions and ambiguity flags."""
        state = ClarificationState(
            user_input="Transfer about five hundred",
            intent=IntentType.ACTION,
            confidence=0.9,
            messages=[],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            metadata={},
            parameters={},
            execution_results={},
            response="",
            missing_params=["source_account"],
            missing_critical_params=["source_account"],
            parameter_priorities=["source_account"],
            clarification_attempts=0,
            max_attempts=3,
            extracted_parameters={"amount": 500.0},
            normalization_suggestions={
                "amount": "User said 'about' - confirm exact amount"
            },
            ambiguity_flags={
                "amount": "approximation"
            },
            clarification_history=[]
        )
        
        # Test that the template building includes normalization and ambiguity info
        instructions = QuestionTemplate.build_instruction_prompt(
            ["source_account"],
            state.normalization_suggestions,
            state.ambiguity_flags,
            state.intent
        )
        
        # Even though we're asking about source_account, the normalization/ambiguity 
        # for amount shouldn't appear since it's not in the slot_names
        assert "source_account" in instructions
        assert "transfer" in instructions.lower() or "account" in instructions.lower()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 