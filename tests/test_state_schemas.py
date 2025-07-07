"""
Comprehensive unit tests for state schemas with hierarchical inheritance.

Tests cover:
- Schema inheritance patterns
- Automatic state transitions
- Message management and node tracking
- Conversation pruning and context
- Validation functions
- Error handling scenarios
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4
from pydantic import ValidationError

from src.schemas.state_schemas import (
    BaseState,
    MainState,
    ClarificationState,
    OrchestratorState,
    Message,
    MessageRole,
    IntentType,
    StateTransitions,
    MessageManager,
    ConversationContext,
    StateValidator,
    MAX_MESSAGE_LENGTH,
    MAX_CONVERSATION_HISTORY,
    MAX_CLARIFICATION_ATTEMPTS,
)


class TestMessage:
    """Test Message schema with role validation and metadata tracking."""
    
    def test_message_creation(self):
        """Test basic message creation with all fields."""
        message = Message(
            role=MessageRole.USER,
            content="Test message content",
            metadata={"node_name": "test_node", "extra": "data"}
        )
        
        assert message.role == MessageRole.USER
        assert message.content == "Test message content"
        assert message.node_name == "test_node"
        assert message.metadata["extra"] == "data"
        assert isinstance(message.timestamp, datetime)
    
    def test_message_content_validation(self):
        """Test message content validation and trimming."""
        # Valid content
        message = Message(role=MessageRole.USER, content="  Valid content  ")
        assert message.content == "Valid content"
        
        # Empty content should raise error
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="")
        
        # Whitespace-only content should raise error
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="   ")
    
    def test_message_length_validation(self):
        """Test message content length limits."""
        # Valid length
        valid_content = "a" * MAX_MESSAGE_LENGTH
        message = Message(role=MessageRole.USER, content=valid_content)
        assert len(message.content) == MAX_MESSAGE_LENGTH
        
        # Too long should raise error
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="a" * (MAX_MESSAGE_LENGTH + 1))
    
    def test_message_properties(self):
        """Test message role checking properties."""
        user_msg = Message(role=MessageRole.USER, content="User message")
        assistant_msg = Message(role=MessageRole.ASSISTANT, content="Assistant message")
        system_msg = Message(role=MessageRole.SYSTEM, content="System message")
        
        assert user_msg.is_user_message
        assert not user_msg.is_assistant_message
        assert not user_msg.is_system_message
        
        assert assistant_msg.is_assistant_message
        assert not assistant_msg.is_user_message
        assert not assistant_msg.is_system_message
        
        assert system_msg.is_system_message
        assert not system_msg.is_user_message
        assert not system_msg.is_assistant_message


class TestBaseState:
    """Test BaseState schema with core conversation and intent tracking."""
    
    def test_base_state_creation(self):
        """Test basic state creation with required fields."""
        state = BaseState(
            user_input="Test user input",
            intent=IntentType.DATA_FETCH,
            confidence=0.85
        )
        
        assert state.user_input == "Test user input"
        assert state.intent == IntentType.DATA_FETCH
        assert state.confidence == 0.85
        assert isinstance(state.session_id, str)
        assert isinstance(state.timestamp, datetime)
        assert state.messages == []
        assert state.metadata == {}
    
    def test_user_input_validation(self):
        """Test user input validation and trimming."""
        # Valid input
        state = BaseState(user_input="  Valid input  ")
        assert state.user_input == "Valid input"
        
        # Empty input should raise error
        with pytest.raises(ValidationError):
            BaseState(user_input="")
        
        # Whitespace-only input should raise error
        with pytest.raises(ValidationError):
            BaseState(user_input="   ")
    
    def test_confidence_validation(self):
        """Test confidence score range validation."""
        # Valid confidence values
        BaseState(user_input="Test", confidence=0.0)
        BaseState(user_input="Test", confidence=0.5)
        BaseState(user_input="Test", confidence=1.0)
        BaseState(user_input="Test", confidence=None)
        
        # Invalid confidence values
        with pytest.raises(ValidationError):
            BaseState(user_input="Test", confidence=-0.1)
        
        with pytest.raises(ValidationError):
            BaseState(user_input="Test", confidence=1.1)
    
    def test_session_id_validation(self):
        """Test session ID UUID format validation."""
        # Valid UUID
        valid_uuid = str(uuid4())
        state = BaseState(user_input="Test", session_id=valid_uuid)
        assert state.session_id == valid_uuid
        
        # Invalid UUID should raise error
        with pytest.raises(ValidationError):
            BaseState(user_input="Test", session_id="invalid-uuid")
    
    def test_message_history_pruning(self):
        """Test automatic message history pruning."""
        # Create messages exceeding limit
        messages = [
            Message(role=MessageRole.USER, content=f"Message {i}")
            for i in range(MAX_CONVERSATION_HISTORY + 5)
        ]
        
        state = BaseState(user_input="Test", messages=messages)
        assert len(state.messages) == MAX_CONVERSATION_HISTORY
        assert state.messages[0].content == f"Message 5"  # Oldest messages pruned


class TestMainState:
    """Test MainState schema extending BaseState."""
    
    def test_main_state_inheritance(self):
        """Test MainState inherits all BaseState functionality."""
        state = MainState(
            user_input="Test input",
            intent=IntentType.ACTION,
            confidence=0.75,
            parameters={"key": "value"},
            response="Test response"
        )
        
        # BaseState fields
        assert state.user_input == "Test input"
        assert state.intent == IntentType.ACTION
        assert state.confidence == 0.75
        
        # MainState specific fields
        assert state.parameters == {"key": "value"}
        assert state.response == "Test response"
        assert state.execution_results is None
    
    def test_parameters_flexibility(self):
        """Test flexible parameter structure."""
        state = MainState(
            user_input="Test",
            parameters={
                "string_param": "value",
                "number_param": 42,
                "list_param": [1, 2, 3],
                "nested_param": {"inner": "value"}
            }
        )
        
        assert state.parameters["string_param"] == "value"
        assert state.parameters["number_param"] == 42
        assert state.parameters["list_param"] == [1, 2, 3]
        assert state.parameters["nested_param"]["inner"] == "value"


class TestClarificationState:
    """Test ClarificationState schema for parameter collection."""
    
    def test_clarification_state_inheritance(self):
        """Test ClarificationState inherits MainState functionality."""
        state = ClarificationState(
            user_input="Test input",
            missing_params=["param1", "param2"],
            clarification_attempts=1,
            missing_critical_params=["param1"],
            parameter_priorities=["param1", "param2"],
            normalization_suggestions={"acc": "account"},
            ambiguity_flags={"amount": "ambiguous value"},
            clarification_history=[{"turn": 1, "question": "What account?", "answer": "Checking"}],
            extracted_parameters={"param1": "value1"}
        )
        
        # Inherited fields
        assert state.user_input == "Test input"
        assert state.parameters == {}
        
        # Clarification specific fields
        assert state.missing_params == ["param1", "param2"]
        assert state.missing_critical_params == ["param1"]
        assert state.parameter_priorities == ["param1", "param2"]
        assert state.normalization_suggestions == {"acc": "account"}
        assert state.ambiguity_flags == {"amount": "ambiguous value"}
        assert state.clarification_history == [{"turn": 1, "question": "What account?", "answer": "Checking"}]
        assert state.clarification_attempts == 1
        assert state.max_attempts == MAX_CLARIFICATION_ATTEMPTS
        assert state.extracted_parameters == {"param1": "value1"}
    
    def test_clarification_attempts_validation(self):
        """Test clarification attempts validation against max limit."""
        # Valid attempts
        ClarificationState(user_input="Test", clarification_attempts=0)
        ClarificationState(user_input="Test", clarification_attempts=3)
        
        # Invalid attempts exceeding max
        with pytest.raises(ValidationError):
            ClarificationState(
                user_input="Test",
                clarification_attempts=4,
                max_attempts=3
            )
    
    def test_clarification_properties(self):
        """Test clarification state helper properties."""
        # Can attempt clarification
        state = ClarificationState(
            user_input="Test",
            clarification_attempts=1,
            max_attempts=3,
            missing_params=["param1"]
        )
        assert state.can_attempt_clarification
        assert state.has_missing_params
        
        # Cannot attempt more clarifications
        state_max_attempts = ClarificationState(
            user_input="Test",
            clarification_attempts=3,
            max_attempts=3
        )
        assert not state_max_attempts.can_attempt_clarification
        
        # No missing params
        state_complete = ClarificationState(
            user_input="Test",
            missing_params=[]
        )
        assert not state_complete.has_missing_params

    def test_default_values(self):
        """Test default values for new ClarificationState fields."""
        state = ClarificationState(user_input="Test")
        assert state.missing_critical_params == []
        assert state.parameter_priorities == []
        assert state.normalization_suggestions == {}
        assert state.ambiguity_flags == {}
        assert state.clarification_history == []
        assert state.extracted_parameters == {}


class TestOrchestratorState:
    """Test OrchestratorState schema for tool execution."""
    
    def test_orchestrator_state_inheritance(self):
        """Test OrchestratorState inherits MainState functionality."""
        state = OrchestratorState(
            user_input="Test input",
            extracted_params={"param": "value"},
            execution_plan={"tools": ["tool1"]},
            tool_results={"tool1": {"status": "success"}},
            final_response="Final response"
        )
        
        # Inherited fields
        assert state.user_input == "Test input"
        assert state.parameters == {}
        
        # Orchestrator specific fields
        assert state.extracted_params == {"param": "value"}
        assert state.execution_plan == {"tools": ["tool1"]}
        assert state.tool_results == {"tool1": {"status": "success"}}
        assert state.final_response == "Final response"


class TestStateTransitions:
    """Test automatic state transitions and parameter merging."""
    
    @pytest.fixture
    def main_state(self):
        """Create a MainState for testing transitions."""
        return MainState(
            user_input="Test input",
            intent=IntentType.DATA_FETCH,
            confidence=0.80,
            session_id=str(uuid4()),
            parameters={"existing": "param"},
            messages=[Message(role=MessageRole.USER, content="User message")]
        )
    
    def test_to_clarification_state(self, main_state):
        """Test conversion from MainState to ClarificationState."""
        clarification_state = StateTransitions.to_clarification_state(main_state)
        
        # Inherited fields preserved
        assert clarification_state.user_input == main_state.user_input
        assert clarification_state.intent == main_state.intent
        assert clarification_state.confidence == main_state.confidence
        assert clarification_state.session_id == main_state.session_id
        assert clarification_state.parameters == main_state.parameters
        assert clarification_state.messages == main_state.messages
        
        # Clarification specific fields initialized
        assert clarification_state.missing_params == []
        assert clarification_state.clarification_attempts == 0
        assert clarification_state.extracted_parameters == {}
        assert clarification_state.missing_critical_params == []
        assert clarification_state.parameter_priorities == []
        assert clarification_state.normalization_suggestions == {}
        assert clarification_state.ambiguity_flags == {}
        assert clarification_state.clarification_history == []
    
    def test_from_clarification_state(self, main_state):
        """Test conversion from ClarificationState back to MainState with parameter merging."""
        # Create clarification state with extracted parameters
        clarification_state = StateTransitions.to_clarification_state(main_state)
        clarification_state.extracted_parameters = {"new": "param", "another": "value"}
        
        # Convert back to MainState
        result_state = StateTransitions.from_clarification_state(clarification_state)
        
        # Inherited fields preserved
        assert result_state.user_input == clarification_state.user_input
        assert result_state.session_id == clarification_state.session_id
        
        # Parameters merged correctly
        expected_params = {"existing": "param", "new": "param", "another": "value"}
        assert result_state.parameters == expected_params
    
    def test_to_orchestrator_state(self, main_state):
        """Test conversion from MainState to OrchestratorState."""
        orchestrator_state = StateTransitions.to_orchestrator_state(main_state)
        
        # Inherited fields preserved
        assert orchestrator_state.user_input == main_state.user_input
        assert orchestrator_state.intent == main_state.intent
        assert orchestrator_state.session_id == main_state.session_id
        
        # Parameters copied to extracted_params
        assert orchestrator_state.extracted_params == main_state.parameters
        
        # Orchestrator specific fields initialized
        assert orchestrator_state.execution_plan is None
        assert orchestrator_state.tool_results is None
        assert orchestrator_state.final_response is None
    
    def test_from_orchestrator_state(self, main_state):
        """Test conversion from OrchestratorState back to MainState."""
        # Create orchestrator state with execution results
        orchestrator_state = StateTransitions.to_orchestrator_state(main_state)
        orchestrator_state.tool_results = {"tool1": {"result": "success"}}
        orchestrator_state.final_response = "Execution complete"
        
        # Convert back to MainState
        result_state = StateTransitions.from_orchestrator_state(orchestrator_state)
        
        # Inherited fields preserved
        assert result_state.user_input == orchestrator_state.user_input
        assert result_state.session_id == orchestrator_state.session_id
        
        # Execution results merged
        assert result_state.parameters == orchestrator_state.extracted_params
        assert result_state.execution_results == orchestrator_state.tool_results
        assert result_state.response == orchestrator_state.final_response


class TestMessageManager:
    """Test message management and conversation handling utilities."""
    
    @pytest.fixture
    def base_state(self):
        """Create a BaseState for testing message management."""
        return BaseState(user_input="Test input")
    
    def test_add_user_message(self, base_state):
        """Test adding user messages with automatic timestamp."""
        updated_state = MessageManager.add_user_message(base_state, "User message content")
        
        assert len(updated_state.messages) == 1
        message = updated_state.messages[0]
        assert message.role == MessageRole.USER
        assert message.content == "User message content"
        assert message.metadata["source"] == "user_input"
        assert isinstance(message.timestamp, datetime)
    
    def test_add_assistant_message(self, base_state):
        """Test adding assistant messages with node tracking."""
        updated_state = MessageManager.add_assistant_message(
            base_state,
            "Assistant response",
            "test_node"
        )
        
        assert len(updated_state.messages) == 1
        message = updated_state.messages[0]
        assert message.role == MessageRole.ASSISTANT
        assert message.content == "Assistant response"
        assert message.metadata["node_name"] == "test_node"
        assert message.metadata["source"] == "assistant_response"
    
    def test_add_system_message(self, base_state):
        """Test adding system messages for debugging."""
        updated_state = MessageManager.add_system_message(
            base_state,
            "System debug message",
            "debug_node"
        )
        
        assert len(updated_state.messages) == 1
        message = updated_state.messages[0]
        assert message.role == MessageRole.SYSTEM
        assert message.content == "System debug message"
        assert message.metadata["node_name"] == "debug_node"
        assert message.metadata["source"] == "system_debug"
    
    def test_automatic_conversation_pruning(self, base_state):
        """Test automatic pruning when adding messages."""
        # Add messages up to the limit
        state = base_state
        for i in range(MAX_CONVERSATION_HISTORY + 5):
            state = MessageManager.add_user_message(state, f"Message {i}")
        
        assert len(state.messages) == MAX_CONVERSATION_HISTORY
        assert state.messages[0].content == "Message 5"  # Oldest messages pruned
        assert state.messages[-1].content == f"Message {MAX_CONVERSATION_HISTORY + 4}"
    
    def test_manual_conversation_pruning(self, base_state):
        """Test manual conversation pruning."""
        # Add many messages
        state = base_state
        for i in range(15):
            state = MessageManager.add_user_message(state, f"Message {i}")
        
        # Manually prune to 10 messages
        pruned_state = MessageManager.prune_conversation(state, max_messages=10)
        
        assert len(pruned_state.messages) == 10
        assert pruned_state.messages[0].content == "Message 5"
        assert pruned_state.messages[-1].content == "Message 14"


class TestConversationContext:
    """Test conversation context and summarization utilities."""
    
    @pytest.fixture
    def state_with_messages(self):
        """Create a state with varied conversation history."""
        state = BaseState(user_input="Test input")
        
        # Add mixed message types
        for i in range(10):
            state = MessageManager.add_user_message(state, f"User message {i}")
            state = MessageManager.add_assistant_message(
                state,
                f"Assistant response {i}",
                f"node_{i % 3}"
            )
            if i % 3 == 0:
                state = MessageManager.add_system_message(
                    state,
                    f"System debug {i}",
                    f"debug_node_{i}"
                )
        
        return state
    
    def test_get_recent_context(self, state_with_messages):
        """Test extracting recent messages for context."""
        recent_context = ConversationContext.get_recent_context(state_with_messages, 5)
        
        assert len(recent_context) == 5
        assert all(isinstance(msg, Message) for msg in recent_context)
        # Should be the last 5 messages
        assert recent_context[-1] == state_with_messages.messages[-1]
    
    def test_summarize_conversation(self, state_with_messages):
        """Test conversation summarization."""
        summary = ConversationContext.summarize_conversation(state_with_messages)
        
        assert isinstance(summary, str)
        assert "Conversation Summary" in summary
        assert "Total messages:" in summary
        assert "User messages:" in summary
        assert "Assistant messages:" in summary
        assert "System messages:" in summary
        assert state_with_messages.session_id[:8] in summary
    
    def test_get_messages_by_role(self, state_with_messages):
        """Test filtering messages by role."""
        user_messages = ConversationContext.get_messages_by_role(
            state_with_messages,
            MessageRole.USER
        )
        assistant_messages = ConversationContext.get_messages_by_role(
            state_with_messages,
            MessageRole.ASSISTANT
        )
        system_messages = ConversationContext.get_messages_by_role(
            state_with_messages,
            MessageRole.SYSTEM
        )
        
        # Due to automatic pruning to MAX_CONVERSATION_HISTORY (20), 
        # the exact counts may vary but should be balanced
        total_messages = len(user_messages) + len(assistant_messages) + len(system_messages)
        assert total_messages == len(state_with_messages.messages)
        assert len(user_messages) > 0  # Should have user messages
        assert len(assistant_messages) > 0  # Should have assistant messages
        assert len(system_messages) > 0  # Should have some system messages
        
        assert all(msg.role == MessageRole.USER for msg in user_messages)
        assert all(msg.role == MessageRole.ASSISTANT for msg in assistant_messages)
        assert all(msg.role == MessageRole.SYSTEM for msg in system_messages)
    
    def test_get_messages_by_node(self, state_with_messages):
        """Test filtering messages by node name."""
        node_0_messages = ConversationContext.get_messages_by_node(
            state_with_messages,
            "node_0"
        )
        
        assert len(node_0_messages) > 0
        assert all(msg.node_name == "node_0" for msg in node_0_messages)


class TestStateValidator:
    """Test comprehensive state validation functions."""
    
    def test_validate_base_state(self):
        """Test BaseState validation."""
        # Valid state
        valid_state = BaseState(
            user_input="Valid input",
            confidence=0.5,
            session_id=str(uuid4())
        )
        assert StateValidator.validate_base_state(valid_state)
        
        # Invalid confidence
        invalid_confidence_state = BaseState(user_input="Test")
        invalid_confidence_state.confidence = 1.5
        with pytest.raises(ValueError, match="Confidence must be between"):
            StateValidator.validate_base_state(invalid_confidence_state)
        
        # Invalid session ID
        invalid_session_state = BaseState(user_input="Test")
        invalid_session_state.session_id = "invalid-uuid"
        with pytest.raises(ValueError, match="Invalid session_id format"):
            StateValidator.validate_base_state(invalid_session_state)
    
    def test_validate_clarification_state(self):
        """Test ClarificationState validation."""
        # Valid clarification state
        valid_state = ClarificationState(
            user_input="Valid input",
            clarification_attempts=2,
            max_attempts=3
        )
        assert StateValidator.validate_clarification_state(valid_state)
        
        # Invalid attempts exceeding max
        invalid_state = ClarificationState(user_input="Test")
        invalid_state.clarification_attempts = 5
        invalid_state.max_attempts = 3
        with pytest.raises(ValueError, match="exceed max_attempts"):
            StateValidator.validate_clarification_state(invalid_state)
    
    def test_validate_state_transition(self):
        """Test state transition validation."""
        session_id = str(uuid4())
        
        from_state = BaseState(
            user_input="Test",
            session_id=session_id,
            intent=IntentType.DATA_FETCH
        )
        
        # Valid transition (same session ID, compatible intent)
        to_state = MainState(
            user_input="Test",
            session_id=session_id,
            intent=IntentType.DATA_FETCH
        )
        assert StateValidator.validate_state_transition(from_state, to_state)
        
        # Invalid transition (different session ID)
        invalid_to_state = MainState(
            user_input="Test",
            session_id=str(uuid4()),  # Different session ID
            intent=IntentType.DATA_FETCH
        )
        with pytest.raises(ValueError, match="Session ID must be preserved"):
            StateValidator.validate_state_transition(from_state, invalid_to_state)


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_empty_conversation_context(self):
        """Test conversation utilities with empty message history."""
        empty_state = BaseState(user_input="Test")
        
        # Empty context
        context = ConversationContext.get_recent_context(empty_state, 5)
        assert context == []
        
        # Empty summary
        summary = ConversationContext.summarize_conversation(empty_state)
        assert "No conversation history" in summary
        
        # Empty filters
        user_messages = ConversationContext.get_messages_by_role(empty_state, MessageRole.USER)
        assert user_messages == []
    
    def test_parameter_merging_edge_cases(self):
        """Test parameter merging with overlapping keys."""
        main_state = MainState(
            user_input="Test",
            parameters={"key1": "original", "key2": "value2"}
        )
        
        clarification_state = StateTransitions.to_clarification_state(main_state)
        clarification_state.extracted_parameters = {
            "key1": "updated",  # Override existing
            "key3": "new"       # Add new
        }
        
        merged_state = StateTransitions.from_clarification_state(clarification_state)
        
        # Clarified params should override original
        assert merged_state.parameters["key1"] == "updated"
        assert merged_state.parameters["key2"] == "value2"
        assert merged_state.parameters["key3"] == "new"
    
    def test_message_metadata_flexibility(self):
        """Test flexible message metadata handling."""
        message = Message(
            role=MessageRole.ASSISTANT,
            content="Test",
            metadata={
                "node_name": "test_node",
                "custom_field": "custom_value",
                "nested": {"inner": "value"},
                "list_data": [1, 2, 3]
            }
        )
        
        assert message.node_name == "test_node"
        assert message.metadata["custom_field"] == "custom_value"
        assert message.metadata["nested"]["inner"] == "value"
        assert message.metadata["list_data"] == [1, 2, 3] 