import pytest
from unittest.mock import patch
from uuid import uuid4
from vyuu_copilot_v2.nodes.exit_with_partial_data_node import exit_with_partial_data_node
from vyuu_copilot_v2.schemas.state_schemas import ClarificationState, IntentType


class TestExitWithPartialDataNode:
    """Test suite for ExitWithPartialData node."""
    
    @pytest.fixture
    def base_state(self):
        """Base ClarificationState for testing."""
        return ClarificationState(
            session_id=str(uuid4()),
            user_input="transfer money",
            intent=IntentType.ACTION,
            extracted_parameters={"amount": None, "source_account": "checking"},
            missing_params=["amount", "target_account"],
            missing_critical_params=["amount"],
            parameter_priorities=["amount", "target_account"],
            normalization_suggestions={},
            ambiguity_flags={},
            clarification_attempts=3,
            max_attempts=3,
            clarification_history=[
                {
                    "question": "What amount would you like to transfer?",
                    "user_response": "some money",
                    "targeted_param": "amount",
                    "attempt": 1,
                    "exit_condition": False
                },
                {
                    "question": "Could you specify the exact dollar amount?",
                    "user_response": "not sure",
                    "targeted_param": "amount", 
                    "attempt": 2,
                    "exit_condition": False
                },
                {
                    "question": "I'm sorry, I still need to know amount before I can proceed. Let me know when you're ready.",
                    "user_response": None,
                    "targeted_param": "max_attempts_exit",
                    "attempt": 3,
                    "exit_condition": True
                }
            ],
            metadata={
                "clarification_status": "max_attempts_reached",
                "exit_message": "I'm sorry, I still need to know amount before I can proceed. Let me know when you're ready.",
                "exit_reason": "Reached maximum clarification attempts (3)",
                "remaining_missing_params": ["amount", "target_account"],
                "remaining_critical_params": ["amount"]
            }
        )
    
    def test_successful_exit_with_partial_data(self, base_state, caplog):
        """Test successful exit handling with proper state and metadata."""
        with patch('src.nodes.exit_with_partial_data_node.logger') as mock_logger:
            # Call the node
            exit_message, updated_state = exit_with_partial_data_node(base_state)
            
            # Verify returned exit message
            expected_message = "I'm sorry, I still need to know amount before I can proceed. Let me know when you're ready."
            assert exit_message == expected_message
            
            # Verify state updates
            assert updated_state.metadata["clarification_subgraph_closed"] == True
            assert updated_state.metadata["clarification_status"] == "max_attempts_reached"
            assert updated_state.metadata["exit_message"] == expected_message
            
            # Verify all other metadata preserved
            assert updated_state.metadata["exit_reason"] == base_state.metadata["exit_reason"]
            assert updated_state.metadata["remaining_missing_params"] == base_state.metadata["remaining_missing_params"]
            assert updated_state.metadata["remaining_critical_params"] == base_state.metadata["remaining_critical_params"]
            
            # Verify other state fields unchanged
            assert updated_state.session_id == base_state.session_id
            assert updated_state.clarification_attempts == base_state.clarification_attempts
            assert updated_state.clarification_history == base_state.clarification_history
            
            # Verify logging calls (session ID will be truncated to first 8 chars)
            session_prefix = base_state.session_id[:8]
            mock_logger.info.assert_any_call(f"[ExitWithPartialData] Processing exit for session {session_prefix}")
            mock_logger.info.assert_any_call(f"[ExitWithPartialData] Exiting clarification subgraph: {expected_message}")
            mock_logger.info.assert_any_call(f"[ExitWithPartialData] Subgraph closed for session {session_prefix}")
    
    def test_wrong_clarification_status_assertion_error(self, base_state):
        """Test assertion error when clarification_status is not 'max_attempts_reached'."""
        # Set wrong status
        base_state.metadata["clarification_status"] = "awaiting_user_response"
        
        with pytest.raises(AssertionError) as exc_info:
            exit_with_partial_data_node(base_state)
        
        assert "Expected clarification_status 'max_attempts_reached'" in str(exc_info.value)
        assert "got 'awaiting_user_response'" in str(exc_info.value)
    
    def test_missing_clarification_status_assertion_error(self, base_state):
        """Test assertion error when clarification_status is missing."""
        # Remove clarification_status
        del base_state.metadata["clarification_status"]
        
        with pytest.raises(AssertionError) as exc_info:
            exit_with_partial_data_node(base_state)
        
        assert "Expected clarification_status 'max_attempts_reached'" in str(exc_info.value)
        assert "got 'None'" in str(exc_info.value)
    
    def test_missing_exit_message_key_error(self, base_state):
        """Test KeyError when exit_message is missing from metadata."""
        # Remove exit_message
        del base_state.metadata["exit_message"]
        
        with patch('src.nodes.exit_with_partial_data_node.logger') as mock_logger:
            with pytest.raises(KeyError) as exc_info:
                exit_with_partial_data_node(base_state)
            
            assert "Required 'exit_message' not found in state.metadata" in str(exc_info.value)
            session_prefix = base_state.session_id[:8]
            mock_logger.error.assert_called_once_with(
                f"[ExitWithPartialData] Missing exit_message in metadata for session {session_prefix}"
            )
    
    def test_exit_message_with_critical_params(self, base_state):
        """Test exit with specific critical parameters in message."""
        # Update state with multiple critical params
        base_state.metadata["exit_message"] = "I'm sorry, I still need to know amount, target_account before I can proceed. Let me know when you're ready."
        base_state.metadata["remaining_critical_params"] = ["amount", "target_account"]
        
        exit_message, updated_state = exit_with_partial_data_node(base_state)
        
        expected_message = "I'm sorry, I still need to know amount, target_account before I can proceed. Let me know when you're ready."
        assert exit_message == expected_message
        assert updated_state.metadata["clarification_subgraph_closed"] == True
    
    def test_exit_message_with_generic_fallback(self, base_state):
        """Test exit with generic fallback message."""
        # Update state with generic message
        base_state.metadata["exit_message"] = "I'm sorry, I still need to know some information before I can proceed. Let me know when you're ready."
        base_state.metadata["remaining_critical_params"] = []
        
        exit_message, updated_state = exit_with_partial_data_node(base_state)
        
        expected_message = "I'm sorry, I still need to know some information before I can proceed. Let me know when you're ready."
        assert exit_message == expected_message
        assert updated_state.metadata["clarification_subgraph_closed"] == True
    
    def test_state_immutability(self, base_state):
        """Test that original state is not modified."""
        original_metadata = base_state.metadata.copy()
        original_attempts = base_state.clarification_attempts
        
        exit_message, updated_state = exit_with_partial_data_node(base_state)
        
        # Verify original state unchanged
        assert base_state.metadata == original_metadata
        assert base_state.clarification_attempts == original_attempts
        assert "clarification_subgraph_closed" not in base_state.metadata
        
        # Verify new state has changes
        assert updated_state.metadata["clarification_subgraph_closed"] == True
        assert updated_state is not base_state
    
    def test_all_metadata_preserved(self, base_state):
        """Test that all existing metadata is preserved in updated state."""
        # Add extra metadata
        base_state.metadata.update({
            "custom_field": "custom_value",
            "another_field": 42,
            "nested_data": {"key": "value"}
        })
        
        exit_message, updated_state = exit_with_partial_data_node(base_state)
        
        # Verify all original metadata preserved
        for key, value in base_state.metadata.items():
            assert updated_state.metadata[key] == value
        
        # Verify new field added
        assert updated_state.metadata["clarification_subgraph_closed"] == True
    
    def test_session_id_truncation_in_logs(self, base_state):
        """Test that session IDs are properly truncated in log messages."""
        # Use specific UUID for predictable truncation testing
        test_uuid = "12345678-9abc-def0-1234-567890abcdef"
        base_state.session_id = test_uuid
        
        with patch('src.nodes.exit_with_partial_data_node.logger') as mock_logger:
            exit_with_partial_data_node(base_state)
            
            # Verify truncated session ID in logs (first 8 characters)
            mock_logger.info.assert_any_call("[ExitWithPartialData] Processing exit for session 12345678")
            mock_logger.info.assert_any_call("[ExitWithPartialData] Subgraph closed for session 12345678")
    
    def test_return_type_annotation(self, base_state):
        """Test that return type matches annotation."""
        result = exit_with_partial_data_node(base_state)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)  # exit_message
        assert isinstance(result[1], ClarificationState)  # updated_state 