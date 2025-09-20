"""
Tests for Tool Execution Node

Tests the tool execution node functionality including:
- Single step execution
- Multi-step execution with transactions
- Error handling and retries
- State management and metadata
- Integration with tool registry
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone
from uuid import uuid4

from vyuu_copilot_v2.schemas.state_schemas import OrchestratorState, IntentType
from vyuu_copilot_v2.schemas.generated_intent_schemas import IntentCategory
from vyuu_copilot_v2.tools.base import ToolResponse
from vyuu_copilot_v2.nodes.tool_execution_node import (
    tool_execution_node,
    ExecutionStepResult,
    ToolExecutionError,
    _execute_single_step,
    _execute_with_transaction,
    _execute_without_transaction,
    _determine_execution_strategy
)


def create_test_orchestrator_state(
    user_input: str = "Test input",
    intent: IntentType = IntentType.DATA_FETCH,
    extracted_params: dict = None,
    execution_plan: dict = None
) -> OrchestratorState:
    """Create a test OrchestratorState."""
    # Handle the case where we explicitly want no execution plan
    if execution_plan is None:
        execution_plan_field = None
    else:
        execution_plan_field = execution_plan or {
            "steps": [
                {
                    "tool_name": "db_query",
                    "operation": "get_account_balance",
                    "params": {"account_name": "checking"}
                }
            ]
        }
    
    return OrchestratorState(
        user_input=user_input,
        intent=intent,
        confidence=0.85,
        session_id=str(uuid4()),
        extracted_params=extracted_params or {"user_id": "test-user"},
        execution_plan=execution_plan_field
    )


class TestExecutionStepResult:
    """Test ExecutionStepResult class."""
    
    def test_execution_step_result_creation(self):
        """Test creating ExecutionStepResult."""
        result = ExecutionStepResult(0, "db_query", "get_account_balance")
        
        assert result.step_index == 0
        assert result.tool_name == "db_query"
        assert result.operation == "get_account_balance"
        assert result.success is False
        assert result.data is None
        assert result.error is None
        assert result.execution_time_ms == 0.0
        assert result.retry_count == 0
    
    def test_execution_timing(self):
        """Test execution timing functionality."""
        result = ExecutionStepResult(0, "db_query", "get_account_balance")
        
        result.start_execution()
        assert result.start_time is not None
        
        # Add a small delay to ensure execution time > 0
        import time
        time.sleep(0.001)
        
        result.end_execution(True, {"balance": 1000.0})
        assert result.end_time is not None
        assert result.success is True
        assert result.data == {"balance": 1000.0}
        assert result.execution_time_ms > 0
    
    def test_to_dict_conversion(self):
        """Test converting to dictionary."""
        result = ExecutionStepResult(1, "db_action", "transfer_money")
        result.start_execution()
        
        # Add a small delay to ensure execution time > 0
        import time
        time.sleep(0.001)
        
        result.end_execution(True, {"transaction_id": "123"})
        
        result_dict = result.to_dict()
        
        assert result_dict["step_index"] == 1
        assert result_dict["tool_name"] == "db_action"
        assert result_dict["operation"] == "transfer_money"
        assert result_dict["success"] is True
        assert result_dict["data"] == {"transaction_id": "123"}
        assert result_dict["execution_time_ms"] > 0
        assert result_dict["retry_count"] == 0


class TestExecutionStrategy:
    """Test execution strategy determination."""
    
    def test_single_step_no_transaction(self):
        """Test single step uses no transaction."""
        steps = [
            {"tool_name": "db_query", "operation": "get_account_balance"}
        ]
        
        strategy = _determine_execution_strategy(steps)
        assert strategy == "no_transaction"
    
    def test_multi_step_uses_transaction(self):
        """Test multi-step uses transaction."""
        steps = [
            {"tool_name": "db_query", "operation": "get_account_balance"},
            {"tool_name": "db_query", "operation": "get_transaction_history"}
        ]
        
        strategy = _determine_execution_strategy(steps)
        assert strategy == "transaction"
    
    def test_single_action_uses_transaction(self):
        """Test single action step uses transaction."""
        steps = [
            {"tool_name": "db_action", "operation": "transfer_money"}
        ]
        
        strategy = _determine_execution_strategy(steps)
        assert strategy == "transaction"


class TestSingleStepExecution:
    """Test single step execution functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_step_execution(self):
        """Test successful step execution."""
        step = {
            "tool_name": "db_query",
            "operation": "get_account_balance",
            "params": {"account_name": "checking"}
        }
        extracted_params = {"user_id": "test-user"}
        
        with patch('src.nodes.tool_execution_node.get_tool') as mock_get_tool, \
             patch('src.nodes.tool_execution_node.get_tool_schema') as mock_get_schema:
            
            # Mock tool
            mock_tool = AsyncMock()
            mock_tool.invoke.return_value = {
                "success": True,
                "data": {"balance": 1000.0},
                "error": None,
                "tool_name": "db_query",
                "execution_time_ms": 50.0
            }
            mock_get_tool.return_value = mock_tool
            
            # Mock schema
            mock_schema = MagicMock()
            mock_schema.return_value.model_dump.return_value = {
                "user_id": "test-user",
                "account_name": "checking"
            }
            mock_get_schema.return_value = mock_schema
            
            result = await _execute_single_step(step, 0, extracted_params)
            
            assert result.success is True
            assert result.data == {"balance": 1000.0}
            assert result.error is None
            assert result.retry_count == 0
            assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_step_with_invalid_tool(self):
        """Test step execution with invalid tool name."""
        step = {
            "tool_name": "invalid_tool",
            "operation": "get_account_balance",
            "params": {}
        }
        extracted_params = {"user_id": "test-user"}
        
        result = await _execute_single_step(step, 0, extracted_params)
        
        assert result.success is False
        assert "not found in registry" in result.error
    
    @pytest.mark.asyncio
    async def test_step_with_parameter_validation_error(self):
        """Test step execution with parameter validation error."""
        step = {
            "tool_name": "db_query",
            "operation": "get_account_balance",
            "params": {}
        }
        extracted_params = {}
        
        with patch('src.nodes.tool_execution_node.get_tool_schema') as mock_get_schema:
            mock_get_schema.side_effect = Exception("Validation error")
            
            result = await _execute_single_step(step, 0, extracted_params)
            
            assert result.success is False
            assert "Parameter validation failed" in result.error
    
    @pytest.mark.asyncio
    async def test_step_with_tool_execution_error(self):
        """Test step execution with tool execution error."""
        step = {
            "tool_name": "db_query",
            "operation": "get_account_balance",
            "params": {"account_name": "checking"}
        }
        extracted_params = {"user_id": "test-user"}
        
        with patch('src.nodes.tool_execution_node.get_tool') as mock_get_tool, \
             patch('src.nodes.tool_execution_node.get_tool_schema') as mock_get_schema:
            
            # Mock tool that raises exception
            mock_tool = AsyncMock()
            mock_tool.invoke.side_effect = Exception("Database connection failed")
            mock_get_tool.return_value = mock_tool
            
            # Mock schema
            mock_schema = MagicMock()
            mock_schema.return_value.model_dump.return_value = {
                "user_id": "test-user",
                "account_name": "checking"
            }
            mock_get_schema.return_value = mock_schema
            
            result = await _execute_single_step(step, 0, extracted_params)
            
            assert result.success is False
            assert "Tool execution failed" in result.error
            assert result.retry_count == 3  # Max retries attempted


class TestMultiStepExecution:
    """Test multi-step execution functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_without_transaction_success(self):
        """Test successful execution without transaction."""
        steps = [
            {
                "tool_name": "db_query",
                "operation": "get_account_balance",
                "params": {"account_name": "checking"}
            },
            {
                "tool_name": "db_query",
                "operation": "get_transaction_history",
                "params": {"account_name": "checking"}
            }
        ]
        extracted_params = {"user_id": "test-user"}
        
        with patch('src.nodes.tool_execution_node._execute_single_step') as mock_execute:
            # Mock successful executions
            result1 = ExecutionStepResult(0, "db_query", "get_account_balance")
            result1.start_execution()
            result1.end_execution(True, {"balance": 1000.0})
            
            result2 = ExecutionStepResult(1, "db_query", "get_transaction_history")
            result2.start_execution()
            result2.end_execution(True, {"transactions": []})
            
            mock_execute.side_effect = [result1, result2]
            
            results, errors = await _execute_without_transaction(steps, extracted_params)
            
            assert len(results) == 2
            assert len(errors) == 0
            assert mock_execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_without_transaction_partial_failure(self):
        """Test execution without transaction with partial failure."""
        steps = [
            {
                "tool_name": "db_query",
                "operation": "get_account_balance",
                "params": {"account_name": "checking"}
            },
            {
                "tool_name": "db_query",
                "operation": "get_transaction_history",
                "params": {"account_name": "checking"}
            }
        ]
        extracted_params = {"user_id": "test-user"}
        
        with patch('src.nodes.tool_execution_node._execute_single_step') as mock_execute:
            # Mock first success, second failure
            result1 = ExecutionStepResult(0, "db_query", "get_account_balance")
            result1.start_execution()
            result1.end_execution(True, {"balance": 1000.0})
            
            result2 = ExecutionStepResult(1, "db_query", "get_transaction_history")
            result2.start_execution()
            result2.end_execution(False, error="Database error")
            
            mock_execute.side_effect = [result1, result2]
            
            results, errors = await _execute_without_transaction(steps, extracted_params)
            
            assert len(results) == 2
            assert len(errors) == 1
            assert "Step 2 failed" in errors[0]
    
    @pytest.mark.asyncio
    async def test_execute_with_transaction_success(self):
        """Test successful execution with transaction."""
        steps = [
            {
                "tool_name": "db_action",
                "operation": "transfer_money",
                "params": {"amount": 100}
            }
        ]
        extracted_params = {"user_id": "test-user"}
        
        with patch('src.nodes.tool_execution_node.get_financial_service') as mock_get_service, \
             patch('src.nodes.tool_execution_node._execute_single_step') as mock_execute:
            
            # Mock financial service and transaction
            mock_service = MagicMock()
            mock_repo = MagicMock()
            mock_transaction = MagicMock()
            
            mock_get_service.return_value = mock_service
            mock_service.account_repo = mock_repo
            mock_repo.transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
            mock_repo.transaction.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock successful execution
            result = ExecutionStepResult(0, "db_action", "transfer_money")
            result.start_execution()
            result.end_execution(True, {"transaction_id": "123"})
            mock_execute.return_value = result
            
            results, errors = await _execute_with_transaction(steps, extracted_params)
            
            assert len(results) == 1
            assert len(errors) == 0
            assert results[0].success is True
    
    @pytest.mark.asyncio
    async def test_execute_with_transaction_failure(self):
        """Test execution with transaction failure."""
        steps = [
            {
                "tool_name": "db_action",
                "operation": "transfer_money",
                "params": {"amount": 100}
            }
        ]
        extracted_params = {"user_id": "test-user"}
        
        with patch('src.nodes.tool_execution_node.get_financial_service') as mock_get_service, \
             patch('src.nodes.tool_execution_node._execute_single_step') as mock_execute:
            
            # Mock financial service and transaction
            mock_service = MagicMock()
            mock_repo = MagicMock()
            mock_transaction = MagicMock()
            
            mock_get_service.return_value = mock_service
            mock_service.account_repo = mock_repo
            mock_repo.transaction.return_value.__aenter__ = AsyncMock(return_value=mock_transaction)
            mock_repo.transaction.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock failed execution
            result = ExecutionStepResult(0, "db_action", "transfer_money")
            result.start_execution()
            result.end_execution(False, error="Insufficient funds")
            mock_execute.return_value = result
            
            with pytest.raises(ToolExecutionError) as exc_info:
                await _execute_with_transaction(steps, extracted_params)
            
            assert "Transaction failed" in str(exc_info.value)


class TestToolExecutionNode:
    """Test the main tool execution node."""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful tool execution."""
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        with patch('src.nodes.tool_execution_node._execute_without_transaction') as mock_execute:
            # Mock successful execution
            result = ExecutionStepResult(0, "db_query", "get_account_balance")
            result.start_execution()
            result.end_execution(True, {"balance": 1000.0})
            
            mock_execute.return_value = ([result], [])
            
            updated_state = await tool_execution_node(state)
            
            assert updated_state.tool_results is not None
            assert "step_0" in updated_state.tool_results
            assert updated_state.metadata["execution_status"] == "success"
            assert updated_state.metadata["steps_completed"] == 1
            assert updated_state.metadata["steps_failed"] == 0
            assert updated_state.metadata["total_execution_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_execution_with_no_plan(self):
        """Test execution with no execution plan."""
        state = create_test_orchestrator_state(execution_plan=None)
        
        # Mock the tool execution to avoid hitting real database
        with patch('src.nodes.tool_execution_node._execute_without_transaction') as mock_execute, \
             patch('src.nodes.tool_execution_node._execute_with_transaction') as mock_execute_transaction:
            
            updated_state = await tool_execution_node(state)
            
            assert updated_state.metadata["execution_status"] == "failure"
            assert "No execution plan provided" in updated_state.metadata["execution_errors"][0]
    
    @pytest.mark.asyncio
    async def test_execution_with_empty_steps(self):
        """Test execution with empty steps."""
        state = create_test_orchestrator_state(
            execution_plan={"steps": []}
        )
        
        updated_state = await tool_execution_node(state)
        
        assert updated_state.metadata["execution_status"] == "failure"
        assert "contains no steps" in updated_state.metadata["execution_errors"][0]
    
    @pytest.mark.asyncio
    async def test_partial_failure_execution(self):
        """Test execution with partial failure."""
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    },
                    {
                        "tool_name": "db_query",
                        "operation": "get_transaction_history",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        # Mock the transaction execution to avoid hitting real database
        with patch('src.nodes.tool_execution_node._execute_with_transaction') as mock_execute:
            # Mock first success, second failure
            result1 = ExecutionStepResult(0, "db_query", "get_account_balance")
            result1.start_execution()
            result1.end_execution(True, {"balance": 1000.0})
            
            result2 = ExecutionStepResult(1, "db_query", "get_transaction_history")
            result2.start_execution()
            result2.end_execution(False, error="Database error")
            
            mock_execute.return_value = ([result1, result2], ["Step 2 failed: Database error"])
            
            updated_state = await tool_execution_node(state)
            
            assert updated_state.metadata["execution_status"] == "partial_failure"
            assert updated_state.metadata["steps_completed"] == 1
            assert updated_state.metadata["steps_failed"] == 1
            assert len(updated_state.metadata["execution_errors"]) == 1
    
    @pytest.mark.asyncio
    async def test_complete_failure_execution(self):
        """Test execution with complete failure."""
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        with patch('src.nodes.tool_execution_node._execute_without_transaction') as mock_execute:
            # Mock failed execution
            result = ExecutionStepResult(0, "db_query", "get_account_balance")
            result.start_execution()
            result.end_execution(False, error="Database connection failed")
            
            mock_execute.return_value = ([result], ["Step 1 failed: Database connection failed"])
            
            updated_state = await tool_execution_node(state)
            
            assert updated_state.metadata["execution_status"] == "failure"
            assert updated_state.metadata["steps_completed"] == 0
            assert updated_state.metadata["steps_failed"] == 1
    
    @pytest.mark.asyncio
    async def test_execution_strategy_selection(self):
        """Test execution strategy selection."""
        # Test single step query (no transaction)
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        with patch('src.nodes.tool_execution_node._execute_without_transaction') as mock_execute:
            result = ExecutionStepResult(0, "db_query", "get_account_balance")
            result.start_execution()
            result.end_execution(True, {"balance": 1000.0})
            mock_execute.return_value = ([result], [])
            
            updated_state = await tool_execution_node(state)
            
            assert updated_state.metadata["execution_strategy"] == "no_transaction"
        
        # Test multi-step (transaction)
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    },
                    {
                        "tool_name": "db_query",
                        "operation": "get_transaction_history",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        with patch('src.nodes.tool_execution_node._execute_with_transaction') as mock_execute:
            result1 = ExecutionStepResult(0, "db_query", "get_account_balance")
            result1.start_execution()
            result1.end_execution(True, {"balance": 1000.0})
            result2 = ExecutionStepResult(1, "db_query", "get_transaction_history")
            result2.start_execution()
            result2.end_execution(True, {"transactions": []})
            mock_execute.return_value = ([result1, result2], [])
            
            updated_state = await tool_execution_node(state)
            
            assert updated_state.metadata["execution_strategy"] == "transaction"
    
    @pytest.mark.asyncio
    async def test_node_exception_handling(self):
        """Test node exception handling."""
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        # Mock both execution methods to ensure we hit the exception handling
        with patch('src.nodes.tool_execution_node._execute_without_transaction') as mock_execute_no_tx, \
             patch('src.nodes.tool_execution_node._execute_with_transaction') as mock_execute_tx:
            
            # Make the execution method raise an exception
            mock_execute_no_tx.side_effect = Exception("Unexpected error")
            mock_execute_tx.side_effect = Exception("Unexpected error")
            
            updated_state = await tool_execution_node(state)
            
            assert updated_state.metadata["execution_status"] == "failure"
            assert "Tool execution failed" in updated_state.metadata["execution_errors"][0]
            assert updated_state.metadata["total_execution_time_ms"] > 0


class TestToolResultsStructure:
    """Test tool results structure and organization."""
    
    @pytest.mark.asyncio
    async def test_tool_results_structure(self):
        """Test tool results are properly structured."""
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        with patch('src.nodes.tool_execution_node._execute_without_transaction') as mock_execute:
            result = ExecutionStepResult(0, "db_query", "get_account_balance")
            result.start_execution()
            result.end_execution(True, {"balance": 1000.0})
            
            mock_execute.return_value = ([result], [])
            
            updated_state = await tool_execution_node(state)
            
            # Check step-based key
            assert "step_0" in updated_state.tool_results
            step_result = updated_state.tool_results["step_0"]
            assert step_result["tool_name"] == "db_query"
            assert step_result["operation"] == "get_account_balance"
            assert step_result["success"] is True
            assert step_result["data"] == {"balance": 1000.0}
            
            # Check tool-based key
            assert "db_query" in updated_state.tool_results
            tool_result = updated_state.tool_results["db_query"]
            assert tool_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_execution_times_structure(self):
        """Test execution times are properly tracked."""
        state = create_test_orchestrator_state(
            execution_plan={
                "steps": [
                    {
                        "tool_name": "db_query",
                        "operation": "get_account_balance",
                        "params": {"account_name": "checking"}
                    }
                ]
            }
        )
        
        with patch('src.nodes.tool_execution_node._execute_without_transaction') as mock_execute:
            result = ExecutionStepResult(0, "db_query", "get_account_balance")
            result.start_execution()
            result.end_execution(True, {"balance": 1000.0})
            result.execution_time_ms = 150.5  # Set specific time
            
            mock_execute.return_value = ([result], [])
            
            updated_state = await tool_execution_node(state)
            
            assert "step_0" in updated_state.metadata["execution_times"]
            assert updated_state.metadata["execution_times"]["step_0"] == 150.5
            assert updated_state.metadata["total_execution_time_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 