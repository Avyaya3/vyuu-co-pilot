"""
Tool Execution Node for Direct Orchestrator Subgraph

Executes tools defined in execution_plan by calling each registered tool in sequence,
collecting standardized ToolResponse objects, and handling errors, retries, and
database transactions where appropriate.

Features:
- Sequential tool execution with error handling
- Retry logic with exponential backoff
- Transaction management for multi-step operations
- Comprehensive execution metadata and timing
- Integration with existing tool registry and financial service
- Consistent state management and transitions
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from ..schemas.state_schemas import OrchestratorState, MessageManager
from ..tools import TOOL_REGISTRY, get_tool, get_tool_schema
# from ..services import get_financial_service  # Temporarily disabled
from ..tools.base import ToolResponse

logger = logging.getLogger(__name__)

# Configuration constants
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0  # seconds
RETRY_MAX_DELAY = 10.0  # seconds


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""
    pass


class ExecutionStepResult:
    """Result of executing a single step in the execution plan."""
    
    def __init__(self, step_index: int, tool_name: str, operation: str):
        self.step_index = step_index
        self.tool_name = tool_name
        self.operation = operation
        self.success = False
        self.data = None
        self.error = None
        self.execution_time_ms = 0.0
        self.retry_count = 0
        self.start_time = None
        self.end_time = None
    
    def start_execution(self):
        """Mark the start of execution."""
        self.start_time = time.time()
    
    def end_execution(self, success: bool, data: Any = None, error: str = None):
        """Mark the end of execution with results."""
        self.end_time = time.time()
        self.success = success
        self.data = data
        self.error = error
        
        # Handle case where start_execution wasn't called
        if self.start_time is None:
            self.start_time = self.end_time
        
        self.execution_time_ms = (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "operation": self.operation,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count
        }


async def _execute_single_step(
    step: Dict[str, Any], 
    step_index: int, 
    base_extracted_params: Dict[str, Any] ) -> ExecutionStepResult:
    """
    Execute a single step in the execution plan.
    
    Args:
        step: Step definition from execution plan
        step_index: Index of the step in the plan
        extracted_params: Parameters extracted from user intent
        
    Returns:
        ExecutionStepResult with execution details
    """
    tool_name = step.get("tool_name")
    operation = step.get("operation")
    step_params = step.get("params", {})
    
    # Create result tracker
    result = ExecutionStepResult(step_index, tool_name, operation)
    
    # Validate tool exists
    if tool_name not in TOOL_REGISTRY:
        result.end_execution(False, error=f"Tool '{tool_name}' not found in registry")
        return result
    
    # Get tool instance
    try:
        tool = get_tool(tool_name)
    except ValueError as e:
        result.end_execution(False, error=f"Tool lookup failed: {str(e)}")
        return result
    
    # Merge parameters with safeguards: never allow step params to override critical fields
    # Start with base extracted parameters (which already have enforced user_id)
    merged_params = {**base_extracted_params}
    # Apply step params for non-critical fields only
    for key, value in (step_params or {}).items():
        if key == "user_id":
            # Protect user_id from being overridden by LLM placeholders (e.g., "extracted_from_context")
            # Prefer the extracted user_id if available; otherwise accept provided value
            if "user_id" not in merged_params or not merged_params["user_id"]:
                merged_params["user_id"] = value
        else:
            merged_params[key] = value
    
    # Validate parameters against tool schema
    try:
        tool_schema = get_tool_schema(tool_name)
        validated_params = tool_schema(**merged_params)
        final_params = validated_params.model_dump()
        
        # Debug logging for final parameters
        logger.info(f"Tool execution debug - final_params user_id: {final_params.get('user_id')}")
        logger.info(f"Tool execution debug - final_params keys: {list(final_params.keys())}")
        
    except Exception as e:
        result.end_execution(False, error=f"Parameter validation failed: {str(e)}")
        return result
    
    # Execute with retry logic
    for attempt in range(MAX_RETRY_ATTEMPTS + 1):
        result.retry_count = attempt
        result.start_execution()
        
        try:
            logger.info(
                f"Executing step {step_index + 1}: {tool_name}.{operation} (attempt {attempt + 1})",
                extra={
                    "step_index": step_index,
                    "tool_name": tool_name,
                    "operation": operation,
                    "attempt": attempt + 1,
                    "user_id": final_params.get("user_id")
                }
            )
            
            # Call tool
            response_dict = await tool.invoke(final_params)
            
            # Validate response format
            if not isinstance(response_dict, dict):
                raise ToolExecutionError("Tool returned invalid response format")
            
            # Convert to ToolResponse for validation
            tool_response = ToolResponse(**response_dict)
            
            if tool_response.success:
                result.end_execution(True, data=tool_response.data)
                logger.info(
                    f"Step {step_index + 1} completed successfully",
                    extra={
                        "step_index": step_index,
                        "tool_name": tool_name,
                        "operation": operation,
                        "execution_time_ms": result.execution_time_ms,
                        "user_id": final_params.get("user_id")
                    }
                )
                return result
            else:
                result.end_execution(False, error=tool_response.error)
                
        except Exception as e:
            result.end_execution(False, error=f"Tool execution failed: {str(e)}")
        
        # If this was the last attempt, return the failure
        if attempt == MAX_RETRY_ATTEMPTS:
            logger.error(
                f"Step {step_index + 1} failed after {MAX_RETRY_ATTEMPTS + 1} attempts",
                extra={
                    "step_index": step_index,
                    "tool_name": tool_name,
                    "operation": operation,
                    "final_error": result.error,
                    "user_id": final_params.get("user_id")
                }
            )
            return result
        
        # Calculate delay for next retry (exponential backoff)
        delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
        logger.warning(
            f"Step {step_index + 1} failed, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})",
            extra={
                "step_index": step_index,
                "tool_name": tool_name,
                "operation": operation,
                "error": result.error,
                "retry_delay": delay,
                "user_id": final_params.get("user_id")
            }
        )
        
        await asyncio.sleep(delay)


# Temporarily disabled - financial service not available
# async def _execute_with_transaction(
#     steps: List[Dict[str, Any]], 
#     extracted_params: Dict[str, Any] ) -> Tuple[List[ExecutionStepResult], List[str]]:
#     """
#     Execute multiple steps within a database transaction.
#     
#     Args:
#         steps: List of steps to execute
#         extracted_params: Parameters extracted from user intent
#         
#     Returns:
#         Tuple of (results, errors)
#     """
#     results = []
#     errors = []
#     
#     # financial_service = get_financial_service()  # Temporarily disabled
#     
#     async with financial_service.account_repo.transaction() as conn:
#         logger.info(f"Starting transaction for {len(steps)} steps")
#         
#         for i, step in enumerate(steps):
#             result = await _execute_single_step(step, i, extracted_params)
#             results.append(result)
#             
#             if not result.success:
#                 errors.append(f"Step {i + 1} failed: {result.error}")
#                 
#                 # For now, continue with remaining steps (partial failure)
#                 # In the future, this could be configurable based on step type
#                 logger.warning(
#                     f"Step {i + 1} failed, continuing with remaining steps",
#                     extra={
#                         "step_index": i,
#                         "error": result.error,
#                         "steps_remaining": len(steps) - i - 1
#                     }
#                 )
#         
#         # If any step failed, the transaction will be rolled back
#         if errors:
#             logger.error(f"Transaction will be rolled back due to {len(errors)} failures")
#             raise ToolExecutionError(f"Transaction failed: {'; '.join(errors)}")
#         
#         logger.info("Transaction completed successfully")
#     
#     return results, errors


async def _execute_without_transaction(
    steps: List[Dict[str, Any]], 
    extracted_params: Dict[str, Any] ) -> Tuple[List[ExecutionStepResult], List[str]]:
    """
    Execute steps without database transaction (single step or read-only operations).
    
    Args:
        steps: List of steps to execute
        extracted_params: Parameters extracted from user intent
        
    Returns:
        Tuple of (results, errors)
    """
    results = []
    errors = []
    
    for i, step in enumerate(steps):
        result = await _execute_single_step(step, i, extracted_params)
        results.append(result)
        
        if not result.success:
            errors.append(f"Step {i + 1} failed: {result.error}")
    
    return results, errors


def _determine_execution_strategy(steps: List[Dict[str, Any]]) -> str:
    """
    Determine whether to use transaction-based execution.
    
    Args:
        steps: List of steps to execute
        
    Returns:
        "transaction" or "no_transaction"
    """
    # Use transaction for multi-step plans
    if len(steps) > 1:
        return "transaction"
    
    # Use transaction if any step involves state-changing operations
    state_changing_tools = {"db_action"}
    for step in steps:
        if step.get("tool_name") in state_changing_tools:
            return "transaction"
    
    return "no_transaction"


async def tool_execution_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute tools defined in execution_plan and collect results.
    
    Args:
        state: OrchestratorState with execution_plan from planning node
        
    Returns:
        Updated OrchestratorState with tool_results and execution metadata
    """
    node_name = "tool_execution_node"
    start_time = datetime.now(timezone.utc)
    execution_start = time.time()
    
    logger.info(f"Tool execution node starting for session {state.session_id[:8]}")
    
    # Add system message for tracking
    state = MessageManager.add_system_message(
        state,
        f"Starting tool execution for {state.intent.value if state.intent else 'unknown'} intent",
        node_name
    )
    
    try:
        # Validate execution plan exists
        if not state.execution_plan:
            error_msg = "No execution plan provided"
            logger.error(error_msg)
            return state.model_copy(update={
                "metadata": {
                    **state.metadata,
                    "execution_status": "failure",
                    "execution_errors": [error_msg],
                    "execution_times": {},
                    "total_execution_time_ms": 0.0,
                    "steps_completed": 0,
                    "steps_failed": 0
                }
            })
        
        # Extract steps from execution plan
        steps = state.execution_plan.get("steps", [])
        if not steps:
            error_msg = "Execution plan contains no steps"
            logger.error(error_msg)
            return state.model_copy(update={
                "metadata": {
                    **state.metadata,
                    "execution_status": "failure",
                    "execution_errors": [error_msg],
                    "execution_times": {},
                    "total_execution_time_ms": 0.0,
                    "steps_completed": 0,
                    "steps_failed": 0
                }
            })
        
        logger.info(f"Executing {len(steps)} steps from execution plan")
        
        # Determine execution strategy
        execution_strategy = _determine_execution_strategy(steps)
        logger.info(f"Using execution strategy: {execution_strategy}")
        
        # Prepare base extracted params and enforce real user_id from state metadata
        base_extracted_params = dict(state.extracted_params or {})
        real_user_id = (state.metadata or {}).get("user_id")
        
        # Debug logging for user_id enforcement
        logger.info(f"Tool execution debug - extracted_params user_id: {base_extracted_params.get('user_id')}")
        logger.info(f"Tool execution debug - state metadata user_id: {real_user_id}")
        logger.info(f"Tool execution debug - state metadata keys: {list((state.metadata or {}).keys())}")
        logger.info(f"Tool execution debug - state session_id: {getattr(state, 'session_id', 'NOT_FOUND')}")
        
        if real_user_id:
            base_extracted_params["user_id"] = real_user_id
            logger.info(f"Tool execution debug - enforced user_id: {real_user_id}")
        else:
            logger.warning("Tool execution debug - no real user_id found in state metadata!")
            # Fallback to hardcoded test user for Studio
            fallback_user_id = "cmemx6bqy0000tb3perguhj4m"
            base_extracted_params["user_id"] = fallback_user_id
            logger.warning(f"Tool execution debug - using fallback user_id: {fallback_user_id}")
        
        # Execute steps with safe parameters
        if execution_strategy == "transaction":
            results, errors = await _execute_with_transaction(steps, base_extracted_params)
        else:
            results, errors = await _execute_without_transaction(steps, base_extracted_params)
        
        # Calculate execution statistics
        total_execution_time = (time.time() - execution_start) * 1000
        successful_steps = sum(1 for r in results if r.success)
        failed_steps = len(results) - successful_steps
        
        # Determine final execution status
        if failed_steps == 0:
            execution_status = "success"
        elif successful_steps > 0:
            execution_status = "partial_failure"
        else:
            execution_status = "failure"
        
        # Prepare tool results dictionary
        tool_results = {}
        execution_times = {}
        
        for result in results:
            step_key = f"step_{result.step_index}"
            tool_results[step_key] = result.to_dict()
            execution_times[step_key] = result.execution_time_ms
        
        # Add tool-specific keys for easier access
        for result in results:
            if result.success:
                tool_results[result.tool_name] = result.to_dict()
        
        logger.info(
            f"Tool execution completed: {execution_status}",
            extra={
                "steps_total": len(steps),
                "steps_successful": successful_steps,
                "steps_failed": failed_steps,
                "execution_time_ms": total_execution_time,
                "execution_status": execution_status,
                "session_id": state.session_id
            }
        )
        
        # Update state with results
        updated_state = state.model_copy(update={
            "tool_results": tool_results,
            "metadata": {
                **state.metadata,
                "execution_status": execution_status,
                "execution_errors": errors,
                "execution_times": execution_times,
                "total_execution_time_ms": total_execution_time,
                "steps_completed": successful_steps,
                "steps_failed": failed_steps,
                "execution_strategy": execution_strategy,
                "node_name": node_name,
                "node_timestamp": start_time.isoformat()
            }
        })
        
        # Add completion message
        updated_state = MessageManager.add_system_message(
            updated_state,
            f"Tool execution completed: {execution_status} ({successful_steps}/{len(steps)} steps successful)",
            node_name
        )
        
        return updated_state
        
    except Exception as e:
        execution_time = (time.time() - execution_start) * 1000
        
        logger.error(
            f"Tool execution node failed: {str(e)}",
            extra={
                "error": str(e),
                "execution_time_ms": execution_time,
                "session_id": state.session_id
            }
        )
        
        # Add error message
        state = MessageManager.add_system_message(
            state,
            f"Tool execution failed: {str(e)}",
            node_name
        )
        
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "execution_status": "failure",
                "execution_errors": [f"Tool execution failed: {str(e)}"],
                "execution_times": {},
                "total_execution_time_ms": execution_time,
                "steps_completed": 0,
                "steps_failed": 0,
                "node_name": node_name,
                "node_timestamp": start_time.isoformat()
            }
        }) 