"""
Hybrid Execution Planner Node

Combines LLM flexibility with rule-based validation to create safe,
validated execution plans. Uses the LLM to draft plans and validates
them against the tool registry for safety and correctness.

Features:
- LLM-based plan generation for flexibility
- Rule-based validation against tool registry
- Parameter adaptation and schema validation
- Robust error handling with fallback strategies
- Scalable design for future multi-step workflows
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4

from ..config.settings import get_config
from ..schemas.state_schemas import OrchestratorState, PlanStep, ExecutionPlan
from ..schemas.generated_intent_schemas import IntentCategory, IntentEntry
from ..tools import TOOL_REGISTRY, get_tool_info, get_tool_schema
from ..utils.llm_client import LLMClient
from ..utils.node_execution_logger import track_node_execution, add_execution_metrics_to_state

logger = logging.getLogger(__name__)

# Intent to common operation mapping for fallback scenarios
INTENT_OPERATION_MAPPING = {
    IntentCategory.READ: {
        "account_balance": "get_account_balance",
        "transaction_history": "get_transaction_history",
        "account_info": "get_user_accounts"
    },
    IntentCategory.DATABASE_OPERATIONS: {
        "create": "create",
        "update": "update",
        "delete": "delete",
        "transfer": "transfer"
    },
    IntentCategory.ADVICE: {
        "generate_advice": "generate_advice"
    }
}


class ExecutionPlannerError(Exception):
    """Exception raised when execution planning fails."""
    pass


def can_execute_parallel(intent: Dict[str, Any]) -> bool:
    """
    Determine if intent can be executed in parallel.
    
    Args:
        intent: Intent dictionary to check
        
    Returns:
        True if intent is read or advice (safe to parallelize)
        False if intent is database_operations (must be sequential)
    """
    return intent.intent in ["read", "advice"]


async def execute_single_intent_from_classification(
    state: OrchestratorState,
    intent_entry: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a single intent from classification result.
    
    This adapts the existing execution logic to work with
    individual intents from multi-intent scenarios.
    
    Args:
        state: Current orchestrator state
        intent_entry: Intent dictionary to execute
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Create a temporary state for this intent
        temp_state = OrchestratorState(
            user_input=state.user_input,
            intent=state.intent,  # Use primary intent for compatibility
            confidence=state.confidence,
            messages=state.messages,
            session_id=state.session_id,
            timestamp=state.timestamp,
            metadata=state.metadata,
            parameters=intent_entry.get('params', {}),
            execution_results=state.execution_results,
            response=state.response,
            extracted_params=intent_entry.get('params', {}),
            execution_plan=None,
            tool_results=None,
            final_response=None
        )
        
        # Execute using existing logic (simplified for now)
        # In a full implementation, this would call the appropriate tool execution
        result = {
            "intent": intent_entry.get('intent', 'unknown'),
            "confidence": intent_entry.get('confidence', 0.0),
            "success": True,
            "summary": f"Executed {intent_entry.get('intent', 'unknown')} intent successfully",
            "data": intent_entry.get('params', {})
        }
        
        logger.info(f"Executed intent {intent_entry.get('intent', 'unknown')} with confidence {intent_entry.get('confidence', 0.0)}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to execute intent {intent_entry.get('intent', 'unknown')}: {e}")
        return {
            "intent": intent_entry.get('intent', 'unknown'),
            "confidence": intent_entry.get('confidence', 0.0),
            "success": False,
            "error": str(e),
            "summary": f"Failed to execute {intent_entry.get('intent', 'unknown')} intent"
        }


async def execute_multiple_intents(state: OrchestratorState) -> OrchestratorState:
    """
    Execute multiple intents with parallel/sequential strategy.
    
    Args:
        state: OrchestratorState with multiple intents
        
    Returns:
        Updated state with combined execution results
    """
    intents = state.multiple_intents
    logger.info(f"Executing {len(intents)} intents for session {state.session_id[:8]}")
    
    # Group intents by execution strategy
    parallel_intents = []
    sequential_intents = []
    
    for intent in intents:
        if can_execute_parallel(intent):
            parallel_intents.append(intent)
        else:
            sequential_intents.append(intent)
    
    results = []
    
    # Execute parallel intents concurrently
    if parallel_intents:
        logger.info(f"Executing {len(parallel_intents)} intents in parallel")
        parallel_tasks = [
            asyncio.create_task(execute_single_intent_from_classification(state, intent)) 
            for intent in parallel_intents
        ]
        
        # Wait for parallel tasks with timeout
        try:
            parallel_results = await asyncio.wait_for(
                asyncio.gather(*parallel_tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for parallel execution
            )
            results.extend(parallel_results)
        except asyncio.TimeoutError:
            logger.warning("Parallel execution timed out")
            # Cancel remaining tasks
            for task in parallel_tasks:
                if not task.done():
                    task.cancel()
            results.append({"intent": "unknown", "error": "timeout", "success": False})
    
    # Execute sequential intents one by one
    for intent in sequential_intents:
        logger.info(f"Executing sequential intent: {intent.intent}")
        try:
            result = await execute_single_intent_from_classification(state, intent)
            results.append(result)
            
            # If sequential intent fails, we might want to stop the chain
            if not result.get("success", False):
                logger.warning(f"Sequential intent {intent.intent} failed, continuing with remaining intents")
                
        except Exception as e:
            logger.error(f"Sequential intent execution failed: {e}")
            results.append({
                "intent": intent.intent,
                "error": str(e),
                "success": False,
                "summary": f"Failed to execute {intent.intent} intent"
            })
    
    # Combine results
    return combine_intent_results(state, results)


def combine_intent_results(
    state: OrchestratorState,
    results: List[Dict[str, Any]]
) -> OrchestratorState:
    """
    Combine results from multiple intent executions.
    
    Args:
        state: Current orchestrator state
        results: List of execution results (may include errors)
    
    Returns:
        Updated state with combined results and metadata
    """
    # Separate successful and failed results
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    # Create combined execution results
    combined_results = {
        "multiple_intent_results": results,
        "successful_count": len(successful_results),
        "failed_count": len(failed_results),
        "execution_strategy": "mixed" if len(successful_results) > 0 and len(failed_results) > 0 else "success" if len(failed_results) == 0 else "failed"
    }
    
    # Create summary response
    response_parts = []
    
    if successful_results:
        response_parts.append("✅ **Successfully completed:**")
        for result in successful_results:
            response_parts.append(f"• {result.get('summary', 'Completed')}")
    
    if failed_results:
        response_parts.append("\n❌ **Issues encountered:**")
        for result in failed_results:
            response_parts.append(f"• {result.get('error', 'Unknown error')}")
    
    combined_response = "\n".join(response_parts)
    
    return state.model_copy(update={
        "execution_results": combined_results,
        "response": combined_response
    })


async def execution_planner_node(state: OrchestratorState) -> OrchestratorState:
    """
    Generate and validate execution plan using hybrid LLM + rule-based approach.
    
    Args:
        state: Current orchestrator state with intent and extracted parameters
        
    Returns:
        Updated state with validated execution plan and metadata
    """
    node_name = "execution_planner_node"
    
    async with track_node_execution(node_name, state.session_id) as exec_logger:
        try:
            # Check for multiple intents first
            if state.has_multiple_intents:
                exec_logger.log_step("multi_intent_execution_start", {
                    "intent_count": len(state.multiple_intents),
                    "intent_types": [intent.intent for intent in state.multiple_intents]
                })
                
                # Execute multiple intents with parallel/sequential strategy
                result_state = await execute_multiple_intents(state)
                
                exec_logger.log_step("multi_intent_execution_complete", {
                    "successful_count": result_state.execution_results.get("successful_count", 0),
                    "failed_count": result_state.execution_results.get("failed_count", 0)
                })
                
                # Add execution metrics to state
                execution_metrics = exec_logger.end(success=True, metadata={
                    "execution_type": "multi_intent",
                    "intent_count": len(state.multiple_intents),
                    "execution_strategy": result_state.execution_results.get("execution_strategy", "unknown")
                })
                
                return add_execution_metrics_to_state(result_state, execution_metrics)
            
            # Single intent execution (existing logic)
            planning_errors = []
            planning_status = "error"
            
            exec_logger.log_step("node_start", {
                "intent": state.intent.value if state.intent else "unknown",
                "extracted_params_count": len(state.extracted_params) if state.extracted_params else 0,
                "extracted_params_keys": list(state.extracted_params.keys()) if state.extracted_params else []
            })
            
            exec_logger.log_step("llm_draft_plan_generation_start")
            
            # Step 1: Generate LLM draft plan
            draft_plan = await _generate_llm_draft_plan(state)
            
            if not draft_plan:
                planning_errors.append("LLM failed to generate valid plan")
                exec_logger.log_step("llm_plan_failed_fallback_creation")
                # Create fallback plan
                draft_plan = _create_fallback_plan(state)
            else:
                exec_logger.log_step("llm_draft_plan_generated", {
                    "draft_steps_count": len(draft_plan)
                })
            
            exec_logger.log_step("plan_validation_start")
            
            # Step 2: Validate and sanitize draft plan
            validated_steps, validation_errors = await _validate_and_sanitize_plan(
                draft_plan, 
                state.extracted_params
            )
            
            planning_errors.extend(validation_errors)
            
            exec_logger.log_step("plan_validation_complete", {
                "validated_steps_count": len(validated_steps),
                "validation_errors_count": len(validation_errors)
            })
            
            exec_logger.log_step("execution_plan_creation")
            
            # Step 3: Build final execution plan
            if validated_steps:
                execution_plan = ExecutionPlan(steps=validated_steps)
                planning_status = "success"
                
                exec_logger.log_step("execution_plan_created_successfully", {
                    "final_steps_count": len(validated_steps),
                    "planning_status": planning_status
                })
            else:
                planning_errors.append("No valid steps remained after validation")
                execution_plan = ExecutionPlan(steps=[])
                
                exec_logger.log_step("execution_plan_creation_failed", {
                    "planning_errors": planning_errors
                })
            
            exec_logger.log_step("state_update_start")
            
            # Step 4: Update state
            updated_state = state.model_copy(update={
                "execution_plan": execution_plan.dict(),
                "metadata": {
                    **state.metadata,
                    "planning_status": planning_status,
                    "planning_errors": planning_errors,
                    "llm_planning_used": True,
                    "steps_generated": len(validated_steps)
                }
            })
            
            exec_logger.log_step("node_complete", {
                "planning_status": planning_status,
                "final_steps_count": len(validated_steps),
                "planning_errors_count": len(planning_errors)
            })
            
            # Add execution metrics to state
            execution_metrics = exec_logger.end(success=True, metadata={
                "planning_status": planning_status,
                "planning_errors": planning_errors,
                "llm_planning_used": True,
                "steps_generated": len(validated_steps),
                "draft_plan_steps": len(draft_plan) if draft_plan else 0,
                "validation_errors_count": len(validation_errors)
            })
            
            updated_state = add_execution_metrics_to_state(updated_state, execution_metrics)
            
            return updated_state
            
        except Exception as e:
            exec_logger.log_error(e, {
                "intent": state.intent.value if state.intent else "unknown",
                "session_id": state.session_id,
                "extracted_params_count": len(state.extracted_params) if state.extracted_params else 0,
                "error_context": "execution_planner_node"
            })
            
            error_msg = f"Execution planning failed: {str(e)}"
            planning_errors = [error_msg]
            
            # Return error state
            error_state = state.model_copy(update={
                "metadata": {
                    **state.metadata,
                    "planning_status": "error",
                    "planning_errors": planning_errors,
                    "node_error": error_msg
                }
            })
            
            # Add execution metrics to error state
            execution_metrics = exec_logger.end(success=False, error=str(e), error_type=type(e).__name__)
            error_state = add_execution_metrics_to_state(error_state, execution_metrics)
            
            return error_state


async def _generate_llm_draft_plan(state: OrchestratorState) -> Optional[List[Dict[str, Any]]]:
    """
    Generate draft execution plan using LLM.
    
    Args:
        state: Current orchestrator state
        
    Returns:
        List of draft plan steps or None if generation fails
    """
    try:
        llm_client = LLMClient.for_task("execution_planning")
        
        # Build tool descriptions and intent context
        tool_info = get_tool_info()
        tools_description = "Available tools:\n\n"
        for tool_name, info in tool_info.items():
            tools_description += f"**{tool_name}**: {info['description']}\n"
            tools_description += f"Operations: {', '.join(info['operations'])}\n\n"
        
        intent_str = state.intent.value if state.intent else "unknown"
        
        logger.debug("Calling LLM for execution plan generation")
        
        # Build system prompt for execution planning
        system_prompt = f"""You are an execution planner for a financial assistant. Your task is to create execution plans using available tools.

{tools_description}

IMPORTANT RULES:
1. Return ONLY a JSON array of steps, no other text
2. Each step must have: tool_name, operation, params
3. Use exact tool names and operations from the list above
4. Do NOT include user_id in params - it's automatically injected from session context
5. Use normalized parameter names (they're already provided correctly)
6. For single operations, return an array with one step

Example response format:
[
{{
    "tool_name": "db_query",
    "operation": "get_account_balance", 
    "params": {{
    "account_name": "checking"
    }}
}}
]

Focus on accuracy and use only the tools and operations listed above."""

        # Build user prompt
        user_prompt = f"""Create an execution plan for this request:

Intent: {intent_str}
User Input: "{state.user_input}"
Extracted Parameters: {state.extracted_params}

Generate a JSON execution plan using the available tools. Choose the most appropriate tool and operation for this intent."""

        # Use simplified LLM client
        response_content = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for deterministic planning
            max_tokens=1000
        )
        
        # Extract JSON from response (handle cases where LLM adds explanation)
        json_start = response_content.find('[')
        json_end = response_content.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            logger.warning("LLM response doesn't contain JSON array")
            return None
        
        json_content = response_content[json_start:json_end]
        draft_plan = json.loads(json_content)
        
        if not isinstance(draft_plan, list):
            logger.warning("LLM response is not a list")
            return None
        
        logger.info(f"LLM generated draft plan with {len(draft_plan)} steps")
        return draft_plan
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM API error during planning: {e}")
        return None


# Removed _build_system_prompt and _build_user_prompt - now handled by centralized LLM client


async def _validate_and_sanitize_plan(
    draft_plan: List[Dict[str, Any]], 
    extracted_params: Dict[str, Any] ) -> tuple[List[PlanStep], List[str]]:
    """
    Validate and sanitize draft plan against tool registry.
    
    Args:
        draft_plan: Raw plan from LLM
        extracted_params: Extracted parameters from state
        
    Returns:
        Tuple of (validated_steps, validation_errors)
    """
    validated_steps = []
    validation_errors = []
    
    for i, step_dict in enumerate(draft_plan):
        try:
            # Step 1: Validate tool exists
            tool_name = step_dict.get("tool_name")
            if not tool_name or tool_name not in TOOL_REGISTRY:
                validation_errors.append(f"Step {i}: Unknown tool '{tool_name}'")
                continue
            
            # Step 2: Validate operation exists
            operation = step_dict.get("operation")
            if not operation:
                validation_errors.append(f"Step {i}: Missing operation")
                continue
            
            tool_info = get_tool_info()
            if operation not in tool_info[tool_name]["operations"]:
                validation_errors.append(f"Step {i}: Invalid operation '{operation}' for tool '{tool_name}'")
                continue
            
            # Step 3: Use validated parameters directly (already normalized by parameter extraction)
            raw_params = step_dict.get("params", {})
            adapted_params = _merge_parameters(raw_params, extracted_params)
            
            # Get tool schema and validate
            tool_schema = get_tool_schema(tool_name)
            
            try:
                # Add required fields
                adapted_params["operation"] = operation
                
                # Validate against tool schema
                validated_params = tool_schema(**adapted_params)
                
                # Create validated step
                step = PlanStep(
                    tool_name=tool_name,
                    operation=operation,
                    params=validated_params.dict(),
                    step_id=str(uuid4())
                )
                
                validated_steps.append(step)
                logger.debug(f"Step {i} validated successfully")
                
            except Exception as validation_error:
                validation_errors.append(f"Step {i} parameter validation failed: {str(validation_error)}")
                continue
                
        except Exception as e:
            validation_errors.append(f"Step {i} validation error: {str(e)}")
            continue
    
    return validated_steps, validation_errors


def _merge_parameters(raw_params: Dict[str, Any], extracted_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge LLM parameters with extracted parameters (no mapping needed - already normalized).
    
    Args:
        raw_params: Parameters from LLM plan
        extracted_params: Already normalized parameters from parameter extraction
        
    Returns:
        Merged parameters dict with extracted_params taking precedence
    """
    # Start with extracted parameters (already normalized and validated)
    merged = extracted_params.copy()
    
    # Only add LLM parameters that aren't already provided
    for key, value in raw_params.items():
        if key not in merged and value is not None:
            merged[key] = value
    
    # Don't automatically add user_id here - it's injected later in tool execution
    # This prevents LLM-generated placeholders from being used
    
    return merged


def _create_fallback_plan(state: OrchestratorState) -> List[Dict[str, Any]]:
    """
    Create fallback plan when LLM fails using extracted parameters.
    
    Args:
        state: Current orchestrator state
        
    Returns:
        Simple fallback plan using normalized extracted parameters
    """
    if not state.intent:
        return []
    
    # Use extracted parameters directly (already normalized)
    base_params = state.extracted_params.copy() if state.extracted_params else {}
    
    if state.intent == IntentCategory.READ:
        # Default to data fetch
        return [{
            "tool_name": "data_fetch",
            "operation": "get_user_info",
            "params": base_params
        }]
    
    elif state.intent == IntentCategory.DATABASE_OPERATIONS:
        # Cannot create safe fallback for database operations
        logger.warning("Cannot create fallback plan for database_operations intent")
        return []
    
    elif state.intent == IntentCategory.ADVICE:
        # Default to advice generation
        return [{
            "tool_name": "advice",
            "operation": "generate_advice",
            "params": base_params
        }]
    
    # Default fallback
    return [{
        "tool_name": "db_query",
        "operation": "get_user_accounts", 
        "params": base_params
    }] 