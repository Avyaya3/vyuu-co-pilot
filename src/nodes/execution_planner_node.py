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

import json
import logging
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4

from ..config.settings import get_config
from ..schemas.state_schemas import OrchestratorState, PlanStep, ExecutionPlan
from ..schemas.generated_intent_schemas import IntentCategory
from ..tools import TOOL_REGISTRY, get_tool_info, get_tool_schema
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Intent to common operation mapping for fallback scenarios
INTENT_OPERATION_MAPPING = {
    IntentCategory.DATA_FETCH: {
        "account_balance": "get_account_balance",
        "transaction_history": "get_transaction_history",
        "account_info": "get_user_accounts"
    },
    IntentCategory.AGGREGATE: {
        "spending": "spending_by_category",
        "summary": "monthly_summary", 
        "analysis": "budget_analysis",
        "trends": "transaction_trends"
    },
    IntentCategory.ACTION: {
        "transfer": "transfer_money",
        "create": "create_transaction",
        "update": "update_transaction"
    }
}


class ExecutionPlannerError(Exception):
    """Exception raised when execution planning fails."""
    pass


async def execution_planner_node(state: OrchestratorState) -> OrchestratorState:
    """
    Generate and validate execution plan using hybrid LLM + rule-based approach.
    
    Args:
        state: Current orchestrator state with intent and extracted parameters
        
    Returns:
        Updated state with validated execution plan and metadata
    """
    start_time = time.time()
    planning_errors = []
    planning_status = "error"
    
    logger.info(
        f"Starting execution planning for intent: {state.intent}",
        extra={
            "intent": state.intent.value if state.intent else None,
            "session_id": state.session_id,
            "extracted_params": len(state.extracted_params) if state.extracted_params else 0
        }
    )
    
    try:
        # Step 1: Generate LLM draft plan
        draft_plan = await _generate_llm_draft_plan(state)
        
        if not draft_plan:
            planning_errors.append("LLM failed to generate valid plan")
            # Create fallback plan
            draft_plan = _create_fallback_plan(state)
        
        # Step 2: Validate and sanitize draft plan
        validated_steps, validation_errors = await _validate_and_sanitize_plan(
            draft_plan, 
            state.extracted_params
        )
        
        planning_errors.extend(validation_errors)
        
        # Step 3: Build final execution plan
        if validated_steps:
            execution_plan = ExecutionPlan(steps=validated_steps)
            planning_status = "success"
            
            logger.info(
                f"Execution plan created successfully with {len(validated_steps)} steps",
                extra={
                    "steps_count": len(validated_steps),
                    "planning_time_ms": (time.time() - start_time) * 1000,
                    "session_id": state.session_id
                }
            )
        else:
            planning_errors.append("No valid steps remained after validation")
            execution_plan = ExecutionPlan(steps=[])
            
            logger.warning(
                "No valid steps in execution plan",
                extra={
                    "planning_errors": planning_errors,
                    "session_id": state.session_id
                }
            )
        
        # Step 4: Update state
        updated_state = state.model_copy(update={
            "execution_plan": execution_plan.dict(),
            "metadata": {
                **state.metadata,
                "planning_status": planning_status,
                "planning_errors": planning_errors,
                "planning_time_ms": (time.time() - start_time) * 1000,
                "llm_planning_used": True,
                "steps_generated": len(validated_steps)
            }
        })
        
        return updated_state
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        error_msg = f"Execution planning failed: {str(e)}"
        planning_errors.append(error_msg)
        
        logger.error(
            error_msg,
            extra={
                "error": str(e),
                "intent": state.intent.value if state.intent else None,
                "session_id": state.session_id,
                "execution_time_ms": execution_time
            },
            exc_info=True
        )
        
        # Return error state
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "planning_status": "error",
                "planning_errors": planning_errors,
                "planning_time_ms": execution_time,
                "node_error": error_msg
            }
        })


async def _generate_llm_draft_plan(state: OrchestratorState) -> Optional[List[Dict[str, Any]]]:
    """
    Generate draft execution plan using LLM.
    
    Args:
        state: Current orchestrator state
        
    Returns:
        List of draft plan steps or None if generation fails
    """
    try:
        llm_client = LLMClient()
        
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
    
    if state.intent == IntentCategory.DATA_FETCH:
        # Default to account balance query
        return [{
            "tool_name": "db_query",
            "operation": "get_user_accounts",
            "params": base_params
        }]
    
    elif state.intent == IntentCategory.AGGREGATE:
        # Default to spending analysis
        params = base_params.copy()
        if "days_back" not in params:
            params["days_back"] = 30
        return [{
            "tool_name": "db_aggregate", 
            "operation": "spending_by_category",
            "params": params
        }]
    
    elif state.intent == IntentCategory.ACTION:
        # Cannot create safe fallback for actions
        logger.warning("Cannot create fallback plan for action intent")
        return []
    
    # Default fallback
    return [{
        "tool_name": "db_query",
        "operation": "get_user_accounts", 
        "params": base_params
    }] 