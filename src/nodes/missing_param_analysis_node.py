import logging
import json
from typing import Any, Dict, Optional, List
from src.utils.parameter_config import get_parameter_config
from src.schemas.state_schemas import ClarificationState, IntentType
from src.schemas.generated_intent_schemas import (
    DataFetchParams, 
    AggregateParams, 
    ActionParams,
    INTENT_PARAM_MODELS as GENERATED_INTENT_PARAM_MODELS,
    IntentCategory
)
from src.utils.llm_client import LLMClient
import asyncio

logger = logging.getLogger(__name__)

def get_pydantic_model_for_intent(intent: IntentType) -> Optional[type]:
    """
    Get the Pydantic parameter model for an intent type.
    
    Uses the auto-generated registry as the source of truth, converting
    from IntentType (state schemas) to IntentCategory (generated schemas).
    
    Args:
        intent: IntentType enum value
        
    Returns:
        Pydantic model class or None if not found
    """
    # Convert IntentType to IntentCategory
    intent_mapping = {
        IntentType.DATA_FETCH: IntentCategory.DATA_FETCH,
        IntentType.AGGREGATE: IntentCategory.AGGREGATE,
        IntentType.ACTION: IntentCategory.ACTION,
        IntentType.UNKNOWN: IntentCategory.UNKNOWN,
        IntentType.CLARIFICATION: IntentCategory.CLARIFICATION,
    }
    
    intent_category = intent_mapping.get(intent)
    if intent_category is None:
        logger.warning(f"No mapping found for IntentType: {intent}")
        return None
    
    return GENERATED_INTENT_PARAM_MODELS.get(intent_category)

def normalize_parameter_priorities(parameter_priorities: Any) -> List[str]:
    """
    Normalize parameter_priorities to ensure it's always a list of strings.
    
    Args:
        parameter_priorities: Raw parameter_priorities from LLM (could be dict, list, or other)
        
    Returns:
        Normalized list of parameter names as strings
    """
    if isinstance(parameter_priorities, list):
        # Already a list, convert all items to strings and filter out empty/None
        return [str(item) for item in parameter_priorities if item is not None and str(item).strip()]
    
    elif isinstance(parameter_priorities, dict):
        # Convert dict to list - use keys as parameter names
        # Sort by priority if values are numeric, otherwise use insertion order
        try:
            # Try to sort by numeric priority values
            sorted_items = sorted(parameter_priorities.items(), 
                                key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0,
                                reverse=True)
            return [str(key) for key, _ in sorted_items if key is not None and str(key).strip()]
        except (ValueError, TypeError):
            # Fallback to keys in insertion order
            return [str(key) for key in parameter_priorities.keys() if key is not None and str(key).strip()]
    
    elif isinstance(parameter_priorities, str):
        # Single string, treat as single-item list if not empty
        return [parameter_priorities] if parameter_priorities.strip() else []
    
    else:
        # Unknown type, return empty list
        logger.warning(f"Unknown parameter_priorities type: {type(parameter_priorities)}, value: {parameter_priorities}")
        return []


def normalize_missing_params(missing_params: Any) -> List[str]:
    """
    Normalize missing_params to ensure it's always a list of strings.
    
    Args:
        missing_params: Raw missing_params from LLM
        
    Returns:
        Normalized list of parameter names as strings
    """
    if isinstance(missing_params, list):
        return [str(item) for item in missing_params if item is not None and str(item).strip()]
    elif isinstance(missing_params, str):
        return [missing_params] if missing_params.strip() else []
    else:
        logger.warning(f"Unknown missing_params type: {type(missing_params)}, value: {missing_params}")
        return []

def render_schema(intent: str, schemas: dict) -> str:
    """
    Render a markdown-style schema for the given intent, including slot names, types, and descriptions.
    """
    param_config = get_parameter_config()
    intent_schema = schemas.get(intent, {})
    critical = intent_schema.get("critical", [])
    optional = intent_schema.get("optional", [])
    # Try to get slot descriptions/types from Pydantic models
    pydantic_model = get_pydantic_model_for_intent(IntentType(intent))
    slot_fields = {}
    if pydantic_model:
        for name, field in pydantic_model.model_fields.items():
            slot_fields[name] = {
                "type": str(field.annotation.__name__) if hasattr(field.annotation, "__name__") else str(field.annotation),
                "description": field.description or ""
            }
    def slot_line(slot):
        # Handle both old format (string) and new format (dict)
        if isinstance(slot, str):
            # Old format - simple string
            info = slot_fields.get(slot, {"type": "Any", "description": ""})
            return f"- {slot} ({info['type']}): {info['description']}"
        elif isinstance(slot, dict):
            # New format - dictionary with name, type, description
            slot_name = slot.get('name', 'unknown')
            slot_type = slot.get('type', 'Any')
            slot_description = slot.get('description', '')
            return f"- {slot_name} ({slot_type}): {slot_description}"
        else:
            # Fallback for unexpected format
            return f"- {str(slot)} (Any): No description available"
    lines = []
    if critical:
        lines.append("**Critical slots:**")
        lines += [slot_line(s) for s in critical]
    if optional:
        lines.append("**Optional slots:**")
        lines += [slot_line(s) for s in optional]
    return "\n".join(lines)


async def missing_param_analysis_node(state: ClarificationState) -> ClarificationState:
    """
    Analyze, normalize, and prioritize missing parameters using an LLM.
    Args:
        state: ClarificationState with intent, user input, extracted_parameters, missing_params, etc.
    Returns:
        Updated ClarificationState with enriched slot info for clarification question generation.
    """
    logger.info(f"[MissingParamAnalysis] Starting for session {state.session_id[:8]} intent={state.intent}")
    param_config = get_parameter_config()
    schemas = param_config._config_data.get("intent_parameters", {})
    intent = state.intent.value if hasattr(state.intent, "value") else str(state.intent)
    schema_text = render_schema(intent, schemas)
    
    # Compose prompt with explicit format instructions
    prompt = f"""
        You are a slot-filling assistant. Here is the schema for "{intent}":
        {schema_text}

        User said: "{getattr(state, 'raw_user_input', state.user_input)}"
        Parsed so far: {json.dumps(state.extracted_parameters, ensure_ascii=False)}
        Missing slots: {json.dumps(state.missing_params, ensure_ascii=False)}
        Missing critical slots: {json.dumps(state.missing_critical_params, ensure_ascii=False)}
        Clarification history: {json.dumps(state.clarification_history, ensure_ascii=False)}

        Please return JSON with the following EXACT structure:
        {{
            "extracted_parameters": {{"slot_name": "value" or null}},
            "missing_params": ["slot_name1", "slot_name2"],
            "missing_critical_params": ["critical_slot_name"],
            "parameter_priorities": ["slot_name1", "slot_name2"],
            "normalization_suggestions": {{"user_term": "canonical_value"}},
            "ambiguity_flags": {{"slot_name": "reason"}}
        }}

        IMPORTANT: 
        - parameter_priorities must be a LIST of strings, not a dictionary
        - missing_params must be a LIST of strings
        - missing_critical_params must be a LIST of strings
        - Order parameter_priorities by importance (most important first)
    """
    
    llm_client = LLMClient()
    try:
        # Use simplified LLM client
        response_text = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a careful slot-filling assistant for a financial assistant app. Always return valid JSON with the exact structure requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent analysis
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        logger.info(f"[MissingParamAnalysis] LLM raw response: {response_text}")
        llm_json = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"[MissingParamAnalysis] Failed to parse LLM JSON: {e}")
        return state.model_copy()
    except Exception as e:
        logger.error(f"[MissingParamAnalysis] LLM call failed: {e}")
        return state.model_copy()

    # Validate and normalize output
    # Get slot list from schema (critical+optional)
    critical_slots = schemas.get(intent, {}).get("critical", [])
    optional_slots = schemas.get(intent, {}).get("optional", [])
    slot_list = critical_slots + optional_slots
    
    # Extract slot names (handle both old format strings and new format dicts)
    slot_names = []
    for slot in slot_list:
        if isinstance(slot, str):
            slot_names.append(slot)
        elif isinstance(slot, dict):
            slot_names.append(slot.get('name', ''))
    slot_names = [name for name in slot_names if name]  # Filter out empty strings
    
    # Use Pydantic for type info if available
    pydantic_model = get_pydantic_model_for_intent(IntentType(intent))
    validated_params = {}
    normalization_suggestions = llm_json.get("normalization_suggestions", {})
    ambiguity_flags = llm_json.get("ambiguity_flags", {})
    
    for slot in slot_names:
        value = llm_json.get("extracted_parameters", {}).get(slot, None)
        # Type check using Pydantic if possible
        if pydantic_model and slot in pydantic_model.model_fields and value is not None:
            try:
                # Create a temp instance to validate this field
                temp_data = {slot: value}
                temp_instance = pydantic_model.model_validate(temp_data, strict=False)
                validated_params[slot] = getattr(temp_instance, slot)
            except Exception as e:
                logger.warning(f"[MissingParamAnalysis] Slot '{slot}' failed validation: {e}. Setting to null.")
                validated_params[slot] = None
                ambiguity_flags[slot] = f"Validation error: {e}"
        else:
            validated_params[slot] = value
    
    # Normalize all list fields to ensure consistent format
    normalized_missing_params = normalize_missing_params(llm_json.get("missing_params", []))
    normalized_missing_critical_params = normalize_missing_params(llm_json.get("missing_critical_params", []))
    normalized_parameter_priorities = normalize_parameter_priorities(llm_json.get("parameter_priorities", []))
    
    # Log normalization results for debugging
    logger.info(f"[MissingParamAnalysis] Normalized parameter_priorities: {normalized_parameter_priorities}")
    logger.info(f"[MissingParamAnalysis] Normalized missing_params: {normalized_missing_params}")
    
    # Compose updated state
    updated_state = state.model_copy(update={
        "extracted_parameters": validated_params,
        "missing_params": normalized_missing_params,
        "missing_critical_params": normalized_missing_critical_params,
        "parameter_priorities": normalized_parameter_priorities,
        "normalization_suggestions": normalization_suggestions,
        "ambiguity_flags": ambiguity_flags,
    })
    
    logger.info(f"[MissingParamAnalysis] Completed for session {state.session_id[:8]}")
    return updated_state 