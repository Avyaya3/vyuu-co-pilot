import logging
import json
from typing import Any, Dict, Optional
from src.utils.parameter_config import get_parameter_config
from src.schemas.state_schemas import ClarificationState, IntentType
from src.schemas.intent_schemas import DataFetchParams, AggregateParams, ActionParams
from src.nodes.intent_classification_node import LLMClient
import asyncio

logger = logging.getLogger(__name__)

# Helper: Map intent to Pydantic param model
INTENT_PARAM_MODELS = {
    IntentType.DATA_FETCH: DataFetchParams,
    IntentType.AGGREGATE: AggregateParams,
    IntentType.ACTION: ActionParams,
}

def render_schema(intent: str, schemas: dict) -> str:
    """
    Render a markdown-style schema for the given intent, including slot names, types, and descriptions.
    """
    param_config = get_parameter_config()
    intent_schema = schemas.get(intent, {})
    critical = intent_schema.get("critical", [])
    optional = intent_schema.get("optional", [])
    # Try to get slot descriptions/types from Pydantic models
    pydantic_model = INTENT_PARAM_MODELS.get(IntentType(intent), None)
    slot_fields = {}
    if pydantic_model:
        for name, field in pydantic_model.model_fields.items():
            slot_fields[name] = {
                "type": str(field.annotation.__name__) if hasattr(field.annotation, "__name__") else str(field.annotation),
                "description": field.description or ""
            }
    def slot_line(slot):
        info = slot_fields.get(slot, {"type": "Any", "description": ""})
        return f"- {slot} ({info['type']}): {info['description']}"
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
    # Compose prompt
    prompt = f"""
        You are a slot-filling assistant. Here is the schema for "{intent}":
        {schema_text}

        User said: "{getattr(state, 'raw_user_input', state.user_input)}"
        Parsed so far: {json.dumps(state.extracted_parameters, ensure_ascii=False)}
        Missing slots: {json.dumps(state.missing_params, ensure_ascii=False)}
        Missing critical slots: {json.dumps(state.missing_critical_params, ensure_ascii=False)}
        Clarification history: {json.dumps(state.clarification_history, ensure_ascii=False)}

        Please return JSON with:
        - extracted_parameters: all slots, value or null
        - missing_params & missing_critical_params
        - parameter_priorities
        - normalization_suggestions
        - ambiguity_flags (map of slot_name to reason, only for genuinely ambiguous slots)
    """
    llm_client = LLMClient()
    try:
        # Use OpenAI best practice: system+user prompt, expect JSON
        response = await asyncio.to_thread(
            llm_client.client.chat.completions.create,
            model=llm_client.model,
            temperature=llm_client.temperature,
            messages=[
                {"role": "system", "content": "You are a careful slot-filling assistant for a financial assistant app. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        response_text = response.choices[0].message.content
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
    slot_list = schemas.get(intent, {}).get("critical", []) + schemas.get(intent, {}).get("optional", [])
    # Use Pydantic for type info if available
    pydantic_model = INTENT_PARAM_MODELS.get(IntentType(intent), None)
    validated_params = {}
    normalization_suggestions = llm_json.get("normalization_suggestions", {})
    ambiguity_flags = llm_json.get("ambiguity_flags", {})
    for slot in slot_list:
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
    # Compose updated state
    updated_state = state.model_copy(update={
        "extracted_parameters": validated_params,
        "missing_params": llm_json.get("missing_params", []),
        "missing_critical_params": llm_json.get("missing_critical_params", []),
        "parameter_priorities": llm_json.get("parameter_priorities", []),
        "normalization_suggestions": normalization_suggestions,
        "ambiguity_flags": ambiguity_flags,
    })
    logger.info(f"[MissingParamAnalysis] Completed for session {state.session_id[:8]}")
    return updated_state 