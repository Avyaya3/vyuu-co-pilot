"""
Intent Classification Node for LangGraph with OpenAI Integration.

This module provides the main entry point for intent classification in the LangGraph
orchestration system, using real OpenAI API calls for intelligent classification.
"""

import json
import logging
import os
from typing import Any, Dict, Optional
from datetime import datetime, timezone

import openai
from dotenv import load_dotenv

from vyuu_copilot_v2.schemas.state_schemas import (
    MainState,
    MessageManager,
    IntentType,
    ConversationContext,
)
from vyuu_copilot_v2.schemas.generated_intent_schemas import (
    IntentClassificationResult,
    IntentClassificationError,
    FallbackIntentResult,
    IntentCategory,
    IntentEntry,
    ReadParams,
    DatabaseOperationsParams,
    AdviceParams,
)
from vyuu_copilot_v2.utils.llm_client import LLMClient
from vyuu_copilot_v2.utils.node_execution_logger import track_node_execution, add_execution_metrics_to_state

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# LLMClient moved to src/utils/llm_client.py for centralized access


class IntentClassifier:
    """
    Main intent classifier with OpenAI LLM integration and fallback handling.
    
    This class handles the full intent classification pipeline including
    LLM calls, result parsing, validation, and fallback mechanisms.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the intent classifier with optimized LLM client.
        
        Args:
            llm_client: Optional LLM client instance. If not provided, creates optimized one.
        """
        self.llm_client = llm_client or LLMClient.for_task("intent_classification")
        logger.info("Intent classifier initialized with optimized LLM client for classification")
    
    async def classify_intent(self, user_input: str, conversation_context: str = "") -> IntentClassificationResult:
        """
        Classify user intent using centralized LLM client.
        
        Args:
            user_input: The user's input to classify
            conversation_context: Previous conversation context
            
        Returns:
            Structured intent classification result
            
        Raises:
            IntentClassificationError: When classification fails
        """
        if not user_input or not user_input.strip():
            raise IntentClassificationError(
                "Empty user input provided",
                error_type="validation_error",
                user_input=user_input
            )
        
        try:
            logger.debug(f"Classifying intent for input: '{user_input[:50]}...'")
            
            # Build system prompt for intent classification
            system_prompt = """You are an AI assistant specialized in classifying user intents for a financial management application.

Your task is to analyze user requests and classify them into one or more intent categories:

1. **read**: User wants to retrieve/view existing financial data
- Examples: "Show me my transactions", "What's my balance?", "List my accounts"
- Extract parameters like: entity_type, time_period, account_types, limit, sort_by, order, filters

2. **database_operations**: User wants to perform database operations (create, update, delete, transfer)
- Examples: "Transfer ₹100 to savings", "Pay my electric bill", "Create a new budget", "Delete this transaction"
- Extract parameters like: action_type, entity_type, entity_id, data, user_id

3. **advice**: User wants financial advice or recommendations
- Examples: "How can I save more money?", "What should I invest in?", "Give me budgeting advice"
- Extract parameters like: user_query, context_data, user_id

4. **unknown**: Intent is unclear or doesn't fit the above categories

**MULTI-INTENT DETECTION:**
- A single user request can contain multiple distinct intents
- Example: "Show me my expenses and give me budgeting advice" = [read, advice]
- Example: "Transfer ₹1000 to savings and show me the updated balance" = [database_operations, read]
- Example: "Show my budget, create a savings goal, and give me investment advice" = [read, database_operations, advice]

**RESPONSE FORMAT:**
Return JSON in one of these formats:

**Single Intent (existing format):**
{
"intent": "read|database_operations|advice|unknown",
"confidence": 0.85,
"reasoning": "brief explanation",
"user_input_analysis": "analysis of the input",
"missing_params": ["param1", "param2"],
"clarification_needed": true,
"read_params": { ... },
"database_operations_params": { ... },
"advice_params": { ... }
}

**Multiple Intents (new format):**
{
"multiple_intents": [
    {
        "intent": "read",
        "confidence": 0.9,
        "reasoning": "User wants to view financial data",
        "params": {
            "entity_type": "expenses",
            "time_period": "last_month"
        }
    },
    {
        "intent": "advice",
        "confidence": 0.85,
        "reasoning": "User wants budgeting advice",
        "params": {
            "user_query": "budgeting advice"
        }
    }
]
}

**MULTI-INTENT RULES:**
- Only use multiple_intents array if you detect 2+ distinct intents
- Each intent must have confidence >= 0.5
- Extract parameters for each intent separately in the "params" field
- Set clarification_needed=true if ANY intent needs clarification
- Single intent uses existing format (no array)

**IMPORTANT DATA TYPE RULES:**
- ALL array fields must be arrays, not strings: ["category"] not "category"
- Only include the params object for the matched intent type
- Set null for unused param objects

Set confidence based on clarity: 0.8-1.0 (high), 0.5-0.79 (medium), 0.0-0.49 (low).
Set clarification_needed=true if confidence < 0.7 or critical parameters are missing.

Return only valid JSON. Do not include any text outside the JSON."""

            # Build user prompt
            user_prompt = f"""Classify this user request:

User Input: "{user_input}"

Recent Context:
{conversation_context}

Respond with valid JSON containing intent classification and extracted parameters."""

            # Use simplified LLM client
            response_content = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            llm_result = json.loads(response_content)
            
            # Parse and validate result
            classification_result = self._parse_llm_result(llm_result, user_input)
            
            logger.debug(f"Intent classification successful: {classification_result.intent} (confidence: {classification_result.confidence})")
            
            return classification_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return self._create_fallback_result(e, user_input)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._create_fallback_result(e, user_input)
    
    def _parse_llm_result(self, llm_result: Dict[str, Any], user_input: str) -> IntentClassificationResult:
        """
        Parse LLM result into structured IntentClassificationResult.
        Handles both single and multiple intent responses.
        """
        try:
            # Check for multiple intents
            if "multiple_intents" in llm_result:
                return self._parse_multiple_intents(llm_result, user_input)
            else:
                return self._parse_single_intent(llm_result, user_input)
                
        except Exception as e:
            logger.error(f"Failed to parse LLM result: {e}")
            raise IntentClassificationError(
                f"Invalid LLM response format: {str(e)}",
                error_type="parsing_error",
                user_input=user_input
            )
    
    def _parse_multiple_intents(self, llm_result: Dict[str, Any], user_input: str) -> IntentClassificationResult:
        """Parse multiple intents from LLM response."""
        multiple_intents = []
        
        for intent_data in llm_result["multiple_intents"]:
            intent_entry = IntentEntry(
                intent=intent_data["intent"],
                confidence=float(intent_data.get("confidence", 0.0)),
                reasoning=intent_data.get("reasoning"),
                params=intent_data.get("params", {})
            )
            multiple_intents.append(intent_entry)
        
        # Sort by confidence (highest first)
        multiple_intents.sort(key=lambda x: x.confidence, reverse=True)
        
        # Create primary intent result (highest confidence) for backward compatibility
        primary_intent = multiple_intents[0]
        
        # Parse primary intent parameters
        data_fetch_params = None
        action_params = None
        
        if primary_intent.intent == IntentCategory.READ and primary_intent.params:
            data_fetch_params = ReadParams(**primary_intent.params)
        elif primary_intent.intent == IntentCategory.DATABASE_OPERATIONS and primary_intent.params:
            action_params = DatabaseOperationsParams(**primary_intent.params)
        elif primary_intent.intent == IntentCategory.ADVICE and primary_intent.params:
            action_params = AdviceParams(**primary_intent.params)
        
        # Create result with primary intent for backward compatibility
        result = IntentClassificationResult(
            intent=primary_intent.intent,
            confidence=primary_intent.confidence,
            reasoning=primary_intent.reasoning or "Multi-intent classification",
            read_params=data_fetch_params,
            database_operations_params=action_params if primary_intent.intent == IntentCategory.DATABASE_OPERATIONS else None,
            advice_params=action_params if primary_intent.intent == IntentCategory.ADVICE else None,
            missing_params=[],
            clarification_needed=any(intent.confidence < 0.7 for intent in multiple_intents),
            user_input_analysis=f"Multi-intent analysis of: '{user_input[:50]}...'"
        )
        
        # Store multiple intents as dictionaries for state compatibility
        result.multiple_intents = [intent.model_dump() for intent in multiple_intents]
        
        logger.debug(f"Successfully parsed {len(multiple_intents)} intents: {[i.intent for i in multiple_intents]}")
        return result
    
    def _parse_single_intent(self, llm_result: Dict[str, Any], user_input: str) -> IntentClassificationResult:
        """Parse single intent from LLM response (existing logic)."""
        # Extract intent-specific parameters
        data_fetch_params = None
        action_params = None
        
        if llm_result.get("intent") == IntentCategory.READ and llm_result.get("read_params"):
            data_fetch_params = ReadParams(**llm_result["read_params"])
        elif llm_result.get("intent") == IntentCategory.DATABASE_OPERATIONS and llm_result.get("database_operations_params"):
            action_params = DatabaseOperationsParams(**llm_result["database_operations_params"])
        elif llm_result.get("intent") == IntentCategory.ADVICE and llm_result.get("advice_params"):
            action_params = AdviceParams(**llm_result["advice_params"])
        
        # Create and validate result
        result = IntentClassificationResult(
            intent=llm_result["intent"],
            confidence=llm_result["confidence"],
            reasoning=llm_result.get("reasoning", "Intent classified by OpenAI"),
            read_params=data_fetch_params,
            database_operations_params=action_params if llm_result.get("intent") == IntentCategory.DATABASE_OPERATIONS else None,
            advice_params=action_params if llm_result.get("intent") == IntentCategory.ADVICE else None,
            missing_params=llm_result.get("missing_params", []),
            clarification_needed=llm_result.get("clarification_needed", False),
            user_input_analysis=llm_result.get("user_input_analysis", f"Analysis of: '{user_input[:50]}...'")
        )
        
        logger.debug(f"Successfully parsed single intent: {result.intent} (params: {len(result.extracted_parameters)})")
        return result
    
    def _create_fallback_result(self, error: Exception, user_input: str) -> FallbackIntentResult:
        """Create fallback result when classification fails."""
        logger.info(f"Creating fallback result for error: {type(error).__name__}")
        return FallbackIntentResult.from_error(error, user_input)


# LangGraph Node Implementation
async def intent_classification_node(state: MainState) -> MainState:
    """
    LangGraph node for intent classification using OpenAI.
    
    This is the main entry point that takes a MainState with user_input
    and returns an updated MainState with classified intent, confidence,
    and extracted parameters.
    """
    node_name = "intent_classification_node"
    
    async with track_node_execution(node_name, state.session_id) as exec_logger:
        try:
            exec_logger.log_step("node_start", {
                "user_input_length": len(state.user_input),
                "user_input_preview": state.user_input[:50] + "..." if len(state.user_input) > 50 else state.user_input
            })
            
            # Add system message for tracking
            state = MessageManager.add_system_message(
                state,
                f"Starting OpenAI intent classification for: '{state.user_input[:50]}...'",
                node_name
            )
            
            exec_logger.log_step("conversation_context_extraction")
            
            # Get conversation context for better classification
            recent_messages = ConversationContext.get_recent_context(state, 5)
            conversation_context = "\n".join([
                f"{msg.role}: {msg.content}" for msg in recent_messages
            ])
            
            exec_logger.log_step("classifier_initialization")
            
            # Initialize classifier and classify intent
            classifier = IntentClassifier()
            
            exec_logger.log_step("intent_classification_start", {
                "conversation_context_length": len(conversation_context),
                "recent_messages_count": len(recent_messages)
            })
            
            classification_result = await classifier.classify_intent(
                state.user_input,
                conversation_context
            )
            
            # Check if this is a multi-intent result
            is_multi_intent = hasattr(classification_result, 'multiple_intents') and classification_result.multiple_intents
            
            # Handle extracted_parameters safely for both IntentClassificationResult and FallbackIntentResult
            extracted_params_count = 0
            if hasattr(classification_result, 'extracted_parameters'):
                extracted_params_count = len(classification_result.extracted_parameters)
            elif hasattr(classification_result, 'missing_params'):
                extracted_params_count = len(classification_result.missing_params)
            
            exec_logger.log_step("intent_classification_complete", {
                "classified_intent": classification_result.intent.value,
                "confidence": classification_result.confidence,
                "extracted_params_count": extracted_params_count,
                "requires_clarification": getattr(classification_result, 'requires_clarification', getattr(classification_result, 'clarification_needed', False)),
                "is_multi_intent": is_multi_intent,
                "intent_count": len(classification_result.multiple_intents) if is_multi_intent else 1
            })
            
            # Convert IntentCategory to IntentType for state compatibility
            intent_mapping = {
                IntentCategory.READ: IntentType.READ,
                IntentCategory.DATABASE_OPERATIONS: IntentType.DATABASE_OPERATIONS,
                IntentCategory.ADVICE: IntentType.ADVICE,
                IntentCategory.UNKNOWN: IntentType.UNKNOWN,
            }
            
            exec_logger.log_step("state_update_start")
            
            # Update state with classification results
            state_data = state.model_dump()
            # Remove fields we want to override
            state_data.pop('intent', None)
            state_data.pop('confidence', None)
            state_data.pop('metadata', None)
            state_data.pop('parameters', None)
            state_data.pop('multiple_intents', None)
            
            # Prepare metadata
            metadata = {
                **state.metadata,
                "classification_result": classification_result.model_dump(),
                "llm_provider": "openai",
                "is_multi_intent": is_multi_intent
            }
            
            # Add multi-intent specific metadata
            if is_multi_intent:
                metadata["intent_count"] = len(classification_result.multiple_intents)
                metadata["intent_types"] = [intent.intent for intent in classification_result.multiple_intents]
            
            updated_state = MainState(
                **state_data,
                intent=intent_mapping[classification_result.intent],
                confidence=classification_result.confidence,
                metadata=metadata,
                parameters=classification_result.extracted_parameters,
                multiple_intents=classification_result.multiple_intents if is_multi_intent else None,
            )
            
            exec_logger.log_step("response_message_generation")
            
            # Add assistant message with classification result
            if is_multi_intent:
                intent_types = [intent.intent for intent in classification_result.multiple_intents]
                response_message = (
                    f"I've analyzed your request using OpenAI: '{state.user_input}'. "
                    f"I've detected {len(classification_result.multiple_intents)} intents: "
                    f"{', '.join(intent_types)}. "
                    f"Primary intent is {classification_result.intent.value} "
                    f"with {classification_result.confidence:.0%} confidence."
                )
            else:
                response_message = (
                    f"I've analyzed your request using OpenAI: '{state.user_input}'. "
                    f"I've classified it as a {classification_result.intent.value} intent "
                    f"with {classification_result.confidence:.0%} confidence."
                )
            
            if classification_result.extracted_parameters:
                response_message += f" I've extracted {len(classification_result.extracted_parameters)} parameters."
            
            if classification_result.requires_clarification:
                response_message += " I'll need some additional information to proceed."
            
            updated_state = MessageManager.add_assistant_message(
                updated_state,
                response_message,
                node_name
            )
            
            exec_logger.log_step("node_complete", {
                "final_intent": classification_result.intent.value,
                "final_confidence": classification_result.confidence,
                "final_params_count": len(classification_result.extracted_parameters)
            })
            
            # Add execution metrics to state
            execution_metrics = exec_logger.end(success=True, metadata={
                "classification_result": classification_result.model_dump(),
                "llm_provider": "openai",
                "intent_mapping_applied": True,
                "response_message_generated": True
            })
            
            updated_state = add_execution_metrics_to_state(updated_state, execution_metrics)
            
            return updated_state
            
        except Exception as e:
            exec_logger.log_error(e, {
                "user_input": state.user_input,
                "session_id": state.session_id,
                "error_context": "intent_classification_node"
            })
            
            # Add error tracking to state
            error_state = MainState(
                user_input=state.user_input,
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                messages=state.messages,
                session_id=state.session_id,
                timestamp=state.timestamp,
                metadata={
                    **state.metadata,
                    "error": {
                        "node": node_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "llm_provider": "openai"
                    }
                },
                parameters={},
                execution_results=state.execution_results,
                response=state.response
            )
            
            # Add error message
            error_state = MessageManager.add_system_message(
                error_state,
                f"Error in OpenAI intent classification: {str(e)}",
                node_name
            )
            
            # Add execution metrics to error state
            execution_metrics = exec_logger.end(success=False, error=str(e), error_type=type(e).__name__)
            error_state = add_execution_metrics_to_state(error_state, execution_metrics)
            
            return error_state 