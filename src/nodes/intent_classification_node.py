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

from src.schemas.state_schemas import (
    MainState,
    MessageManager,
    IntentType,
    ConversationContext,
)
from src.schemas.generated_intent_schemas import (
    IntentClassificationResult,
    IntentClassificationError,
    FallbackIntentResult,
    IntentCategory,
    DataFetchParams,
    AggregateParams,
    ActionParams,
)

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenAI LLM client for intent classification.
    
    This class provides real OpenAI API integration for intent classification
    with structured output and comprehensive error handling.
    """
    
    def __init__(self, model: str = "gpt-4-1106-preview", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise IntentClassificationError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.",
                error_type="configuration_error"
            )
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {model}, temperature: {temperature}")
    
    async def classify_intent(self, user_input: str, conversation_context: str = "") -> Dict[str, Any]:
        """
        Call OpenAI LLM for intent classification with structured output.
        
        Args:
            user_input: The user's input to classify
            conversation_context: Previous conversation context
            
        Returns:
            Structured classification result
            
        Raises:
            IntentClassificationError: When LLM call fails
        """
        try:
            logger.debug(f"Calling OpenAI for intent classification: '{user_input[:50]}...'")
            
            # Create system prompt for intent classification
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(user_input, conversation_context)
            
            # Call OpenAI with structured output
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            classification = json.loads(response_text)
            
            # Validate and normalize response
            classification = self._validate_and_normalize_response(classification)
            
            logger.debug(f"OpenAI classification result: {classification['intent']} (confidence: {classification['confidence']})")
            
            return classification
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI JSON response: {e}")
            raise IntentClassificationError(
                f"Invalid JSON response from OpenAI: {str(e)}",
                error_type="parsing_error",
                user_input=user_input
            )
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise IntentClassificationError(
                f"OpenAI API error: {str(e)}",
                error_type="api_error", 
                user_input=user_input
            )
        except Exception as e:
            logger.error(f"Unexpected error in LLM classification: {e}")
            raise IntentClassificationError(
                f"LLM call failed: {str(e)}",
                error_type="llm_error",
                user_input=user_input
            )
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for intent classification."""
        return """You are an AI assistant specialized in classifying user intents for a financial management application.

        Your task is to analyze user requests and classify them into one of these intent categories:

        1. **data_fetch**: User wants to retrieve/view existing financial data
        - Examples: "Show me my transactions", "What's my balance?", "List my accounts"
        - Extract parameters like: entity_type, time_period, account_types, limit, sort_by, order, filters

        2. **aggregate**: User wants to analyze/summarize financial data
        - Examples: "What's my total spending?", "Average monthly expenses", "Count transactions by category"
        - Extract parameters like: metric_type, group_by, time_period, category_filter, account_filter, comparison_period

        3. **action**: User wants to perform an action (transfer, payment, categorization, etc.)
        - Examples: "Transfer $100 to savings", "Pay my electric bill", "Categorize this transaction"
        - Extract parameters like: action_type, amount, source_account, target_account, description, schedule_date, confirmation_required

        4. **unknown**: Intent is unclear or doesn't fit the above categories

        CRITICAL: You must respond with a valid JSON object with this exact structure:
        {
          "intent": "data_fetch|aggregate|action|unknown",
          "confidence": 0.85,
          "reasoning": "brief explanation",
          "user_input_analysis": "analysis of the input",
          "missing_params": ["param1", "param2"],
          "clarification_needed": true,
          "data_fetch_params": {
            "entity_type": "transactions",
            "account_types": ["checking", "savings"],
            "time_period": "last_month",
            "limit": 10,
            "sort_by": "date",
            "order": "desc"
          },
          "aggregate_params": {
            "metric_type": "sum",
            "group_by": ["category", "month"],
            "category_filter": ["groceries", "restaurants"],
            "time_period": "last_year"
          },
          "action_params": {
            "action_type": "transfer",
            "amount": 100.0,
            "source_account": "checking",
            "target_account": "savings"
          }
        }

        **IMPORTANT DATA TYPE RULES:**
        - ALL array fields must be arrays, not strings: ["category"] not "category"
        - group_by must be an array: ["category", "date"] not "category"
        - category_filter must be an array: ["groceries", "dining"] not "groceries"
        - account_types must be an array: ["checking", "savings"] not "checking"
        - missing_params must be an array: ["amount", "account"] not "amount"
        - Only include the params object for the matched intent type
        - Set null for unused param objects

        Set confidence based on clarity: 0.8-1.0 (high), 0.5-0.79 (medium), 0.0-0.49 (low).
        Set clarification_needed=true if confidence < 0.7 or critical parameters are missing."""

    def _create_user_prompt(self, user_input: str, conversation_context: str) -> str:
        """Create user prompt with input and context."""
        prompt_parts = []
        
        if conversation_context:
            prompt_parts.append(f"Previous conversation context:\n{conversation_context}\n")
        
        prompt_parts.append(f"User request to classify: \"{user_input}\"")
        prompt_parts.append("\nAnalyze this request and respond with the required JSON format.")
        
        return "\n".join(prompt_parts)
    
    def _validate_and_normalize_response(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the OpenAI response."""
        # Ensure required fields exist
        required_fields = ["intent", "confidence", "reasoning", "user_input_analysis"]
        for field in required_fields:
            if field not in classification:
                raise ValueError(f"Missing required field: {field}")
        
        # Normalize intent to IntentCategory enum
        intent_str = classification["intent"].lower()
        intent_mapping = {
            "data_fetch": IntentCategory.DATA_FETCH,
            "aggregate": IntentCategory.AGGREGATE,
            "action": IntentCategory.ACTION,
            "unknown": IntentCategory.UNKNOWN
        }
        
        if intent_str not in intent_mapping:
            logger.warning(f"Unknown intent '{intent_str}', defaulting to UNKNOWN")
            classification["intent"] = IntentCategory.UNKNOWN
            classification["confidence"] = min(classification.get("confidence", 0.0), 0.3)
        else:
            classification["intent"] = intent_mapping[intent_str]
        
        # Ensure confidence is in valid range
        confidence = classification.get("confidence", 0.0)
        classification["confidence"] = max(0.0, min(1.0, float(confidence)))
        
        # Ensure optional fields exist
        classification.setdefault("missing_params", [])
        classification.setdefault("clarification_needed", False)
        classification.setdefault("data_fetch_params", None)
        classification.setdefault("aggregate_params", None) 
        classification.setdefault("action_params", None)
        
        # Auto-set clarification_needed based on confidence and missing params
        if (classification["confidence"] < 0.7 or 
            len(classification["missing_params"]) > 0 or
            classification["intent"] == IntentCategory.UNKNOWN):
            classification["clarification_needed"] = True
        
        return classification


class IntentClassifier:
    """
    Main intent classifier with OpenAI LLM integration and fallback handling.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        logger.info("Intent classifier initialized with OpenAI integration")
    
    async def classify_intent(
        self, 
        user_input: str, 
        conversation_context: str = ""
    ) -> IntentClassificationResult:
        """
        Classify user intent with comprehensive error handling.
        
        Args:
            user_input: The user's input to classify
            conversation_context: Previous conversation context for better classification
            
        Returns:
            IntentClassificationResult with intent, confidence, and parameters
            
        Raises:
            IntentClassificationError: When classification fails completely
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Starting intent classification for input: '{user_input[:50]}...'")
            
            # Validate input
            if not user_input or not user_input.strip():
                raise IntentClassificationError(
                    "Empty or whitespace-only user input",
                    error_type="validation_error",
                    user_input=user_input
                )
            
            # Call LLM for classification
            llm_result = await self.llm_client.classify_intent(user_input, conversation_context)
            
            # Convert to structured result
            classification_result = self._parse_llm_result(llm_result, user_input)
            
            # Log success
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                f"Intent classification successful: {classification_result.intent} "
                f"(confidence: {classification_result.confidence:.2f}, "
                f"processing_time: {processing_time:.3f}s)"
            )
            
            return classification_result
            
        except IntentClassificationError:
            # Re-raise known errors
            raise
            
        except Exception as e:
            # Handle unexpected errors with fallback
            logger.error(f"Unexpected error during intent classification: {e}")
            
            try:
                # Attempt fallback classification
                fallback_result = self._create_fallback_result(e, user_input)
                logger.warning(f"Using fallback classification: {fallback_result.intent}")
                return fallback_result.to_classification_result()
                
            except Exception as fallback_error:
                logger.error(f"Fallback classification also failed: {fallback_error}")
                raise IntentClassificationError(
                    f"Classification failed completely: {str(e)}",
                    error_type="complete_failure",
                    user_input=user_input
                )
    
    def _parse_llm_result(self, llm_result: Dict[str, Any], user_input: str) -> IntentClassificationResult:
        """
        Parse LLM result into structured IntentClassificationResult.
        """
        try:
            # Extract intent-specific parameters
            data_fetch_params = None
            aggregate_params = None
            action_params = None
            
            if llm_result.get("intent") == IntentCategory.DATA_FETCH and llm_result.get("data_fetch_params"):
                data_fetch_params = DataFetchParams(**llm_result["data_fetch_params"])
            
            elif llm_result.get("intent") == IntentCategory.AGGREGATE and llm_result.get("aggregate_params"):
                aggregate_params = AggregateParams(**llm_result["aggregate_params"])
            
            elif llm_result.get("intent") == IntentCategory.ACTION and llm_result.get("action_params"):
                action_params = ActionParams(**llm_result["action_params"])
            
            # Create and validate result
            result = IntentClassificationResult(
                intent=llm_result["intent"],
                confidence=llm_result["confidence"],
                reasoning=llm_result.get("reasoning", "Intent classified by OpenAI"),
                data_fetch_params=data_fetch_params,
                aggregate_params=aggregate_params,
                action_params=action_params,
                missing_params=llm_result.get("missing_params", []),
                clarification_needed=llm_result.get("clarification_needed", False),
                user_input_analysis=llm_result.get("user_input_analysis", f"Analysis of: '{user_input[:50]}...'")
            )
            
            logger.debug(f"Successfully parsed LLM result: {result.intent} (params: {len(result.extracted_parameters)})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse LLM result: {e}")
            raise IntentClassificationError(
                f"Invalid LLM response format: {str(e)}",
                error_type="parsing_error",
                user_input=user_input
            )
    
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
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Intent classification node processing session: {state.session_id[:8]}...")
        
        # Add system message for tracking
        state = MessageManager.add_system_message(
            state,
            f"Starting OpenAI intent classification for: '{state.user_input[:50]}...'",
            node_name
        )
        
        # Get conversation context for better classification
        recent_messages = ConversationContext.get_recent_context(state, 5)
        conversation_context = "\n".join([
            f"{msg.role}: {msg.content}" for msg in recent_messages
        ])
        
        # Initialize classifier and classify intent
        classifier = IntentClassifier()
        classification_result = await classifier.classify_intent(
            state.user_input,
            conversation_context
        )
        
        # Convert IntentCategory to IntentType for state compatibility
        intent_mapping = {
            IntentCategory.DATA_FETCH: IntentType.DATA_FETCH,
            IntentCategory.AGGREGATE: IntentType.AGGREGATE,
            IntentCategory.ACTION: IntentType.ACTION,
            IntentCategory.UNKNOWN: IntentType.UNKNOWN,
        }
        
        # Update state with classification results
        state_data = state.model_dump()
        # Remove fields we want to override
        state_data.pop('intent', None)
        state_data.pop('confidence', None)
        state_data.pop('metadata', None)
        state_data.pop('parameters', None)
        
        updated_state = MainState(
            **state_data,
            intent=intent_mapping[classification_result.intent],
            confidence=classification_result.confidence,
            metadata={
                **state.metadata,
                "classification_result": classification_result.model_dump(),
                "node_processing_time": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "llm_provider": "openai"
            },
            parameters=classification_result.extracted_parameters,
        )
        
        # Add assistant message with classification result
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
        
        # Log success
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"OpenAI intent classification node completed successfully: "
            f"{classification_result.intent} (confidence: {classification_result.confidence:.2f}, "
            f"params: {len(classification_result.extracted_parameters)}, "
            f"processing_time: {processing_time:.3f}s)"
        )
        
        return updated_state
        
    except Exception as e:
        # Log error and add to state metadata
        error_message = f"OpenAI intent classification failed: {str(e)}"
        logger.error(error_message)
        
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
        
        return error_state 