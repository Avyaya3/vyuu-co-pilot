"""
User Response Processor Node for LangGraph Clarification Subgraph.

This module processes user responses to clarification questions, extracting and normalizing
slot values using LLM-based parsing with structured JSON output, comprehensive validation,
and state updates for re-validation by the Completeness Validator.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Set
from decimal import Decimal, InvalidOperation

import openai

from src.schemas.state_schemas import ClarificationState, IntentType
from src.nodes.intent_classification_node import LLMClient
from src.utils.parameter_config import get_parameter_config
from src.schemas.generated_intent_schemas import DataFetchParams, AggregateParams, ActionParams

logger = logging.getLogger(__name__)


class ResponseParsingResult:
    """Result container for parsed user response."""
    
    def __init__(
        self,
        slot_values: Dict[str, Any],
        ambiguity_flags: Dict[str, str],
        extraction_confidence: float = 1.0,
        parsing_notes: Optional[str] = None
    ):
        self.slot_values = slot_values
        self.ambiguity_flags = ambiguity_flags
        self.extraction_confidence = extraction_confidence
        self.parsing_notes = parsing_notes


class ValidationResult:
    """Result container for value validation."""
    
    def __init__(
        self,
        is_valid: bool,
        normalized_value: Any = None,
        error_reason: Optional[str] = None
    ):
        self.is_valid = is_valid
        self.normalized_value = normalized_value
        self.error_reason = error_reason


class ValueValidator:
    """Utility class for validating and normalizing different data types."""
    
    @staticmethod
    def validate_numeric(value: Any, slot_name: str) -> ValidationResult:
        """
        Validate and normalize numeric values.
        
        Args:
            value: Input value to validate
            slot_name: Name of the slot for context
            
        Returns:
            ValidationResult with validation status and normalized value
        """
        if value is None:
            return ValidationResult(True, None)
        
        # Handle string representations
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            
            # Handle currency symbols
            value = re.sub(r'[$€£¥₹]', '', value)
            
            # Handle percentage
            if value.endswith('%'):
                try:
                    numeric_val = float(value[:-1]) / 100
                    return ValidationResult(True, numeric_val)
                except ValueError:
                    return ValidationResult(
                        False,
                        None,
                        f"Invalid percentage format: {value}"
                    )
            
            # Try to convert to number
            try:
                if '.' in value:
                    return ValidationResult(True, float(value))
                else:
                    return ValidationResult(True, int(value))
            except ValueError:
                return ValidationResult(
                    False,
                    None,
                    f"Cannot convert '{value}' to number"
                )
        
        # Handle numeric types
        if isinstance(value, (int, float, Decimal)):
            return ValidationResult(True, value)
        
        return ValidationResult(
            False,
            None,
            f"Unsupported numeric type: {type(value)}"
        )
    
    @staticmethod
    def validate_date(value: Any, slot_name: str) -> ValidationResult:
        """
        Validate and normalize date values.
        
        Args:
            value: Input value to validate
            slot_name: Name of the slot for context
            
        Returns:
            ValidationResult with validation status and normalized value
        """
        if value is None:
            return ValidationResult(True, None)
        
        if isinstance(value, str):
            value = value.strip()
            
            # Common date patterns
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
                r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
                r'^\d{1,2}/\d{1,2}/\d{4}$',  # M/D/YYYY
            ]
            
            for pattern in date_patterns:
                if re.match(pattern, value):
                    try:
                        # Parse different formats
                        if '-' in value and len(value.split('-')[0]) == 4:
                            # YYYY-MM-DD format
                            datetime.strptime(value, '%Y-%m-%d')
                            return ValidationResult(True, value)
                        elif '/' in value:
                            # MM/DD/YYYY or M/D/YYYY format
                            datetime.strptime(value, '%m/%d/%Y')
                            return ValidationResult(True, value)
                        elif '-' in value:
                            # MM-DD-YYYY format
                            datetime.strptime(value, '%m-%d-%Y')
                            return ValidationResult(True, value)
                    except ValueError:
                        continue
            
            return ValidationResult(
                False,
                None,
                f"Invalid date format: {value}. Expected formats: YYYY-MM-DD, MM/DD/YYYY, MM-DD-YYYY"
            )
        
        return ValidationResult(
            False,
            None,
            f"Date must be a string, got: {type(value)}"
        )
    
    @staticmethod
    def validate_enum(value: Any, slot_name: str, valid_options: List[str]) -> ValidationResult:
        """
        Validate and normalize enum values.
        
        Args:
            value: Input value to validate
            slot_name: Name of the slot for context
            valid_options: List of valid enum options
            
        Returns:
            ValidationResult with validation status and normalized value
        """
        if value is None:
            return ValidationResult(True, None)
        
        if isinstance(value, str):
            value = value.strip().lower()
            
            # Exact match
            for option in valid_options:
                if value == option.lower():
                    return ValidationResult(True, option)
            
            # Partial match
            for option in valid_options:
                if value in option.lower() or option.lower() in value:
                    return ValidationResult(True, option)
            
            return ValidationResult(
                False,
                None,
                f"'{value}' is not a valid option. Valid options: {valid_options}"
            )
        
        return ValidationResult(
            False,
            None,
            f"Enum value must be a string, got: {type(value)}"
        )
    
    @staticmethod
    def validate_string(value: Any, slot_name: str) -> ValidationResult:
        """
        Validate and normalize string values.
        
        Args:
            value: Input value to validate
            slot_name: Name of the slot for context
            
        Returns:
            ValidationResult with validation status and normalized value
        """
        if value is None:
            return ValidationResult(True, None)
        
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return ValidationResult(
                    False,
                    None,
                    "String value cannot be empty"
                )
            return ValidationResult(True, normalized)
        
        # Convert other types to string
        return ValidationResult(True, str(value))


class UserResponseProcessor:
    """
    LLM-based processor for parsing user responses to clarification questions.
    
    This class handles:
    - Identifying target slots from clarification history
    - LLM-based response parsing with structured output
    - Value validation and normalization
    - State updates and missing parameter recalculation
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.validator = ValueValidator()
    
    def identify_target_slots(self, clarification_history: List[Dict[str, Any]]) -> List[str]:
        """
        Identify which slot(s) the user is answering based on clarification history.
        
        Args:
            clarification_history: List of clarification turns
            
        Returns:
            List of slot names the user is expected to answer
        """
        if not clarification_history:
            logger.warning("No clarification history found")
            return []
        
        last_entry = clarification_history[-1]
        
        # Check for targeted_param field
        if "targeted_param" in last_entry:
            targeted_param = last_entry["targeted_param"]
            if isinstance(targeted_param, str):
                return [targeted_param]
            elif isinstance(targeted_param, list):
                return targeted_param
        
        # Fallback: try to extract from question text
        question = last_entry.get("question", "")
        
        # Simple heuristic: look for quoted parameter names
        quoted_params = re.findall(r'"([^"]+)"', question)
        if quoted_params:
            logger.info(f"Extracted target slots from question text: {quoted_params}")
            return quoted_params
        
        logger.warning(f"Could not identify target slots from last clarification entry: {last_entry}")
        return []
    
    def create_llm_prompt(
        self,
        user_response: str,
        target_slots: List[str],
        normalization_suggestions: Dict[str, str],
        ambiguity_flags: Dict[str, str],
        context: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Create system and user prompts for LLM response parsing.
        
        Args:
            user_response: User's raw response
            target_slots: Expected slot names to fill
            normalization_suggestions: Existing normalization hints
            ambiguity_flags: Current ambiguity information
            context: Additional context including intent and extracted params
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are a parameter extraction assistant. Your job is to parse user responses to clarification questions and extract structured slot values.

        INSTRUCTIONS:
        1. Analyze the user's response to identify values for the specified slot(s)
        2. Handle common variations, synonyms, and informal language
        3. If the user says "I don't know", "not sure", or similar, set the slot to null
        4. If the response is ambiguous or contradictory, flag it for clarification
        5. Return valid JSON with the specified format

        RESPONSE FORMAT:
        {
            "slot_values": {
                "slot_name": "normalized_value_or_null"
            },
            "ambiguity_flags": {
                "slot_name": "reason_if_unclear"
            },
            "extraction_confidence": 0.0-1.0,
            "parsing_notes": "optional_explanation"
        }

        GUIDELINES:
        - Use null for unknown/unclear values, not empty strings
        - Normalize values to standard formats (dates: YYYY-MM-DD, numbers: numeric)
        - Flag ambiguity if multiple interpretations are possible
        - Set confidence based on clarity of user response
        - Include parsing notes for complex extractions"""

        user_prompt_parts = [
            f"USER RESPONSE: \"{user_response}\"",
            f"TARGET SLOTS: {target_slots}",
        ]
        
        if normalization_suggestions:
            suggestions_text = "\n".join([f"  {k}: {v}" for k, v in normalization_suggestions.items()])
            user_prompt_parts.append(f"NORMALIZATION HINTS:\n{suggestions_text}")
        
        if ambiguity_flags:
            ambiguity_text = "\n".join([f"  {k}: {v}" for k, v in ambiguity_flags.items()])
            user_prompt_parts.append(f"PREVIOUS AMBIGUITIES:\n{ambiguity_text}")
        
        if context.get("intent"):
            user_prompt_parts.append(f"INTENT CONTEXT: {context['intent']}")
        
        if context.get("extracted_parameters"):
            existing_params = {k: v for k, v in context["extracted_parameters"].items() if v is not None}
            if existing_params:
                user_prompt_parts.append(f"EXISTING PARAMETERS: {existing_params}")
        
        user_prompt_parts.append("Extract the slot values from the user response:")
        
        return system_prompt, "\n\n".join(user_prompt_parts)
    
    async def parse_user_response(
        self,
        user_response: str,
        target_slots: List[str],
        normalization_suggestions: Dict[str, str],
        ambiguity_flags: Dict[str, str],
        context: Dict[str, Any]
    ) -> ResponseParsingResult:
        """
        Use LLM to parse user response and extract slot values.
        
        Args:
            user_response: User's raw response
            target_slots: Expected slot names to fill
            normalization_suggestions: Existing normalization hints
            ambiguity_flags: Current ambiguity information
            context: Additional context including intent and extracted params
            
        Returns:
            ResponseParsingResult with extracted values and metadata
        """
        try:
            logger.info(f"Parsing user response for slots {target_slots}: '{user_response[:100]}...'")
            
            # Create prompts
            system_prompt, user_prompt = self.create_llm_prompt(
                user_response, target_slots, normalization_suggestions, ambiguity_flags, context
            )
            
            # Call LLM
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                temperature=self.llm_client.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            parsed_result = json.loads(response_text)
            
            # Extract fields with defaults
            slot_values = parsed_result.get("slot_values", {})
            response_ambiguity_flags = parsed_result.get("ambiguity_flags", {})
            extraction_confidence = float(parsed_result.get("extraction_confidence", 1.0))
            parsing_notes = parsed_result.get("parsing_notes")
            
            logger.info(f"LLM extracted slot values: {slot_values}")
            if response_ambiguity_flags:
                logger.warning(f"LLM flagged ambiguities: {response_ambiguity_flags}")
            
            return ResponseParsingResult(
                slot_values=slot_values,
                ambiguity_flags=response_ambiguity_flags,
                extraction_confidence=extraction_confidence,
                parsing_notes=parsing_notes
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON, using fallback parsing: {e}")
            return self._create_fallback_result(user_response, target_slots, "json_decode_error")
        
        except openai.OpenAIError as e:
            logger.warning(f"OpenAI API error, using fallback parsing: {e}")
            return self._create_fallback_result(user_response, target_slots, "openai_api_error")
        
        except Exception as e:
            logger.error(f"Unexpected error during LLM parsing: {e}")
            return self._create_fallback_result(user_response, target_slots, "unexpected_error")
    
    def _create_fallback_result(self, user_response: str, target_slots: List[str], error_type: str = "unknown") -> ResponseParsingResult:
        """
        Create fallback parsing result when LLM fails.
        
        Args:
            user_response: User's response
            target_slots: Target slot names
            error_type: Type of error that triggered fallback
            
        Returns:
            Basic ResponseParsingResult with simple heuristics
        """
        logger.info(f"Creating fallback parsing result due to {error_type}")
        
        slot_values = {}
        ambiguity_flags = {}
        
        # Simple heuristic parsing
        response_lower = user_response.lower().strip()
        
        # Check for "don't know" patterns
        dont_know_patterns = [
            "don't know", "not sure", "unsure", "no idea", "unknown",
            "i don't", "not certain", "unclear", "can't say"
        ]
        
        is_dont_know = any(pattern in response_lower for pattern in dont_know_patterns)
        
        for slot in target_slots:
            if is_dont_know:
                slot_values[slot] = None
                ambiguity_flags[slot] = "User indicated they don't know this information"
            else:
                # Try to extract the response as-is
                slot_values[slot] = user_response.strip() if user_response.strip() else None
                if slot_values[slot]:
                    ambiguity_flags[slot] = f"Fallback parsing due to {error_type} - manual review recommended"
        
        # Adjust confidence based on error type
        confidence_map = {
            "json_decode_error": 0.3,  # LLM responded but format was wrong
            "openai_api_error": 0.2,   # API issues
            "unexpected_error": 0.1,   # Unknown errors
            "unknown": 0.3
        }
        
        return ResponseParsingResult(
            slot_values=slot_values,
            ambiguity_flags=ambiguity_flags,
            extraction_confidence=confidence_map.get(error_type, 0.3),
            parsing_notes=f"Fallback parsing used due to {error_type}"
        )
    
    def get_slot_types_from_schema(self, intent: IntentType) -> Dict[str, str]:
        """
        Extract slot type information from Pydantic schema models.
        
        Args:
            intent: Intent type to get schema for
            
        Returns:
            Dictionary mapping slot names to their types
        """
        # Map intent to Pydantic model
        intent_models = {
            IntentType.DATA_FETCH: DataFetchParams,
            IntentType.AGGREGATE: AggregateParams,
            IntentType.ACTION: ActionParams,
        }
        
        slot_types = {}
        pydantic_model = intent_models.get(intent)
        
        if pydantic_model:
            for field_name, field_info in pydantic_model.model_fields.items():
                # Extract type information from field annotation
                field_type = field_info.annotation
                
                # Handle Optional types and Union types
                if hasattr(field_type, '__origin__'):
                    if field_type.__origin__ is Union:
                        # Get the non-None type from Optional[T]
                        non_none_types = [t for t in field_type.__args__ if t is not type(None)]
                        if non_none_types:
                            field_type = non_none_types[0]
                
                # Map Python types to validation types
                type_name = getattr(field_type, '__name__', str(field_type))
                
                if type_name in ['int', 'float', 'Decimal'] or 'amount' in field_name.lower():
                    slot_types[field_name] = "numeric"
                elif type_name in ['date', 'datetime'] or 'date' in field_name.lower():
                    slot_types[field_name] = "date"
                elif type_name in ['list', 'List']:
                    slot_types[field_name] = "list"
                else:
                    slot_types[field_name] = "string"
        
        return slot_types
    
    def validate_and_normalize_values(
        self,
        slot_values: Dict[str, Any],
        intent: IntentType,
        slot_types: Optional[Dict[str, str]] = None
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """
        Validate and normalize extracted slot values.
        
        Args:
            slot_values: Raw slot values from LLM
            intent: Intent type for schema-based type inference
            slot_types: Optional type hints for validation (overrides schema)
            
        Returns:
            Tuple of (normalized_values, validation_errors)
        """
        normalized_values = {}
        validation_errors = {}
        
        # Get slot types from schema if not provided
        if slot_types is None:
            slot_types = self.get_slot_types_from_schema(intent)
        
        for slot_name, value in slot_values.items():
            if value is None:
                normalized_values[slot_name] = None
                continue
            
            # Get expected type
            expected_type = slot_types.get(slot_name, "string")
            
            # Validate based on type
            if expected_type in ["number", "numeric", "amount", "quantity"]:
                result = self.validator.validate_numeric(value, slot_name)
            elif expected_type in ["date", "datetime", "timestamp"]:
                result = self.validator.validate_date(value, slot_name)
            elif expected_type in ["enum", "choice"] and slot_name in ["status", "type", "category"]:
                # For enum types, we'd need the valid options
                # This is a simplified version
                result = self.validator.validate_string(value, slot_name)
            else:
                result = self.validator.validate_string(value, slot_name)
            
            if result.is_valid:
                normalized_values[slot_name] = result.normalized_value
            else:
                normalized_values[slot_name] = None
                validation_errors[slot_name] = result.error_reason
                logger.warning(f"Validation failed for {slot_name}: {result.error_reason}")
        
        return normalized_values, validation_errors
    
    def recompute_missing_parameters(
        self,
        extracted_parameters: Dict[str, Any],
        intent: IntentType,
        required_params: Optional[List[str]] = None,
        critical_params: Optional[List[str]] = None
    ) -> tuple[List[str], List[str]]:
        """
        Recompute missing parameters after updating extracted values.
        
        Args:
            extracted_parameters: Current extracted parameters
            intent: Intent type for context
            required_params: List of required parameters (if known)
            critical_params: List of critical parameters (if known)
            
        Returns:
            Tuple of (missing_params, missing_critical_params)
        """
        # Load parameters from YAML configuration if not provided
        if required_params is None or critical_params is None:
            param_config = get_parameter_config()
            intent_key = intent.value if hasattr(intent, 'value') else str(intent).lower()
            
            if required_params is None:
                # Get all parameters (critical + optional) as required
                critical_set = param_config.get_critical_parameters(intent_key)
                optional_set = param_config.get_optional_parameters(intent_key)
                required_params = list(critical_set | optional_set)
            
            if critical_params is None:
                critical_params = list(param_config.get_critical_parameters(intent_key))
        
        # Find missing parameters
        provided_params = {k for k, v in extracted_parameters.items() if v is not None}
        missing_params = [p for p in required_params if p not in provided_params]
        missing_critical_params = [p for p in critical_params if p not in provided_params]
        
        logger.info(f"Recomputed missing parameters: {missing_params}")
        logger.info(f"Recomputed missing critical parameters: {missing_critical_params}")
        
        return missing_params, missing_critical_params
    
    def update_clarification_history(
        self,
        clarification_history: List[Dict[str, Any]],
        user_response: str,
        parsing_result: ResponseParsingResult
    ) -> List[Dict[str, Any]]:
        """
        Update clarification history with user response and parsing results.
        
        Args:
            clarification_history: Current history
            user_response: User's response
            parsing_result: Results from parsing
            
        Returns:
            Updated clarification history
        """
        if not clarification_history:
            logger.warning("No clarification history to update")
            return clarification_history
        
        # Update the last entry with user response
        updated_history = clarification_history.copy()
        last_entry = updated_history[-1].copy()
        
        last_entry.update({
            "user_response": user_response,
            "extracted_values": parsing_result.slot_values,
            "extraction_confidence": parsing_result.extraction_confidence,
            "response_timestamp": datetime.now().isoformat(),
        })
        
        if parsing_result.ambiguity_flags:
            last_entry["response_ambiguity_flags"] = parsing_result.ambiguity_flags
        
        if parsing_result.parsing_notes:
            last_entry["parsing_notes"] = parsing_result.parsing_notes
        
        updated_history[-1] = last_entry
        
        logger.info(f"Updated clarification history entry with user response")
        return updated_history


async def user_response_processor_node(
    state: ClarificationState,
    clarification_response: str) -> ClarificationState:
    """
    Process user response to clarification questions and update state.
    
    This node:
    1. Identifies target slots from clarification history
    2. Uses LLM to parse the user response with structured output
    3. Validates and normalizes extracted values
    4. Updates extracted_parameters and recomputes missing parameters
    5. Updates clarification_history with the response
    
    Args:
        state: Current clarification state
        clarification_response: User's response to the clarification question
        
    Returns:
        Updated ClarificationState with new parameter values and metadata
    """
    logger.info(f"Processing user response for session {state.session_id}: '{clarification_response[:100]}...'")
    
    try:
        # Initialize processor
        processor = UserResponseProcessor()
        
        # 1. Identify target slots
        target_slots = processor.identify_target_slots(state.clarification_history)
        if not target_slots:
            logger.error("No target slots identified from clarification history")
            return state.model_copy(update={
                "metadata": {
                    **state.metadata,
                    "error": "Could not identify which parameters to extract from response"
                }
            })
        
        logger.info(f"Target slots for response processing: {target_slots}")
        
        # 2. Parse user response with LLM
        context = {
            "intent": state.intent,
            "extracted_parameters": state.extracted_parameters,
            "user_input": state.user_input
        }
        
        parsing_result = await processor.parse_user_response(
            clarification_response,
            target_slots,
            state.normalization_suggestions,
            state.ambiguity_flags,
            context
        )
        
        # 3. Validate and normalize values
        normalized_values, validation_errors = processor.validate_and_normalize_values(
            parsing_result.slot_values,
            state.intent
        )
        
        # 4. Update extracted parameters
        updated_extracted_parameters = state.extracted_parameters.copy()
        updated_extracted_parameters.update(normalized_values)
        
        # 5. Update ambiguity flags
        updated_ambiguity_flags = state.ambiguity_flags.copy()
        updated_ambiguity_flags.update(parsing_result.ambiguity_flags)
        
        # Add validation errors to ambiguity flags
        for slot, error in validation_errors.items():
            updated_ambiguity_flags[slot] = f"Validation error: {error}"
        
        # 6. Recompute missing parameters
        missing_params, missing_critical_params = processor.recompute_missing_parameters(
            updated_extracted_parameters,
            state.intent
        )
        
        # 7. Update parameter priorities based on new missing parameters
        updated_parameter_priorities = [
            param for param in state.parameter_priorities 
            if param in missing_params
        ]
        # Add any new missing parameters that weren't in original priorities
        for param in missing_params:
            if param not in updated_parameter_priorities:
                updated_parameter_priorities.append(param)
        
        # 8. Update clarification history
        updated_history = processor.update_clarification_history(
            state.clarification_history,
            clarification_response,
            parsing_result
        )
        
        # 9. Create updated state
        updated_state = state.model_copy(update={
            "extracted_parameters": updated_extracted_parameters,
            "missing_params": missing_params,
            "missing_critical_params": missing_critical_params,
            "parameter_priorities": updated_parameter_priorities,
            "normalization_suggestions": state.normalization_suggestions,  # Preserve existing
            "ambiguity_flags": updated_ambiguity_flags,
            "clarification_history": updated_history,
            "metadata": {
                **state.metadata,
                "last_extraction_confidence": parsing_result.extraction_confidence,
                "last_processing_timestamp": datetime.now().isoformat()
            }
        })
        
        logger.info(f"Successfully processed user response")
        logger.info(f"Updated extracted parameters: {updated_extracted_parameters}")
        logger.info(f"Remaining missing parameters: {missing_params}")
        
        return updated_state
        
    except openai.OpenAIError as e:
        logger.warning(f"OpenAI API error during response processing: {e}")
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "error": f"LLM service temporarily unavailable: {str(e)}",
                "error_type": "llm_api_error",
                "fallback_used": True
            }
        })
    
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error during response processing: {e}")
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "error": f"Invalid response format: {str(e)}",
                "error_type": "parsing_error",
                "fallback_used": True
            }
        })
    
    except ValueError as e:
        logger.warning(f"Validation error during response processing: {e}")
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "error": f"Data validation failed: {str(e)}",
                "error_type": "validation_error"
            }
        })
    
    except Exception as e:
        logger.error(f"Unexpected error processing user response: {e}")
        # This is a genuine bug that needs attention
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "error": f"System error - please contact support: {str(e)}",
                "error_type": "system_error",
                "needs_operator_attention": True
            }
        }) 