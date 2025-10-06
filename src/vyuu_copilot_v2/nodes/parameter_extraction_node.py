"""
Parameter Extraction Node for Direct Orchestrator Subgraph.

This module implements comprehensive parameter extraction, validation, and normalization
for the LangGraph intent orchestration system. It takes coarse parameters from intent
classification and produces fully-typed, validated parameter maps ready for execution.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from decimal import Decimal, InvalidOperation

from vyuu_copilot_v2.schemas.state_schemas import OrchestratorState, IntentType, MessageManager, MainState
from vyuu_copilot_v2.schemas.generated_intent_schemas import INTENT_PARAM_MODELS, IntentCategory
from vyuu_copilot_v2.utils.llm_client import LLMClient
from vyuu_copilot_v2.nodes.missing_param_analysis_node import get_pydantic_model_for_intent
from vyuu_copilot_v2.utils.node_execution_logger import track_node_execution, add_execution_metrics_to_state

logger = logging.getLogger(__name__)


class ParameterNormalizer:
    """Handles normalization and enrichment of extracted parameters."""
    
    @staticmethod
    def normalize_date_aliases(value: str) -> Dict[str, str]:
        """
        Convert date aliases into ISO date ranges.
        
        Args:
            value: Date alias like "last month", "last_3_months", etc.
            
        Returns:
            Dictionary with start_date and end_date in ISO format
        """
        now = datetime.now(timezone.utc)
        value_lower = value.lower().strip()
        
        # Define date patterns
        patterns = {
            # Last X days/weeks/months/years
            r'last (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
            r'last (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
            r'last (\d+) months?': lambda m: (now - timedelta(days=30 * int(m.group(1))), now),
            r'last (\d+) years?': lambda m: (now - timedelta(days=365 * int(m.group(1))), now),
            
            # Common aliases
            r'yesterday': lambda m: (now - timedelta(days=1), now - timedelta(days=1)),
            r'last week': lambda m: (now - timedelta(weeks=1), now),
            r'last month': lambda m: (now - timedelta(days=30), now),
            r'last year': lambda m: (now - timedelta(days=365), now),
            r'this week': lambda m: (now - timedelta(days=now.weekday()), now),
            r'this month': lambda m: (now.replace(day=1), now),
            r'this year': lambda m: (now.replace(month=1, day=1), now),
            
            # Year-to-date, month-to-date
            r'ytd|year.to.date': lambda m: (now.replace(month=1, day=1), now),
            r'mtd|month.to.date': lambda m: (now.replace(day=1), now),
        }
        
        for pattern, date_func in patterns.items():
            match = re.search(pattern, value_lower)
            if match:
                start_date, end_date = date_func(match)
                return {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "original_value": value
                }
        
        # If no pattern matches, try to parse as ISO date
        try:
            parsed_date = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return {
                "start_date": parsed_date.isoformat(),
                "end_date": parsed_date.isoformat(),
                "original_value": value
            }
        except ValueError:
            logger.warning(f"Could not parse date alias: {value}")
            return {"original_value": value}
    
    @staticmethod
    def normalize_numeric_value(value: Union[str, int, float]) -> Optional[float]:
        """
        Normalize numeric values from strings with currency symbols, percentages, etc.
        
        Args:
            value: Numeric value as string, int, or float
            
        Returns:
            Normalized float value or None if invalid
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if not isinstance(value, str):
            return None
        
        # Clean the string
        cleaned = value.strip().replace(',', '').replace(' ', '')
        
        # Handle currency symbols
        currency_patterns = [r'^\$', r'USD\s*', r'€', r'£']
        for pattern in currency_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Handle percentages
        if '%' in cleaned:
            cleaned = cleaned.replace('%', '')
            try:
                return float(cleaned) / 100.0
            except ValueError:
                return None
        
        # Try to convert to float
        try:
            return float(cleaned)
        except ValueError:
            # Try Decimal for more precision
            try:
                return float(Decimal(cleaned))
            except InvalidOperation:
                logger.warning(f"Could not normalize numeric value: {value}")
                return None
    
    @staticmethod
    def normalize_list_value(value: Union[str, List[str]]) -> List[str]:
        """
        Normalize list values from various formats.
        
        Args:
            value: List value as string or list
            
        Returns:
            Normalized list of strings
        """
        if isinstance(value, list):
            return [str(item).strip() for item in value if item]
        
        if not isinstance(value, str):
            return []
        
        # Split by common delimiters
        delimiters = [',', ';', '|', ' and ', ' & ']
        result = [value.strip()]
        
        for delimiter in delimiters:
            temp = []
            for item in result:
                temp.extend([part.strip() for part in item.split(delimiter)])
            result = temp
        
        # Remove empty strings and duplicates
        return list(dict.fromkeys([item for item in result if item]))
    
    @staticmethod
    def apply_default_values(parameters: Dict[str, Any], pydantic_model: type) -> Dict[str, Any]:
        """
        Apply default values for optional parameters that are None.
        
        Args:
            parameters: Current parameter dictionary
            pydantic_model: Pydantic model class with field definitions
            
        Returns:
            Parameters with defaults applied
        """
        if not pydantic_model:
            return parameters
        
        defaults = {
            'limit': 100,
            'sort_by': 'date',
            'order': 'desc',
            'metric_type': 'sum',
        }
        
        result = parameters.copy()
        
        for field_name, field_info in pydantic_model.model_fields.items():
            if field_name not in result or result[field_name] is None:
                # Check if field has a default value
                if field_info.default is not None and field_info.default != ...:
                    result[field_name] = field_info.default
                elif field_name in defaults:
                    result[field_name] = defaults[field_name]
        
        return result


class ParameterValidator:
    """Validates cross-field relationships and business rules."""
    
    @staticmethod
    def validate_date_range(parameters: Dict[str, Any]) -> List[str]:
        """
        Validate that start_date <= end_date if both are present.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        start_date = parameters.get('start_date')
        end_date = parameters.get('end_date')
        
        if start_date and end_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                if start_dt > end_dt:
                    errors.append("start_date must be before or equal to end_date")
            except ValueError as e:
                errors.append(f"Invalid date format: {e}")
        
        return errors
    
    @staticmethod
    def validate_numeric_ranges(parameters: Dict[str, Any]) -> List[str]:
        """
        Validate numeric ranges and constraints.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Amount should be positive for most operations
        amount = parameters.get('amount')
        if amount is not None:
            try:
                amount_val = float(amount)
                if amount_val <= 0:
                    errors.append("amount must be greater than 0")
            except (ValueError, TypeError):
                errors.append("amount must be a valid number")
        
        # Limit should be reasonable
        limit = parameters.get('limit')
        if limit is not None:
            try:
                limit_val = int(limit)
                if limit_val < 1:
                    errors.append("limit must be at least 1")
                elif limit_val > 10000:
                    errors.append("limit cannot exceed 10,000")
            except (ValueError, TypeError):
                errors.append("limit must be a valid integer")
        
        return errors
    
    @staticmethod
    def validate_account_relationships(parameters: Dict[str, Any]) -> List[str]:
        """
        Validate account-related constraints.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        source_account = parameters.get('source_account')
        target_account = parameters.get('target_account')
        
        # Source and target accounts should be different
        if source_account and target_account and source_account == target_account:
            errors.append("source_account and target_account must be different")
        
        return errors


class ParameterExtractor:
    """Main parameter extraction class with LLM integration."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient.for_task("parameter_extraction")
        self.normalizer = ParameterNormalizer()
        self.validator = ParameterValidator()
    
    def build_schema_prompt(self, intent: IntentType, pydantic_model: type) -> str:
        """
        Build a detailed schema description for the LLM prompt.
        
        Args:
            intent: Intent type
            pydantic_model: Pydantic model class
            
        Returns:
            Formatted schema description string
        """
        if not pydantic_model:
            return f"No schema available for intent: {intent}"
        
        schema_lines = [f"Schema for {intent.value} intent:"]
        
        for field_name, field_info in pydantic_model.model_fields.items():
            field_type = field_info.annotation
            description = field_info.description or "No description"
            
            # Simplify type names for readability
            type_str = str(field_type)
            type_str = re.sub(r'typing\.', '', type_str)
            type_str = re.sub(r'<class \'([^\']+)\'>', r'\1', type_str)
            
            schema_lines.append(f"  - {field_name} ({type_str}): {description}")
        
        return "\n".join(schema_lines)
    
    def create_extraction_prompt(
        self, 
        intent: IntentType, 
        user_input: str, 
        coarse_parameters: Dict[str, Any],
        pydantic_model: type
    ) -> str:
        """
        Create the LLM prompt for parameter extraction.
        
        Args:
            intent: Intent type
            user_input: Original user input
            coarse_parameters: Parameters from intent classification
            pydantic_model: Pydantic model for validation
            
        Returns:
            Formatted prompt string
        """
        schema_description = self.build_schema_prompt(intent, pydantic_model)
        
        prompt = f"""You are a parameter extraction specialist for a financial management application.

        Your task is to extract and normalize ALL parameters from the user input according to the provided schema.

        {schema_description}

        User Input: "{user_input}"

        Coarse Parameters (from intent classification): {json.dumps(coarse_parameters, indent=2)}

        Instructions:
        1. Extract ALL fields from the schema, setting null for missing values
        2. Normalize values to proper types (strings, numbers, lists, dates)
        3. For date fields, convert aliases like "last month" to specific dates
        4. For numeric fields, remove currency symbols and convert to numbers
        5. For list fields, split comma-separated values into arrays
        6. Be thorough and precise in your extraction
        
        IMPORTANT FIELD MAPPINGS:
        - For liabilities: "amount" = principal/loan amount, "emi" = monthly payment
        - For assets: "current_value" = current worth, "purchase_value" = original cost
        - For savings: "current_balance" = current amount, "target_amount" = goal amount
        - For expenses: "amount" = expense amount
        - For income: "amount" = income amount
        - For goals: "target" = target amount, "current" = current progress
        
        DATE EXTRACTION RULES:
        - "start_date" = when something begins (e.g., "starting January 1, 2022" → 2022-01-01T00:00:00)
        - "end_date" = when something ends (e.g., "ending December 31, 2052" → 2052-12-31T00:00:00)
        - "purchase_date" = when something was bought
        - "maturity_date" = when something matures/expires
        - Always extract dates in ISO format: YYYY-MM-DDTHH:MM:SS
        - Pay careful attention to the order: "starting X, ending Y" means start_date=X, end_date=Y
        
        EXAMPLE DATE EXTRACTION:
        Input: "starting January 1, 2022, ending December 31, 2052"
        Output: {{"start_date": "2022-01-01T00:00:00", "end_date": "2052-12-31T00:00:00"}}
        
        CRITICAL: Extract BOTH dates separately. Do NOT use the same date for both start_date and end_date.

        Return a JSON object with this exact structure:
        {{
        "parameters": {{
            "field1": value1,
            "field2": value2,
            ...
        }},
        "confidence": 0.95,
        "reasoning": "Brief explanation of extraction decisions"
        }}

        CRITICAL: 
        - Include ALL fields from the schema in the parameters object
        - Use null for fields that cannot be determined
        - Ensure all values match the expected types
        - Return valid JSON only"""

        return prompt
    
    async def extract_parameters_with_llm(
        self,
        intent: IntentType,
        user_input: str,
        coarse_parameters: Dict[str, Any],
        pydantic_model: type
    ) -> Tuple[Dict[str, Any], float, str]:
        """
        Extract parameters using LLM with structured output.
        
        Args:
            intent: Intent type
            user_input: Original user input
            coarse_parameters: Parameters from intent classification
            pydantic_model: Pydantic model for validation
            
        Returns:
            Tuple of (parameters, confidence, reasoning)
        """
        try:
            prompt = self.create_extraction_prompt(intent, user_input, coarse_parameters, pydantic_model)
            
            logger.debug(f"Calling LLM for parameter extraction: {intent}")
            
            response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise parameter extraction specialist. Always return valid JSON with complete parameter objects."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for precise extraction
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            
            logger.debug(f"LLM response: {response}")
            
            parsed_response = json.loads(response)
            
            parameters = parsed_response.get("parameters", {})
            confidence = parsed_response.get("confidence", 0.0)
            reasoning = parsed_response.get("reasoning", "No reasoning provided")
            
            return parameters, confidence, reasoning
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return {}, 0.0, f"JSON parsing error: {e}"
        except Exception as e:
            logger.error(f"LLM parameter extraction failed: {e}")
            return {}, 0.0, f"Extraction error: {e}"
    
    def normalize_parameters(self, parameters: Dict[str, Any], pydantic_model: type) -> Dict[str, Any]:
        """
        Apply normalization to extracted parameters.
        
        Args:
            parameters: Raw extracted parameters
            pydantic_model: Pydantic model for field type information
            
        Returns:
            Normalized parameters
        """
        if not pydantic_model:
            return parameters
        
        normalized = {}
        
        for field_name, field_info in pydantic_model.model_fields.items():
            value = parameters.get(field_name)
            
            if value is None:
                normalized[field_name] = None
                continue
            
            field_type = field_info.annotation
            type_str = str(field_type).lower()
            
            # Apply type-specific normalization
            if 'date' in field_name.lower() or 'time' in type_str:
                if isinstance(value, str):
                    date_result = self.normalizer.normalize_date_aliases(value)
                    if 'start_date' in date_result:
                        # Date range detected
                        normalized.update(date_result)
                        continue
                normalized[field_name] = value
            elif 'amount' in field_name.lower() or 'float' in type_str or 'int' in type_str:
                normalized[field_name] = self.normalizer.normalize_numeric_value(value)
            elif 'list' in type_str:
                normalized[field_name] = self.normalizer.normalize_list_value(value)
            else:
                normalized[field_name] = value
        
        # Apply default values
        normalized = self.normalizer.apply_default_values(normalized, pydantic_model)
        
        return normalized
    
    def fix_liability_dates(self, parameters: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """
        Fix liability date extraction by parsing the user input directly.
        
        Args:
            parameters: Current parameters with potentially incorrect dates
            user_input: Original user input to parse
            
        Returns:
            Parameters with corrected dates
        """
        import re
        from datetime import datetime
        
        # Check if both dates are the same (indicating extraction error)
        start_date = parameters.get('start_date')
        end_date = parameters.get('end_date')
        
        if start_date and end_date and start_date == end_date:
            logger.info("Detected duplicate dates in liability extraction, attempting to fix...")
            
            # Parse the user input for date patterns
            # Look for patterns like "starting January 1, 2022" and "ending December 31, 2052"
            start_pattern = r'starting\s+([^,]+,\s*\d{4})'
            end_pattern = r'ending\s+([^,]+,\s*\d{4})'
            
            start_match = re.search(start_pattern, user_input, re.IGNORECASE)
            end_match = re.search(end_pattern, user_input, re.IGNORECASE)
            
            if start_match and end_match:
                try:
                    # Parse the start date
                    start_date_str = start_match.group(1).strip()
                    start_dt = datetime.strptime(start_date_str, "%B %d, %Y")
                    corrected_start = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
                    
                    # Parse the end date
                    end_date_str = end_match.group(1).strip()
                    end_dt = datetime.strptime(end_date_str, "%B %d, %Y")
                    corrected_end = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
                    
                    logger.info(f"Fixed liability dates: start={corrected_start}, end={corrected_end}")
                    
                    return {
                        **parameters,
                        'start_date': corrected_start,
                        'end_date': corrected_end
                    }
                    
                except ValueError as e:
                    logger.warning(f"Could not parse liability dates: {e}")
                    return parameters
            else:
                logger.warning("Could not find start/end date patterns in user input")
                return parameters
        
        return parameters
    
    def validate_parameters(self, parameters: Dict[str, Any], pydantic_model: type) -> Tuple[bool, List[str]]:
        """
        Validate parameters using Pydantic model and custom validators.
        
        Args:
            parameters: Parameters to validate
            pydantic_model: Pydantic model for validation
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        all_errors = []
        
        # Pydantic validation
        if pydantic_model:
            try:
                # Remove None values for validation
                validation_params = {k: v for k, v in parameters.items() if v is not None}
                pydantic_model.model_validate(validation_params)
            except Exception as e:
                all_errors.append(f"Pydantic validation: {e}")
        
        # Custom cross-field validations
        all_errors.extend(self.validator.validate_date_range(parameters))
        all_errors.extend(self.validator.validate_numeric_ranges(parameters))
        all_errors.extend(self.validator.validate_account_relationships(parameters))
        
        return len(all_errors) == 0, all_errors


async def parameter_extraction_node(state: MainState) -> OrchestratorState:
    """
    LangGraph node for parameter extraction in the Direct Orchestrator subgraph.
    
    This is the FIRST NODE in the Direct Orchestrator subgraph, so it handles
    the MainState → OrchestratorState conversion internally.
    
    Args:
        state: MainState from decision router with routing decision
        
    Returns:
        OrchestratorState with extracted_params and metadata
    """
    node_name = "parameter_extraction_node"
    
    async with track_node_execution(node_name, state.session_id) as exec_logger:
        try:
            exec_logger.log_step("node_start", {
                "intent": state.intent.value if state.intent else "unknown",
                "user_input_length": len(state.user_input),
                "coarse_parameters_count": len(state.parameters) if state.parameters else 0
            })
            
            exec_logger.log_step("state_conversion_start")
            
            # Convert MainState to OrchestratorState (first node in subgraph)
            from vyuu_copilot_v2.schemas.state_schemas import StateTransitions
            orchestrator_state = StateTransitions.to_orchestrator_state(state)
            
            exec_logger.log_step("state_conversion_complete", {
                "converted_to_orchestrator_state": True
            })
            
            # Add system message for tracking
            orchestrator_state = MessageManager.add_system_message(
                orchestrator_state,
                f"Entering Direct Orchestrator subgraph - starting parameter extraction for {state.intent} intent",
                node_name
            )
            
            exec_logger.log_step("pydantic_model_lookup")
            
            # Get Pydantic model for the intent
            pydantic_model = get_pydantic_model_for_intent(orchestrator_state.intent)
            
            if not pydantic_model:
                error_msg = f"No Pydantic model found for intent: {orchestrator_state.intent}"
                exec_logger.log_error(Exception(error_msg), {"intent": orchestrator_state.intent.value})
                
                error_state = orchestrator_state.model_copy(update={
                    "metadata": {
                        **orchestrator_state.metadata,
                        "extraction_status": "error",
                        "extraction_errors": [error_msg],
                        "extraction_confidence": 0.0
                    }
                })
                
                execution_metrics = exec_logger.end(success=False, error=error_msg, error_type="ModelNotFoundError")
                error_state = add_execution_metrics_to_state(error_state, execution_metrics)
                return error_state
            
            exec_logger.log_step("extractor_initialization")
            
            # Initialize extractor
            extractor = ParameterExtractor()
            
            exec_logger.log_step("llm_parameter_extraction_start", {
                "pydantic_model": pydantic_model.__name__ if hasattr(pydantic_model, '__name__') else str(pydantic_model)
            })
            
            # Extract parameters using LLM
            raw_parameters, confidence, reasoning = await extractor.extract_parameters_with_llm(
                orchestrator_state.intent,
                orchestrator_state.user_input,
                orchestrator_state.parameters,  # Coarse parameters from intent classification
                pydantic_model
            )
            
            exec_logger.log_step("llm_parameter_extraction_complete", {
                "raw_parameters_count": len(raw_parameters),
                "confidence": confidence,
                "reasoning_length": len(reasoning) if reasoning else 0
            })
            
            exec_logger.log_step("parameter_normalization_start")
            
            # Normalize parameters
            normalized_parameters = extractor.normalize_parameters(raw_parameters, pydantic_model)
            
            exec_logger.log_step("parameter_normalization_complete", {
                "normalized_parameters_count": len(normalized_parameters)
            })
            
            exec_logger.log_step("liability_date_processing")
            
            # Post-process date extraction for liability entities
            if (orchestrator_state.intent == IntentCategory.DATABASE_OPERATIONS and 
                normalized_parameters.get('entity_type') == 'liability' and
                'start_date' in normalized_parameters and 'end_date' in normalized_parameters):
                normalized_parameters = extractor.fix_liability_dates(
                    normalized_parameters, orchestrator_state.user_input
                )
                exec_logger.log_step("liability_dates_fixed")
            
            exec_logger.log_step("user_id_enrichment")
            
            # Always include user_id from the state if it's not already present
            if 'user_id' in pydantic_model.model_fields and 'user_id' not in normalized_parameters:
                user_id = orchestrator_state.metadata.get('user_id')
                if user_id:
                    normalized_parameters['user_id'] = user_id
                    exec_logger.log_step("user_id_added_from_state", {"user_id_length": len(user_id)})
                else:
                    exec_logger.log_step("user_id_not_found_in_state")
            
            exec_logger.log_step("parameter_validation_start")
            
            # Validate parameters
            is_valid, validation_errors = extractor.validate_parameters(normalized_parameters, pydantic_model)
            
            exec_logger.log_step("parameter_validation_complete", {
                "is_valid": is_valid,
                "validation_errors_count": len(validation_errors)
            })
            
            # Determine extraction status
            extraction_status = "success" if is_valid and confidence > 0.7 else "incomplete"
            
            exec_logger.log_step("state_update_start")
            
            # Update state with extracted parameters
            updated_state = orchestrator_state.model_copy(update={
                "extracted_params": normalized_parameters,
                "metadata": {
                    **orchestrator_state.metadata,
                    "extraction_status": extraction_status,
                    "extraction_errors": validation_errors,
                    "extraction_confidence": confidence,
                    "extraction_reasoning": reasoning,
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
            
            exec_logger.log_step("response_message_generation")
            
            # Add assistant message with extraction results
            parameter_count = len([v for v in normalized_parameters.values() if v is not None])
            response_message = (
                f"Extracted {parameter_count} parameters for {orchestrator_state.intent.value} intent "
                f"with {confidence:.0%} confidence. Status: {extraction_status}."
            )
            
            if validation_errors:
                response_message += f" Found {len(validation_errors)} validation issues."
            
            updated_state = MessageManager.add_assistant_message(
                updated_state,
                response_message,
                node_name
            )
            
            exec_logger.log_step("node_complete", {
                "extraction_status": extraction_status,
                "parameter_count": parameter_count,
                "confidence": confidence,
                "validation_errors_count": len(validation_errors)
            })
            
            # Add execution metrics to state
            execution_metrics = exec_logger.end(success=True, metadata={
                "extraction_status": extraction_status,
                "extraction_errors": validation_errors,
                "extraction_confidence": confidence,
                "extraction_reasoning": reasoning,
                "parameter_count": parameter_count,
                "is_valid": is_valid,
                "pydantic_model": pydantic_model.__name__ if hasattr(pydantic_model, '__name__') else str(pydantic_model)
            })
            
            updated_state = add_execution_metrics_to_state(updated_state, execution_metrics)
            
            return updated_state
            
        except Exception as e:
            exec_logger.log_error(e, {
                "intent": state.intent.value if state.intent else "unknown",
                "session_id": state.session_id,
                "error_context": "parameter_extraction_node"
            })
            
            # Convert to orchestrator state for error handling
            try:
                from vyuu_copilot_v2.schemas.state_schemas import StateTransitions
                error_orchestrator_state = StateTransitions.to_orchestrator_state(state)
            except Exception:
                # If state conversion fails, create minimal orchestrator state
                from vyuu_copilot_v2.schemas.state_schemas import OrchestratorState
                error_orchestrator_state = OrchestratorState(
                    **state.model_dump(),
                    extracted_params={},
                    execution_plan=None,
                    tool_results=None,
                    final_response=None
                )
            
            # Add error to state
            error_state = error_orchestrator_state.model_copy(update={
                "metadata": {
                    **error_orchestrator_state.metadata,
                    "extraction_status": "error",
                    "extraction_errors": [str(e)],
                    "extraction_confidence": 0.0
                }
            })
            
            # Add error message
            error_state = MessageManager.add_assistant_message(
                error_state,
                f"Parameter extraction failed: {str(e)}",
                node_name
            )
            
            # Add execution metrics to error state
            execution_metrics = exec_logger.end(success=False, error=str(e), error_type=type(e).__name__)
            error_state = add_execution_metrics_to_state(error_state, execution_metrics)
            
            return error_state 