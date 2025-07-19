import logging
import re
from typing import List, Dict, Any, Optional, Set, Union, Literal
from src.schemas.state_schemas import ClarificationState, IntentType
from src.utils.llm_client import LLMClient
import asyncio

logger = logging.getLogger(__name__)

# Define exit signal type for clarification subgraph
ClarificationResult = Union[tuple[str, ClarificationState], tuple[Literal["EXIT_WITH_PARTIAL_DATA"], ClarificationState]]


class QuestionTemplate:
    """Template system for generating different types of clarification questions."""
    
    @staticmethod
    def build_context_prompt(state: ClarificationState) -> str:
        """Build context section of the prompt."""
        context_lines = [
            f"Context:",
            f"• Intent: {state.intent.value}",
            f"• User input: \"{state.user_input}\""
        ]
        
        # Add already provided parameters
        if state.extracted_parameters:
            provided_params = {k: v for k, v in state.extracted_parameters.items() if v is not None}
            if provided_params:
                context_lines.append(f"• Already provided: {provided_params}")
        
        # Add previously asked questions
        if state.clarification_history:
            past_questions = [entry.get("question", "N/A") for entry in state.clarification_history]
            context_lines.append(f"• Previously asked: {past_questions}")
        
        return "\n".join(context_lines)
    
    @staticmethod
    def build_instruction_prompt(
        slot_names: List[str], 
        normalization_suggestions: Dict[str, str],
        ambiguity_flags: Dict[str, str],
        intent: IntentType
    ) -> str:
        """Build instruction section with slot-specific guidance."""
        
        if len(slot_names) == 1:
            slot_name = slot_names[0]
            instructions = [f"Generate a clear, friendly question asking the user for their \"{slot_name}\"."]
            
            # Add normalization guidance
            if slot_name in normalization_suggestions:
                suggestion = normalization_suggestions[slot_name]
                instructions.append(f"Clarification note: {suggestion}")
            
            # Add ambiguity guidance
            if slot_name in ambiguity_flags:
                ambiguity_reason = ambiguity_flags[slot_name]
                instructions.append(f"Ambiguity note: The {slot_name} may be ambiguous because {ambiguity_reason}")
            
        else:
            # Multiple slots - bundle them
            slot_list = ", ".join(slot_names)
            instructions = [f"Generate a clear, friendly question asking the user for: {slot_list}."]
            instructions.append("Try to ask for multiple related pieces of information in one question when appropriate.")
        
        # Add intent-specific guidance
        intent_guidance = QuestionTemplate._get_intent_guidance(intent, slot_names)
        if intent_guidance:
            instructions.append(f"Intent guidance: {intent_guidance}")
        
        return "\n".join(instructions)
    
    @staticmethod
    def _get_intent_guidance(intent: IntentType, slot_names: List[str]) -> Optional[str]:
        """Get intent-specific guidance for question generation."""
        guidance_map = {
            IntentType.ACTION: {
                "amount": "Ask for a specific dollar amount (e.g., '$500', '$1,250.50').",
                "source_account": "Ask which account to transfer/withdraw from (e.g., 'checking', 'savings').",
                "target_account": "Ask which account to transfer/deposit to.",
                "description": "Ask for an optional description or memo for the transaction."
            },
            IntentType.DATA_FETCH: {
                "entity_type": "Ask what type of data they want to see (e.g., 'transactions', 'balances', 'statements').",
                "time_period": "Ask for a specific time range (e.g., 'last month', 'January 2024', 'last 30 days').",
                "account_types": "Ask which accounts to include (e.g., 'checking', 'savings', 'credit cards').",
                "limit": "Ask how many results they want to see (e.g., 'top 10', 'all transactions')."
            },
            IntentType.AGGREGATE: {
                "metric_type": "Ask what kind of analysis they want (e.g., 'total spending', 'average', 'breakdown').",
                "group_by": "Ask how they want the data grouped (e.g., 'by category', 'by month', 'by account').",
                "categories": "Ask which spending categories to include (e.g., 'groceries', 'entertainment')."
            }
        }
        
        intent_map = guidance_map.get(intent, {})
        for slot_name in slot_names:
            if slot_name in intent_map:
                return intent_map[slot_name]
        
        return None


class ClarificationQuestionGenerator:
    """Generates contextual clarification questions using LLM."""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    def select_next_slots(self, state: ClarificationState, max_slots: int = 2) -> List[str]:
        """
        Select the next slot(s) to ask about based on priorities and history.
        
        Args:
            state: Current clarification state
            max_slots: Maximum number of slots to ask about in one question
            
        Returns:
            List of slot names to ask about next
        """
        # Get slots already asked about
        asked_slots: Set[str] = set()
        for entry in state.clarification_history:
            targeted_param = entry.get("targeted_param")
            if targeted_param:
                asked_slots.add(targeted_param)
        
        # Parse parameter_priorities to get ordered list
        priority_list = self._parse_priorities(state.parameter_priorities)
        
        # Filter out already asked slots
        remaining_slots = [slot for slot in priority_list if slot not in asked_slots]
        
        if not remaining_slots:
            # Fallback to missing_critical_params if no priorities
            remaining_slots = [slot for slot in state.missing_critical_params if slot not in asked_slots]
        
        if not remaining_slots:
            # Final fallback to missing_params
            remaining_slots = [slot for slot in state.missing_params if slot not in asked_slots]
        
        # Return up to max_slots
        return remaining_slots[:max_slots]
    
    def _parse_priorities(self, parameter_priorities: Any) -> List[str]:
        """Parse parameter_priorities into a flat ordered list."""
        if isinstance(parameter_priorities, list):
            return parameter_priorities
        elif isinstance(parameter_priorities, dict):
            # Handle different priority structures from LLM
            ordered_slots = []
            # Try different common keys
            priority_keys = ['critical', 'high_priority', 'high', 'medium_priority', 'medium', 'low_priority', 'low', 'optional']
            for key in priority_keys:
                if key in parameter_priorities:
                    slots = parameter_priorities[key]
                    if isinstance(slots, list):
                        ordered_slots.extend(slots)
                    elif isinstance(slots, str):
                        ordered_slots.append(slots)
            return ordered_slots
        else:
            return []
    
    async def generate_question(self, state: ClarificationState) -> str:
        """
        Generate a clarification question using LLM.
        
        Args:
            state: Current clarification state
            
        Returns:
            Generated question string
        """
        # Select next slots to ask about
        slot_names = self.select_next_slots(state)
        
        if not slot_names:
            logger.warning(f"[ClarificationQuestionGenerator] No slots to ask about for session {state.session_id[:8]}")
            return "I need some additional information to help you. Could you provide more details?"
        
        logger.info(f"[ClarificationQuestionGenerator] Generating question for slots: {slot_names}")
        
        # Build prompt
        context_prompt = QuestionTemplate.build_context_prompt(state)
        instruction_prompt = QuestionTemplate.build_instruction_prompt(
            slot_names,
            state.normalization_suggestions,
            state.ambiguity_flags,
            state.intent
        )
        
        full_prompt = f"""
{context_prompt}

{instruction_prompt}

Requirements:
- Generate exactly one clear, natural question
- Make it conversational and friendly
- End with a question mark
- Keep it concise (1-2 sentences max)
- Don't mention technical terms like "slot" or "parameter"
- Make it specific to the financial context

Return only the question text, nothing else.
"""
        
        try:
            # Call LLM using simplified client
            question = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant. Generate clear, friendly clarification questions for users."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,  # Slightly higher temperature for natural questions
                max_tokens=150
            )
            logger.info(f"[ClarificationQuestionGenerator] Raw LLM response: {question}")
            
            # Post-process the question
            processed_question = self._post_process_question(question)
            
            logger.info(f"[ClarificationQuestionGenerator] Generated question for session {state.session_id[:8]}: {processed_question}")
            return processed_question
            
        except Exception as e:
            logger.error(f"[ClarificationQuestionGenerator] LLM call failed: {e}")
            # Fallback to template-based question
            return self._generate_fallback_question(slot_names, state.intent)
    
    def _post_process_question(self, question: str) -> str:
        """
        Post-process the LLM-generated question.
        
        Args:
            question: Raw question from LLM
            
        Returns:
            Cleaned and validated question
        """
        # Remove quotes if present
        question = question.strip('"\'')
        
        # Remove any leading/trailing whitespace
        question = question.strip()
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        # Capitalize first letter
        if question:
            question = question[0].upper() + question[1:]
        
        # Remove any multiple spaces
        question = re.sub(r'\s+', ' ', question)
        
        return question
    
    def _generate_fallback_question(self, slot_names: List[str], intent: IntentType) -> str:
        """Generate a fallback question if LLM fails."""
        if len(slot_names) == 1:
            slot_name = slot_names[0]
            # Simple template-based questions
            templates = {
                "amount": "What amount would you like to transfer?",
                "source_account": "Which account would you like to transfer from?",
                "target_account": "Which account would you like to transfer to?",
                "entity_type": "What type of information would you like to see?",
                "time_period": "What time period are you interested in?",
                "account_types": "Which accounts should I include?",
                "description": "Would you like to add a description for this transaction?"
            }
            return templates.get(slot_name, f"Could you please provide your {slot_name.replace('_', ' ')}?")
        else:
            slot_list = ", ".join([slot.replace('_', ' ') for slot in slot_names])
            return f"I need a bit more information: {slot_list}. Could you provide these details?"


async def clarification_question_generator_node(state: ClarificationState) -> ClarificationResult:
    """
    Generate a clarification question and update the state.
    
    This node is part of the clarification subgraph flow:
    1. Entry State → Missing Parameter Analysis → Clarification Question Generator
    2. Question to User → User Response Processor → Completeness Validator
    3. [Complete?] → Yes → Exit to Direct Orchestrator
    4. [Complete?] → No → Loop back to Missing Parameter Analysis
    5. Max Attempts → Exit with Partial Data
    
    Args:
        state: Current ClarificationState
        
    Returns:
        Either:
        - Tuple of (generated_question, updated_state) to continue clarification
        - Tuple of ("EXIT_WITH_PARTIAL_DATA", updated_state) to exit subgraph
    """
    logger.info(f"[ClarificationQuestionGenerator] Starting for session {state.session_id[:8]}")
    
    # Check if we've exceeded max attempts - exit with partial data per user flow
    if state.clarification_attempts >= state.max_attempts:
        logger.warning(f"[ClarificationQuestionGenerator] Max attempts ({state.max_attempts}) reached for session {state.session_id[:8]} - exiting with partial data")
        
        # Create exit message based on what's still missing
        missing_params_text = ", ".join(state.missing_critical_params) if state.missing_critical_params else "some information"
        exit_message = f"I'm sorry, I still need to know {missing_params_text} before I can proceed. Let me know when you're ready."
        
        # Update state with exit status
        exit_state = state.model_copy(update={
            "metadata": {
                **state.metadata,
                "clarification_status": "max_attempts_reached",
                "exit_message": exit_message,
                "exit_reason": f"Reached maximum clarification attempts ({state.max_attempts})",
                "remaining_missing_params": state.missing_params,
                "remaining_critical_params": state.missing_critical_params
            },
            "clarification_history": state.clarification_history + [{
                "question": exit_message,
                "user_response": None,
                "targeted_param": "max_attempts_exit",
                "attempt": state.clarification_attempts,
                "exit_condition": True
            }]
        })
        
        logger.info(f"[ClarificationQuestionGenerator] Exiting clarification subgraph with partial data for session {state.session_id[:8]}")
        return "EXIT_WITH_PARTIAL_DATA", exit_state
    
    try:
        generator = ClarificationQuestionGenerator()
        
        # Generate the question
        question = await generator.generate_question(state)
        
        # Determine which slots this question targets
        targeted_slots = generator.select_next_slots(state)
        targeted_param = targeted_slots[0] if targeted_slots else "unknown"
        
        # Update state for continued clarification
        updated_state = state.model_copy(update={
            "clarification_attempts": state.clarification_attempts + 1,
            "clarification_history": state.clarification_history + [{
                "question": question,
                "user_response": None,  # Will be filled by User Response Processor
                "targeted_param": targeted_param,
                "attempt": state.clarification_attempts + 1,
                "exit_condition": False
            }],
            "metadata": {
                **state.metadata,
                "clarification_status": "awaiting_user_response",
                "current_question": question,
                "targeted_slots": targeted_slots
            }
        })
        
        logger.info(f"[ClarificationQuestionGenerator] Generated question for session {state.session_id[:8]}: {question}")
        return question, updated_state
        
    except Exception as e:
        logger.error(f"[ClarificationQuestionGenerator] Failed for session {state.session_id[:8]}: {e}")
        
        # Fallback question and state update
        fallback_question = "I need some additional information to help you. Could you provide more details?"
        updated_state = state.model_copy(update={
            "clarification_attempts": state.clarification_attempts + 1,
            "clarification_history": state.clarification_history + [{
                "question": fallback_question,
                "user_response": None,
                "targeted_param": "error_fallback",
                "attempt": state.clarification_attempts + 1,
                "exit_condition": False
            }],
            "metadata": {
                **state.metadata,
                "clarification_status": "error_fallback",
                "error": f"Question generation failed: {str(e)}"
            }
        })
        
        return fallback_question, updated_state 