"""
Response Synthesis Node for Direct Orchestrator Subgraph

Generates natural language responses from tool execution results using LLM calls
with intent-specific formatting and comprehensive error handling.

Features:
- LLM-based response generation with intent-specific prompts
- Data formatting utilities for financial information
- Tool result processing and aggregation
- Graceful error handling with fallback responses
- Context-aware conversation management
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from ..schemas.state_schemas import OrchestratorState, MessageManager
from ..schemas.generated_intent_schemas import IntentCategory, IntentEntry
from ..utils.llm_client import LLMClient
from ..utils.data_formatters import DataFormatter
from ..utils.node_execution_logger import track_node_execution, add_execution_metrics_to_state

logger = logging.getLogger(__name__)

# Base system prompt with intent-specific sections
BASE_SYSTEM_PROMPT = """You are a helpful financial assistant. Generate responses based on the intent type.

{intent_specific_guidelines}

RESPONSE GUIDELINES:
- Be conversational and helpful
- Format financial data clearly (use ₹ for amounts, % for percentages)
- Present data in a structured, easy-to-read format
- Include relevant context and insights
- Keep responses concise but informative"""

# Intent-specific guidelines
INTENT_GUIDELINES = {
    IntentCategory.READ: "Focus on presenting data clearly and informatively. Highlight key information and make it easy to understand.",
    IntentCategory.DATABASE_OPERATIONS: "Focus on confirming actions and providing next steps. Be clear about what was done and any follow-up needed.",
    IntentCategory.ADVICE: """Focus on presenting advice with mathematical transparency. Present all three risk levels clearly with their calculations. Structure the response to show:
1. Each risk level recommendation
2. Mathematical calculations and assumptions
3. Expected outcomes

Make calculations easy to understand and verify."""
}


class ResponseSynthesizer:
    """
    LLM-based response synthesizer with intent-specific formatting.

    This class handles:
    - Intent-aware response generation using LLM
    - Data formatting for financial information
    - Tool result processing and aggregation
    - Context-aware conversation management
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient.for_task("response_synthesis")
        self.formatter = DataFormatter()

    async def synthesize_response(
        self, state: OrchestratorState, tool_results: Dict[str, Any]
    ) -> str:
        """
        Synthesize natural language response from tool results.

        Args:
            state: Current orchestrator state with context
            tool_results: Results from tool execution

        Returns:
            Generated natural language response
        """
        try:
            # Step 1: Process and format tool results
            processed_data = self._process_tool_results(tool_results)

            # Step 2: Generate intent-specific response
            if state.intent == IntentCategory.READ:
                response = await self._generate_read_response(
                    state, processed_data
                )
            elif state.intent == IntentCategory.DATABASE_OPERATIONS:
                response = await self._generate_database_operations_response(state, processed_data)
            elif state.intent == IntentCategory.ADVICE:
                response = await self._generate_advice_response(state, processed_data)
            else:
                response = await self._generate_generic_response(state, processed_data)

            return response

        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            return self._generate_fallback_response(state, tool_results)

    def _process_tool_results(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and aggregate tool execution results.

        Args:
            tool_results: Raw tool execution results

        Returns:
            Processed and aggregated data
        """
        processed_data = {
            "aggregated_data": {},
            "execution_summary": {
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "has_data": False,
            },
            "failed_steps": [],
        }

        if not tool_results:
            return processed_data

        # Process each step
        for step_key, step_result in tool_results.items():
            processed_data["execution_summary"]["total_steps"] += 1

            if step_result.get("success", False):
                processed_data["execution_summary"]["successful_steps"] += 1
                step_data = step_result.get("data", {})
                
                if step_data:
                    processed_data["execution_summary"]["has_data"] = True
                    self._aggregate_step_data(
                        processed_data["aggregated_data"], step_data
                    )
            else:
                processed_data["execution_summary"]["failed_steps"] += 1
                processed_data["failed_steps"].append({
                    "step": step_key,
                    "error": step_result.get("error", "Unknown error")
                })

        return processed_data

    def _aggregate_step_data(self, aggregated_data: Dict[str, Any], step_data: Dict[str, Any]):
        """
        Aggregate data from individual steps.

        Args:
            aggregated_data: Accumulated data from previous steps
            step_data: Data from current step
        """
        if "balance" in step_data:
            # Store under "balances" and directly for compatibility
            aggregated_data["balances"] = aggregated_data.get("balances", {})
            aggregated_data["balances"].update(step_data)
            aggregated_data["balance"] = step_data["balance"]
        elif "transactions" in step_data:
            # Extend transaction lists
            aggregated_data["transactions"] = aggregated_data.get("transactions", [])
            aggregated_data["transactions"].extend(step_data["transactions"])
        elif "accounts" in step_data:
            # Extend account lists
            aggregated_data["accounts"] = aggregated_data.get("accounts", [])
            aggregated_data["accounts"].extend(step_data["accounts"])
        elif "spending_analysis" in step_data:
            # Merge spending analysis
            aggregated_data["spending_analysis"] = aggregated_data.get("spending_analysis", {})
            aggregated_data["spending_analysis"].update(step_data["spending_analysis"])
        elif "monthly_summary" in step_data:
            # Merge monthly summaries
            aggregated_data["monthly_summary"] = aggregated_data.get("monthly_summary", {})
            aggregated_data["monthly_summary"].update(step_data["monthly_summary"])
        else:
            # Generic data merging
            for key, value in step_data.items():
                if key not in aggregated_data:
                    aggregated_data[key] = value
                elif isinstance(value, list):
                    aggregated_data[key].extend(value)
                elif isinstance(value, dict):
                    aggregated_data[key].update(value)

    async def _generate_read_response(
        self, state: OrchestratorState, processed_data: Dict[str, Any]
    ) -> str:
        """
        Generate response for data fetch intents.

        Args:
            state: Current orchestrator state
            processed_data: Processed tool results

        Returns:
            Generated response for data fetch intent
        """
        try:
            # Get intent-specific guidelines
            intent_guidelines = INTENT_GUIDELINES.get(state.intent, "")
            
            # Create system prompt
            system_prompt = BASE_SYSTEM_PROMPT.format(
                intent_specific_guidelines=intent_guidelines
            )

            # Format data for LLM
            formatted_data = self.formatter.format_data_for_llm(
                processed_data["aggregated_data"]
            )

            # Create user prompt
            user_prompt = f"""Generate a helpful response for this data fetch request:

User Request: {state.user_input}

Available Data:
{formatted_data}

Please provide a clear, informative response that addresses the user's request."""

            # Generate response using LLM
            response = await self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            return response

        except Exception as e:
            logger.error(f"LLM-based data fetch response generation failed: {e}")
            return self._generate_template_data_fetch_response(processed_data)

    async def _generate_database_operations_response(
        self, state: OrchestratorState, processed_data: Dict[str, Any]
    ) -> str:
        """
        Generate response for database operations intents.

        Args:
            state: Current orchestrator state
            processed_data: Processed tool results

        Returns:
            Generated response for database operations intent
        """
        # Check if any steps failed
        if processed_data.get("failed_steps", []):
            return self._generate_action_error_response(processed_data)

        try:
            # Get intent-specific guidelines
            intent_guidelines = INTENT_GUIDELINES.get(state.intent, "")
            
            # Create system prompt
            system_prompt = BASE_SYSTEM_PROMPT.format(
                intent_specific_guidelines=intent_guidelines
            )

            # Format data for LLM
            formatted_data = self.formatter.format_data_for_llm(
                processed_data["aggregated_data"]
            )

            # Create user prompt
            user_prompt = f"""Generate a confirmation response for this database operation request:

User Request: {state.user_input}

Operation Results:
{formatted_data}

Please confirm what was done and provide any relevant next steps."""

            # Generate response using LLM
            response = await self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            return response

        except Exception as e:
            logger.error(f"LLM-based database operations response generation failed: {e}")
            return self._generate_template_database_operations_response(processed_data)

    async def _generate_advice_response(
        self, state: OrchestratorState, processed_data: Dict[str, Any]
    ) -> str:
        """
        Generate response for advice intents.

        Args:
            state: Current orchestrator state
            processed_data: Processed tool results

        Returns:
            Generated response for advice intent
        """
        # Check if any steps failed
        if processed_data.get("failed_steps", []):
            return self._generate_action_error_response(processed_data)

        try:
            # For advice, we want to return the raw advice data directly
            # instead of generating a confirmation response
            aggregated_data = processed_data.get("aggregated_data", {})
            
            # Look for advice data in the aggregated results
            if "advice" in aggregated_data:
                advice_data = aggregated_data["advice"]
                if isinstance(advice_data, dict) and "advice" in advice_data:
                    # Return the raw advice text with calculations
                    return advice_data["advice"]
                elif isinstance(advice_data, str):
                    # Return the advice string directly
                    return advice_data
            
            # Fallback: format the data nicely if no direct advice found
            formatted_data = self.formatter.format_data_for_llm(aggregated_data)
            return f"Here's your financial advice:\n\n{formatted_data}"

        except Exception as e:
            logger.error(f"Advice response generation failed: {e}")
            # Fallback to template response
            return self._generate_template_action_response(processed_data)

    async def _generate_generic_response(
        self, state: OrchestratorState, processed_data: Dict[str, Any]
    ) -> str:
        """
        Generate generic response for unknown or unclear intents.

        Args:
            state: Current orchestrator state
            processed_data: Processed tool results

        Returns:
            Generated generic response
        """
        try:
            # Use LLM to make it conversational
            system_prompt = """You are a helpful financial assistant. Generate a conversational response for the user's request.

RESPONSE GUIDELINES:
- Be conversational and helpful
- Address the user's request clearly
- If data is available, present it in a user-friendly way
- If no data is available, explain what you tried to do
- Keep the tone friendly and professional"""

            # Format data for LLM
            formatted_data = self.formatter.format_data_for_llm(
                processed_data["aggregated_data"]
            )

            # Create user prompt
            user_prompt = f"""Generate a conversational response for this request:

User Request: {state.user_input}

Available Data:
{formatted_data}

Please provide a helpful, conversational response that addresses the user's request."""

            # Generate response using LLM
            response = await self.llm_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            return response

        except Exception as e:
            logger.error(f"LLM-based generic response generation failed: {e}")
            # Fallback to simple response
            if processed_data["execution_summary"]["has_data"]:
                return f"I've processed your request: '{state.user_input}'. Here are the results:\n\n{json.dumps(processed_data['aggregated_data'], indent=2)}"
            else:
                return f"I processed your request: '{state.user_input}', but couldn't find any relevant data. Please check your parameters and try again."

    def _generate_template_data_fetch_response(
        self, processed_data: Dict[str, Any]
    ) -> str:
        """Generate template-based data fetch response when LLM fails."""
        data = processed_data["aggregated_data"]
        
        if not data:
            return "I couldn't retrieve the requested data. Please check your parameters and try again."

        response_parts = ["Here's the information you requested:"]

        # Handle balance data
        if "balances" in data or "balance" in data:
            balance_data = data.get("balances", data.get("balance", {}))
            if balance_data:
                response_parts.append("\n**Account Balances:**")
                for account, balance in balance_data.items():
                    if isinstance(balance, (int, float)):
                        response_parts.append(f"- {account}: {self.formatter.format_currency(balance)}")

        # Handle transaction data
        if "transactions" in data:
            transactions = data["transactions"]
            if transactions:
                response_parts.append(f"\n**Recent Transactions ({len(transactions)} found):**")
                for i, tx in enumerate(transactions[:5]):  # Show first 5
                    amount = tx.get("amount", 0)
                    description = tx.get("description", "Unknown")
                    date = tx.get("date", "Unknown date")
                    response_parts.append(
                        f"- {self.formatter.format_currency(amount)} - {description} ({date})"
                    )

        # Handle account data
        if "accounts" in data:
            accounts = data["accounts"]
            if accounts:
                response_parts.append(f"\n**Accounts ({len(accounts)} found):**")
                for account in accounts[:3]:  # Show first 3
                    name = account.get("name", "Unknown")
                    account_type = account.get("type", "Unknown type")
                    response_parts.append(f"- {name} ({account_type})")

        return "\n".join(response_parts)

    def _generate_template_aggregate_response(
        self, processed_data: Dict[str, Any]
    ) -> str:
        """Generate template-based aggregate response when LLM fails."""
        data = processed_data["aggregated_data"]
        
        if not data:
            return "I couldn't analyze the requested data. Please check your parameters and try again."

        response_parts = ["Here's the analysis of your data:"]

        # Handle spending analysis
        if "spending_analysis" in data:
            spending = data["spending_analysis"]
            if spending:
                response_parts.append("\n**Spending Analysis:**")
                for category, amount in spending.items():
                    if isinstance(amount, (int, float)):
                        response_parts.append(f"- {category}: {self.formatter.format_currency(amount)}")

        # Handle monthly summary
        if "monthly_summary" in data:
            summary = data["monthly_summary"]
            if summary:
                response_parts.append("\n**Monthly Summary:**")
                for month, data in summary.items():
                    if isinstance(data, dict):
                        income = data.get("income", 0)
                        expenses = data.get("expenses", 0)
                        response_parts.append(
                            f"- {month}: Income {self.formatter.format_currency(income)}, "
                            f"Expenses {self.formatter.format_currency(expenses)}"
                        )

        # Handle balance data
        if "balances" in data or "balance" in data:
            balance_data = data.get("balances", data.get("balance", {}))
            if balance_data:
                response_parts.append("\n**Current Balances:**")
                for account, balance in balance_data.items():
                    if isinstance(balance, (int, float)):
                        response_parts.append(f"- {account}: {self.formatter.format_currency(balance)}")

        return "\n".join(response_parts)

    def _generate_template_action_response(
        self, processed_data: Dict[str, Any]
    ) -> str:
        """Generate template-based action response when LLM fails."""
        data = processed_data["aggregated_data"]
        
        if not data:
            return "The requested action has been completed successfully."

        response_parts = ["The requested action has been completed successfully."]

        # Add any relevant data from the action
        if "transaction_id" in data:
            response_parts.append(f"\nTransaction ID: {data['transaction_id']}")
        
        if "status" in data:
            response_parts.append(f"Status: {data['status']}")

        return "\n".join(response_parts)

    def _generate_action_error_response(self, processed_data: Dict[str, Any]) -> str:
        """Generate error response for failed actions."""
        failed_steps = processed_data.get("failed_steps", [])
        
        if not failed_steps:
            return "The requested action couldn't be completed. Please try again."

        error_messages = []
        for step in failed_steps:
            error = step.get("error", "Unknown error")
            error_messages.append(f"- {error}")

        return f"I'm sorry, but the requested action couldn't be completed.\n\nErrors encountered:\n" + "\n".join(error_messages) + "\n\nPlease check your information and try again."

    def _generate_template_database_operations_response(self, processed_data: Dict[str, Any]) -> str:
        """
        Generate template response for database operations when LLM fails.
        
        Args:
            processed_data: Processed tool results
            
        Returns:
            Template response string
        """
        execution_summary = processed_data.get("execution_summary", {})
        successful_steps = execution_summary.get("successful_steps", 0)
        failed_steps = execution_summary.get("failed_steps", 0)
        
        if successful_steps > 0 and failed_steps == 0:
            return "Your database operation has been completed successfully."
        elif successful_steps > 0 and failed_steps > 0:
            return f"Your database operation was partially completed. {successful_steps} operations succeeded, but {failed_steps} operations failed."
        else:
            return "I'm sorry, but the database operation could not be completed. Please check your information and try again."

    def _generate_fallback_response(
        self, state: OrchestratorState, tool_results: Dict[str, Any]
    ) -> str:
        """
        Generate fallback response when synthesis fails.

        Args:
            state: Current orchestrator state
            tool_results: Raw tool results

        Returns:
            Fallback response
        """
        if not tool_results:
            return f"I'm sorry, but I couldn't process your request: '{state.user_input}'. Please try again."

        # Count successful vs failed steps
        total_steps = len(tool_results)
        successful_steps = sum(1 for result in tool_results.values() if result.get("success", False))

        if successful_steps == 0:
            return f"I'm sorry, but I couldn't complete your request: '{state.user_input}'. All operations failed. Please check your parameters and try again."
        elif successful_steps < total_steps:
            return f"I partially completed your request: '{state.user_input}'. Some operations succeeded while others failed. Please check the results and try again if needed."
        else:
            return f"I've completed your request: '{state.user_input}'. All operations were successful."


async def synthesize_multi_intent_response(state: OrchestratorState) -> OrchestratorState:
    """
    Synthesize structured response for multiple intents.
    
    Args:
        state: OrchestratorState with multiple intent results
        
    Returns:
        Updated state with structured multi-intent response
    """
    results = state.execution_results.get("multiple_intent_results", [])
    
    # Group results by intent type
    intent_responses = {
        "read": [],
        "database_operations": [],
        "advice": []
    }
    
    for result in results:
        intent_type = result.get("intent", "unknown")
        if intent_type in intent_responses:
            intent_responses[intent_type].append(result)
    
    # Build structured response
    response_sections = []
    
    # Read results
    if intent_responses["read"]:
        response_sections.append("📊 **Data Retrieved:**")
        for result in intent_responses["read"]:
            if result.get("success", False):
                response_sections.append(f"• {result.get('summary', 'Data retrieved successfully')}")
            else:
                response_sections.append(f"• ❌ {result.get('error', 'Failed to retrieve data')}")
    
    # Database operations results  
    if intent_responses["database_operations"]:
        response_sections.append("\n💾 **Database Operations:**")
        for result in intent_responses["database_operations"]:
            if result.get("success", False):
                response_sections.append(f"• {result.get('summary', 'Operation completed successfully')}")
            else:
                response_sections.append(f"• ❌ {result.get('error', 'Operation failed')}")
    
    # Advice results
    if intent_responses["advice"]:
        response_sections.append("\n💡 **Financial Advice:**")
        for result in intent_responses["advice"]:
            if result.get("success", False):
                response_sections.append(f"• {result.get('summary', 'Advice provided successfully')}")
            else:
                response_sections.append(f"• ❌ {result.get('error', 'Failed to provide advice')}")
    
    # Add summary if we have mixed results
    successful_count = state.execution_results.get("successful_count", 0)
    failed_count = state.execution_results.get("failed_count", 0)
    
    if successful_count > 0 and failed_count > 0:
        response_sections.insert(0, f"I've processed your request with {successful_count + failed_count} actions. Here's what happened:\n")
    elif successful_count > 0:
        response_sections.insert(0, f"✅ I've successfully completed all {successful_count} requested actions:\n")
    elif failed_count > 0:
        response_sections.insert(0, f"❌ I encountered issues with all {failed_count} requested actions:\n")
    
    final_response = "\n".join(response_sections)
    
    return state.model_copy(update={"final_response": final_response})


async def response_synthesis_node(state: OrchestratorState) -> OrchestratorState:
    """
    Generate natural language response from tool execution results.

    Args:
        state: OrchestratorState with tool_results and execution context

    Returns:
        Updated OrchestratorState with final_response
    """
    node_name = "response_synthesis_node"
    
    async with track_node_execution(node_name, state.session_id) as exec_logger:
        try:
            # Check for multiple intents first
            if state.has_multiple_intents:
                exec_logger.log_step("multi_intent_response_synthesis_start", {
                    "intent_count": len(state.multiple_intents),
                    "intent_types": [intent.intent for intent in state.multiple_intents]
                })
                
                # Add system message for tracking
                state = MessageManager.add_system_message(
                    state,
                    f"Starting multi-intent response synthesis for {len(state.multiple_intents)} intents",
                    node_name,
                )
                
                # Synthesize multi-intent response
                result_state = await synthesize_multi_intent_response(state)
                
                exec_logger.log_step("multi_intent_response_synthesis_complete", {
                    "response_length": len(result_state.final_response or ""),
                    "successful_count": result_state.execution_results.get("successful_count", 0),
                    "failed_count": result_state.execution_results.get("failed_count", 0)
                })
                
                # Add execution metrics to state
                execution_metrics = exec_logger.end(success=True, metadata={
                    "synthesis_type": "multi_intent",
                    "intent_count": len(state.multiple_intents),
                    "response_length": len(result_state.final_response or "")
                })
                
                return add_execution_metrics_to_state(result_state, execution_metrics)
            
            # Single intent response synthesis (existing logic)
            exec_logger.log_step("node_start", {
                "intent": state.intent.value if state.intent else "unknown",
                "tool_results_exists": bool(state.tool_results),
                "tool_results_count": len(state.tool_results) if state.tool_results else 0
            })

            # Add system message for tracking
            state = MessageManager.add_system_message(
                state,
                f"Starting response synthesis for {state.intent.value if state.intent else 'unknown'} intent",
                node_name,
            )

            exec_logger.log_step("tool_results_validation")

            # Validate tool results exist
            if not state.tool_results:
                error_msg = "No tool results available for response synthesis"
                exec_logger.log_error(Exception(error_msg), {"error_type": "missing_tool_results"})
                
                error_state = state.model_copy(
                    update={
                        "final_response": "I'm sorry, but I couldn't process your request. Please try again.",
                        "metadata": {
                            **state.metadata,
                            "synthesis_status": "error",
                            "synthesis_errors": [error_msg],
                            "synthesis_time_ms": 0.0,
                            "node_name": node_name,
                            "node_timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    }
                )
                
                execution_metrics = exec_logger.end(success=False, error=error_msg, error_type="MissingToolResultsError")
                error_state = add_execution_metrics_to_state(error_state, execution_metrics)
                return error_state

            exec_logger.log_step("synthesizer_initialization")

            # Initialize synthesizer
            synthesizer = ResponseSynthesizer()

            exec_logger.log_step("response_generation_start", {
                "tool_results_keys": list(state.tool_results.keys()) if state.tool_results else []
            })

            # Generate response
            final_response = await synthesizer.synthesize_response(
                state, state.tool_results
            )

            exec_logger.log_step("response_generation_complete", {
                "response_length": len(final_response),
                "response_preview": final_response[:100] + "..." if len(final_response) > 100 else final_response
            })

            exec_logger.log_step("state_update_start")

            # Update state with final response
            updated_state = state.model_copy(
                update={
                    "final_response": final_response,
                    "metadata": {
                        **state.metadata,
                        "synthesis_status": "success",
                        "node_name": node_name,
                        "node_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )

            exec_logger.log_step("node_complete", {
                "synthesis_status": "success",
                "response_length": len(final_response),
                "final_response_generated": True
            })

            # Add execution metrics to state
            execution_metrics = exec_logger.end(success=True, metadata={
                "synthesis_status": "success",
                "response_length": len(final_response),
                "tool_results_count": len(state.tool_results) if state.tool_results else 0,
                "intent": state.intent.value if state.intent else "unknown"
            })

            updated_state = add_execution_metrics_to_state(updated_state, execution_metrics)

            return updated_state

        except Exception as e:
            exec_logger.log_error(e, {
                "intent": state.intent.value if state.intent else "unknown",
                "session_id": state.session_id,
                "tool_results_exists": bool(state.tool_results),
                "error_context": "response_synthesis_node"
            })
            
            # Return state with error response
            error_state = state.model_copy(
                update={
                    "final_response": "I'm sorry, but I encountered an error while processing your request. Please try again.",
                    "metadata": {
                        **state.metadata,
                        "synthesis_status": "error",
                        "synthesis_errors": [str(e)],
                        "node_name": node_name,
                        "node_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
            
            # Add execution metrics to error state
            execution_metrics = exec_logger.end(success=False, error=str(e), error_type=type(e).__name__)
            error_state = add_execution_metrics_to_state(error_state, execution_metrics)
            
            return error_state
