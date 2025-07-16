#!/usr/bin/env python3
"""
Demo script for Completeness Validator Node.

This script demonstrates the completeness validator node functionality, showing
how it determines whether parameter collection is complete, incomplete, or has
reached maximum attempts based on rule-based validation.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from uuid import uuid4

from src.nodes.completeness_validator_node import completeness_validator_node
from src.schemas.state_schemas import ClarificationState, IntentType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_scenario_header(scenario_title: str, description: str):
    """Print formatted scenario header."""
    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario_title}")
    print("=" * 80)
    print(f"Description: {description}")
    print("-" * 80)


def print_validation_result(before: ClarificationState, status: str, after: ClarificationState):
    """Print detailed validation result."""
    print(f"\n📥 INPUT STATE:")
    print(f"   Intent: {before.intent}")
    print(f"   User Input: '{before.user_input}'")
    print(f"   Attempts: {before.clarification_attempts}/{before.max_attempts}")
    print(f"   Missing Critical Params: {before.missing_critical_params}")
    print(f"   Extracted Parameters: {dict(before.extracted_parameters)}")
    
    print(f"\n🔍 VALIDATION RESULT:")
    print(f"   Status: {status}")
    print(f"   Clarification Status: {after.metadata.get('clarification_status', 'N/A')}")
    
    if status == "complete":
        print(f"   Completion Reason: {after.metadata.get('completion_reason', 'N/A')}")
        print(f"   Total Attempts Used: {after.metadata.get('total_attempts_used', 'N/A')}")
    elif status == "incomplete":
        print(f"   Remaining Critical Params: {after.metadata.get('remaining_critical_params', 'N/A')}")
        print(f"   Attempts Remaining: {after.metadata.get('attempts_remaining', 'N/A')}")
    elif status == "max_attempts_reached":
        print(f"   Exit Message: {after.metadata.get('exit_message', 'N/A')}")
        print(f"   Missing at Exit: {after.metadata.get('missing_critical_params_at_exit', 'N/A')}")
        print(f"   Partial Data Available: {after.metadata.get('partial_data_available', 'N/A')}")


def create_sample_state(
    intent: IntentType,
    user_input: str,
    extracted_parameters: Dict[str, Any],
    missing_critical_params: list[str],
    clarification_attempts: int,
    max_attempts: int = 3,
    clarification_history: list[Dict[str, Any]] = None
) -> ClarificationState:
    """Create a sample clarification state for testing."""
    return ClarificationState(
        session_id=str(uuid4()),
        user_input=user_input,
        intent=intent,
        extracted_parameters=extracted_parameters,
        missing_params=missing_critical_params,  # Simplified - same as critical
        missing_critical_params=missing_critical_params,
        parameter_priorities=missing_critical_params,
        normalization_suggestions={},
        ambiguity_flags={},
        clarification_history=clarification_history or [],
        clarification_attempts=clarification_attempts,
        max_attempts=max_attempts
    )


async def demo_scenario_complete():
    """Demo: All critical parameters collected - completion."""
    print_scenario_header(
        "ALL CRITICAL PARAMETERS COLLECTED",
        "User wants to transfer money and all required parameters have been collected."
    )
    
    # Create state with all critical parameters present
    state = create_sample_state(
        intent=IntentType.ACTION,
        user_input="Transfer $500 from checking to savings",
        extracted_parameters={
            "action_type": "transfer",
            "amount": 500.0,
            "source_account": "checking",
            "target_account": "savings"
        },
        missing_critical_params=[],  # No missing critical params
        clarification_attempts=1,
        clarification_history=[
            {
                "question": "Which account would you like to transfer from?",
                "user_response": "My checking account",
                "targeted_param": "source_account",
                "attempt": 1
            }
        ]
    )
    
    status, updated_state = await completeness_validator_node(state)
    print_validation_result(state, status, updated_state)
    
    print(f"\n✅ RESULT: Parameter collection COMPLETE. Ready to proceed to orchestrator.")


async def demo_scenario_incomplete():
    """Demo: Missing critical parameters - continue clarification."""
    print_scenario_header(
        "MISSING CRITICAL PARAMETERS",
        "User wants to fetch data but entity type is still missing."
    )
    
    # Create state with missing critical parameters (entity_type is critical for DATA_FETCH)
    state = create_sample_state(
        intent=IntentType.DATA_FETCH,
        user_input="Show me data from last month",
        extracted_parameters={
            "entity_type": None,  # Missing critical parameter
            "time_period": "last_month",
            "account_types": ["checking", "savings"]
        },
        missing_critical_params=["entity_type"],
        clarification_attempts=1,
        clarification_history=[
            {
                "question": "What time period would you like to see data for?",
                "user_response": "Last month",
                "targeted_param": "time_period",
                "attempt": 1
            }
        ]
    )
    
    status, updated_state = await completeness_validator_node(state)
    print_validation_result(state, status, updated_state)
    
    print(f"\n🔄 RESULT: Parameter collection INCOMPLETE. Will continue clarification loop.")


async def demo_scenario_max_attempts_with_missing():
    """Demo: Max attempts reached with missing parameters - exit with partial data."""
    print_scenario_header(
        "MAX ATTEMPTS REACHED WITH MISSING PARAMETERS",
        "Bot has asked 3 questions but user hasn't provided metric type information."
    )
    
    # Create state at max attempts with missing critical params (metric_type is critical for AGGREGATE)
    state = create_sample_state(
        intent=IntentType.AGGREGATE,
        user_input="I don't know what kind of analysis",
        extracted_parameters={
            "metric_type": None,  # Still missing after max attempts (critical for AGGREGATE)
            "time_period": "last_month",
            "group_by": ["category"],
            "account_filter": ["checking"]
        },
        missing_critical_params=["metric_type"],
        clarification_attempts=3,  # At max attempts
        max_attempts=3,
        clarification_history=[
            {
                "question": "What time period would you like to analyze?",
                "user_response": "Last month",
                "targeted_param": "time_period",
                "attempt": 1
            },
            {
                "question": "What type of analysis would you like - sum, average, count?",
                "user_response": "I'm not sure",
                "targeted_param": "metric_type",
                "attempt": 2
            },
            {
                "question": "Would you like to see totals, averages, or count of transactions?",
                "user_response": "I don't know what kind of analysis",
                "targeted_param": "metric_type",
                "attempt": 3
            }
        ]
    )
    
    status, updated_state = await completeness_validator_node(state)
    print_validation_result(state, status, updated_state)
    
    print(f"\n🚪 RESULT: MAX ATTEMPTS reached. Will exit clarification with partial data.")


async def demo_scenario_max_attempts_complete():
    """Demo: Max attempts reached but no missing parameters - treat as complete."""
    print_scenario_header(
        "MAX ATTEMPTS REACHED BUT COMPLETE",
        "Bot has used all attempts but managed to collect all critical parameters."
    )
    
    # Create state at max attempts but with all critical params
    state = create_sample_state(
        intent=IntentType.DATA_FETCH,
        user_input="Last 6 months from all accounts",
        extracted_parameters={
            "entity_type": "transactions",
            "time_period": "last_6_months",
            "account_types": ["checking", "savings"],
            "limit": 50
        },
        missing_critical_params=[],  # All critical params collected
        clarification_attempts=3,  # At max attempts
        max_attempts=3,
        clarification_history=[
            {
                "question": "What type of data would you like to see?",
                "user_response": "My transactions",
                "targeted_param": "entity_type",
                "attempt": 1
            },
            {
                "question": "What time period are you interested in?",
                "user_response": "The last 6 months",
                "targeted_param": "time_period",
                "attempt": 2
            },
            {
                "question": "Which accounts should I include?",
                "user_response": "All of them - checking and savings",
                "targeted_param": "account_types",
                "attempt": 3
            }
        ]
    )
    
    status, updated_state = await completeness_validator_node(state)
    print_validation_result(state, status, updated_state)
    
    print(f"\n✅ RESULT: All critical parameters collected despite max attempts. Ready to proceed.")


async def demo_scenario_quality_validation_failure():
    """Demo: Quality validation fails - treat as incomplete."""
    print_scenario_header(
        "QUALITY VALIDATION FAILURE",
        "All critical parameters present but some have invalid/placeholder values."
    )
    
    # Create state with invalid parameter values (action_type is critical and has placeholder value)
    state = create_sample_state(
        intent=IntentType.ACTION,
        user_input="Do some action",
        extracted_parameters={
            "action_type": "unknown",  # Invalid placeholder value for critical param
            "amount": 100.0,
            "source_account": "checking",
            "target_account": "savings"
        },
        missing_critical_params=[],  # No missing params but quality issues
        clarification_attempts=1
    )
    
    status, updated_state = await completeness_validator_node(state)
    print_validation_result(state, status, updated_state)
    
    print(f"\n❌ RESULT: Quality validation FAILED. Will continue clarification to fix invalid values.")


async def demo_scenario_at_max_attempts_edge_case():
    """Demo: Edge case where attempts equal max attempts."""
    print_scenario_header(
        "AT MAX ATTEMPTS EDGE CASE",
        "System has reached max attempts exactly - immediate exit required."
    )
    
    # Create state exactly at max attempts (since Pydantic requires max_attempts >= 1)
    state = create_sample_state(
        intent=IntentType.DATA_FETCH,
        user_input="Show me some data",
        extracted_parameters={
            "entity_type": None  # Missing critical parameter
        },
        missing_critical_params=["entity_type"],
        clarification_attempts=1,  # At max attempts
        max_attempts=1  # Minimum allowed by Pydantic
    )
    
    status, updated_state = await completeness_validator_node(state)
    print_validation_result(state, status, updated_state)
    
    print(f"\n🚪 RESULT: At max attempts - immediate exit with partial data.")


async def main():
    """Run all completeness validator demo scenarios."""
    print("🔍 COMPLETENESS VALIDATOR NODE DEMO")
    print("=" * 80)
    print("This demo shows how the Completeness Validator determines whether")
    print("parameter collection is complete, incomplete, or has reached max attempts.")
    print("\nThe validator uses pure rule-based logic without LLM calls:")
    print("• Checks missing_critical_params list")
    print("• Enforces max_attempts limit")
    print("• Performs lightweight quality validation")
    print("• Updates metadata for downstream routing")
    
    # Run all scenarios
    await demo_scenario_complete()
    await demo_scenario_incomplete()
    await demo_scenario_max_attempts_with_missing()
    await demo_scenario_max_attempts_complete()
    await demo_scenario_quality_validation_failure()
    await demo_scenario_at_max_attempts_edge_case()
    
    print("\n" + "=" * 80)
    print("✅ COMPLETENESS VALIDATOR DEMO COMPLETED")
    print("=" * 80)
    print("\nKey takeaways:")
    print("• Rule-based validation without LLM dependency")
    print("• Three possible outcomes: complete, incomplete, max_attempts_reached")
    print("• Quality validation catches invalid/placeholder values")
    print("• Proper metadata updates for downstream routing")
    print("• Graceful handling of edge cases and error conditions")


if __name__ == "__main__":
    asyncio.run(main()) 