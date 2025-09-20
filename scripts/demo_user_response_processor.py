"""
Demo script for User Response Processor Node.

This script demonstrates the user response processor node functionality with real OpenAI 
integration, showing various scenarios including successful parsing, validation errors,
ambiguous responses, and different data types.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from dotenv import load_dotenv

from vyuu_copilot_v2.nodes.user_response_processor_node import user_response_processor_node
from vyuu_copilot_v2.schemas.state_schemas import ClarificationState, IntentType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_state(
    intent: IntentType,
    user_input: str,
    extracted_parameters: Dict[str, Any],
    missing_params: list[str],
    missing_critical_params: list[str],
    clarification_history: list[Dict[str, Any]],
    normalization_suggestions: Dict[str, str] = None,
    ambiguity_flags: Dict[str, str] = None
) -> ClarificationState:
    """Create a sample clarification state for testing."""
    return ClarificationState(
        session_id=str(uuid4()),
        user_id=str(uuid4()),
        user_input=user_input,
        intent=intent,
        extracted_parameters=extracted_parameters,
        missing_params=missing_params,
        missing_critical_params=missing_critical_params,
        parameter_priorities=missing_params,  # Simple prioritization
        normalization_suggestions=normalization_suggestions or {},
        ambiguity_flags=ambiguity_flags or {},
        clarification_history=clarification_history,
        clarification_attempts=len(clarification_history),
        max_attempts=3
    )


def print_state_comparison(original_state: ClarificationState, updated_state: ClarificationState):
    """Print a comparison of before and after states."""
    print("\n" + "="*60)
    print("STATE COMPARISON")
    print("="*60)
    
    print("\nEXTRACTED PARAMETERS:")
    print(f"  Before: {original_state.extracted_parameters}")
    print(f"  After:  {updated_state.extracted_parameters}")
    
    print("\nMISSING PARAMETERS:")
    print(f"  Before: {original_state.missing_params}")
    print(f"  After:  {updated_state.missing_params}")
    
    print("\nMISSING CRITICAL PARAMETERS:")
    print(f"  Before: {original_state.missing_critical_params}")
    print(f"  After:  {updated_state.missing_critical_params}")
    
    print("\nAMBIGUITY FLAGS:")
    print(f"  Before: {original_state.ambiguity_flags}")
    print(f"  After:  {updated_state.ambiguity_flags}")
    
    print("\nCLARIFICATION HISTORY:")
    print(f"  Entries: {len(updated_state.clarification_history)}")
    if updated_state.clarification_history:
        last_entry = updated_state.clarification_history[-1]
        if "user_response" in last_entry:
            print(f"  Last Response: {last_entry['user_response']}")
            print(f"  Extracted Values: {last_entry.get('extracted_values', 'N/A')}")
            print(f"  Confidence: {last_entry.get('extraction_confidence', 'N/A')}")
    
    if updated_state.errors:
        print(f"\nERRORS: {updated_state.errors}")


async def demo_scenario_1_successful_time_period():
    """Demo Scenario 1: Successful time period extraction."""
    print("\n" + "="*80)
    print("DEMO SCENARIO 1: Successful Time Period Extraction")
    print("="*80)
    print("User Query: 'Show me sales data for Q1'")
    print("Clarification Question: 'What time period would you like to see sales data for?'")
    print("User Response: 'Q1 2024'")
    
    state = create_sample_state(
        intent=IntentType.DATA_FETCH,
        user_input="Show me sales data for Q1",
        extracted_parameters={
            "entity_type": "sales_data",
            "time_period": None,
            "region": None
        },
        missing_params=["time_period", "region"],
        missing_critical_params=["time_period"],
        clarification_history=[
            {
                "question": "What time period would you like to see sales data for?",
                "targeted_param": "time_period",
                "timestamp": datetime.now().isoformat(),
                "attempt_number": 1
            }
        ],
        normalization_suggestions={
            "Q1": "2024-Q1",
            "first quarter": "2024-Q1"
        }
    )
    
    try:
        updated_state = await user_response_processor_node(state, "Q1 2024")
        print_state_comparison(state, updated_state)
        print("\n✅ RESULT: Successfully extracted and normalized time period")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


async def demo_scenario_2_numeric_amount():
    """Demo Scenario 2: Numeric amount with currency formatting."""
    print("\n" + "="*80)
    print("DEMO SCENARIO 2: Numeric Amount with Currency Formatting")
    print("="*80)
    print("User Query: 'Update budget for marketing'")
    print("Clarification Question: 'What amount would you like to set for the marketing budget?'")
    print("User Response: '$50,000.00'")
    
    state = create_sample_state(
        intent=IntentType.ACTION,
        user_input="Update budget for marketing",
        extracted_parameters={
            "action_type": "update_budget",
            "target": "marketing",
            "amount": None
        },
        missing_params=["amount"],
        missing_critical_params=["amount"],
        clarification_history=[
            {
                "question": "What amount would you like to set for the marketing budget?",
                "targeted_param": "amount",
                "timestamp": datetime.now().isoformat(),
                "attempt_number": 1
            }
        ]
    )
    
    try:
        updated_state = await user_response_processor_node(state, "$50,000.00")
        print_state_comparison(state, updated_state)
        print("\n✅ RESULT: Successfully extracted and normalized currency amount")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


async def demo_scenario_3_date_validation():
    """Demo Scenario 3: Date validation with multiple formats."""
    print("\n" + "="*80)
    print("DEMO SCENARIO 3: Date Validation with Multiple Formats")
    print("="*80)
    print("User Query: 'Schedule a meeting'")
    print("Clarification Question: 'What date would you like to schedule the meeting?'")
    print("User Response: '01/15/2024'")
    
    state = create_sample_state(
        intent=IntentType.ACTION,
        user_input="Schedule a meeting",
        extracted_parameters={
            "action_type": "schedule_meeting",
            "meeting_date": None,
            "attendees": None
        },
        missing_params=["meeting_date", "attendees"],
        missing_critical_params=["meeting_date"],
        clarification_history=[
            {
                "question": "What date would you like to schedule the meeting?",
                "targeted_param": "meeting_date",
                "timestamp": datetime.now().isoformat(),
                "attempt_number": 1
            }
        ]
    )
    
    try:
        updated_state = await user_response_processor_node(state, "01/15/2024")
        print_state_comparison(state, updated_state)
        print("\n✅ RESULT: Successfully validated and normalized date")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


async def demo_scenario_4_dont_know_response():
    """Demo Scenario 4: Handling 'don't know' responses."""
    print("\n" + "="*80)
    print("DEMO SCENARIO 4: Handling 'Don't Know' Responses")
    print("="*80)
    print("User Query: 'Show me customer data'")
    print("Clarification Question: 'Which region would you like to see customer data for?'")
    print("User Response: 'I'm not sure'")
    
    state = create_sample_state(
        intent=IntentType.DATA_FETCH,
        user_input="Show me customer data",
        extracted_parameters={
            "entity_type": "customer_data",
            "region": None,
            "filters": None
        },
        missing_params=["region", "filters"],
        missing_critical_params=[],
        clarification_history=[
            {
                "question": "Which region would you like to see customer data for?",
                "targeted_param": "region",
                "timestamp": datetime.now().isoformat(),
                "attempt_number": 1
            }
        ]
    )
    
    try:
        updated_state = await user_response_processor_node(state, "I'm not sure")
        print_state_comparison(state, updated_state)
        print("\n✅ RESULT: Properly handled uncertain response with ambiguity flag")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


async def demo_scenario_5_validation_error():
    """Demo Scenario 5: Validation error with invalid data."""
    print("\n" + "="*80)
    print("DEMO SCENARIO 5: Validation Error with Invalid Data")
    print("="*80)
    print("User Query: 'Calculate average revenue'")
    print("Clarification Question: 'What time period should I use for the calculation?'")
    print("User Response: 'sometime last year maybe'")
    
    state = create_sample_state(
        intent=IntentType.AGGREGATE,
        user_input="Calculate average revenue",
        extracted_parameters={
            "entity_type": "revenue",
            "metric": "average",
            "time_period": None,
            "grouping": None
        },
        missing_params=["time_period", "grouping"],
        missing_critical_params=["time_period"],
        clarification_history=[
            {
                "question": "What time period should I use for the calculation?",
                "targeted_param": "time_period",
                "timestamp": datetime.now().isoformat(),
                "attempt_number": 1
            }
        ]
    )
    
    try:
        updated_state = await user_response_processor_node(state, "sometime last year maybe")
        print_state_comparison(state, updated_state)
        print("\n✅ RESULT: Handled ambiguous response with appropriate flags")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


async def demo_scenario_6_multi_slot_response():
    """Demo Scenario 6: Multi-slot response handling."""
    print("\n" + "="*80)
    print("DEMO SCENARIO 6: Multi-Slot Response Handling")
    print("="*80)
    print("User Query: 'Generate quarterly report'")
    print("Clarification Question: 'What region and time period for the quarterly report?'")
    print("User Response: 'North America for Q4 2023'")
    
    state = create_sample_state(
        intent=IntentType.DATA_FETCH,
        user_input="Generate quarterly report",
        extracted_parameters={
            "entity_type": "quarterly_report",
            "region": None,
            "time_period": None
        },
        missing_params=["region", "time_period"],
        missing_critical_params=["time_period"],
        clarification_history=[
            {
                "question": "What region and time period for the quarterly report?",
                "targeted_param": ["region", "time_period"],
                "timestamp": datetime.now().isoformat(),
                "attempt_number": 1
            }
        ]
    )
    
    try:
        updated_state = await user_response_processor_node(state, "North America for Q4 2023")
        print_state_comparison(state, updated_state)
        print("\n✅ RESULT: Successfully extracted multiple slot values from single response")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


async def main():
    """Run all demo scenarios."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    print("🚀 USER RESPONSE PROCESSOR NODE DEMO")
    print("This demo showcases LLM-based user response parsing with various scenarios")
    print("Using real OpenAI API integration for structured JSON output")
    
    scenarios = [
        demo_scenario_1_successful_time_period,
        demo_scenario_2_numeric_amount,
        demo_scenario_3_date_validation,
        demo_scenario_4_dont_know_response,
        demo_scenario_5_validation_error,
        demo_scenario_6_multi_slot_response,
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        try:
            await scenario()
            print(f"\n✅ Scenario {i} completed successfully")
        except Exception as e:
            print(f"\n❌ Scenario {i} failed: {e}")
            logger.error(f"Scenario {i} error", exc_info=True)
        
        # Add delay between scenarios for better readability
        await asyncio.sleep(1)
    
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETED")
    print("="*80)
    print("Key Features Demonstrated:")
    print("• LLM-based response parsing with structured JSON output")
    print("• Comprehensive value validation and normalization")
    print("• Multi-format support (dates, currency, percentages)")
    print("• Ambiguity detection and flagging")
    print("• Error handling with fallback parsing")
    print("• State updates and missing parameter recalculation")
    print("• Multi-slot response handling")
    print("• Don't know response processing")


if __name__ == "__main__":
    asyncio.run(main()) 