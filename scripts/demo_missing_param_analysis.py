#!/usr/bin/env python3
"""
Demo script for Missing Parameter Analysis Node.

Shows the transformation of ClarificationState before and after
the missing parameter analysis node execution using REAL OpenAI API calls.
"""

import asyncio
import json
import uuid
import os
from datetime import datetime, timezone

from vyuu_copilot_v2.schemas.state_schemas import ClarificationState, IntentType, Message, MessageRole
from vyuu_copilot_v2.nodes.missing_param_analysis_node import missing_param_analysis_node


def print_state_comparison(before: ClarificationState, after: ClarificationState):
    """Print a detailed comparison of states before and after processing."""
    
    print("=" * 80)
    print("MISSING PARAMETER ANALYSIS NODE - STATE TRANSFORMATION")
    print("=" * 80)
    
    print("\n📥 INPUT STATE (ClarificationState):")
    print("-" * 50)
    print(f"Intent: {before.intent}")
    print(f"User Input: '{before.user_input}'")
    print(f"Confidence: {before.confidence}")
    print(f"Session ID: {before.session_id[:8]}...")
    print(f"Current Parameters: {json.dumps(before.parameters, indent=2)}")
    print(f"Missing Params: {before.missing_params}")
    print(f"Missing Critical Params: {before.missing_critical_params}")
    print(f"Extracted Parameters: {json.dumps(before.extracted_parameters, indent=2)}")
    print(f"Parameter Priorities: {before.parameter_priorities}")
    print(f"Normalization Suggestions: {before.normalization_suggestions}")
    print(f"Ambiguity Flags: {before.ambiguity_flags}")
    print(f"Clarification History: {before.clarification_history}")
    
    print("\n📤 OUTPUT STATE (ClarificationState):")
    print("-" * 50)
    print(f"Intent: {after.intent}")
    print(f"User Input: '{after.user_input}'")
    print(f"Confidence: {after.confidence}")
    print(f"Session ID: {after.session_id[:8]}...")
    print(f"Current Parameters: {json.dumps(after.parameters, indent=2)}")
    print(f"Missing Params: {after.missing_params}")
    print(f"Missing Critical Params: {after.missing_critical_params}")
    print(f"Extracted Parameters: {json.dumps(after.extracted_parameters, indent=2)}")
    print(f"Parameter Priorities: {after.parameter_priorities}")
    print(f"Normalization Suggestions: {json.dumps(after.normalization_suggestions, indent=2)}")
    print(f"Ambiguity Flags: {json.dumps(after.ambiguity_flags, indent=2)}")
    print(f"Clarification History: {after.clarification_history}")
    
    print("\n🔄 KEY CHANGES:")
    print("-" * 50)
    if before.extracted_parameters != after.extracted_parameters:
        print("✅ extracted_parameters: Updated with normalized values")
    if before.missing_params != after.missing_params:
        print("✅ missing_params: Refined list of still-missing slots")
    if before.missing_critical_params != after.missing_critical_params:
        print("✅ missing_critical_params: Updated critical missing slots")
    if before.parameter_priorities != after.parameter_priorities:
        print("✅ parameter_priorities: Ordered list for question generation")
    if before.normalization_suggestions != after.normalization_suggestions:
        print("✅ normalization_suggestions: User phrases → canonical values")
    if before.ambiguity_flags != after.ambiguity_flags:
        print("✅ ambiguity_flags: Slots needing disambiguation")


async def demo_action_intent_scenario():
    """Demo scenario: ACTION intent with partial parameters."""
    
    print("\n🎬 SCENARIO 1: ACTION Intent - Transfer Money")
    print("User said: 'Transfer $500 from checking to my emergency fund'")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found. Skipping scenario...")
        return
    
    # Create initial ClarificationState (as would come from decision router)
    initial_state = ClarificationState(
        user_input="Transfer $500 from checking to my emergency fund",
        intent=IntentType.ACTION,
        confidence=0.95,
        messages=[
            Message(role=MessageRole.USER, content="Transfer $500 from checking to my emergency fund"),
            Message(role=MessageRole.SYSTEM, content="Starting intent classification..."),
            Message(role=MessageRole.ASSISTANT, content="Classified as ACTION with 95% confidence")
        ],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={
            "routing_decision": "clarification",
            "routing_reason": "missing_critical_params"
        },
        parameters={
            "action_type": "transfer",
            "amount": 500.0,
            "source_account": "checking"
        },
        execution_results={},
        response="",
        # ClarificationState specific fields
        missing_params=["target_account", "description"],
        missing_critical_params=["target_account"],
        clarification_attempts=0,
        max_attempts=3,
        extracted_parameters={},  # Will be populated by analysis node
        parameter_priorities=[],  # Will be populated by analysis node
        normalization_suggestions={},  # Will be populated by analysis node
        ambiguity_flags={},  # Will be populated by analysis node
        clarification_history=[]  # Will be populated by analysis node
    )
    
    try:
        # Execute the node with REAL LLM call
        print("🤖 Calling OpenAI API...")
        result_state = await missing_param_analysis_node(initial_state)
        print("✅ LLM call completed successfully!")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return
    
    print_state_comparison(initial_state, result_state)


async def demo_data_fetch_scenario():
    """Demo scenario: DATA_FETCH intent with ambiguous parameters."""
    
    print("\n🎬 SCENARIO 2: DATA_FETCH Intent - Ambiguous Request")
    print("User said: 'Show me transactions from last month'")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found. Skipping scenario...")
        return
    
    # Create initial ClarificationState
    initial_state = ClarificationState(
        user_input="Show me transactions from last month",
        intent=IntentType.DATA_FETCH,
        confidence=0.75,
        messages=[
            Message(role=MessageRole.USER, content="Show me transactions from last month"),
            Message(role=MessageRole.SYSTEM, content="Starting intent classification..."),
            Message(role=MessageRole.ASSISTANT, content="Classified as DATA_FETCH with 75% confidence")
        ],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={
            "routing_decision": "clarification",
            "routing_reason": "missing_critical_params"
        },
        parameters={
            "time_period": "last_month"
        },
        execution_results={},
        response="",
        # ClarificationState specific fields
        missing_params=["entity_type", "account_types", "limit"],
        missing_critical_params=["entity_type"],
        clarification_attempts=0,
        max_attempts=3,
        extracted_parameters={},
        parameter_priorities=[],
        normalization_suggestions={},
        ambiguity_flags={},
        clarification_history=[]
    )
    
    try:
        # Execute the node with REAL LLM call
        print("🤖 Calling OpenAI API...")
        result_state = await missing_param_analysis_node(initial_state)
        print("✅ LLM call completed successfully!")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return
    
    print_state_comparison(initial_state, result_state)


async def demo_follow_up_scenario():
    """Demo scenario: Follow-up clarification with history."""
    
    print("\n🎬 SCENARIO 3: Follow-up Clarification with History")
    print("User originally said: 'Transfer money'")
    print("Bot asked: 'How much would you like to transfer?'")
    print("User replied: 'About five hundred dollars'")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found. Skipping scenario...")
        return
    
    # Create initial ClarificationState with history
    initial_state = ClarificationState(
        user_input="About five hundred dollars",
        intent=IntentType.ACTION,
        confidence=0.95,
        messages=[
            Message(role=MessageRole.USER, content="Transfer money"),
            Message(role=MessageRole.ASSISTANT, content="How much would you like to transfer?"),
            Message(role=MessageRole.USER, content="About five hundred dollars")
        ],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={
            "routing_decision": "clarification",
            "routing_reason": "missing_critical_params"
        },
        parameters={
            "action_type": "transfer"
        },
        execution_results={},
        response="",
        # ClarificationState specific fields
        missing_params=["amount", "source_account", "target_account"],
        missing_critical_params=["amount", "source_account", "target_account"],
        clarification_attempts=1,
        max_attempts=3,
        extracted_parameters={
            "action_type": "transfer"
        },
        parameter_priorities=["amount", "source_account", "target_account"],
        normalization_suggestions={},
        ambiguity_flags={},
        clarification_history=[
            {
                "question": "How much would you like to transfer?",
                "user_response": "About five hundred dollars",
                "targeted_param": "amount",
                "attempt": 1
            }
        ]
    )
    
    try:
        # Execute the node with REAL LLM call
        print("🤖 Calling OpenAI API...")
        result_state = await missing_param_analysis_node(initial_state)
        print("✅ LLM call completed successfully!")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return
    
    print_state_comparison(initial_state, result_state)


async def main():
    """Run all demo scenarios."""
    print("🚀 MISSING PARAMETER ANALYSIS NODE - OUTPUT EXAMPLES")
    print("This shows how the node enriches ClarificationState with LLM-driven analysis")
    print("Using REAL OpenAI API calls for authentic results!")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ No OpenAI API key found.")
        print("Please set your OPENAI_API_KEY environment variable to run this demo.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    await demo_action_intent_scenario()
    await demo_data_fetch_scenario()
    await demo_follow_up_scenario()
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("The Missing Parameter Analysis Node successfully:")
    print("• Normalizes and validates extracted parameters using Pydantic models")
    print("• Refines missing parameter lists based on LLM analysis")
    print("• Prioritizes missing parameters for optimal question ordering")
    print("• Provides normalization suggestions for user phrase mapping")
    print("• Flags ambiguous values that need clarification")
    print("• Preserves clarification history for context-aware processing")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main()) 