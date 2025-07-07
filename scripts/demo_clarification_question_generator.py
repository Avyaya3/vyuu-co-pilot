#!/usr/bin/env python3
"""
Demo script for Clarification Question Generator Node.

Shows how the node generates contextual clarification questions based on 
Missing Parameter Analysis output using REAL OpenAI API calls.
"""

import asyncio
import json
import uuid
import os
from datetime import datetime, timezone

from src.schemas.state_schemas import ClarificationState, IntentType, Message, MessageRole
from src.nodes.clarification_question_generator_node import clarification_question_generator_node


def print_question_generation(before: ClarificationState, question: str, after: ClarificationState):
    """Print detailed view of question generation process."""
    
    print("=" * 80)
    print("CLARIFICATION QUESTION GENERATOR - QUESTION GENERATION")
    print("=" * 80)
    
    print("\n📥 INPUT STATE (ClarificationState):")
    print("-" * 50)
    print(f"Intent: {before.intent}")
    print(f"User Input: '{before.user_input}'")
    print(f"Clarification Attempts: {before.clarification_attempts}/{before.max_attempts}")
    print(f"Extracted Parameters: {json.dumps(before.extracted_parameters, indent=2)}")
    print(f"Missing Params: {before.missing_params}")
    print(f"Missing Critical Params: {before.missing_critical_params}")
    print(f"Parameter Priorities: {before.parameter_priorities}")
    print(f"Normalization Suggestions: {json.dumps(before.normalization_suggestions, indent=2)}")
    print(f"Ambiguity Flags: {json.dumps(before.ambiguity_flags, indent=2)}")
    print(f"Clarification History ({len(before.clarification_history)} entries):")
    for i, entry in enumerate(before.clarification_history, 1):
        print(f"  {i}. Q: {entry.get('question', 'N/A')}")
        print(f"     A: {entry.get('user_response', 'Not answered yet')}")
        print(f"     Target: {entry.get('targeted_param', 'N/A')}")
    
    print(f"\n🤖 GENERATED QUESTION:")
    print("-" * 50)
    print(f"'{question}'")
    
    print(f"\n📤 OUTPUT STATE (ClarificationState):")
    print("-" * 50)
    print(f"Clarification Attempts: {after.clarification_attempts}/{after.max_attempts}")
    print(f"Clarification History ({len(after.clarification_history)} entries):")
    for i, entry in enumerate(after.clarification_history, 1):
        print(f"  {i}. Q: {entry.get('question', 'N/A')}")
        print(f"     A: {entry.get('user_response', 'Not answered yet')}")
        print(f"     Target: {entry.get('targeted_param', 'N/A')}")
        print(f"     Attempt: {entry.get('attempt', 'N/A')}")
    
    print("\n🔄 KEY CHANGES:")
    print("-" * 50)
    if after.clarification_attempts > before.clarification_attempts:
        print("✅ clarification_attempts: Incremented")
    if len(after.clarification_history) > len(before.clarification_history):
        print("✅ clarification_history: New question added")
        new_entry = after.clarification_history[-1]
        print(f"   - Question: '{new_entry.get('question', 'N/A')}'")
        print(f"   - Targeting: {new_entry.get('targeted_param', 'N/A')}")


async def demo_action_intent_first_question():
    """Demo: ACTION intent - First clarification question."""
    
    print("\n🎬 SCENARIO 1: ACTION Intent - First Clarification Question")
    print("User said: 'Transfer money to my emergency fund'")
    print("Missing Parameter Analysis identified missing: amount, source_account")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found. Skipping scenario...")
        return
    
    # Create ClarificationState as it would come from Missing Parameter Analysis
    initial_state = ClarificationState(
        user_input="Transfer money to my emergency fund",
        intent=IntentType.ACTION,
        confidence=0.95,
        messages=[
            Message(role=MessageRole.USER, content="Transfer money to my emergency fund"),
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
            "target_account": "emergency fund"
        },
        execution_results={},
        response="",
        # From Missing Parameter Analysis
        missing_params=["amount", "source_account"],
        missing_critical_params=["amount", "source_account"],
        clarification_attempts=0,
        max_attempts=3,
        extracted_parameters={
            "action_type": "transfer",
            "amount": None,
            "source_account": None,
            "target_account": "emergency fund",
            "description": None
        },
        parameter_priorities=["amount", "source_account"],  # LLM prioritized amount first
        normalization_suggestions={},
        ambiguity_flags={},
        clarification_history=[]
    )
    
    try:
        # Generate the question with REAL LLM call
        print("🤖 Calling OpenAI API to generate question...")
        question, result_state = await clarification_question_generator_node(initial_state)
        print("✅ Question generated successfully!")
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        return
    
    print_question_generation(initial_state, question, result_state)


async def demo_data_fetch_with_ambiguity():
    """Demo: DATA_FETCH intent with ambiguity flags."""
    
    print("\n🎬 SCENARIO 2: DATA_FETCH Intent - With Ambiguity Resolution")
    print("User said: 'Show me transactions from last month'")
    print("Missing Parameter Analysis found ambiguity in time_period and missing account_types")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found. Skipping scenario...")
        return
    
    # Create ClarificationState with ambiguity flags
    initial_state = ClarificationState(
        user_input="Show me transactions from last month",
        intent=IntentType.DATA_FETCH,
        confidence=0.80,
        messages=[
            Message(role=MessageRole.USER, content="Show me transactions from last month"),
            Message(role=MessageRole.ASSISTANT, content="I'll help you find your transactions.")
        ],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={
            "entity_type": "transactions",
            "time_period": "last_month"
        },
        execution_results={},
        response="",
        # From Missing Parameter Analysis
        missing_params=["account_types"],
        missing_critical_params=[],  # entity_type filled
        clarification_attempts=0,
        max_attempts=3,
        extracted_parameters={
            "entity_type": "transactions",
            "time_period": "2024-01",  # LLM normalized
            "account_types": None,
            "limit": 50,  # LLM suggested default
            "sort_by": None,
            "order": None
        },
        parameter_priorities=["account_types"],
        normalization_suggestions={
            "last_month": "January 2024"
        },
        ambiguity_flags={
            "time_period": "User said 'last month' - could mean different months depending on current date",
            "limit": "No limit specified - using default of 50"
        },
        clarification_history=[]
    )
    
    try:
        # Generate the question with REAL LLM call
        print("🤖 Calling OpenAI API to generate question...")
        question, result_state = await clarification_question_generator_node(initial_state)
        print("✅ Question generated successfully!")
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        return
    
    print_question_generation(initial_state, question, result_state)


async def demo_follow_up_clarification():
    """Demo: Follow-up question with conversation history."""
    
    print("\n🎬 SCENARIO 3: Follow-up Clarification with History")
    print("User originally said: 'Transfer money'")
    print("Bot asked: 'What amount would you like to transfer?'")
    print("User replied: 'About five hundred'")
    print("Missing Parameter Analysis processed the amount, now needs source_account")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found. Skipping scenario...")
        return
    
    # Create ClarificationState with existing history
    initial_state = ClarificationState(
        user_input="About five hundred",  # Latest user input
        intent=IntentType.ACTION,
        confidence=0.95,
        messages=[
            Message(role=MessageRole.USER, content="Transfer money"),
            Message(role=MessageRole.ASSISTANT, content="What amount would you like to transfer?"),
            Message(role=MessageRole.USER, content="About five hundred")
        ],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={
            "action_type": "transfer"
        },
        execution_results={},
        response="",
        # From Missing Parameter Analysis after processing "about five hundred"
        missing_params=["source_account", "target_account"],
        missing_critical_params=["source_account", "target_account"],
        clarification_attempts=1,  # Already asked one question
        max_attempts=3,
        extracted_parameters={
            "action_type": "transfer",
            "amount": 500.0,  # Normalized from "about five hundred"
            "source_account": None,
            "target_account": None,
            "description": None
        },
        parameter_priorities=["source_account", "target_account"],
        normalization_suggestions={
            "about five hundred": "500.0"
        },
        ambiguity_flags={
            "amount": "User said 'about' - may not be exact amount"
        },
        clarification_history=[
            {
                "question": "What amount would you like to transfer?",
                "user_response": "About five hundred",
                "targeted_param": "amount",
                "attempt": 1
            }
        ]
    )
    
    try:
        # Generate the follow-up question with REAL LLM call
        print("🤖 Calling OpenAI API to generate follow-up question...")
        question, result_state = await clarification_question_generator_node(initial_state)
        print("✅ Follow-up question generated successfully!")
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        return
    
    print_question_generation(initial_state, question, result_state)


async def demo_max_attempts_scenario():
    """Demo: Max attempts reached scenario - Exit with partial data."""
    
    print("\n🎬 SCENARIO 4: Max Attempts Reached - Exit with Partial Data")
    print("User has been asked 3 questions but still missing critical information")
    print("System should exit clarification subgraph with partial data per user flow")
    
    # Create ClarificationState at max attempts
    initial_state = ClarificationState(
        user_input="I need help with my account",
        intent=IntentType.UNKNOWN,
        confidence=0.40,
        messages=[],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={},
        execution_results={},
        response="",
        missing_params=["entity_type", "action_type"],
        missing_critical_params=["entity_type", "action_type"],
        clarification_attempts=3,  # At max
        max_attempts=3,
        extracted_parameters={},
        parameter_priorities=["entity_type", "action_type"],
        normalization_suggestions={},
        ambiguity_flags={},
        clarification_history=[
            {"question": "What would you like to do with your account?", "user_response": "Something", "targeted_param": "action_type", "attempt": 1},
            {"question": "What type of information do you need?", "user_response": "Help", "targeted_param": "entity_type", "attempt": 2},
            {"question": "Could you be more specific about what you're looking for?", "user_response": "I need help", "targeted_param": "general", "attempt": 3}
        ]
    )
    
    try:
        # This should trigger the exit with partial data behavior
        result, result_state = await clarification_question_generator_node(initial_state)
        
        if result == "EXIT_WITH_PARTIAL_DATA":
            print("✅ Exit with partial data triggered successfully!")
            print("\n🚪 EXIT SIGNAL DETAILS:")
            print("-" * 50)
            print(f"Exit Signal: {result}")
            print(f"Exit Message: {result_state.metadata.get('exit_message', 'N/A')}")
            print(f"Exit Reason: {result_state.metadata.get('exit_reason', 'N/A')}")
            print(f"Remaining Missing Params: {result_state.metadata.get('remaining_missing_params', [])}")
            print(f"Remaining Critical Params: {result_state.metadata.get('remaining_critical_params', [])}")
            print(f"Clarification Status: {result_state.metadata.get('clarification_status', 'N/A')}")
            
            print("\n📤 FINAL CLARIFICATION STATE:")
            print("-" * 50)
            print(f"Attempts: {result_state.clarification_attempts}/{result_state.max_attempts}")
            print(f"Exit Condition: {result_state.clarification_history[-1].get('exit_condition', False)}")
            print(f"Partial Parameters Available: {json.dumps(result_state.extracted_parameters, indent=2)}")
        else:
            print(f"❌ Unexpected result: {result}")
            print_question_generation(initial_state, result, result_state)
            
    except Exception as e:
        print(f"❌ Max attempts handling failed: {e}")
        return


async def demo_aggregate_intent():
    """Demo: AGGREGATE intent with multiple missing parameters."""
    
    print("\n🎬 SCENARIO 5: AGGREGATE Intent - Multiple Missing Parameters")
    print("User said: 'How much did I spend on food?'")
    print("Missing Parameter Analysis needs time_period and wants to clarify categories")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found. Skipping scenario...")
        return
    
    # Create ClarificationState for aggregate intent
    initial_state = ClarificationState(
        user_input="How much did I spend on food?",
        intent=IntentType.AGGREGATE,
        confidence=0.85,
        messages=[
            Message(role=MessageRole.USER, content="How much did I spend on food?"),
        ],
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        metadata={},
        parameters={
            "metric_type": "total_spending"
        },
        execution_results={},
        response="",
        # From Missing Parameter Analysis
        missing_params=["time_period", "categories"],
        missing_critical_params=["time_period"],
        clarification_attempts=0,
        max_attempts=3,
        extracted_parameters={
            "metric_type": "total_spending",
            "categories": ["food", "groceries", "restaurants"],  # LLM inferred food categories
            "time_period": None,
            "group_by": None,
            "account_types": None
        },
        parameter_priorities=["time_period", "categories"],  # Flattened for state schema
        normalization_suggestions={
            "food": "food, groceries, restaurants"
        },
        ambiguity_flags={
            "categories": "User said 'food' - could include groceries, restaurants, fast food, etc."
        },
        clarification_history=[]
    )
    
    try:
        # Generate the question with REAL LLM call
        print("🤖 Calling OpenAI API to generate question...")
        question, result_state = await clarification_question_generator_node(initial_state)
        print("✅ Question generated successfully!")
    except Exception as e:
        print(f"❌ Question generation failed: {e}")
        return
    
    print_question_generation(initial_state, question, result_state)


async def main():
    """Run all demo scenarios."""
    print("🚀 CLARIFICATION QUESTION GENERATOR NODE - DEMO")
    print("This shows how the node generates contextual clarification questions")
    print("Using REAL OpenAI API calls for authentic question generation!")
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ No OpenAI API key found.")
        print("Please set your OPENAI_API_KEY environment variable to run this demo.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    await demo_action_intent_first_question()
    await demo_data_fetch_with_ambiguity()
    await demo_follow_up_clarification()
    await demo_max_attempts_scenario()
    await demo_aggregate_intent()
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("The Clarification Question Generator Node successfully:")
    print("• Generates contextual questions based on parameter priorities")
    print("• Avoids asking about parameters already in clarification history")
    print("• Incorporates normalization suggestions and ambiguity flags")
    print("• Provides intent-specific guidance and examples")
    print("• Exits with partial data when max attempts reached (per user flow)")
    print("• Uses natural, conversational language appropriate for financial context")
    print("• Updates clarification state with attempts and history tracking")
    print("• Returns proper exit signals for clarification subgraph orchestration")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main()) 