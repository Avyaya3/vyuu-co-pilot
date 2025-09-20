#!/usr/bin/env python3
"""
Demo script for hierarchical state schemas with LangGraph intent orchestration.

This script demonstrates:
1. State inheritance hierarchy
2. Automatic state transitions
3. Message tracking with node metadata
4. Conversation management and pruning
5. Parameter collection and merging
6. Real-world usage patterns

Run with: python scripts/demo_state_schemas.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vyuu_copilot_v2.schemas.state_schemas import (
    BaseState,
    MainState,
    ClarificationState,
    OrchestratorState,
    Message,
    MessageRole,
    IntentType,
    StateTransitions,
    MessageManager,
    ConversationContext,
    StateValidator,
)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_state_info(state, title: str):
    """Print formatted state information."""
    print(f"\n{title}:")
    print(f"  Session ID: {state.session_id[:8]}...")
    print(f"  User Input: {state.user_input}")
    print(f"  Intent: {state.intent}")
    print(f"  Confidence: {state.confidence}")
    print(f"  Messages: {len(state.messages)}")
    
    if hasattr(state, 'parameters'):
        print(f"  Parameters: {state.parameters}")
    
    if hasattr(state, 'missing_params'):
        print(f"  Missing Params: {state.missing_params}")
        print(f"  Clarification Attempts: {state.clarification_attempts}/{state.max_attempts}")
        print(f"  Extracted Params: {state.extracted_parameters}")
    
    if hasattr(state, 'execution_plan'):
        print(f"  Execution Plan: {state.execution_plan}")
        print(f"  Tool Results: {state.tool_results}")
        print(f"  Final Response: {state.final_response}")


def demo_basic_state_creation():
    """Demonstrate basic state creation and validation."""
    print_section("1. Basic State Creation and Validation")
    
    # Create initial BaseState
    base_state = BaseState(
        user_input="Show me my account balances for the last 3 months",
        intent=IntentType.DATA_FETCH,
        confidence=0.85
    )
    
    print_state_info(base_state, "Initial BaseState")
    
    # Validate the state
    try:
        StateValidator.validate_base_state(base_state)
        print("✅ State validation passed")
    except ValueError as e:
        print(f"❌ State validation failed: {e}")
    
    return base_state


def demo_message_management(state):
    """Demonstrate message management with node tracking."""
    print_section("2. Message Management with Node Tracking")
    
    # Add user message
    state = MessageManager.add_user_message(
        state,
        "Show me my account balances for the last 3 months"
    )
    
    # Add assistant message with node tracking
    state = MessageManager.add_assistant_message(
        state,
        "I'll help you fetch your account balances. Let me classify your intent first.",
        "intent_classification_node"
    )
    
    # Add system message for debugging
    state = MessageManager.add_system_message(
        state,
        f"Intent classified as {state.intent} with confidence {state.confidence}",
        "intent_classification_node"
    )
    
    print(f"Messages after adding: {len(state.messages)}")
    for i, msg in enumerate(state.messages):
        print(f"  {i+1}. [{msg.role}] {msg.content[:50]}..." + 
              (f" (from: {msg.node_name})" if msg.node_name else ""))
    
    return state


def demo_state_inheritance():
    """Demonstrate state inheritance hierarchy."""
    print_section("3. State Inheritance Hierarchy")
    
    # Create MainState from BaseState
    main_state = MainState(
        user_input="Show me top 5 transactions by amount",
        intent=IntentType.DATA_FETCH,
        confidence=0.92,
        parameters={
            "query_type": "transactions",
            "limit": 5,
            "sort_by": "amount",
            "order": "desc"
        }
    )
    
    print_state_info(main_state, "MainState with Parameters")
    
    return main_state


def demo_clarification_flow(main_state):
    """Demonstrate clarification state flow."""
    print_section("4. Clarification State Flow")
    
    # Convert to clarification state
    clarification_state = StateTransitions.to_clarification_state(main_state)
    clarification_state.missing_params = ["time_period", "account_types"]
    clarification_state.clarification_attempts = 1
    
    print_state_info(clarification_state, "ClarificationState (Missing Parameters)")
    
    # Simulate clarification collection
    clarification_state.extracted_parameters = {
        "time_period": "last_6_months",
        "account_types": ["checking", "savings"]
    }
    clarification_state.clarification_attempts = 2
    
    print_state_info(clarification_state, "ClarificationState (After Collection)")
    
    # Convert back to MainState with merged parameters
    merged_main_state = StateTransitions.from_clarification_state(clarification_state)
    
    print_state_info(merged_main_state, "MainState (After Parameter Merging)")
    print(f"🔄 Original parameters: {main_state.parameters}")
    print(f"🔄 Extracted parameters: {clarification_state.extracted_parameters}")
    print(f"🔄 Merged parameters: {merged_main_state.parameters}")
    
    return merged_main_state


def demo_orchestrator_flow(main_state):
    """Demonstrate orchestrator state flow."""
    print_section("5. Orchestrator State Flow")
    
    # Convert to orchestrator state
    orchestrator_state = StateTransitions.to_orchestrator_state(main_state)
    
    # Set execution plan
    orchestrator_state.execution_plan = {
        "tools": ["supabase_query_tool", "data_formatter_tool", "response_generator_tool"],
        "sequence": [
            {"tool": "supabase_query_tool", "action": "fetch_transactions"},
            {"tool": "data_formatter_tool", "action": "format_results"},
            {"tool": "response_generator_tool", "action": "generate_response"}
        ],
        "parameters": orchestrator_state.extracted_params
    }
    
    print_state_info(orchestrator_state, "OrchestratorState (With Execution Plan)")
    
    # Simulate tool execution
    orchestrator_state.tool_results = {
        "supabase_query_tool": {
            "status": "success",
            "data": [
                {"id": "1", "amount": 2500.00, "description": "Salary Deposit", "date": "2024-01-15"},
                {"id": "2", "amount": -1200.00, "description": "Rent Payment", "date": "2024-01-01"},
                {"id": "3", "amount": -85.50, "description": "Grocery Store", "date": "2024-01-03"},
                {"id": "4", "amount": -45.00, "description": "Gas Station", "date": "2024-01-05"},
                {"id": "5", "amount": -120.00, "description": "Utilities", "date": "2024-01-10"}
            ],
            "count": 5,
            "execution_time": 150
        },
        "data_formatter_tool": {
            "status": "success",
            "formatted_data": "Formatted transaction list with currency formatting",
            "execution_time": 25
        },
        "response_generator_tool": {
            "status": "success",
            "response": "Here are your top 5 transactions by amount from the last 6 months...",
            "execution_time": 75
        }
    }
    
    orchestrator_state.final_response = "Here are your top 5 transactions by amount from the last 6 months:\n\n1. Salary Deposit: +$2,500.00 (Jan 15)\n2. Rent Payment: -$1,200.00 (Jan 01)\n3. Utilities: -$120.00 (Jan 10)\n4. Grocery Store: -$85.50 (Jan 03)\n5. Gas Station: -$45.00 (Jan 05)\n\nTotal execution time: 250ms"
    
    print_state_info(orchestrator_state, "OrchestratorState (After Execution)")
    
    # Convert back to MainState
    final_main_state = StateTransitions.from_orchestrator_state(orchestrator_state)
    
    print_state_info(final_main_state, "Final MainState (With Results)")
    
    return final_main_state


def demo_conversation_management():
    """Demonstrate conversation management and context."""
    print_section("6. Conversation Management and Context")
    
    # Create state with multiple messages
    state = BaseState(user_input="Complex financial query")
    
    # Simulate a conversation
    conversations = [
        ("user", "What's my current financial status?"),
        ("assistant", "I'll help you check your financial status. Let me gather your account information.", "financial_analyzer_node"),
        ("system", "Fetching account data from database", "data_fetch_node"),
        ("assistant", "I found 3 accounts. Would you like a summary or detailed breakdown?", "financial_analyzer_node"),
        ("user", "Give me a detailed breakdown please"),
        ("assistant", "Here's your detailed financial breakdown...", "report_generator_node"),
        ("user", "Can you also show me my spending trends?"),
        ("assistant", "I'll analyze your spending trends from the last 6 months.", "trend_analyzer_node"),
        ("system", "Analyzing transaction patterns", "trend_analyzer_node"),
        ("assistant", "Your spending trends show a 15% increase in dining expenses.", "trend_analyzer_node"),
    ]
    
    for role, content, *node_info in conversations:
        if role == "user":
            state = MessageManager.add_user_message(state, content)
        elif role == "assistant":
            node_name = node_info[0] if node_info else "unknown_node"
            state = MessageManager.add_assistant_message(state, content, node_name)
        elif role == "system":
            node_name = node_info[0] if node_info else "system_node"
            state = MessageManager.add_system_message(state, content, node_name)
    
    print(f"Total messages in conversation: {len(state.messages)}")
    
    # Get conversation summary
    summary = ConversationContext.summarize_conversation(state)
    print(f"\nConversation Summary:\n{summary}")
    
    # Get recent context
    recent_context = ConversationContext.get_recent_context(state, 3)
    print(f"\nRecent Context (last 3 messages):")
    for i, msg in enumerate(recent_context):
        print(f"  {i+1}. [{msg.role}] {msg.content[:40]}...")
    
    # Filter by role
    user_messages = ConversationContext.get_messages_by_role(state, MessageRole.USER)
    print(f"\nUser messages: {len(user_messages)}")
    
    # Filter by node
    analyzer_messages = ConversationContext.get_messages_by_node(state, "financial_analyzer_node")
    print(f"Messages from financial_analyzer_node: {len(analyzer_messages)}")
    
    return state


def demo_conversation_pruning():
    """Demonstrate automatic conversation pruning."""
    print_section("7. Automatic Conversation Pruning")
    
    state = BaseState(user_input="Test pruning")
    
    # Add more messages than the limit
    print(f"Adding 25 messages (limit is {ConversationContext.get_recent_context.__defaults__[0]} in recent context)...")
    
    for i in range(25):
        state = MessageManager.add_user_message(state, f"User message {i}")
    
    print(f"Messages after adding 25: {len(state.messages)} (auto-pruned)")
    print(f"First message: {state.messages[0].content}")
    print(f"Last message: {state.messages[-1].content}")
    
    return state


def demo_validation_scenarios():
    """Demonstrate comprehensive validation scenarios."""
    print_section("8. Validation Scenarios")
    
    print("Testing validation scenarios:")
    
    # Test 1: Invalid confidence
    try:
        BaseState(user_input="Test", confidence=1.5)
        print("❌ Should have failed confidence validation")
    except Exception as e:
        print(f"✅ Confidence validation caught: {type(e).__name__}")
    
    # Test 2: Invalid clarification attempts
    try:
        ClarificationState(
            user_input="Test",
            clarification_attempts=5,
            max_attempts=3
        )
        print("❌ Should have failed clarification attempts validation")
    except Exception as e:
        print(f"✅ Clarification attempts validation caught: {type(e).__name__}")
    
    # Test 3: Invalid user input
    try:
        BaseState(user_input="   ")
        print("❌ Should have failed user input validation")
    except Exception as e:
        print(f"✅ User input validation caught: {type(e).__name__}")
    
    # Test 4: Valid state transition
    state1 = BaseState(user_input="Test")
    state2 = MainState(user_input="Test", session_id=state1.session_id)
    
    try:
        StateValidator.validate_state_transition(state1, state2)
        print("✅ State transition validation passed")
    except Exception as e:
        print(f"❌ State transition validation failed: {e}")


def main():
    """Run the complete demo."""
    print("🚀 LangGraph State Schema Demonstration")
    print("=" * 60)
    
    # 1. Basic state creation
    base_state = demo_basic_state_creation()
    
    # 2. Message management
    state_with_messages = demo_message_management(base_state)
    
    # 3. State inheritance
    main_state = demo_state_inheritance()
    
    # 4. Clarification flow
    clarified_state = demo_clarification_flow(main_state)
    
    # 5. Orchestrator flow
    final_state = demo_orchestrator_flow(clarified_state)
    
    # 6. Conversation management
    conversation_state = demo_conversation_management()
    
    # 7. Conversation pruning
    pruned_state = demo_conversation_pruning()
    
    # 8. Validation scenarios
    demo_validation_scenarios()
    
    print_section("Demo Complete!")
    print("✅ All state schema features demonstrated successfully!")
    print("\nKey features showcased:")
    print("  • Hierarchical inheritance (BaseState → MainState → specialized states)")
    print("  • Automatic state transitions with parameter merging")
    print("  • Message tracking with node metadata")
    print("  • Conversation management and automatic pruning")
    print("  • Comprehensive validation")
    print("  • Real-world usage patterns")
    print(f"\nFinal state session: {final_state.session_id[:8]}...")
    print(f"Final response: {final_state.response[:60]}...")


if __name__ == "__main__":
    main() 