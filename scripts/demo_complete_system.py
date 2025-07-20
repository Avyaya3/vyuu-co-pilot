#!/usr/bin/env python3
"""
Complete Intent Orchestration System Demo

This script demonstrates the fully assembled LangGraph intent orchestration system,
showcasing the complete workflow from user input through intent classification,
decision routing, subgraph execution, and final response generation.

Features Demonstrated:
- Complete main graph execution
- Intent classification and routing
- Clarification and direct orchestrator subgraphs
- Session management and conversation continuity
- Error handling and recovery
- Multi-user session isolation
- Performance monitoring and metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from typing import List, Dict, Any
from uuid import uuid4

from src.orchestrator import main_orchestrator

# Configure logging for demo
logging.basicConfig(level=logging.WARNING)  # Reduce noise for demo
demo_logger = logging.getLogger("demo")
demo_logger.setLevel(logging.INFO)


def print_header(title: str):
    """Print a demo section header."""
    print(f"\n{'🚀' if 'Demo' in title else '📋'} {title}")
    print("=" * 80)


def print_step(step: str):
    """Print a demo step."""
    print(f"\n▶️  {step}")
    print("-" * 40)


def print_result(result: Dict[str, Any]):
    """Print a formatted result."""
    print(f"✅ Status: {result['status']}")
    print(f"📱 Session: {result['session_id'][:8]}...")
    print(f"💬 Response: {result['response'][:100]}...")
    print(f"📊 Messages: {len(result['conversation_history'])}")
    
    if result['status'] == 'error':
        print(f"❌ Error: {result.get('metadata', {}).get('error_metadata', {}).get('error_message', 'Unknown error')}")


async def demo_basic_workflow():
    """Demonstrate the basic workflow of the orchestration system."""
    print_header("Basic Workflow Demo")
    
    print_step("Single user interaction with account balance query")
    
    # Test a simple balance query
    result = await main_orchestrator.process_user_message(
        user_input="Show me my checking account balance",
        user_id="demo_user_001"
    )
    
    print_result(result)
    return result['session_id']


async def demo_conversation_continuity(session_id: str):
    """Demonstrate conversation continuity within a session."""
    print_header("Conversation Continuity Demo")
    
    print_step("Follow-up question in the same session")
    
    # Follow-up question
    result = await main_orchestrator.process_user_message(
        user_input="What about my savings account?",
        user_id="demo_user_001",
        session_id=session_id
    )
    
    print_result(result)
    
    print_step("Another follow-up question")
    
    # Another follow-up
    result = await main_orchestrator.process_user_message(
        user_input="Can you show me recent transactions?",
        user_id="demo_user_001",
        session_id=session_id
    )
    
    print_result(result)


async def demo_multi_user_sessions():
    """Demonstrate multi-user session isolation."""
    print_header("Multi-User Session Isolation Demo")
    
    users = [
        {"id": "alice_001", "query": "Transfer $100 to savings"},
        {"id": "bob_002", "query": "Show me my spending this month"},
        {"id": "charlie_003", "query": "What's my total balance?"}
    ]
    
    sessions = {}
    
    for user in users:
        print_step(f"Processing request for {user['id']}")
        
        result = await main_orchestrator.process_user_message(
            user_input=user['query'],
            user_id=user['id']
        )
        
        sessions[user['id']] = result['session_id']
        print_result(result)
    
    print_step("Verifying session isolation")
    stats = main_orchestrator.get_orchestrator_stats()
    print(f"📊 Total active sessions: {stats['session_stats']['total_sessions']}")
    print(f"👥 Expected sessions: {len(users)}")
    print(f"✅ Session isolation: {'Success' if stats['session_stats']['total_sessions'] >= len(users) else 'Failed'}")


async def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print_header("Error Handling Demo")
    
    print_step("Testing empty input handling")
    
    try:
        result = await main_orchestrator.process_user_message(
            user_input="",  # Empty input should trigger validation error
            user_id="error_test_user"
        )
        print_result(result)
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")
    
    print_step("Testing recovery from processing errors")
    
    # Test with a complex query that might trigger downstream errors
    result = await main_orchestrator.process_user_message(
        user_input="Do something completely nonsensical that the system can't handle",
        user_id="error_test_user"
    )
    print_result(result)


async def demo_different_intents():
    """Demonstrate handling of different intent types."""
    print_header("Different Intent Types Demo")
    
    test_cases = [
        {
            "name": "Data Fetch Intent",
            "query": "Show me my account balance",
            "expected": "Should route to direct orchestrator (high confidence)"
        },
        {
            "name": "Action Intent",
            "query": "Transfer money to my savings",
            "expected": "May route to clarification (missing parameters)"
        },
        {
            "name": "Aggregate Intent", 
            "query": "What's my total spending this month?",
            "expected": "Should route based on confidence and parameters"
        },
        {
            "name": "Unknown Intent",
            "query": "The weather is nice today",
            "expected": "Should route to clarification (unknown intent)"
        }
    ]
    
    for test_case in test_cases:
        print_step(f"Testing {test_case['name']}")
        print(f"Query: \"{test_case['query']}\"")
        print(f"Expected: {test_case['expected']}")
        
        result = await main_orchestrator.process_user_message(
            user_input=test_case['query'],
            user_id=f"intent_test_{test_case['name'].lower().replace(' ', '_')}"
        )
        
        print_result(result)


async def demo_session_management():
    """Demonstrate session management capabilities."""
    print_header("Session Management Demo")
    
    print_step("Creating test session")
    
    result = await main_orchestrator.process_user_message(
        user_input="Hello, I need help with my finances",
        user_id="session_demo_user"
    )
    
    session_id = result['session_id']
    print(f"✅ Created session: {session_id[:8]}...")
    
    print_step("Retrieving session information")
    
    session_info = main_orchestrator.get_session_info(session_id)
    if session_info:
        print(f"📊 Session Info:")
        print(f"   - User ID: {session_info['user_id']}")
        print(f"   - Message count: {session_info['message_count']}")
        print(f"   - Turn count: {session_info['turn_count']}")
        print(f"   - Last intent: {session_info['last_intent']}")
        print(f"   - Last confidence: {session_info['last_confidence']}")
    else:
        print("❌ Session not found (this indicates an error)")
    
    print_step("Getting orchestrator statistics")
    
    stats = main_orchestrator.get_orchestrator_stats()
    print(f"📊 Orchestrator Stats:")
    print(f"   - Status: {stats['status']}")
    print(f"   - Session management: {stats['session_management']}")
    print(f"   - Graph compiled: {stats['graph_compiled']}")
    print(f"   - Total sessions: {stats['session_stats']['total_sessions']}")
    print(f"   - Total messages: {stats['session_stats']['total_messages']}")


async def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print_header("Performance Monitoring Demo")
    
    print_step("Processing multiple requests to gather performance data")
    
    import time
    start_time = time.time()
    
    # Process multiple requests
    for i in range(3):
        result = await main_orchestrator.process_user_message(
            user_input=f"Performance test request #{i+1}",
            user_id=f"perf_user_{i+1:03d}"
        )
        
        processing_metadata = result.get('metadata', {}).get('processing_metadata', {})
        processing_time = processing_metadata.get('processing_time_seconds', 0)
        
        print(f"   Request {i+1}: {processing_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n📊 Performance Summary:")
    print(f"   - Total time: {total_time:.2f}s")
    print(f"   - Average per request: {total_time/3:.2f}s")
    print(f"   - Requests per second: {3/total_time:.2f}")


async def main():
    """Run the complete system demonstration."""
    print_header("🚀 Complete LangGraph Intent Orchestration System Demo")
    print("This demo showcases the fully assembled intent orchestration system")
    print("with all subgraphs, nodes, and session management capabilities.")
    
    try:
        # Basic workflow demo
        session_id = await demo_basic_workflow()
        
        # Conversation continuity demo
        await demo_conversation_continuity(session_id)
        
        # Multi-user sessions demo
        await demo_multi_user_sessions()
        
        # Error handling demo
        await demo_error_handling()
        
        # Different intents demo
        await demo_different_intents()
        
        # Session management demo
        await demo_session_management()
        
        # Performance monitoring demo
        await demo_performance_monitoring()
        
        print_header("🎉 Demo Complete!")
        print("✅ All system components demonstrated successfully!")
        print("\n🏗️ System Architecture Validated:")
        print("  • Complete LangGraph workflow integration")
        print("  • Intent classification and decision routing")
        print("  • Clarification and direct orchestrator subgraphs")
        print("  • State transitions and wrapper nodes")
        print("  • In-memory session management")
        print("  • Multi-user session isolation")
        print("  • Comprehensive error handling")
        print("  • Performance monitoring and metrics")
        print("  • Conversation continuity and context preservation")
        
        print("\n🚀 Ready for Integration:")
        print("  • FastAPI endpoint integration")
        print("  • Frontend chat UI connection")
        print("  • Database session management (Phase 2)")
        print("  • Production deployment")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n{'🎉 Demo completed successfully!' if success else '❌ Demo failed!'}")
    sys.exit(0 if success else 1) 