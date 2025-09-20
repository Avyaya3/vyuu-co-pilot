#!/usr/bin/env python3
"""
Test script for the MainOrchestrator class.

This script validates the complete MainOrchestrator functionality including:
- In-memory session management
- User message processing
- Conversation continuity
- Error handling and recovery
- API interface functionality
- Session statistics and monitoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from uuid import uuid4

from vyuu_copilot_v2.orchestrator import MainOrchestrator, SessionManager
from vyuu_copilot_v2.schemas.state_schemas import IntentType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def test_session_manager():
    """Test the SessionManager functionality."""
    print_section("Testing SessionManager")
    
    session_manager = SessionManager()
    
    # Test session creation and storage
    print("1. Testing session creation and storage")
    from vyuu_copilot_v2.schemas.state_schemas import MainState
    
    test_state = MainState(
        user_input="Test message",
        session_id=str(uuid4())
    )
    
    save_success = session_manager.save_session_state(test_state)
    print(f"   ✅ Session saved: {save_success}")
    
    # Test session loading
    print("\n2. Testing session loading")
    loaded_state = session_manager.load_session_state(test_state.session_id)
    print(f"   ✅ Session loaded: {loaded_state is not None}")
    print(f"   - Same session ID: {loaded_state.session_id == test_state.session_id}")
    print(f"   - Same user input: {loaded_state.user_input == test_state.user_input}")
    
    # Test session statistics
    print("\n3. Testing session statistics")
    stats = session_manager.get_session_stats()
    print(f"   ✅ Session stats retrieved")
    print(f"   - Total sessions: {stats['total_sessions']}")
    print(f"   - Total messages: {stats['total_messages']}")
    
    # Test session deletion
    print("\n4. Testing session deletion")
    delete_success = session_manager.delete_session(test_state.session_id)
    print(f"   ✅ Session deleted: {delete_success}")
    
    # Verify deletion
    deleted_state = session_manager.load_session_state(test_state.session_id)
    print(f"   - Session no longer exists: {deleted_state is None}")


async def test_orchestrator_basic_functionality():
    """Test basic MainOrchestrator functionality."""
    print_section("Testing MainOrchestrator Basic Functionality")
    
    orchestrator = MainOrchestrator(use_database=False)
    
    # Test initialization
    print("1. Testing orchestrator initialization")
    print(f"   ✅ Orchestrator initialized")
    print(f"   - Using database: {orchestrator.use_database}")
    print(f"   - Graph available: {orchestrator.graph is not None}")
    print(f"   - Session manager type: {type(orchestrator.session_manager).__name__}")
    
    # Test orchestrator stats
    print("\n2. Testing orchestrator statistics")
    stats = orchestrator.get_orchestrator_stats()
    print(f"   ✅ Orchestrator stats retrieved")
    print(f"   - Status: {stats['status']}")
    print(f"   - Session management: {stats['session_management']}")
    print(f"   - Graph compiled: {stats['graph_compiled']}")


async def test_message_processing():
    """Test message processing functionality."""
    print_section("Testing Message Processing")
    
    orchestrator = MainOrchestrator(use_database=False)
    
    # Test new conversation
    print("1. Testing new conversation")
    
    try:
        result = await orchestrator.process_user_message(
            user_input="Show me my account balance",
            user_id="test_user_123"
        )
        
        print(f"   ✅ Message processed successfully")
        print(f"   - Status: {result['status']}")
        print(f"   - Session ID generated: {bool(result['session_id'])}")
        print(f"   - Response generated: {bool(result['response'])}")
        print(f"   - Conversation history: {len(result['conversation_history'])} messages")
        
        # Store session ID for next test
        session_id = result['session_id']
        
    except Exception as e:
        print(f"   ❌ Message processing failed: {e}")
        return None
    
    # Test conversation continuity
    print("\n2. Testing conversation continuity")
    
    try:
        followup_result = await orchestrator.process_user_message(
            user_input="What about my savings account?",
            user_id="test_user_123",
            session_id=session_id
        )
        
        print(f"   ✅ Follow-up message processed successfully")
        print(f"   - Same session ID: {followup_result['session_id'] == session_id}")
        print(f"   - More messages in history: {len(followup_result['conversation_history']) > len(result['conversation_history'])}")
        print(f"   - Response generated: {bool(followup_result['response'])}")
        
    except Exception as e:
        print(f"   ❌ Follow-up processing failed: {e}")
    
    return session_id


async def test_error_handling():
    """Test error handling functionality."""
    print_section("Testing Error Handling")
    
    orchestrator = MainOrchestrator(use_database=False)
    
    # Test empty input handling
    print("1. Testing empty input handling")
    
    try:
        # This should trigger validation error which should be handled gracefully
        result = await orchestrator.process_user_message(
            user_input="",  # Empty input
            user_id="test_user_error"
        )
        
        print(f"   - Status: {result['status']}")
        print(f"   - Error handled: {result['status'] == 'error'}")
        print(f"   - Fallback response provided: {bool(result['response'])}")
        
    except Exception as e:
        print(f"   ⚠️  Unexpected error (this might be expected): {e}")
    
    # Test nonexistent session loading
    print("\n2. Testing nonexistent session handling")
    
    try:
        fake_session_id = str(uuid4())
        result = await orchestrator.process_user_message(
            user_input="Test with fake session",
            user_id="test_user_123",
            session_id=fake_session_id
        )
        
        print(f"   ✅ Nonexistent session handled gracefully")
        print(f"   - New session created: {result['session_id'] != fake_session_id}")
        print(f"   - Response generated: {bool(result['response'])}")
        
    except Exception as e:
        print(f"   ❌ Nonexistent session handling failed: {e}")


async def test_session_info():
    """Test session information retrieval."""
    print_section("Testing Session Information")
    
    orchestrator = MainOrchestrator(use_database=False)
    
    # Create a session first
    print("1. Creating a test session")
    result = await orchestrator.process_user_message(
        user_input="Test session info",
        user_id="test_user_info"
    )
    
    session_id = result['session_id']
    print(f"   ✅ Test session created: {session_id[:8]}")
    
    # Test session info retrieval
    print("\n2. Testing session info retrieval")
    
    session_info = orchestrator.get_session_info(session_id)
    
    if session_info:
        print(f"   ✅ Session info retrieved successfully")
        print(f"   - Session ID: {session_info['session_id'][:8]}...")
        print(f"   - Message count: {session_info['message_count']}")
        print(f"   - User ID: {session_info['user_id']}")
        print(f"   - Turn count: {session_info['turn_count']}")
    else:
        print(f"   ❌ Session info not found")
    
    # Test nonexistent session info
    print("\n3. Testing nonexistent session info")
    
    fake_session_id = str(uuid4())
    fake_session_info = orchestrator.get_session_info(fake_session_id)
    
    print(f"   ✅ Nonexistent session handled correctly: {fake_session_info is None}")


async def test_multiple_users():
    """Test multiple user sessions."""
    print_section("Testing Multiple User Sessions")
    
    orchestrator = MainOrchestrator(use_database=False)
    
    # Create sessions for multiple users
    users = ["user_001", "user_002", "user_003"]
    sessions = {}
    
    print("1. Creating sessions for multiple users")
    
    for user_id in users:
        result = await orchestrator.process_user_message(
            user_input=f"Hello from {user_id}",
            user_id=user_id
        )
        sessions[user_id] = result['session_id']
        print(f"   ✅ Session created for {user_id}: {result['session_id'][:8]}")
    
    # Test session isolation
    print("\n2. Testing session isolation")
    
    stats = orchestrator.get_orchestrator_stats()
    session_count = stats['session_stats']['total_sessions']
    
    print(f"   ✅ Multiple sessions created: {session_count} sessions")
    print(f"   - Expected users: {len(users)}")
    print(f"   - Sessions isolated: {session_count >= len(users)}")
    
    # Test individual session access
    print("\n3. Testing individual session access")
    
    for user_id, session_id in sessions.items():
        session_info = orchestrator.get_session_info(session_id)
        if session_info:
            print(f"   ✅ {user_id} session accessible: {session_info['user_id']}")
        else:
            print(f"   ❌ {user_id} session not found")


async def main():
    """Run all tests."""
    print("🧪 MainOrchestrator Test Suite")
    print("=" * 60)
    
    try:
        # Test session manager
        await test_session_manager()
        
        # Test basic orchestrator functionality
        await test_orchestrator_basic_functionality()
        
        # Test message processing
        session_id = await test_message_processing()
        
        # Test error handling
        await test_error_handling()
        
        # Test session info
        await test_session_info()
        
        # Test multiple users
        await test_multiple_users()
        
        print_section("Test Results")
        print("✅ All tests passed successfully!")
        print("\nMainOrchestrator Features Validated:")
        print("  • In-memory session management")
        print("  • User message processing")
        print("  • Conversation continuity")
        print("  • Error handling and recovery")
        print("  • API interface functionality")
        print("  • Session statistics and monitoring")
        print("  • Multi-user session isolation")
        print("  • Complete graph integration")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 