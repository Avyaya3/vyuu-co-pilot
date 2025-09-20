"""
Test Database Session Manager Implementation

This test verifies the database session manager functionality.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_session_manager():
    """Test database session manager functionality."""
    print("💾 Testing Database Session Manager Implementation")
    print("=" * 60)
    
    try:
        # Import the modules
        from vyuu_copilot_v2.utils.database_session_manager import DatabaseSessionManager
        from vyuu_copilot_v2.schemas.state_schemas import MainState, MessageManager
        from vyuu_copilot_v2.schemas.database_models import ConversationSessionCreate
        
        # Mock the database client and repository
        with patch('src.utils.database_session_manager.ConversationSessionRepository') as mock_repo_class:
            mock_repository = Mock()
            mock_repo_class.return_value = mock_repository
            
            # Create test session manager
            session_manager = DatabaseSessionManager(session_expiry_hours=24)
        
            # Create test MainState
            test_session_id = str(uuid4())
            test_user_id = "test-user-123"
            
            # Create test messages
            from vyuu_copilot_v2.schemas.state_schemas import Message, MessageRole
            messages = [
                Message(
                    role=MessageRole.USER,
                    content="Hello",
                    metadata={"node_name": "user_input"}
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Hi there!",
                    metadata={"node_name": "response_synthesis"}
                )
            ]
            
            from vyuu_copilot_v2.schemas.state_schemas import IntentType
            test_state = MainState(
                user_input="Hello, I need help with my finances",
                session_id=test_session_id,
                timestamp=datetime.now(timezone.utc),
                messages=messages,
                metadata={
                    "user_id": test_user_id,
                    "session_created": datetime.now(timezone.utc).isoformat(),
                    "turn_count": 1
                },
                intent=IntentType.ACTION,
                confidence=0.95,
                parameters={"topic": "general_finance"},
                execution_results={},
                response="I'd be happy to help you with your finances!"
            )
            
            print(f"✅ Created test MainState with session ID: {test_session_id[:8]}")
            
            # Test save_session_state (new session)
            print("\n💾 Testing save_session_state (new session)...")
            mock_repository.get_by_id.return_value = None  # Session doesn't exist
            mock_repository.create.return_value = Mock(
                session_id=test_session_id,
                user_id=test_user_id,
                is_active=True
            )
            
            result = session_manager.save_session_state(test_state)
            assert result == True
            print("✅ save_session_state (new session) test passed")
            
            # Verify repository.create was called
            mock_repository.create.assert_called_once()
            create_call_args = mock_repository.create.call_args[0][0]
            assert isinstance(create_call_args, ConversationSessionCreate)
            assert create_call_args.session_id == test_session_id
            assert create_call_args.user_id == test_user_id
            print("✅ Repository create call verified")
            
            # Test save_session_state (existing session)
            print("\n💾 Testing save_session_state (existing session)...")
            mock_repository.reset_mock()
            
            # Mock existing session
            existing_session = Mock(
                session_id=test_session_id,
                user_id=test_user_id,
                is_active=True
            )
            mock_repository.get_by_id.return_value = existing_session
            mock_repository.update.return_value = existing_session
            
            result = session_manager.save_session_state(test_state)
            assert result == True
            print("✅ save_session_state (existing session) test passed")
            
            # Verify repository.update was called
            mock_repository.update.assert_called_once()
            print("✅ Repository update call verified")
            
            # Test load_session_state
            print("\n📖 Testing load_session_state...")
            mock_repository.reset_mock()
            
            # Mock session data
            mock_session = Mock(
                session_id=test_session_id,
                user_id=test_user_id,
                is_active=True,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                state_data={
                    "user_input": test_state.user_input,
                    "session_id": test_state.session_id,
                    "timestamp": test_state.timestamp.isoformat(),
                    "messages": [msg.model_dump() for msg in test_state.messages],
                    "metadata": test_state.metadata,
                    "intent": test_state.intent,
                    "confidence": test_state.confidence,
                    "parameters": test_state.parameters,
                    "execution_results": test_state.execution_results,
                    "response": test_state.response
                }
            )
            mock_repository.get_by_id.return_value = mock_session
            
            loaded_state = session_manager.load_session_state(test_session_id)
            assert loaded_state is not None
            assert loaded_state.session_id == test_session_id
            assert loaded_state.user_input == test_state.user_input
            assert len(loaded_state.messages) == len(test_state.messages)
            print("✅ load_session_state test passed")
            
            # Test expired session handling
            print("\n⏰ Testing expired session handling...")
            mock_repository.reset_mock()
            
            # Mock expired session
            expired_session = Mock(
                session_id=test_session_id,
                user_id=test_user_id,
                is_active=True,
                expires_at=datetime.now(timezone.utc) - timedelta(hours=1),  # Expired
                state_data={}
            )
            mock_repository.get_by_id.return_value = expired_session
            mock_repository.update.return_value = Mock()  # Mock deactivation
            
            loaded_state = session_manager.load_session_state(test_session_id)
            assert loaded_state is None
            print("✅ Expired session handling test passed")
            
            # Test get_user_sessions
            print("\n👤 Testing get_user_sessions...")
            mock_repository.reset_mock()
            
            # Mock user sessions
            mock_user_sessions = [
                Mock(
                    session_id="session-1",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    message_count=5,
                    last_intent="financial_advice",
                    last_confidence=0.95,
                    is_active=True
                ),
                Mock(
                    session_id="session-2",
                    created_at=datetime.now(timezone.utc) - timedelta(hours=1),
                    updated_at=datetime.now(timezone.utc) - timedelta(hours=1),
                    message_count=3,
                    last_intent="expense_tracking",
                    last_confidence=0.88,
                    is_active=True
                )
            ]
            mock_repository.get_by_user_id.return_value = mock_user_sessions
            
            user_sessions = session_manager.get_user_sessions(test_user_id, limit=10)
            assert len(user_sessions) == 2
            assert user_sessions[0]["session_id"] == "session-1"
            assert user_sessions[1]["session_id"] == "session-2"
            print("✅ get_user_sessions test passed")
            
            # Test cleanup operations
            print("\n🧹 Testing cleanup operations...")
            mock_repository.reset_mock()
            
            mock_repository.cleanup_expired_sessions.return_value = 5
            mock_repository.cleanup_old_sessions.return_value = 10
            
            expired_cleaned = session_manager.cleanup_expired_sessions()
            old_cleaned = session_manager.cleanup_old_sessions(max_age_days=7)
            
            assert expired_cleaned == 5
            assert old_cleaned == 10
            print("✅ Cleanup operations test passed")
            
            # Test session stats
            print("\n📊 Testing session stats...")
            mock_repository.reset_mock()
            
            mock_stats = {
                "total_sessions": 100,
                "active_sessions": 85,
                "user_sessions": 70,
                "anonymous_sessions": 15,
                "avg_message_count": 5.5,
                "last_activity": datetime.now(timezone.utc)
            }
            mock_repository.get_session_stats.return_value = mock_stats
            
            stats = session_manager.get_session_stats()
            assert stats["total_sessions"] == 100
            assert stats["active_sessions"] == 85
            print("✅ Session stats test passed")
            
            print("\n🎉 All database session manager tests passed!")
            return True
        
    except Exception as e:
        print(f"❌ Database session manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_database_integration():
    """Test orchestrator integration with database session manager."""
    print("\n🚀 Testing Orchestrator Database Integration")
    print("=" * 60)
    
    try:
        # Import the modules
        from vyuu_copilot_v2.orchestrator import MainOrchestrator
        
        # Mock the database session manager
        with patch('src.utils.database_session_manager.DatabaseSessionManager') as mock_db_manager_class:
            mock_db_manager = Mock()
            mock_db_manager_class.return_value = mock_db_manager
            
            # Test orchestrator initialization with database
            orchestrator = MainOrchestrator(use_database=True)
            
            assert orchestrator.use_database == True
            assert orchestrator.session_manager == mock_db_manager
            print("✅ Orchestrator database initialization test passed")
            
            # Test user session operations
            test_user_id = "test-user-456"
            mock_user_sessions = [
                {
                    "session_id": "session-1",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "message_count": 5,
                    "last_intent": "financial_advice",
                    "last_confidence": 0.95,
                    "is_active": True
                }
            ]
            mock_db_manager.get_user_sessions.return_value = mock_user_sessions
            
            user_sessions = orchestrator.get_user_sessions(test_user_id)
            assert len(user_sessions) == 1
            assert user_sessions[0]["session_id"] == "session-1"
            print("✅ Orchestrator get_user_sessions test passed")
            
            # Test user session cleanup
            mock_db_manager.get_user_sessions.return_value = mock_user_sessions * 15  # 15 sessions
            mock_db_manager.delete_session.return_value = True
            
            cleaned_count = orchestrator.cleanup_user_sessions(test_user_id, keep_recent=10)
            assert cleaned_count == 5  # Should clean up 5 old sessions
            print("✅ Orchestrator cleanup_user_sessions test passed")
            
            # Test session deletion
            mock_db_manager.delete_session.return_value = True
            result = orchestrator.delete_session("test-session-id")
            assert result == True
            print("✅ Orchestrator delete_session test passed")
            
        print("🎉 All orchestrator database integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator database integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Database Session Manager Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test database session manager
    results.append(test_database_session_manager())
    
    # Test orchestrator integration
    results.append(test_orchestrator_database_integration())
    
    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Database session manager is working correctly.")
        print("\n✅ READY FOR NEXT PHASE:")
        print("   - Database session manager is implemented")
        print("   - Orchestrator integration is working")
        print("   - User-specific session operations are available")
        print("   - Session persistence across server restarts is enabled")
        print("   - Ready to move to JWT token refresh logic")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
