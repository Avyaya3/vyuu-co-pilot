"""
Comprehensive Test for JWT Authentication Flow with Custom Tokens

This test verifies the complete authentication flow from NextJS to LangGraph execution,
including custom JWT token handling, user ID extraction, and session management.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
import jwt
from datetime import datetime, timedelta, timezone

# Import the modules we need to test
from src.api import app
from src.utils.auth import SupabaseAuth, verify_jwt_token, get_current_user, TokenValidationError
from src.orchestrator import MainOrchestrator, SessionManager
from src.schemas.state_schemas import MainState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomJWTTestSuite:
    """
    Test suite for custom JWT token authentication flow.
    """
    
    def __init__(self):
        self.test_user_id = "test-user-123"
        self.test_email = "test@example.com"
        self.test_secret = "test-secret-key"
        self.test_issuer = "vyuu-copilot"
        self.test_audience = "vyuu-users"
        
    def create_custom_jwt_token(
        self, 
        user_id: str = None, 
        email: str = None,
        expires_in_hours: int = 24,
        secret: str = None,
        issuer: str = None,
        audience: str = None
    ) -> str:
        """
        Create a custom JWT token for testing.
        
        Args:
            user_id: User ID to include in token
            email: Email to include in token
            expires_in_hours: Token expiration time in hours
            secret: Secret key for signing
            issuer: Token issuer
            audience: Token audience
            
        Returns:
            Signed JWT token string
        """
        user_id = user_id or self.test_user_id
        email = email or self.test_email
        secret = secret or self.test_secret
        issuer = issuer or self.test_issuer
        audience = audience or self.test_audience
        
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,  # Subject (user ID)
            "email": email,
            "iat": now,  # Issued at
            "exp": now + timedelta(hours=expires_in_hours),  # Expiration
            "iss": issuer,  # Issuer
            "aud": audience,  # Audience
            "user_id": user_id,  # Custom claim for user ID
            "role": "authenticated",
            "user_metadata": {
                "name": "Test User",
                "preferences": {"theme": "dark"}
            }
        }
        
        token = jwt.encode(payload, secret, algorithm="HS256")
        logger.info(f"Created custom JWT token for user {user_id}")
        return token
    
    def create_expired_jwt_token(self) -> str:
        """Create an expired JWT token for testing."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": self.test_user_id,
            "email": self.test_email,
            "iat": now - timedelta(hours=25),  # Issued 25 hours ago
            "exp": now - timedelta(hours=1),   # Expired 1 hour ago
            "iss": self.test_issuer,
            "aud": self.test_audience,
            "user_id": self.test_user_id,
            "role": "authenticated"
        }
        
        token = jwt.encode(payload, self.test_secret, algorithm="HS256")
        logger.info(f"Created expired JWT token for user {self.test_user_id}")
        return token
    
    def create_invalid_jwt_token(self) -> str:
        """Create an invalid JWT token for testing."""
        return "invalid.jwt.token"


class TestCustomJWTAuthentication:
    """
    Test custom JWT authentication flow.
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.test_suite = CustomJWTTestSuite()
        self.client = TestClient(app)
        
    def test_custom_jwt_token_creation(self):
        """Test that we can create custom JWT tokens."""
        token = self.test_suite.create_custom_jwt_token()
        
        # Verify token structure
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts
        
        # Decode and verify payload
        decoded = jwt.decode(token, self.test_suite.test_secret, algorithms=["HS256"])
        assert decoded["user_id"] == self.test_suite.test_user_id
        assert decoded["email"] == self.test_suite.test_email
        assert decoded["iss"] == self.test_suite.test_issuer
        
        logger.info("✅ Custom JWT token creation test passed")
    
    def test_expired_jwt_token_creation(self):
        """Test that we can create expired JWT tokens."""
        token = self.test_suite.create_expired_jwt_token()
        
        # Verify token is expired
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(token, self.test_suite.test_secret, algorithms=["HS256"])
        
        logger.info("✅ Expired JWT token creation test passed")
    
    @patch('src.utils.auth.get_auth_manager')
    def test_current_auth_does_not_handle_custom_tokens(self, mock_auth_manager):
        """Test that current auth system doesn't handle custom tokens."""
        # Mock the auth manager
        mock_auth = Mock()
        mock_auth_manager.return_value = mock_auth
        
        # Create custom token
        custom_token = self.test_suite.create_custom_jwt_token()
        
        # Mock Supabase auth to fail (as expected for custom tokens)
        mock_auth.extract_token_from_header.return_value = custom_token
        mock_auth.verify_supabase_jwt.side_effect = TokenValidationError("Custom token not supported")
        
        # Test that custom token fails with current system
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(verify_jwt_token(f"Bearer {custom_token}"))
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        logger.info("✅ Current auth system correctly rejects custom tokens")
    
    def test_chat_endpoint_without_auth(self):
        """Test chat endpoint without authentication."""
        response = self.client.post("/chat", json={
            "message": "Hello, I need help with my finances",
            "session_id": None
        })
        
        # Should work without auth (optional dependency)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        
        logger.info("✅ Chat endpoint works without authentication")
    
    def test_chat_endpoint_with_invalid_custom_token(self):
        """Test chat endpoint with invalid custom token."""
        custom_token = self.test_suite.create_custom_jwt_token()
        
        response = self.client.post("/chat", 
            json={"message": "Hello, I need help with my finances"},
            headers={"Authorization": f"Bearer {custom_token}"}
        )
        
        # Should fail with current system
        assert response.status_code == 401
        logger.info("✅ Chat endpoint correctly rejects custom tokens with current system")


class TestSessionPersistence:
    """
    Test session persistence requirements.
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = MainOrchestrator(use_database=False)
        
    def test_current_session_management_is_in_memory(self):
        """Test that current session management is in-memory only."""
        # Create a session
        session_id = "test-session-123"
        user_id = "test-user-456"
        
        # Process a message to create session
        result = asyncio.run(self.orchestrator.process_user_message(
            user_input="Hello, I need help",
            user_id=user_id,
            session_id=session_id
        ))
        
        assert result["session_id"] == session_id
        assert result["status"] == "success"
        
        # Verify session exists in memory
        session_info = self.orchestrator.get_session_info(session_id)
        assert session_info is not None
        assert session_info["user_id"] == user_id
        
        # Test that sessions don't persist across orchestrator restarts
        new_orchestrator = MainOrchestrator(use_database=False)
        session_info_after_restart = new_orchestrator.get_session_info(session_id)
        assert session_info_after_restart is None
        
        logger.info("✅ Current session management is in-memory only (doesn't persist)")
    
    def test_database_session_management_not_implemented(self):
        """Test that database session management is not implemented."""
        with pytest.raises(NotImplementedError):
            MainOrchestrator(use_database=True)
        
        logger.info("✅ Database session management is not yet implemented")


class TestUserContextIntegration:
    """
    Test user context integration in LangGraph execution.
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = MainOrchestrator(use_database=False)
        
    def test_user_id_passed_to_orchestrator(self):
        """Test that user ID is passed to orchestrator."""
        user_id = "test-user-789"
        session_id = "test-session-789"
        
        result = asyncio.run(self.orchestrator.process_user_message(
            user_input="I want to track my expenses",
            user_id=user_id,
            session_id=session_id
        ))
        
        # Verify user ID is stored in session
        session_info = self.orchestrator.get_session_info(session_id)
        assert session_info["user_id"] == user_id
        
        # Verify user ID is in metadata
        assert "user_id" in result["metadata"]
        
        logger.info("✅ User ID is passed to orchestrator and stored in session")
    
    def test_user_context_preserved_across_messages(self):
        """Test that user context is preserved across multiple messages."""
        user_id = "test-user-persistent"
        session_id = "test-session-persistent"
        
        # First message
        result1 = asyncio.run(self.orchestrator.process_user_message(
            user_input="My name is John",
            user_id=user_id,
            session_id=session_id
        ))
        
        # Second message in same session
        result2 = asyncio.run(self.orchestrator.process_user_message(
            user_input="What's my name?",
            user_id=user_id,
            session_id=session_id
        ))
        
        # Verify same session ID
        assert result1["session_id"] == result2["session_id"]
        
        # Verify user context is preserved
        session_info = self.orchestrator.get_session_info(session_id)
        assert session_info["user_id"] == user_id
        assert session_info["turn_count"] == 2  # Two messages
        
        logger.info("✅ User context is preserved across multiple messages")


def run_comprehensive_test():
    """
    Run the comprehensive test suite and generate a report.
    """
    print("🔍 Running Comprehensive JWT Authentication Flow Test")
    print("=" * 60)
    
    test_results = {
        "custom_jwt_tests": [],
        "session_persistence_tests": [],
        "user_context_tests": [],
        "missing_implementations": [],
        "recommendations": []
    }
    
    # Test Custom JWT Authentication
    print("\n📋 Testing Custom JWT Token Handling...")
    try:
        custom_jwt_test = TestCustomJWTAuthentication()
        custom_jwt_test.setup_method()
        
        # Test token creation
        custom_jwt_test.test_custom_jwt_token_creation()
        test_results["custom_jwt_tests"].append("✅ Custom JWT token creation works")
        
        # Test expired token creation
        custom_jwt_test.test_expired_jwt_token_creation()
        test_results["custom_jwt_tests"].append("✅ Expired JWT token creation works")
        
        # Test current auth system limitations
        custom_jwt_test.test_current_auth_does_not_handle_custom_tokens()
        test_results["custom_jwt_tests"].append("✅ Current auth system correctly rejects custom tokens")
        
        # Test API endpoint behavior
        custom_jwt_test.test_chat_endpoint_without_auth()
        test_results["custom_jwt_tests"].append("✅ Chat endpoint works without authentication")
        
        custom_jwt_test.test_chat_endpoint_with_invalid_custom_token()
        test_results["custom_jwt_tests"].append("✅ Chat endpoint correctly rejects custom tokens")
        
    except Exception as e:
        test_results["custom_jwt_tests"].append(f"❌ Custom JWT test failed: {e}")
    
    # Test Session Persistence
    print("\n💾 Testing Session Persistence...")
    try:
        session_test = TestSessionPersistence()
        session_test.setup_method()
        
        # Test in-memory session management
        session_test.test_current_session_management_is_in_memory()
        test_results["session_persistence_tests"].append("✅ Current session management is in-memory only")
        
        # Test database session management
        session_test.test_database_session_management_not_implemented()
        test_results["session_persistence_tests"].append("✅ Database session management is not implemented")
        
    except Exception as e:
        test_results["session_persistence_tests"].append(f"❌ Session persistence test failed: {e}")
    
    # Test User Context Integration
    print("\n👤 Testing User Context Integration...")
    try:
        user_context_test = TestUserContextIntegration()
        user_context_test.setup_method()
        
        # Test user ID passing
        user_context_test.test_user_id_passed_to_orchestrator()
        test_results["user_context_tests"].append("✅ User ID is passed to orchestrator")
        
        # Test user context preservation
        user_context_test.test_user_context_preserved_across_messages()
        test_results["user_context_tests"].append("✅ User context is preserved across messages")
        
    except Exception as e:
        test_results["user_context_tests"].append(f"❌ User context test failed: {e}")
    
    # Identify Missing Implementations
    print("\n🚨 Identifying Missing Implementations...")
    
    test_results["missing_implementations"] = [
        "❌ Custom JWT token verification (currently only supports Supabase JWT)",
        "❌ Database-backed session persistence (currently in-memory only)",
        "❌ User-specific session retrieval and management",
        "❌ Custom JWT token refresh logic",
        "❌ User context deep integration in LangGraph nodes"
    ]
    
    # Generate Recommendations
    print("\n💡 Generating Recommendations...")
    
    test_results["recommendations"] = [
        "🔧 Implement custom JWT verification in SupabaseAuth class",
        "🔧 Create database session manager to replace in-memory storage",
        "🔧 Add user-specific session operations (get_user_sessions, cleanup_user_sessions)",
        "🔧 Implement JWT token refresh endpoint for expired tokens",
        "🔧 Enhance LangGraph state schemas to include user context",
        "🔧 Add user-specific personalization in response generation",
        "🔧 Implement user session cleanup and expiration policies"
    ]
    
    # Print Results
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for category, results in test_results.items():
        if category == "missing_implementations" or category == "recommendations":
            print(f"\n{category.upper().replace('_', ' ')}:")
            for result in results:
                print(f"  {result}")
        else:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for result in results:
                print(f"  {result}")
    
    print("\n🎯 NEXT STEPS")
    print("=" * 60)
    print("1. Implement custom JWT verification for NextJS tokens")
    print("2. Create database session manager for persistence")
    print("3. Add user-specific session management")
    print("4. Enhance user context integration in LangGraph")
    print("5. Test the complete flow end-to-end")
    
    return test_results


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    # Save results to file
    with open("test_auth_flow_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: test_auth_flow_results.json")
