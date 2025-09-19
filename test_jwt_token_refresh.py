"""
Test JWT Token Refresh Implementation

This test verifies the JWT token refresh functionality.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import jwt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_jwt_token_refresh():
    """Test JWT token refresh functionality."""
    print("🔄 Testing JWT Token Refresh Implementation")
    print("=" * 60)
    
    try:
        # Import the modules
        from src.utils.auth import SupabaseAuth
        
        # Mock configuration
        mock_config = Mock()
        mock_config.custom_jwt.secret = "test-secret-key-for-jwt-refresh"
        mock_config.custom_jwt.issuer = "vyuu-copilot"
        mock_config.custom_jwt.audience = "vyuu-users"
        
        # Mock database client
        mock_db_client = Mock()
        
        # Create test auth manager
        with patch('src.utils.auth.get_config', return_value=mock_config), \
             patch('src.utils.auth.get_db_client', return_value=mock_db_client):
            
            auth_manager = SupabaseAuth()
            
            # Test user data
            test_user_id = "test-user-refresh-123"
            test_email = "refresh@example.com"
            
            # Create initial token
            print("\n🔑 Creating initial JWT token...")
            initial_token = auth_manager.create_custom_jwt_token(
                user_id=test_user_id,
                email=test_email,
                expires_in_hours=1  # Short expiration for testing
            )
            
            print(f"✅ Created initial token: {initial_token[:50]}...")
            
            # Verify initial token
            initial_payload = auth_manager.verify_custom_jwt(
                token=initial_token,
                secret=mock_config.custom_jwt.secret,
                issuer=mock_config.custom_jwt.issuer,
                audience=mock_config.custom_jwt.audience
            )
            
            assert initial_payload["user_id"] == test_user_id
            assert initial_payload["email"] == test_email
            print("✅ Initial token verification passed")
            
            # Test token refresh
            print("\n🔄 Testing token refresh...")
            refreshed_token = auth_manager.refresh_custom_jwt_token(
                old_token=initial_token,
                expires_in_hours=24
            )
            
            print(f"✅ Created refreshed token: {refreshed_token[:50]}...")
            
            # Verify refreshed token
            refreshed_payload = auth_manager.verify_custom_jwt(
                token=refreshed_token,
                secret=mock_config.custom_jwt.secret,
                issuer=mock_config.custom_jwt.issuer,
                audience=mock_config.custom_jwt.audience
            )
            
            assert refreshed_payload["user_id"] == test_user_id
            assert refreshed_payload["email"] == test_email
            assert refreshed_payload["token_type"] == "custom_jwt"
            print("✅ Refreshed token verification passed")
            
            # Verify tokens are different
            assert initial_token != refreshed_token
            print("✅ Tokens are different (refresh worked)")
            
            # Test refresh with expired token
            print("\n⏰ Testing refresh with expired token...")
            
            # Create an expired token
            now = datetime.now(timezone.utc)
            expired_payload = {
                "sub": test_user_id,
                "email": test_email,
                "iat": now - timedelta(hours=25),
                "exp": now - timedelta(hours=1),  # Expired
                "iss": mock_config.custom_jwt.issuer,
                "aud": mock_config.custom_jwt.audience,
                "user_id": test_user_id,
                "role": "authenticated",
                "user_metadata": {"name": "Test User"}
            }
            
            expired_token = jwt.encode(expired_payload, mock_config.custom_jwt.secret, algorithm="HS256")
            
            # Refresh the expired token
            refreshed_from_expired = auth_manager.refresh_custom_jwt_token(
                old_token=expired_token,
                expires_in_hours=24
            )
            
            # Verify the refreshed token works
            refreshed_from_expired_payload = auth_manager.verify_custom_jwt(
                token=refreshed_from_expired,
                secret=mock_config.custom_jwt.secret,
                issuer=mock_config.custom_jwt.issuer,
                audience=mock_config.custom_jwt.audience
            )
            
            assert refreshed_from_expired_payload["user_id"] == test_user_id
            assert refreshed_from_expired_payload["email"] == test_email
            print("✅ Expired token refresh passed")
            
            # Test refresh with invalid token
            print("\n❌ Testing refresh with invalid token...")
            try:
                auth_manager.refresh_custom_jwt_token(
                    old_token="invalid.token.here",
                    expires_in_hours=24
                )
                print("❌ Invalid token refresh should have failed!")
                return False
            except Exception as e:
                if "Invalid token" in str(e) or "Cannot refresh" in str(e):
                    print("✅ Invalid token refresh properly rejected")
                else:
                    print(f"❌ Unexpected error with invalid token: {e}")
                    return False
            
            print("\n🎉 All JWT token refresh tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ JWT token refresh test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fastapi_refresh_endpoint():
    """Test FastAPI token refresh endpoint."""
    print("\n🚀 Testing FastAPI Token Refresh Endpoint")
    print("=" * 60)
    
    try:
        # Import the modules
        from src.api import app
        from fastapi.testclient import TestClient
        
        # Mock configuration
        mock_config = Mock()
        mock_config.custom_jwt.secret = "test-secret-key-for-jwt-refresh"
        mock_config.custom_jwt.issuer = "vyuu-copilot"
        mock_config.custom_jwt.audience = "vyuu-users"
        
        # Mock database client
        mock_db_client = Mock()
        
        # Create test client
        client = TestClient(app)
        
        # Test with mocked dependencies
        with patch('src.utils.auth.get_auth_manager') as mock_get_auth_manager, \
             patch('src.config.settings.get_config', return_value=mock_config), \
             patch('src.utils.auth.get_db_client', return_value=mock_db_client):
            
            # Mock auth manager
            mock_auth_manager = Mock()
            mock_auth_manager.config = mock_config
            
            # Create a test token
            test_user_id = "test-user-api-456"
            test_email = "api@example.com"
            
            now = datetime.now(timezone.utc)
            test_payload = {
                "sub": test_user_id,
                "email": test_email,
                "iat": now,
                "exp": now + timedelta(hours=1),
                "iss": mock_config.custom_jwt.issuer,
                "aud": mock_config.custom_jwt.audience,
                "user_id": test_user_id,
                "role": "authenticated"
            }
            
            test_token = jwt.encode(test_payload, mock_config.custom_jwt.secret, algorithm="HS256")
            new_token = jwt.encode({
                **test_payload,
                "exp": now + timedelta(hours=24)
            }, mock_config.custom_jwt.secret, algorithm="HS256")
            
            mock_auth_manager.refresh_custom_jwt_token.return_value = new_token
            mock_get_auth_manager.return_value = mock_auth_manager
            
            # Test token refresh endpoint
            response = client.post("/auth/refresh", json={"token": test_token})
            
            assert response.status_code == 200
            data = response.json()
            
            assert "new_token" in data
            assert data["user_id"] == test_user_id
            assert data["expires_in_hours"] == 24
            assert data["status"] == "success"
            
            print("✅ FastAPI token refresh endpoint test passed")
            
            # Test with invalid token
            response = client.post("/auth/refresh", json={"token": "invalid.token"})
            assert response.status_code == 401
            
            print("✅ FastAPI invalid token handling test passed")
            
        print("🎉 All FastAPI token refresh endpoint tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI token refresh endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 JWT Token Refresh Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test JWT token refresh
    results.append(test_jwt_token_refresh())
    
    # Test FastAPI endpoint
    results.append(test_fastapi_refresh_endpoint())
    
    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! JWT token refresh is working correctly.")
        print("\n✅ READY FOR NEXT PHASE:")
        print("   - JWT token refresh is implemented")
        print("   - FastAPI endpoint is working")
        print("   - Expired token handling is working")
        print("   - Ready to move to user context integration")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
