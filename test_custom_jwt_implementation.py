"""
Test Custom JWT Implementation

This test verifies the custom JWT token verification functionality.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import jwt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_custom_jwt_verification():
    """Test custom JWT token verification functionality."""
    print("🔐 Testing Custom JWT Verification Implementation")
    print("=" * 60)
    
    # Mock configuration
    mock_config = Mock()
    mock_config.custom_jwt.secret = "test-secret-key-for-jwt-verification"
    mock_config.custom_jwt.issuer = "vyuu-copilot"
    mock_config.custom_jwt.audience = "vyuu-users"
    
    # Mock database client
    mock_db_client = Mock()
    
    # Create test token
    test_user_id = "test-user-123"
    test_email = "test@example.com"
    
    now = datetime.now(timezone.utc)
    payload = {
        "sub": test_user_id,
        "email": test_email,
        "iat": now,
        "exp": now + timedelta(hours=24),
        "iss": "vyuu-copilot",
        "aud": "vyuu-users",
        "user_id": test_user_id,
        "role": "authenticated",
        "user_metadata": {
            "name": "Test User",
            "preferences": {"theme": "dark"}
        }
    }
    
    token = jwt.encode(payload, mock_config.custom_jwt.secret, algorithm="HS256")
    print(f"✅ Created test JWT token: {token[:50]}...")
    
    # Test the custom JWT verification
    try:
        # Import the auth module
        from vyuu_copilot_v2.utils.auth import SupabaseAuth
        
        # Create auth manager with mocked dependencies
        with patch('src.utils.auth.get_config', return_value=mock_config), \
             patch('src.utils.auth.get_db_client', return_value=mock_db_client):
            
            auth_manager = SupabaseAuth()
            
            # Test custom JWT verification
            result = auth_manager.verify_custom_jwt(
                token=token,
                secret=mock_config.custom_jwt.secret,
                issuer=mock_config.custom_jwt.issuer,
                audience=mock_config.custom_jwt.audience
            )
            
            print("✅ Custom JWT verification successful!")
            print(f"  User ID: {result['user_id']}")
            print(f"  Email: {result['email']}")
            print(f"  Role: {result['role']}")
            print(f"  Token Type: {result['token_type']}")
            
            # Verify expected fields
            assert result['user_id'] == test_user_id
            assert result['email'] == test_email
            assert result['role'] == "authenticated"
            assert result['token_type'] == "custom_jwt"
            assert result['iss'] == "vyuu-copilot"
            assert result['aud'] == "vyuu-users"
            
            print("✅ All assertions passed!")
            
    except Exception as e:
        print(f"❌ Custom JWT verification failed: {e}")
        return False
    
    # Test expired token
    print("\n🕐 Testing expired token handling...")
    expired_payload = {
        "sub": test_user_id,
        "email": test_email,
        "iat": now - timedelta(hours=25),
        "exp": now - timedelta(hours=1),  # Expired 1 hour ago
        "iss": "vyuu-copilot",
        "aud": "vyuu-users",
        "user_id": test_user_id,
        "role": "authenticated"
    }
    
    expired_token = jwt.encode(expired_payload, mock_config.custom_jwt.secret, algorithm="HS256")
    
    try:
        with patch('src.utils.auth.get_config', return_value=mock_config), \
             patch('src.utils.auth.get_db_client', return_value=mock_db_client):
            
            auth_manager = SupabaseAuth()
            auth_manager.verify_custom_jwt(
                token=expired_token,
                secret=mock_config.custom_jwt.secret,
                issuer=mock_config.custom_jwt.issuer,
                audience=mock_config.custom_jwt.audience
            )
            print("❌ Expired token was not properly rejected!")
            return False
            
    except Exception as e:
        if "expired" in str(e).lower():
            print("✅ Expired token properly rejected!")
        else:
            print(f"❌ Unexpected error with expired token: {e}")
            return False
    
    # Test invalid issuer
    print("\n🔒 Testing invalid issuer handling...")
    invalid_issuer_payload = {
        "sub": test_user_id,
        "email": test_email,
        "iat": now,
        "exp": now + timedelta(hours=24),
        "iss": "invalid-issuer",  # Wrong issuer
        "aud": "vyuu-users",
        "user_id": test_user_id,
        "role": "authenticated"
    }
    
    invalid_issuer_token = jwt.encode(invalid_issuer_payload, mock_config.custom_jwt.secret, algorithm="HS256")
    
    try:
        with patch('src.utils.auth.get_config', return_value=mock_config), \
             patch('src.utils.auth.get_db_client', return_value=mock_db_client):
            
            auth_manager = SupabaseAuth()
            auth_manager.verify_custom_jwt(
                token=invalid_issuer_token,
                secret=mock_config.custom_jwt.secret,
                issuer=mock_config.custom_jwt.issuer,
                audience=mock_config.custom_jwt.audience
            )
            print("❌ Invalid issuer was not properly rejected!")
            return False
            
    except Exception as e:
        if "issuer" in str(e).lower():
            print("✅ Invalid issuer properly rejected!")
        else:
            print(f"❌ Unexpected error with invalid issuer: {e}")
            return False
    
    print("\n🎉 All custom JWT tests passed!")
    return True


def test_fastapi_integration():
    """Test FastAPI integration with custom JWT tokens."""
    print("\n🚀 Testing FastAPI Integration")
    print("=" * 60)
    
    # Mock configuration
    mock_config = Mock()
    mock_config.custom_jwt.secret = "test-secret-key-for-jwt-verification"
    mock_config.custom_jwt.issuer = "vyuu-copilot"
    mock_config.custom_jwt.audience = "vyuu-users"
    
    # Mock database client
    mock_db_client = Mock()
    
    # Create test token
    test_user_id = "test-user-456"
    test_email = "test456@example.com"
    
    now = datetime.now(timezone.utc)
    payload = {
        "sub": test_user_id,
        "email": test_email,
        "iat": now,
        "exp": now + timedelta(hours=24),
        "iss": "vyuu-copilot",
        "aud": "vyuu-users",
        "user_id": test_user_id,
        "role": "authenticated"
    }
    
    token = jwt.encode(payload, mock_config.custom_jwt.secret, algorithm="HS256")
    
    try:
        # Import the auth module
        from vyuu_copilot_v2.utils.auth import verify_jwt_token
        
        # Test FastAPI dependency
        with patch('src.utils.auth.get_auth_manager') as mock_get_auth_manager, \
             patch('src.utils.auth.get_config', return_value=mock_config), \
             patch('src.utils.auth.get_db_client', return_value=mock_db_client):
            
            # Mock auth manager
            mock_auth_manager = Mock()
            mock_auth_manager.config = mock_config
            mock_auth_manager.extract_token_from_header.return_value = token
            from vyuu_copilot_v2.utils.auth import TokenValidationError
            mock_auth_manager.verify_supabase_jwt.side_effect = TokenValidationError("Not a Supabase token")
            mock_auth_manager.verify_custom_jwt.return_value = {
                "user_id": test_user_id,
                "email": test_email,
                "role": "authenticated",
                "token_type": "custom_jwt"
            }
            mock_get_auth_manager.return_value = mock_auth_manager
            
            # Test the FastAPI dependency
            result = asyncio.run(verify_jwt_token(f"Bearer {token}"))
            
            print("✅ FastAPI integration successful!")
            print(f"  User ID: {result['user_id']}")
            print(f"  Email: {result['email']}")
            print(f"  Token Type: {result['token_type']}")
            
            # Verify the auth manager was called correctly
            mock_auth_manager.extract_token_from_header.assert_called_once_with(f"Bearer {token}")
            mock_auth_manager.verify_custom_jwt.assert_called_once()
            
            print("✅ FastAPI dependency calls verified!")
            
    except Exception as e:
        print(f"❌ FastAPI integration test failed: {e}")
        return False
    
    print("🎉 FastAPI integration test passed!")
    return True


def main():
    """Run all tests."""
    print("🧪 Custom JWT Implementation Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test custom JWT verification
    results.append(test_custom_jwt_verification())
    
    # Test FastAPI integration
    results.append(test_fastapi_integration())
    
    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Custom JWT implementation is working correctly.")
        print("\n✅ READY FOR NEXT PHASE:")
        print("   - Custom JWT token verification is implemented")
        print("   - FastAPI integration is working")
        print("   - Error handling is in place")
        print("   - Ready to move to database session management")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
