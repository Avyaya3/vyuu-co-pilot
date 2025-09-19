#!/usr/bin/env python3
"""
Test the simplified custom JWT only authentication flow.

This test verifies:
1. Only custom JWT verification is supported
2. No Supabase JWT verification attempts
3. Clean, efficient authentication flow
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import our modules
from src.utils.auth import SupabaseAuth, get_auth_manager


class TestSimplifiedCustomJWTFlow:
    """Test simplified custom JWT only authentication."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock environment variables for custom JWT only
        self.test_env = {
            "CUSTOM_JWT_SECRET": "test-custom-jwt-secret-key-12345",
            "CUSTOM_JWT_ISSUER": "https://your-nextjs-app.com",
            "CUSTOM_JWT_AUDIENCE": "vyuu-copilot-api",
            "SUPABASE_URL": "https://test-project.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-supabase-service-role-key-12345",
        }
        
        # Apply environment variables
        for key, value in self.test_env.items():
            os.environ[key] = value
    
    def teardown_method(self):
        """Clean up test environment."""
        # Remove test environment variables
        for key in self.test_env.keys():
            if key in os.environ:
                del os.environ[key]
    
    def test_custom_jwt_only_verification(self):
        """Test that only custom JWT verification is attempted."""
        print("\n🔍 Testing Custom JWT Only Verification")
        print("=" * 50)
        
        with patch('src.utils.auth.get_auth_manager') as mock_get_auth:
            auth_manager = SupabaseAuth()
            mock_get_auth.return_value = auth_manager
            
            # Mock custom JWT verification to succeed
            with patch.object(auth_manager, 'verify_custom_jwt') as mock_verify_custom:
                mock_verify_custom.return_value = {
                    "user_id": "nextjs-user-123",
                    "email": "user@nextjs-app.com",
                    "role": "authenticated"
                }
                
                # Mock Supabase JWT verification (should not be called)
                with patch.object(auth_manager, 'verify_supabase_jwt') as mock_verify_supabase:
                    mock_verify_supabase.side_effect = Exception("Should not be called")
                    
                    # Test JWT verification
                    try:
                        from src.utils.auth import verify_jwt_token
                        result = asyncio.run(verify_jwt_token("Bearer test-custom-jwt-token"))
                        
                        if result and result.get("token_type") == "custom_jwt":
                            print(f"✅ Custom JWT verification succeeded")
                            print(f"   User ID: {result.get('user_id')}")
                            print(f"   Token Type: {result.get('token_type')}")
                            print(f"   Supabase verification was NOT called (simplified!)")
                            
                            # Verify Supabase verification was never called
                            if not mock_verify_supabase.called:
                                print(f"✅ Supabase JWT verification was never attempted")
                                return True
                            else:
                                print(f"❌ Supabase JWT verification was called (unexpected)")
                                return False
                        else:
                            print(f"❌ Unexpected result: {result}")
                            return False
                            
                    except Exception as e:
                        print(f"❌ JWT verification failed: {e}")
                        return False
    
    def test_custom_jwt_verification_failure(self):
        """Test custom JWT verification failure handling."""
        print("\n❌ Testing Custom JWT Verification Failure")
        print("=" * 50)
        
        with patch('src.utils.auth.get_auth_manager') as mock_get_auth:
            auth_manager = SupabaseAuth()
            mock_get_auth.return_value = auth_manager
            
            # Mock custom JWT verification to fail
            with patch.object(auth_manager, 'verify_custom_jwt') as mock_verify_custom:
                from src.utils.auth import TokenValidationError
                mock_verify_custom.side_effect = TokenValidationError("Custom JWT verification failed")
                
                # Mock Supabase JWT verification (should not be called)
                with patch.object(auth_manager, 'verify_supabase_jwt') as mock_verify_supabase:
                    mock_verify_supabase.side_effect = Exception("Should not be called")
                    
                    # Test JWT verification
                    try:
                        from src.utils.auth import verify_jwt_token
                        result = asyncio.run(verify_jwt_token("Bearer invalid-token"))
                        
                        print(f"❌ JWT verification should have failed but didn't")
                        return False
                        
                    except Exception as e:
                        if "Invalid token" in str(e):
                            print(f"✅ Custom JWT verification failed as expected")
                            print(f"   Error: {str(e)}")
                            
                            # Verify Supabase verification was never called
                            if not mock_verify_supabase.called:
                                print(f"✅ Supabase JWT verification was never attempted")
                                return True
                            else:
                                print(f"❌ Supabase JWT verification was called (unexpected)")
                                return False
                        else:
                            print(f"❌ Unexpected error: {e}")
                            return False
    
    def test_no_authorization_header(self):
        """Test behavior when no authorization header is provided."""
        print("\n🚫 Testing No Authorization Header")
        print("=" * 50)
        
        with patch('src.utils.auth.get_auth_manager') as mock_get_auth:
            auth_manager = SupabaseAuth()
            mock_get_auth.return_value = auth_manager
            
            # Test JWT verification with no authorization header
            try:
                from src.utils.auth import verify_jwt_token
                result = asyncio.run(verify_jwt_token(None))
                
                if result is None:
                    print(f"✅ No authorization header handled correctly")
                    print(f"   Result: {result}")
                    return True
                else:
                    print(f"❌ Unexpected result: {result}")
                    return False
                    
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return False
    
    def test_missing_custom_jwt_config(self):
        """Test behavior when custom JWT configuration is missing."""
        print("\n⚠️  Testing Missing Custom JWT Configuration")
        print("=" * 50)
        
        # Remove custom JWT configuration
        if "CUSTOM_JWT_SECRET" in os.environ:
            del os.environ["CUSTOM_JWT_SECRET"]
        
        with patch('src.utils.auth.get_auth_manager') as mock_get_auth:
            auth_manager = SupabaseAuth()
            mock_get_auth.return_value = auth_manager
            
            # Mock the config to have no custom_jwt
            mock_config = MagicMock()
            mock_config.custom_jwt = None
            auth_manager.config = mock_config
            
            # Test JWT verification with missing config
            try:
                from src.utils.auth import verify_jwt_token
                result = asyncio.run(verify_jwt_token("Bearer test-token"))
                
                print(f"❌ JWT verification should have failed but didn't")
                return False
                
            except Exception as e:
                if "Custom JWT configuration not available" in str(e):
                    print(f"✅ Missing custom JWT configuration handled correctly")
                    print(f"   Error: {str(e)}")
                    return True
                else:
                    print(f"❌ Unexpected error: {e}")
                    return False


def run_simplified_custom_jwt_tests():
    """Run all simplified custom JWT tests."""
    print("🧪 Simplified Custom JWT Only Test Suite")
    print("=" * 60)
    
    test_suite = TestSimplifiedCustomJWTFlow()
    
    # Run tests
    tests = [
        ("Custom JWT Only Verification", test_suite.test_custom_jwt_only_verification),
        ("Custom JWT Verification Failure", test_suite.test_custom_jwt_verification_failure),
        ("No Authorization Header", test_suite.test_no_authorization_header),
        ("Missing Custom JWT Configuration", test_suite.test_missing_custom_jwt_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 50)
        
        try:
            test_suite.setup_method()
            result = test_func()
            test_suite.teardown_method()
            
            if result:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            test_suite.teardown_method()
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Custom JWT only authentication is working correctly.")
        print("\n✅ SIMPLIFICATION COMPLETE:")
        print("   ✅ Only custom JWT verification supported")
        print("   ✅ No Supabase JWT verification attempts")
        print("   ✅ Clean, efficient authentication flow")
        print("   ✅ Proper error handling")
        print("\n🚀 READY FOR NEXTJS INTEGRATION!")
        print("   Just set CUSTOM_JWT_SECRET and you're good to go!")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_simplified_custom_jwt_tests()
    sys.exit(0 if success else 1)
