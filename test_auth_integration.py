#!/usr/bin/env python3
"""
Test script for authentication integration with existing Supabase auth.

This script demonstrates how to test the API with different authentication scenarios.
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any, Optional


class AuthIntegrationTester:
    """Test client for authentication integration scenarios."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def test_without_auth(self) -> bool:
        """Test API without authentication (public access)."""
        print("🔓 Testing without authentication...")
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "message": "Hello, I need help with my finances"
                }
                
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Unauthenticated request successful")
                        print(f"   Response: {data['response'][:100]}...")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Unauthenticated request failed: {response.status}")
                        print(f"   Error: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Unauthenticated request error: {e}")
                return False
    
    async def test_with_invalid_token(self) -> bool:
        """Test API with invalid JWT token."""
        print("\n🔒 Testing with invalid JWT token...")
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "message": "Hello, I need help with my finances"
                }
                
                # Use an invalid token
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer invalid.jwt.token"
                }
                
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 401:
                        print(f"✅ Invalid token correctly rejected (401)")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Invalid token not rejected: {response.status}")
                        print(f"   Response: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Invalid token test error: {e}")
                return False
    
    async def test_with_malformed_header(self) -> bool:
        """Test API with malformed Authorization header."""
        print("\n🔒 Testing with malformed Authorization header...")
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "message": "Hello, I need help with my finances"
                }
                
                # Use malformed header
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "InvalidFormat token"
                }
                
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 401:
                        print(f"✅ Malformed header correctly rejected (401)")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Malformed header not rejected: {response.status}")
                        print(f"   Response: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Malformed header test error: {e}")
                return False
    
    async def test_with_valid_supabase_token(self, token: str) -> bool:
        """Test API with valid Supabase JWT token."""
        print("\n🔒 Testing with valid Supabase JWT token...")
        
        if not token:
            print("❌ No valid token provided for testing")
            return False
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "message": "Hello, I need help with my finances",
                    "user_id": "test-user-id"  # This will be overridden by the token
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
                
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Valid token request successful")
                        print(f"   Response: {data['response'][:100]}...")
                        print(f"   User context available: {bool(data.get('metadata', {}).get('user_id'))}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Valid token request failed: {response.status}")
                        print(f"   Error: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Valid token test error: {e}")
                return False
    
    async def test_user_context_extraction(self, token: str) -> bool:
        """Test that user context is properly extracted from token."""
        print("\n👤 Testing user context extraction...")
        
        if not token:
            print("❌ No valid token provided for testing")
            return False
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "message": "What's my user information?"
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}"
                }
                
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ User context extraction successful")
                        
                        # Check if user information is available in metadata
                        metadata = data.get('metadata', {})
                        if 'user_id' in metadata:
                            print(f"   User ID: {metadata['user_id']}")
                        if 'email' in metadata:
                            print(f"   Email: {metadata['email']}")
                        if 'role' in metadata:
                            print(f"   Role: {metadata['role']}")
                        
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ User context extraction failed: {response.status}")
                        print(f"   Error: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ User context test error: {e}")
                return False
    
    async def run_auth_tests(self, valid_token: Optional[str] = None) -> bool:
        """Run all authentication integration tests."""
        print("🔐 Starting Authentication Integration Tests")
        print("=" * 60)
        
        tests = [
            self.test_without_auth,
            self.test_with_invalid_token,
            self.test_with_malformed_header,
        ]
        
        # Add token-based tests if token is provided
        if valid_token:
            tests.extend([
                lambda: self.test_with_valid_supabase_token(valid_token),
                lambda: self.test_user_context_extraction(valid_token)
            ])
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if await test():
                    passed += 1
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 Auth Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All authentication tests passed!")
            return True
        else:
            print("⚠️ Some authentication tests failed.")
            return False


def get_supabase_token_from_env() -> Optional[str]:
    """Get a test Supabase token from environment variables."""
    import os
    
    # You can set this in your .env file for testing
    return os.getenv('TEST_SUPABASE_TOKEN')


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Vyuu Copilot v2 Authentication Integration")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--token",
        help="Valid Supabase JWT token for testing (or set TEST_SUPABASE_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or get_supabase_token_from_env()
    
    if not token:
        print("ℹ️  No valid Supabase token provided. Running basic auth tests only.")
        print("   To test with a real token, either:")
        print("   - Pass --token <your-supabase-jwt-token>")
        print("   - Set TEST_SUPABASE_TOKEN environment variable")
        print()
    
    tester = AuthIntegrationTester(args.url)
    success = await tester.run_auth_tests(token)
    
    if success:
        print("\n✅ Authentication integration is working correctly!")
        print("\nNext steps:")
        print("1. Use your existing Supabase JWT tokens in your chatbot app")
        print("2. Pass them in the Authorization header: 'Bearer <token>'")
        print("3. The API will automatically extract user context")
    else:
        print("\n❌ Authentication integration needs attention.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())


