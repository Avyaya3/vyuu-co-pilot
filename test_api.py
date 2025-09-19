#!/usr/bin/env python3
"""
Test script for the Vyuu Copilot v2 API.

This script tests the API endpoints to ensure they're working correctly.
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any


class APITester:
    """Test client for the Vyuu Copilot v2 API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
    
    async def test_health(self) -> bool:
        """Test the health endpoint."""
        print("🔍 Testing health endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Health check passed: {data['status']}")
                        print(f"   Orchestrator status: {data['orchestrator_status']}")
                        return True
                    else:
                        print(f"❌ Health check failed: {response.status}")
                        return False
            except Exception as e:
                print(f"❌ Health check error: {e}")
                return False
    
    async def test_simple_chat(self) -> bool:
        """Test the simple chat endpoint."""
        print("\n💬 Testing simple chat endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "message": "Hello, I need help with my finances"
                }
                
                async with session.post(
                    f"{self.base_url}/chat/simple",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Simple chat test passed")
                        print(f"   Response: {data['response'][:100]}...")
                        print(f"   Session ID: {data['session_id']}")
                        self.session_id = data['session_id']
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Simple chat test failed: {response.status}")
                        print(f"   Error: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Simple chat test error: {e}")
                return False
    
    async def test_full_chat(self) -> bool:
        """Test the full chat endpoint."""
        print("\n💬 Testing full chat endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                payload = {
                    "message": "What should I do to improve my financial situation?",
                    "session_id": self.session_id,
                    "conversation_history": [
                        {
                            "role": "user",
                            "content": "Hello, I need help with my finances",
                            "timestamp": "2024-01-01T10:00:00Z"
                        }
                    ]
                }
                
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Full chat test passed")
                        print(f"   Response: {data['response'][:100]}...")
                        print(f"   Status: {data['status']}")
                        print(f"   Processing time: {data.get('processing_time_ms', 'N/A')}ms")
                        print(f"   Conversation history length: {len(data['conversation_history'])}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Full chat test failed: {response.status}")
                        print(f"   Error: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Full chat test error: {e}")
                return False
    
    async def test_session_history(self) -> bool:
        """Test the session history endpoint."""
        print("\n📚 Testing session history endpoint...")
        
        if not self.session_id:
            print("❌ No session ID available for history test")
            return False
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/sessions/{self.session_id}/history"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Session history test passed")
                        print(f"   History length: {len(data)} messages")
                        for i, msg in enumerate(data):
                            print(f"   Message {i+1}: {msg['role']} - {msg['content'][:50]}...")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Session history test failed: {response.status}")
                        print(f"   Error: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Session history test error: {e}")
                return False
    
    async def test_clear_session(self) -> bool:
        """Test the clear session endpoint."""
        print("\n🗑️ Testing clear session endpoint...")
        
        if not self.session_id:
            print("❌ No session ID available for clear test")
            return False
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.delete(
                    f"{self.base_url}/sessions/{self.session_id}"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Clear session test passed")
                        print(f"   Message: {data['message']}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Clear session test failed: {response.status}")
                        print(f"   Error: {error_text}")
                        return False
            except Exception as e:
                print(f"❌ Clear session test error: {e}")
                return False
    
    async def run_all_tests(self) -> bool:
        """Run all API tests."""
        print("🚀 Starting Vyuu Copilot v2 API Tests")
        print("=" * 50)
        
        tests = [
            self.test_health,
            self.test_simple_chat,
            self.test_full_chat,
            self.test_session_history,
            self.test_clear_session
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if await test():
                    passed += 1
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("⚠️ Some tests failed. Check the API server and configuration.")
            return False


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Vyuu Copilot v2 API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
