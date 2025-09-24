#!/usr/bin/env python3
"""
Test script to simulate frontend consumption of streaming endpoint.
This simulates how your Vercel frontend will consume the /chat/stream endpoint.
"""

import asyncio
import json
import aiohttp
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def test_streaming_endpoint():
    """Test the streaming endpoint as a frontend would."""
    print("🧪 Testing Streaming Endpoint (Frontend Simulation)")
    print("=" * 60)
    
    # Test data (same as what your frontend would send)
    test_request = {
        "message": "Create a liability: 'Home Mortgage' for ₹300000 with ₹2000 monthly EMI at 3.5% interest rate, starting January 1, 2022, ending December 31, 2052",
        "session_id": None,
        "conversation_history": [],
        "user_id": "cmfobnc6v0000xit8qoy6gbrj",
        "financial_data": None
    }
    
    print(f"Request: {json.dumps(test_request, indent=2)}")
    print("\nStreaming Response:")
    print("-" * 60)
    
    try:
        # Simulate frontend making request to streaming endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8000/chat/stream',
                json=test_request,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    print(f"❌ HTTP Error: {response.status}")
                    print(await response.text())
                    return
                
                print(f"✅ Connected to stream (Status: {response.status})")
                print("-" * 60)
                
                # Read the stream line by line (SSE format)
                event_count = 0
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        try:
                            # Parse the JSON data
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            event_data = json.loads(data_str)
                            event_count += 1
                            
                            print(f"Event {event_count}: {json.dumps(event_data, indent=2)}")
                            print("-" * 40)
                            
                            # Stop after reasonable number of events
                            if event_count > 25:
                                print("Stopping test after 25 events...")
                                break
                                
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse event: {line_str}")
                            print(f"Error: {e}")
                
                print(f"\n✅ Streaming test completed with {event_count} events")
                
    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to server. Make sure the FastAPI server is running on localhost:8000")
        print("Run: python -m uvicorn src.vyuu_copilot_v2.api:app --host 0.0.0.0 --port 8000 --reload")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("🚀 Starting Frontend Streaming Simulation")
    print("=" * 60)
    print("Note: Make sure your FastAPI server is running!")
    print("Command: python -m uvicorn src.vyuu_copilot_v2.api:app --host 0.0.0.0 --port 8000 --reload")
    print("=" * 60)
    
    try:
        await test_streaming_endpoint()
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
