#!/usr/bin/env python3
"""
Test script to verify improved streaming implementation with heartbeats.
This tests the fixes for Vercel timeout issues.
"""

import asyncio
import json
import aiohttp
import sys
import os
import time
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def test_improved_streaming():
    """Test the improved streaming endpoint with heartbeats."""
    print("🧪 Testing Improved Streaming Endpoint (Vercel Timeout Fix)")
    print("=" * 70)
    
    # Test data
    test_request = {
        "message": "Create a liability: 'Car Loan' for ₹500000 with ₹15000 monthly EMI at 4.2% interest rate, starting March 1, 2023, ending March 1, 2028",
        "session_id": None,
        "conversation_history": [],
        "user_id": "cmfobnc6v0000xit8qoy6gbrj",
        "financial_data": None
    }
    
    print(f"Request: {json.dumps(test_request, indent=2)}")
    print("\nStreaming Response with Heartbeats:")
    print("-" * 70)
    
    try:
        start_time = time.time()
        first_chunk_time = None
        
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
                print(f"⏱️  Connection established in {time.time() - start_time:.3f}s")
                print("-" * 70)
                
                # Read the stream line by line (SSE format)
                event_count = 0
                heartbeat_count = 0
                processing_events = 0
                response_chunks = 0
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        try:
                            # Parse the JSON data
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            event_data = json.loads(data_str)
                            event_count += 1
                            
                            # Track timing of first meaningful chunk
                            if first_chunk_time is None and event_data.get('type') in ['processing', 'heartbeat']:
                                first_chunk_time = time.time() - start_time
                                print(f"🚀 First meaningful chunk received in {first_chunk_time:.3f}s")
                            
                            # Categorize events
                            event_type = event_data.get('type', 'unknown')
                            if event_type == 'heartbeat':
                                heartbeat_count += 1
                                print(f"💓 Heartbeat {heartbeat_count}: {event_data.get('message', 'N/A')}")
                            elif event_type == 'processing':
                                processing_events += 1
                                print(f"⚙️  Processing {processing_events}: {event_data.get('message', 'N/A')} (Stage: {event_data.get('stage', 'N/A')})")
                            elif event_type == 'response_chunk':
                                response_chunks += 1
                                content = event_data.get('content', '')[:50]
                                print(f"📝 Response chunk {response_chunks}: {content}...")
                            elif event_type == 'connection':
                                print(f"🔗 Connection: {event_data.get('status', 'N/A')}")
                            elif event_type == 'response_complete':
                                print(f"✅ Response complete: {event_data.get('session_id', 'N/A')}")
                            elif event_type == 'stream_complete':
                                print(f"🏁 Stream complete")
                            elif event_type == 'error':
                                print(f"❌ Error: {event_data.get('message', 'N/A')}")
                            else:
                                print(f"📦 Event {event_count}: {json.dumps(event_data, indent=2)}")
                            
                            print("-" * 40)
                            
                            # Stop after reasonable number of events
                            if event_count > 30:
                                print("Stopping test after 30 events...")
                                break
                                
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse event: {line_str}")
                            print(f"Error: {e}")
                
                total_time = time.time() - start_time
                print(f"\n📊 Streaming Test Summary:")
                print(f"   Total events: {event_count}")
                print(f"   Heartbeats: {heartbeat_count}")
                print(f"   Processing events: {processing_events}")
                print(f"   Response chunks: {response_chunks}")
                print(f"   First chunk time: {first_chunk_time:.3f}s" if first_chunk_time else "   First chunk time: N/A")
                print(f"   Total time: {total_time:.3f}s")
                
                # Vercel timeout check
                if first_chunk_time and first_chunk_time > 25:
                    print(f"⚠️  WARNING: First chunk took {first_chunk_time:.3f}s (Vercel timeout is 25s)")
                elif first_chunk_time:
                    print(f"✅ First chunk time is within Vercel limits ({first_chunk_time:.3f}s < 25s)")
                
                print(f"\n✅ Improved streaming test completed")
                
    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to server. Make sure the FastAPI server is running on localhost:8000")
        print("Run: python -m uvicorn src.vyuu_copilot_v2.api:app --host 0.0.0.0 --port 8000 --reload")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("🚀 Testing Improved Streaming Implementation")
    print("=" * 70)
    print("This test verifies the fixes for Vercel timeout issues:")
    print("1. Immediate connection event")
    print("2. Heartbeat mechanism")
    print("3. Faster first meaningful chunk")
    print("4. Better error handling")
    print("=" * 70)
    
    try:
        await test_improved_streaming()
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
