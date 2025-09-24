#!/bin/bash
# Test script for streaming endpoint using curl

echo "🧪 Testing Streaming Endpoint with curl"
echo "========================================"

# Test data
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "Create a liability: Home Mortgage for $300000 with $2000 monthly EMI at 3.5% interest rate, starting January 1, 2022, ending December 31, 2052",
    "session_id": null,
    "conversation_history": [],
    "user_id": "cmfobnc6v0000xit8qoy6gbrj",
    "financial_data": null
  }' \
  --no-buffer

echo ""
echo "✅ curl test completed"
