#!/usr/bin/env python3
"""
Manual testing script for multi-intent scenarios.

This script tests various multi-intent scenarios to validate the implementation.
"""

import asyncio
import json
import aiohttp
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_multi_intent_scenario(scenario_name: str, message: str, expected_intents: list):
    """Test a specific multi-intent scenario."""
    
    print(f"\n🧪 Testing: {scenario_name}")
    print("=" * 50)
    print(f"📝 Message: {message}")
    print(f"🎯 Expected Intents: {', '.join(expected_intents)}")
    print()
    
    api_request = {
        "message": message,
        "financialData": {
            "user": {
                "id": "test_user_123",
                "name": "Test User",
                "monthly_income": 50000,
                "monthly_expenses": 40000
            },
            "dashboardMetrics": {
                "savingsRate": 0.2,
                "monthlyIncome": 50000,
                "monthlyExpenses": 40000
            }
        },
        "userId": "test_user_123"
    }
    
    # Wait for server
    await asyncio.sleep(2)
    
    api_url = "http://localhost:8000/chat"
    
    try:
        async with aiohttp.ClientSession() as session:
            print("🌐 Sending multi-intent request...")
            start_time = time.time()
            
            async with session.post(
                api_url,
                json=api_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                processing_time = time.time() - start_time
                
                print(f"📊 Response Status: {response.status}")
                print(f"⏱️  Processing Time: {processing_time:.2f}s")
                print()
                
                if response.status == 200:
                    result = await response.json()
                    
                    print("✅ Response received:")
                    print("-" * 40)
                    print(result.get('response', 'No response'))
                    print("-" * 40)
                    
                    # Check metadata for multi-intent information
                    metadata = result.get('metadata', {})
                    if metadata.get('is_multi_intent'):
                        print(f"\n🎉 Multi-intent detected!")
                        print(f"   Intent Count: {metadata.get('intent_count', 'Unknown')}")
                        print(f"   Intent Types: {metadata.get('intent_types', 'Unknown')}")
                        
                        # Check if expected intents were detected
                        detected_intents = metadata.get('intent_types', [])
                        if all(intent in detected_intents for intent in expected_intents):
                            print(f"   ✅ All expected intents detected: {expected_intents}")
                        else:
                            print(f"   ⚠️  Expected: {expected_intents}, Detected: {detected_intents}")
                    else:
                        print(f"\n⚠️  Single intent detected (expected multi-intent)")
                    
                    # Check execution results
                    if 'execution_results' in result:
                        exec_results = result['execution_results']
                        if 'multiple_intent_results' in exec_results:
                            print(f"\n📋 Execution Results:")
                            print(f"   Successful: {exec_results.get('successful_count', 0)}")
                            print(f"   Failed: {exec_results.get('failed_count', 0)}")
                            print(f"   Strategy: {exec_results.get('execution_strategy', 'Unknown')}")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Error {response.status}: {error_text}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_all_multi_intent_scenarios():
    """Test all multi-intent scenarios."""
    
    print("🚀 Multi-Intent Functionality Test Suite")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Read + Advice",
            "message": "Show me my expenses and give me budgeting advice",
            "expected_intents": ["read", "advice"]
        },
        {
            "name": "Database Operations + Read",
            "message": "Transfer ₹1000 to savings and show me the updated balance",
            "expected_intents": ["database_operations", "read"]
        },
        {
            "name": "All Three Intent Types",
            "message": "Show my budget, create a savings goal, and give me investment advice",
            "expected_intents": ["read", "database_operations", "advice"]
        },
        {
            "name": "Multiple Read Operations",
            "message": "Show me my transactions and also display my account balances",
            "expected_intents": ["read", "read"]
        },
        {
            "name": "Advice + Database Operations",
            "message": "Give me financial advice and help me set up automatic savings",
            "expected_intents": ["advice", "database_operations"]
        }
    ]
    
    for scenario in scenarios:
        await test_multi_intent_scenario(
            scenario["name"],
            scenario["message"],
            scenario["expected_intents"]
        )
        await asyncio.sleep(1)  # Brief pause between tests
    
    print("\n🎯 Multi-Intent Test Suite Complete!")
    print("=" * 60)


async def test_single_intent_compatibility():
    """Test that single intent flows still work (backward compatibility)."""
    
    print("\n🔄 Testing Backward Compatibility (Single Intents)")
    print("=" * 50)
    
    single_intent_scenarios = [
        {
            "name": "Single Read Intent",
            "message": "Show me my expenses"
        },
        {
            "name": "Single Database Operations Intent",
            "message": "Transfer ₹500 to my savings account"
        },
        {
            "name": "Single Advice Intent",
            "message": "Give me budgeting advice"
        }
    ]
    
    for scenario in single_intent_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        print(f"📝 Message: {scenario['message']}")
        
        api_request = {
            "message": scenario["message"],
            "financialData": {
                "user": {
                    "id": "test_user_123",
                    "name": "Test User",
                    "monthly_income": 50000,
                    "monthly_expenses": 40000
                }
            },
            "userId": "test_user_123"
        }
        
        api_url = "http://localhost:8000/chat"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    json=api_request,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        metadata = result.get('metadata', {})
                        
                        if not metadata.get('is_multi_intent', False):
                            print("   ✅ Single intent correctly detected")
                        else:
                            print("   ⚠️  Multi-intent detected (unexpected)")
                        
                        print(f"   Response: {result.get('response', 'No response')[:100]}...")
                    else:
                        print(f"   ❌ Error {response.status}")
                        
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        await asyncio.sleep(1)


async def main():
    """Main test function."""
    
    print("🔧 Multi-Intent Implementation Test Suite")
    print("=" * 60)
    print("This script tests the multi-intent functionality implementation.")
    print("Make sure the API server is running on localhost:8000")
    print()
    
    # Test multi-intent scenarios
    await test_all_multi_intent_scenarios()
    
    # Test backward compatibility
    await test_single_intent_compatibility()
    
    print("\n🎉 All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
