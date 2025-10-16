#!/usr/bin/env python3
"""
Test script for three-level risk advice functionality.
This script tests the enhanced advice tool with various queries to validate
the three-level output and mathematical calculation quality.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Mock financial data for testing
MOCK_FINANCIAL_DATA = {
    "user": {
        "id": "test_user_123",
        "name": "Test User",
        "email": "test@example.com",
        "monthly_income": 100000,
        "monthly_expenses": 60000,
        "risk_profile": "moderate"
    },
    "assets": [
        {
            "id": "asset_1",
            "name": "Savings Account",
            "type": "savings",
            "balance": 200000,
            "currency": "INR"
        },
        {
            "id": "asset_2", 
            "name": "Investment Portfolio",
            "type": "investment",
            "balance": 300000,
            "currency": "INR"
        }
    ],
    "liabilities": [
        {
            "id": "liability_1",
            "name": "Home Loan",
            "type": "home_loan",
            "balance": 2000000,
            "emi": 25000,
            "currency": "INR"
        }
    ],
    "goals": [
        {
            "id": "goal_1",
            "name": "Emergency Fund",
            "target_amount": 500000,
            "current_amount": 200000,
            "currency": "INR"
        },
        {
            "id": "goal_2",
            "name": "Retirement Fund",
            "target_amount": 10000000,
            "current_amount": 300000,
            "currency": "INR"
        }
    ],
    "income": [
        {
            "id": "income_1",
            "source": "Salary",
            "amount": 100000,
            "frequency": "monthly",
            "currency": "INR"
        }
    ],
    "expenses": [
        {
            "id": "expense_1",
            "category": "Food & Dining",
            "amount": 20000,
            "frequency": "monthly",
            "currency": "INR"
        },
        {
            "id": "expense_2",
            "category": "Transportation",
            "amount": 10000,
            "frequency": "monthly", 
            "currency": "INR"
        },
        {
            "id": "expense_3",
            "category": "EMI",
            "amount": 25000,
            "frequency": "monthly",
            "currency": "INR"
        }
    ],
    "stocks": [
        {
            "id": "stock_1",
            "symbol": "RELIANCE",
            "quantity": 20,
            "current_price": 2500,
            "currency": "INR"
        }
    ],
    "stockTrades": [],
    "closedPositions": [],
    "insurance": [
        {
            "id": "insurance_1",
            "type": "life_insurance",
            "premium": 8000,
            "coverage": 2000000,
            "currency": "INR"
        }
    ],
    "savings": [
        {
            "id": "savings_1",
            "name": "Emergency Fund",
            "balance": 200000,
            "currency": "INR"
        }
    ],
    "dashboardMetrics": {
        "net_worth": 500000,
        "monthly_income": 100000,
        "monthly_expenses": 60000,
        "savings_rate": 0.4,
        "total_assets": 500000,
        "total_liabilities": 2000000,
        "currency": "INR"
    }
}

# Test queries for three-level advice
TEST_QUERIES = [
    "Give me investment advice for my portfolio",
    "How should I invest ₹50,000 for maximum returns?",
    "Help me plan for retirement with my current savings",
    "What's the best way to grow my emergency fund?",
    "I want to start investing but don't know where to begin",
    "How can I optimize my current investment strategy?",
    "What should I do with my bonus of ₹1,00,000?",
    "Help me create a diversified investment portfolio"
]

async def test_advice_tool():
    """Test the advice tool with various queries."""
    try:
        from src.vyuu_copilot_v2.tools.advice_tool import advice_tool
        
        print("🧪 Testing Three-Level Risk Advice Tool")
        print("=" * 60)
        print()
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"📋 Test {i}: {query}")
            print("-" * 50)
            
            # Prepare parameters
            params = {
                "user_query": query,
                "user_id": "test_user_123",
                "financial_data": MOCK_FINANCIAL_DATA
            }
            
            try:
                # Call the advice tool
                result = await advice_tool.invoke(params)
                
                if result.get("success", False):
                    advice_data = result.get("data", {})
                    advice_text = advice_data.get("advice", "")
                    
                    print("✅ Success!")
                    print(f"Risk Levels: {advice_data.get('risk_levels', 'Not specified')}")
                    print(f"Has Calculations: {advice_data.get('has_calculations', False)}")
                    print(f"Context Used: {advice_data.get('context_used', False)}")
                    print()
                    
                    # Check if response contains three risk levels
                    risk_levels_found = []
                    if "HIGH RISK" in advice_text.upper():
                        risk_levels_found.append("HIGH")
                    if "MEDIUM RISK" in advice_text.upper():
                        risk_levels_found.append("MEDIUM")
                    if "LOW RISK" in advice_text.upper():
                        risk_levels_found.append("LOW")
                    
                    print(f"Risk Levels Found: {risk_levels_found}")
                    
                    # Check for mathematical calculations
                    has_calculations = any(indicator in advice_text for indicator in [
                        "₹", "×", "=", "%", "calculation", "formula", "return", "projection"
                    ])
                    print(f"Contains Calculations: {has_calculations}")
                    
                    # Show first 200 characters of advice
                    print(f"Advice Preview: {advice_text[:200]}...")
                    
                else:
                    print("❌ Failed!")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Exception: {e}")
            
            print()
            print("=" * 60)
            print()
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

def print_test_summary():
    """Print summary of what to expect from the tests."""
    print("🎯 Three-Level Risk Advice Test Summary")
    print("=" * 50)
    print()
    print("Expected Behavior:")
    print("• Each query should generate advice with 3 risk levels")
    print("• HIGH RISK: Aggressive growth strategies")
    print("• MEDIUM RISK: Balanced approach")
    print("• LOW RISK: Conservative strategies")
    print()
    print("Mathematical Elements:")
    print("• Step-by-step calculations")
    print("• Formulas (compound interest, ROI, etc.)")
    print("• Expected returns with numbers")
    print("• Personalized using financial data")
    print()
    print("Key Metrics Used:")
    print(f"• Monthly Disposable Income: ₹{MOCK_FINANCIAL_DATA['user']['monthly_income'] - MOCK_FINANCIAL_DATA['user']['monthly_expenses']:,}")
    print(f"• Current Net Worth: ₹{MOCK_FINANCIAL_DATA['dashboardMetrics']['net_worth']:,}")
    print(f"• Current Savings Rate: {MOCK_FINANCIAL_DATA['dashboardMetrics']['savings_rate']*100:.1f}%")
    print(f"• Risk Profile: {MOCK_FINANCIAL_DATA['user']['risk_profile']}")
    print()

if __name__ == "__main__":
    print_test_summary()
    
    # Run the async test
    asyncio.run(test_advice_tool())
