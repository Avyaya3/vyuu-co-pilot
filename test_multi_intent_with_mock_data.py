#!/usr/bin/env python3
"""
Test script for multi-intent functionality with mock financial data.
This script provides mock data to work around the DataFetchTool None error.
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
        "email": "test@example.com"
    },
    "assets": [
        {
            "id": "asset_1",
            "name": "Savings Account",
            "type": "savings",
            "balance": 50000,
            "currency": "INR"
        },
        {
            "id": "asset_2", 
            "name": "Investment Portfolio",
            "type": "investment",
            "balance": 100000,
            "currency": "INR"
        }
    ],
    "liabilities": [
        {
            "id": "liability_1",
            "name": "Credit Card",
            "type": "credit_card",
            "balance": 15000,
            "currency": "INR"
        }
    ],
    "goals": [
        {
            "id": "goal_1",
            "name": "Emergency Fund",
            "target_amount": 100000,
            "current_amount": 50000,
            "currency": "INR"
        }
    ],
    "income": [
        {
            "id": "income_1",
            "source": "Salary",
            "amount": 80000,
            "frequency": "monthly",
            "currency": "INR"
        }
    ],
    "expenses": [
        {
            "id": "expense_1",
            "category": "Food & Dining",
            "amount": 15000,
            "frequency": "monthly",
            "currency": "INR"
        },
        {
            "id": "expense_2",
            "category": "Transportation",
            "amount": 8000,
            "frequency": "monthly", 
            "currency": "INR"
        }
    ],
    "stocks": [
        {
            "id": "stock_1",
            "symbol": "RELIANCE",
            "quantity": 10,
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
            "premium": 5000,
            "coverage": 1000000,
            "currency": "INR"
        }
    ],
    "savings": [
        {
            "id": "savings_1",
            "name": "Emergency Fund",
            "balance": 50000,
            "currency": "INR"
        }
    ],
    "dashboardMetrics": {
        "net_worth": 135000,
        "monthly_income": 80000,
        "monthly_expenses": 23000,
        "savings_rate": 0.71,
        "currency": "INR"
    }
}

def create_test_state_with_mock_data(user_input: str) -> Dict[str, Any]:
    """
    Create a test state with mock financial data for multi-intent testing.
    
    Args:
        user_input: The user query to test
        
    Returns:
        State dictionary with mock financial data
    """
    return {
        "user_input": user_input,
        "intent": "unknown",  # Will be classified
        "confidence": 0.0,
        "messages": [
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {"source": "user_input"}
            }
        ],
        "session_id": f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "user_id": "test_user_123",
            "financial_data": MOCK_FINANCIAL_DATA,  # This is the key addition!
            "test_mode": True,
            "mock_data_provided": True
        },
        "parameters": {},
        "execution_results": None,
        "response": None,
        "multiple_intents": None
    }

# Test queries for multi-intent functionality
TEST_QUERIES = [
    "Show me my expenses and give me budgeting advice",
    "Transfer ₹1000 to savings and show me the updated balance", 
    "Show my budget, create a savings goal, and give me investment advice",
    "Show me my transactions and also display my account balances",
    "Give me financial advice and help me set up automatic savings"
]

def print_test_instructions():
    """Print instructions for testing multi-intent functionality."""
    print("🧪 Multi-Intent Testing with Mock Data")
    print("=" * 50)
    print()
    print("To test multi-intent functionality in LangGraph Studio:")
    print()
    print("1. Open LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024")
    print()
    print("2. Use these test queries:")
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"   {i}. {query}")
    print()
    print("3. The mock financial data includes:")
    print("   • Assets: ₹1,50,000 (Savings + Investment)")
    print("   • Liabilities: ₹15,000 (Credit Card)")
    print("   • Monthly Income: ₹80,000")
    print("   • Monthly Expenses: ₹23,000")
    print("   • Net Worth: ₹1,35,000")
    print()
    print("4. Expected behavior:")
    print("   • Intent Classification: Should detect multiple intents")
    print("   • Routing: Should route to direct_orchestrator")
    print("   • Execution: Should show parallel/sequential execution")
    print("   • Response: Should show structured multi-intent response")
    print()
    print("5. If you still see 'Financial data is None' errors:")
    print("   • Check that the state includes 'financial_data' in metadata")
    print("   • Verify the DataFetchTool is receiving the mock data")
    print()

if __name__ == "__main__":
    print_test_instructions()
    
    # Create example state for one query
    example_state = create_test_state_with_mock_data(TEST_QUERIES[0])
    print("📋 Example state structure:")
    print(json.dumps(example_state, indent=2, default=str))
