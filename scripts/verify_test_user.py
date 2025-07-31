#!/usr/bin/env python3
"""
Script to verify the test user exists and has data for LangGraph Studio testing.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.repositories import RepositoryFactory
from src.schemas.database_models import User


async def verify_test_user():
    """Verify the test user exists and has data."""
    print("🔍 Verifying Test User for LangGraph Studio")
    print("=" * 50)
    
    # Get test user ID from environment
    test_user_id = os.getenv("STUDIO_TEST_USER_ID")
    if not test_user_id:
        print("❌ STUDIO_TEST_USER_ID environment variable not set")
        print("Please add STUDIO_TEST_USER_ID=0575867a-743a-4f26-99b3-95b87d116d7b to your .env file")
        return False
    
    print(f"✅ Test User ID: {test_user_id}")
    
    try:
        # Initialize repositories
        repo_factory = RepositoryFactory()
        repo_factory.initialize()
        
        user_repo = repo_factory.get_user_repository()
        account_repo = repo_factory.get_account_repository()
        transaction_repo = repo_factory.get_transaction_repository()
        goal_repo = repo_factory.get_goal_repository()
        
        # Check if user exists
        user = await user_repo.get_by_id(test_user_id)
        if not user:
            print(f"❌ User {test_user_id} not found in database")
            return False
        
        print(f"✅ User found: {user.name} ({user.email})")
        
        # Check user's accounts
        accounts = await account_repo.get_user_accounts(test_user_id)
        print(f"✅ User has {len(accounts)} accounts")
        
        if accounts:
            total_balance = await account_repo.get_user_total_balance(test_user_id)
            print(f"✅ Total balance: ${total_balance:,.2f}")
        
        # Check user's transactions
        transactions = await transaction_repo.get_user_transactions(test_user_id, limit=5)
        print(f"✅ User has {len(transactions)} recent transactions")
        
        # Check user's goals
        goals = await goal_repo.get_user_goals(test_user_id)
        print(f"✅ User has {len(goals)} goals")
        
        print("\n🎉 Test user verification complete!")
        print("This user is ready for LangGraph Studio testing.")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying test user: {e}")
        return False


async def main():
    """Main function."""
    success = await verify_test_user()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 