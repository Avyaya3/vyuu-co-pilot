"""
Simple Test for JWT Authentication Flow Analysis

This test analyzes the current authentication flow without requiring full application startup.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomJWTTestSuite:
    """
    Test suite for custom JWT token authentication flow.
    """
    
    def __init__(self):
        self.test_user_id = "test-user-123"
        self.test_email = "test@example.com"
        self.test_secret = "test-secret-key"
        self.test_issuer = "vyuu-copilot"
        self.test_audience = "vyuu-users"
        
    def create_custom_jwt_token(
        self, 
        user_id: str = None, 
        email: str = None,
        expires_in_hours: int = 24,
        secret: str = None,
        issuer: str = None,
        audience: str = None
    ) -> str:
        """
        Create a custom JWT token for testing.
        """
        user_id = user_id or self.test_user_id
        email = email or self.test_email
        secret = secret or self.test_secret
        issuer = issuer or self.test_issuer
        audience = audience or self.test_audience
        
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,  # Subject (user ID)
            "email": email,
            "iat": now,  # Issued at
            "exp": now + timedelta(hours=expires_in_hours),  # Expiration
            "iss": issuer,  # Issuer
            "aud": audience,  # Audience
            "user_id": user_id,  # Custom claim for user ID
            "role": "authenticated",
            "user_metadata": {
                "name": "Test User",
                "preferences": {"theme": "dark"}
            }
        }
        
        token = jwt.encode(payload, secret, algorithm="HS256")
        logger.info(f"Created custom JWT token for user {user_id}")
        return token
    
    def create_expired_jwt_token(self) -> str:
        """Create an expired JWT token for testing."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": self.test_user_id,
            "email": self.test_email,
            "iat": now - timedelta(hours=25),  # Issued 25 hours ago
            "exp": now - timedelta(hours=1),   # Expired 1 hour ago
            "iss": self.test_issuer,
            "aud": self.test_audience,
            "user_id": self.test_user_id,
            "role": "authenticated"
        }
        
        token = jwt.encode(payload, self.test_secret, algorithm="HS256")
        logger.info(f"Created expired JWT token for user {self.test_user_id}")
        return token


def analyze_current_auth_implementation():
    """
    Analyze the current authentication implementation.
    """
    print("🔍 Analyzing Current Authentication Implementation")
    print("=" * 60)
    
    # Read the auth.py file to understand current implementation
    try:
        with open("src/utils/auth.py", "r") as f:
            auth_code = f.read()
        
        print("📋 Current Authentication Implementation Analysis:")
        print("-" * 50)
        
        # Check for Supabase-specific JWT verification
        if "verify_supabase_jwt" in auth_code:
            print("✅ Found: Supabase JWT verification method")
            if "self.client.auth.get_user(token)" in auth_code:
                print("  → Uses Supabase client to verify tokens")
                print("  → This will NOT work with custom JWT tokens")
        
        # Check for token extraction
        if "extract_token_from_header" in auth_code:
            print("✅ Found: Token extraction from Authorization header")
            if "Bearer" in auth_code:
                print("  → Expects 'Bearer <token>' format")
        
        # Check for FastAPI dependencies
        if "verify_jwt_token" in auth_code:
            print("✅ Found: FastAPI JWT verification dependency")
        if "get_current_user" in auth_code:
            print("✅ Found: FastAPI current user dependency")
        if "require_authentication" in auth_code:
            print("✅ Found: FastAPI required authentication dependency")
        
        print("\n🚨 ISSUE IDENTIFIED:")
        print("  The current implementation only supports Supabase JWT tokens")
        print("  Custom JWT tokens from NextJS will be rejected")
        
    except FileNotFoundError:
        print("❌ Could not read auth.py file")
    
    return True


def analyze_session_management():
    """
    Analyze the current session management implementation.
    """
    print("\n💾 Analyzing Current Session Management")
    print("=" * 60)
    
    try:
        with open("src/orchestrator.py", "r") as f:
            orchestrator_code = f.read()
        
        print("📋 Current Session Management Analysis:")
        print("-" * 50)
        
        # Check for session manager
        if "class SessionManager" in orchestrator_code:
            print("✅ Found: SessionManager class")
            if "self._sessions: Dict[str, MainState] = {}" in orchestrator_code:
                print("  → Uses in-memory dictionary storage")
                print("  → Sessions will NOT persist across server restarts")
        
        # Check for database session management
        if "use_database: bool = False" in orchestrator_code:
            print("✅ Found: Database session management option")
            if "NotImplementedError" in orchestrator_code:
                print("  → Database session management is NOT implemented")
        
        # Check for user ID handling
        if "user_id" in orchestrator_code:
            print("✅ Found: User ID parameter handling")
            if "metadata" in orchestrator_code and "user_id" in orchestrator_code:
                print("  → User ID is stored in session metadata")
        
        print("\n🚨 ISSUES IDENTIFIED:")
        print("  1. Sessions are stored in-memory only")
        print("  2. Sessions will be lost on server restart")
        print("  3. No user-specific session management")
        
    except FileNotFoundError:
        print("❌ Could not read orchestrator.py file")
    
    return True


def test_custom_jwt_token_creation():
    """
    Test custom JWT token creation and validation.
    """
    print("\n🔐 Testing Custom JWT Token Creation")
    print("=" * 60)
    
    test_suite = CustomJWTTestSuite()
    
    try:
        # Test valid token creation
        token = test_suite.create_custom_jwt_token()
        print("✅ Successfully created custom JWT token")
        
        # Test token structure
        parts = token.split('.')
        if len(parts) == 3:
            print("✅ Token has correct JWT structure (3 parts)")
        else:
            print("❌ Token has incorrect structure")
        
        # Test token decoding
        decoded = jwt.decode(token, test_suite.test_secret, algorithms=["HS256"])
        print("✅ Successfully decoded custom JWT token")
        
        # Verify payload contents
        expected_fields = ["sub", "email", "iat", "exp", "iss", "aud", "user_id", "role"]
        for field in expected_fields:
            if field in decoded:
                print(f"✅ Token contains {field}: {decoded[field]}")
            else:
                print(f"❌ Token missing {field}")
        
        # Test expired token
        expired_token = test_suite.create_expired_jwt_token()
        try:
            jwt.decode(expired_token, test_suite.test_secret, algorithms=["HS256"])
            print("❌ Expired token was not properly rejected")
        except jwt.ExpiredSignatureError:
            print("✅ Expired token properly rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom JWT token test failed: {e}")
        return False


def generate_implementation_plan():
    """
    Generate a detailed implementation plan.
    """
    print("\n📋 IMPLEMENTATION PLAN")
    print("=" * 60)
    
    plan = {
        "phase_1": {
            "title": "Custom JWT Token Support",
            "tasks": [
                "Add custom JWT verification method to SupabaseAuth class",
                "Implement JWT token decoding with custom secret",
                "Add token validation (expiration, issuer, audience)",
                "Update FastAPI dependencies to support custom tokens",
                "Add error handling for invalid/expired custom tokens"
            ],
            "files_to_modify": [
                "src/utils/auth.py",
                "src/api.py"
            ]
        },
        "phase_2": {
            "title": "Database Session Persistence",
            "tasks": [
                "Create database session manager class",
                "Implement session storage in PostgreSQL",
                "Add session retrieval by user ID",
                "Implement session cleanup and expiration",
                "Update MainOrchestrator to use database sessions"
            ],
            "files_to_modify": [
                "src/orchestrator.py",
                "src/repositories/ (new session repository)",
                "src/schemas/database_models.py"
            ]
        },
        "phase_3": {
            "title": "User Context Integration",
            "tasks": [
                "Enhance state schemas with user context",
                "Add user-specific personalization",
                "Implement user session management endpoints",
                "Add user-specific data access in LangGraph nodes"
            ],
            "files_to_modify": [
                "src/schemas/state_schemas.py",
                "src/nodes/ (various node files)",
                "src/api.py"
            ]
        }
    }
    
    for phase, details in plan.items():
        print(f"\n{phase.upper().replace('_', ' ')}: {details['title']}")
        print("-" * 40)
        for i, task in enumerate(details['tasks'], 1):
            print(f"  {i}. {task}")
        print(f"\n  Files to modify: {', '.join(details['files_to_modify'])}")
    
    return plan


def main():
    """
    Main test function.
    """
    print("🚀 JWT Authentication Flow Analysis")
    print("=" * 60)
    
    results = {
        "current_auth_analysis": analyze_current_auth_implementation(),
        "session_management_analysis": analyze_session_management(),
        "custom_jwt_test": test_custom_jwt_token_creation(),
        "implementation_plan": generate_implementation_plan()
    }
    
    print("\n📊 SUMMARY")
    print("=" * 60)
    print("✅ Current system supports Supabase JWT tokens only")
    print("❌ Custom JWT tokens from NextJS will be rejected")
    print("❌ Sessions are in-memory only (no persistence)")
    print("❌ No user-specific session management")
    print("✅ User ID is passed to orchestrator and stored in metadata")
    
    print("\n🎯 IMMEDIATE NEXT STEPS:")
    print("1. Implement custom JWT verification in SupabaseAuth class")
    print("2. Create database session manager for persistence")
    print("3. Add user-specific session operations")
    print("4. Test the complete flow end-to-end")
    
    # Save results
    with open("auth_flow_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed analysis saved to: auth_flow_analysis_results.json")
    
    return results


if __name__ == "__main__":
    main()
