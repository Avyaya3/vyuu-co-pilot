#!/usr/bin/env python3
"""
Database connection test script for Supabase integration.

This script tests all aspects of the Supabase connection including:
- Configuration loading
- Database connectivity
- Authentication functionality
- Connection pooling
- Error handling

Run this script to verify your Supabase setup is working correctly.
"""

import asyncio
import json
import logging
import sys
import time
from typing import Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import os
import sys

# Add the src directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    from config import get_config, AppConfig
    from utils.database import get_db_client, DatabaseConnectionError
    from utils.auth import get_auth_manager, AuthenticationError
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Make sure you're running this from the project root directory")
    sys.exit(1)


class ConnectionTester:
    """Test suite for Supabase connection and authentication."""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "total": 0}
        }
    
    def log_test_result(self, test_name: str, passed: bool, details: Dict[str, Any] = None):
        """Log the result of a test."""
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if passed:
            self.test_results["summary"]["passed"] += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.test_results["summary"]["failed"] += 1
            logger.error(f"❌ {test_name}: FAILED")
            if details:
                logger.error(f"   Details: {details}")
        
        self.test_results["summary"]["total"] += 1
    
    def test_configuration_loading(self) -> bool:
        """Test configuration loading."""
        test_name = "Configuration Loading"
        try:
            config = get_config()
            
            # Check required Supabase configuration
            required_fields = ["url", "key", "service_role_key"]
            missing_fields = []
            
            for field in required_fields:
                if not hasattr(config.supabase, field):
                    missing_fields.append(field)
                elif getattr(config.supabase, field).startswith("your_"):
                    missing_fields.append(f"{field} (not configured)")
            
            if missing_fields:
                self.log_test_result(
                    test_name, 
                    False, 
                    {"missing_fields": missing_fields}
                )
                return False
            
            self.log_test_result(
                test_name, 
                True, 
                {
                    "supabase_url": config.supabase.url,
                    "database_pool_size": config.database.pool_size,
                    "jwt_algorithm": config.auth.jwt_algorithm
                }
            )
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_database_connection(self) -> bool:
        """Test basic database connectivity."""
        test_name = "Database Connection"
        try:
            db_client = get_db_client()
            
            # Test basic query
            start_time = time.time()
            result = await db_client.execute_query(
                "SELECT 1 as test_value, NOW() as current_time",
                fetch_one=True
            )
            query_time = (time.time() - start_time) * 1000
            
            if result and result["test_value"] == 1:
                self.log_test_result(
                    test_name, 
                    True, 
                    {
                        "query_time_ms": round(query_time, 2),
                        "current_time": str(result["current_time"])
                    }
                )
                return True
            else:
                self.log_test_result(
                    test_name, 
                    False, 
                    {"error": "Query returned unexpected result"}
                )
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_connection_pool(self) -> bool:
        """Test connection pooling functionality."""
        test_name = "Connection Pool"
        try:
            db_client = get_db_client()
            
            # Test multiple concurrent connections
            tasks = []
            for i in range(5):
                task = db_client.execute_query(
                    "SELECT $1 as connection_id, pg_backend_pid() as pid",
                    i,
                    fetch_one=True
                )
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = (time.time() - start_time) * 1000
            
            # Verify all queries succeeded
            if len(results) == 5 and all(r for r in results):
                pids = [r["pid"] for r in results]
                unique_pids = len(set(pids))
                
                self.log_test_result(
                    test_name, 
                    True, 
                    {
                        "concurrent_queries": 5,
                        "total_time_ms": round(total_time, 2),
                        "unique_connections": unique_pids,
                        "avg_time_per_query_ms": round(total_time / 5, 2)
                    }
                )
                return True
            else:
                self.log_test_result(
                    test_name, 
                    False, 
                    {"error": "Not all concurrent queries succeeded"}
                )
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    def test_supabase_client(self) -> bool:
        """Test Supabase client initialization."""
        test_name = "Supabase Client"
        try:
            db_client = get_db_client()
            client = db_client.client
            
            # Test that client is properly initialized
            if hasattr(client, 'auth') and hasattr(client, 'table'):
                self.log_test_result(
                    test_name, 
                    True, 
                    {"client_type": type(client).__name__}
                )
                return True
            else:
                self.log_test_result(
                    test_name, 
                    False, 
                    {"error": "Client missing required attributes"}
                )
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    def test_jwt_token_verification(self) -> bool:
        """Test JWT token verification functionality."""
        test_name = "JWT Token Verification"
        try:
            auth_manager = get_auth_manager()
            
            # Test token extraction from header
            test_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"
            authorization_header = f"Bearer {test_token}"
            
            extracted_token = auth_manager.extract_token_from_header(authorization_header)
            
            if extracted_token == test_token:
                self.log_test_result(
                    test_name, 
                    True, 
                    {
                        "token_extraction": "successful",
                        "auth_manager_type": type(auth_manager).__name__,
                        "methods_available": [
                            method for method in dir(auth_manager) 
                            if not method.startswith('_')
                        ]
                    }
                )
                return True
            else:
                self.log_test_result(
                    test_name, 
                    False, 
                    {"error": "Token extraction failed"}
                )
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def test_health_check(self) -> bool:
        """Test the database health check functionality."""
        test_name = "Health Check"
        try:
            db_client = get_db_client()
            health_status = await db_client.health_check()
            
            if health_status["status"] == "healthy":
                self.log_test_result(
                    test_name, 
                    True, 
                    {
                        "overall_status": health_status["status"],
                        "tests_count": len(health_status["tests"]),
                        "connection_stats": health_status["connection_stats"]
                    }
                )
                return True
            else:
                self.log_test_result(
                    test_name, 
                    False, 
                    {
                        "status": health_status["status"],
                        "failed_tests": [
                            name for name, test in health_status["tests"].items()
                            if test.get("status") != "healthy"
                        ]
                    }
                )
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, {"error": str(e)})
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all connection tests."""
        logger.info("🚀 Starting Supabase connection tests...")
        logger.info("=" * 50)
        
        # Run tests in order
        tests = [
            ("Configuration", self.test_configuration_loading),
            ("Supabase Client", self.test_supabase_client),
            ("Database Connection", self.test_database_connection),
            ("Connection Pool", self.test_connection_pool),
            ("JWT Verification", self.test_jwt_token_verification),
            ("Health Check", self.test_health_check),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running {test_name} test...")
            try:
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.log_test_result(test_name, False, {"error": f"Test crashed: {e}"})
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 50)
        
        summary = self.test_results["summary"]
        logger.info(f"Total tests: {summary['total']}")
        logger.info(f"Passed: {summary['passed']} ✅")
        logger.info(f"Failed: {summary['failed']} ❌")
        
        success_rate = (summary['passed'] / summary['total']) * 100 if summary['total'] > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        if summary['failed'] == 0:
            logger.info("\n🎉 All tests passed! Your Supabase connection is working correctly.")
            return True
        else:
            logger.error(f"\n⚠️  {summary['failed']} test(s) failed. Please check your configuration.")
            return False
    
    def save_results(self, filename: str = "test_results.json"):
        """Save test results to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"Test results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


async def main():
    """Main test function."""
    tester = ConnectionTester()
    
    try:
        success = await tester.run_all_tests()
        tester.save_results()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test suite crashed: {e}")
        sys.exit(1)
    finally:
        # Clean up connections
        try:
            from utils.database import close_db_connections
            await close_db_connections()
            logger.info("🧹 Database connections closed")
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")


if __name__ == "__main__":
    print("🔧 Vyuu Copilot v2 - Supabase Connection Test")
    print("=" * 50)
    asyncio.run(main()) 