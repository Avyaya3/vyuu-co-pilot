"""
Database connection utility for Supabase integration.

This module provides connection pooling, retry logic, health checks, and 
database utilities for the Supabase PostgreSQL database.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator
import time
from datetime import datetime

from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import asyncpg
from asyncpg import Pool, Connection
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from ..config import get_config
except ImportError:
    from config import get_config

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class SupabaseClient:
    """
    Supabase client wrapper with connection pooling and retry logic.
    """
    
    def __init__(self):
        self.config = get_config()
        self._client: Optional[Client] = None
        self._pg_pool: Optional[Pool] = None
        self._connection_stats = {
            "total_connections": 0,
            "failed_connections": 0,
            "last_connection_time": None,
            "last_failure_time": None,
        }
    
    @property
    def client(self) -> Client:
        """
        Get the Supabase client instance.
        
        Returns:
            Client: The Supabase client instance.
            
        Raises:
            DatabaseConnectionError: If client initialization fails.
        """
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def _initialize_client(self) -> None:
        """Initialize the Supabase client with configuration."""
        try:
            options = ClientOptions(
                postgrest_client_timeout=self.config.supabase.timeout,
                storage_client_timeout=self.config.supabase.timeout,
            )
            
            self._client = create_client(
                self.config.supabase.url,
                self.config.supabase.key,
                options=options
            )
            
            self._connection_stats["last_connection_time"] = datetime.utcnow()
            self._connection_stats["total_connections"] += 1
            
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            self._connection_stats["failed_connections"] += 1
            self._connection_stats["last_failure_time"] = datetime.utcnow()
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise DatabaseConnectionError(f"Client initialization failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    async def get_postgres_pool(self) -> Pool:
        """
        Get or create the PostgreSQL connection pool.
        
        Returns:
            Pool: The asyncpg connection pool.
            
        Raises:
            DatabaseConnectionError: If pool creation fails.
        """
        if self._pg_pool is None:
            await self._create_postgres_pool()
        return self._pg_pool
    
    async def _create_postgres_pool(self) -> None:
        """Create the PostgreSQL connection pool."""
        try:
            # Configure SSL for Supabase connection
            ssl_config = {
                'ssl': 'require',  # Require SSL but don't verify certificate
                'server_settings': {
                    'application_name': 'vyuu_copilot_v2'
                }
            }
            
            self._pg_pool = await asyncpg.create_pool(
                self.config.database.url,
                min_size=self.config.database.pool_size // 2,
                max_size=self.config.database.pool_size,
                max_queries=50000,
                max_inactive_connection_lifetime=self.config.database.pool_recycle,
                timeout=self.config.database.pool_timeout,
                command_timeout=60,
                statement_cache_size=0,  # Disable prepared statements for Supavisor compatibility
                **ssl_config
            )
            
            logger.info(
                f"PostgreSQL connection pool created with size {self.config.database.pool_size}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise DatabaseConnectionError(f"Pool creation failed: {e}")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """
        Get a database connection from the pool.
        
        Yields:
            Connection: Database connection from the pool.
            
        Raises:
            DatabaseConnectionError: If connection acquisition fails.
        """
        pool = await self.get_postgres_pool()
        connection = None
        
        try:
            connection = await pool.acquire(timeout=self.config.database.pool_timeout)
            yield connection
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseConnectionError(f"Connection failed: {e}")
            
        finally:
            if connection:
                await pool.release(connection)
    
    async def execute_query(
        self, 
        query: str, 
        *args, 
        fetch_one: bool = False,
        fetch_all: bool = True
    ) -> Optional[Any]:
        """
        Execute a database query with automatic connection management.
        
        Args:
            query: SQL query to execute.
            *args: Query parameters.
            fetch_one: Whether to fetch only one row.
            fetch_all: Whether to fetch all rows.
            
        Returns:
            Query results or None.
            
        Raises:
            DatabaseConnectionError: If query execution fails.
        """
        async with self.get_connection() as conn:
            try:
                if fetch_one:
                    result = await conn.fetchrow(query, *args)
                elif fetch_all:
                    result = await conn.fetch(query, *args)
                else:
                    result = await conn.execute(query, *args)
                
                return result
                
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseConnectionError(f"Query failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.
        
        Returns:
            Dict containing health check results.
        """
        health_status = {
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "connection_stats": self._connection_stats.copy(),
            "tests": {}
        }
        
        try:
            # Test Supabase client - just check if we can access the client
            start_time = time.time()
            # Simple test to see if client is accessible
            client = self.client
            has_auth = hasattr(client, 'auth')
            has_table = hasattr(client, 'table')
            supabase_latency = (time.time() - start_time) * 1000
            
            if has_auth and has_table:
                health_status["tests"]["supabase_client"] = {
                    "status": "healthy",
                    "latency_ms": round(supabase_latency, 2),
                    "features": {"auth": has_auth, "table": has_table}
                }
            else:
                health_status["tests"]["supabase_client"] = {
                    "status": "unhealthy",
                    "error": "Client missing required features"
                }
            
        except Exception as e:
            health_status["tests"]["supabase_client"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            # Test PostgreSQL pool
            start_time = time.time()
            result = await self.execute_query("SELECT 1 as health_check", fetch_one=True)
            pg_latency = (time.time() - start_time) * 1000
            
            health_status["tests"]["postgresql_pool"] = {
                "status": "healthy",
                "latency_ms": round(pg_latency, 2),
                "result": dict(result) if result else None
            }
            
        except Exception as e:
            health_status["tests"]["postgresql_pool"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall status
        all_tests_healthy = all(
            test.get("status") == "healthy" 
            for test in health_status["tests"].values()
        )
        health_status["status"] = "healthy" if all_tests_healthy else "unhealthy"
        
        return health_status
    
    async def close(self) -> None:
        """Close all database connections."""
        if self._pg_pool:
            await self._pg_pool.close()
            logger.info("PostgreSQL connection pool closed")
        
        # Supabase client doesn't need explicit closing
        self._client = None
        logger.info("Database connections closed")


# Global database client instance
db_client: Optional[SupabaseClient] = None


def get_db_client() -> SupabaseClient:
    """
    Get the global database client instance.
    
    Returns:
        SupabaseClient: The database client instance.
    """
    global db_client
    if db_client is None:
        db_client = SupabaseClient()
    return db_client


async def close_db_connections() -> None:
    """Close all database connections."""
    global db_client
    if db_client:
        await db_client.close()
        db_client = None 