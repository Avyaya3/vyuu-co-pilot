"""
Configuration management for the Vyuu Copilot v2 application.

This module handles all configuration settings including Supabase connection,
database settings, and application configuration using Pydantic settings.
"""

from typing import Optional
from pydantic import BaseSettings, validator
import os


class SupabaseConfig(BaseSettings):
    """Supabase-specific configuration settings."""
    
    url: str
    key: str
    service_role_key: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    
    class Config:
        env_prefix = "SUPABASE_"
        case_sensitive = False
    
    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Validate Supabase URL format."""
        if not v.startswith("https://") or "supabase.co" not in v:
            raise ValueError("Invalid Supabase URL format")
        return v


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False
    
    @validator("pool_size")
    def validate_pool_size(cls, v: int) -> int:
        """Validate connection pool size."""
        if v < 1 or v > 100:
            raise ValueError("Pool size must be between 1 and 100")
        return v


class AuthConfig(BaseSettings):
    """Authentication configuration."""
    
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    class Config:
        env_prefix = ""
        case_sensitive = False
    
    @validator("jwt_secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Validate JWT secret key."""
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v


class APIConfig(BaseSettings):
    """API server configuration."""
    
    host: str = "localhost"
    port: int = 8000
    environment: str = "development"
    debug: bool = True
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "json"
    
    class Config:
        env_prefix = "LOG_"
        case_sensitive = False


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    environment: str = "development"
    debug: bool = True
    
    # Sub-configurations
    supabase: SupabaseConfig
    database: DatabaseConfig
    auth: AuthConfig
    api: APIConfig
    logging: LoggingConfig
    
    def __init__(self, **kwargs):
        # Initialize sub-configurations
        supabase_config = SupabaseConfig()
        database_config = DatabaseConfig()
        auth_config = AuthConfig()
        api_config = APIConfig()
        logging_config = LoggingConfig()
        
        super().__init__(
            supabase=supabase_config,
            database=database_config,
            auth=auth_config,
            api=api_config,
            logging=logging_config,
            **kwargs
        )
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"


# Global configuration instance
config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.
    
    Returns:
        AppConfig: The application configuration instance.
    """
    global config
    if config is None:
        config = AppConfig()
    return config


def reload_config() -> AppConfig:
    """
    Reload the configuration from environment variables.
    
    Returns:
        AppConfig: The reloaded application configuration instance.
    """
    global config
    config = AppConfig()
    return config 

# JWT verification moved to src/utils/auth.py for better organization 