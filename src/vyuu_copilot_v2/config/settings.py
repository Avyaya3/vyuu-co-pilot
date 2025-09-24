"""
Configuration management for the Vyuu Copilot v2 application.

This module handles all configuration settings including Supabase connection,
database settings, and application configuration using Pydantic settings.
"""

from typing import Optional, Annotated, List
from pydantic_settings import BaseSettings
from pydantic import validator, Field
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SupabaseConfig(BaseSettings):
    """Supabase-specific configuration settings."""
    
    url: str = "https://placeholder.supabase.co"
    key: str = "placeholder_key"
    service_role_key: str = "placeholder_service_role_key"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 1
    
    class Config:
        env_prefix = "SUPABASE_"
        case_sensitive = False
        extra = "ignore"
    
    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Validate Supabase URL format."""
        if not v.startswith("https://") or "supabase.co" not in v:
            raise ValueError("Invalid Supabase URL format")
        return v


class CustomJWTConfig(BaseSettings):
    """Custom JWT token configuration for NextJS integration."""
    
    secret: str = Field("your-custom-jwt-secret-key-here", description="Secret key for signing/verifying custom JWT tokens")
    issuer: Optional[str] = Field(None, description="Expected token issuer")
    audience: Optional[str] = Field(None, description="Expected token audience")
    algorithm: str = Field("HS256", description="JWT signing algorithm")
    expiration_hours: int = Field(24, description="Token expiration time in hours")
    
    class Config:
        env_prefix = "CUSTOM_JWT_"
        case_sensitive = False
        extra = "ignore"
    
    @validator("secret")
    def validate_secret(cls, v: str) -> str:
        """Validate JWT secret is not empty."""
        if not v or len(v) < 16:
            raise ValueError("JWT secret must be at least 16 characters long")
        return v


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    
    url: str = Field("postgresql://placeholder:placeholder@localhost:5432/placeholder", alias='DATABASE_URL')  # Map to DATABASE_URL env var
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False
        extra = "ignore"
    
    @validator("pool_size")
    def validate_pool_size(cls, v: int) -> int:
        """Validate connection pool size."""
        if v < 1 or v > 100:
            raise ValueError("Pool size must be between 1 and 100")
        return v


class AuthConfig(BaseSettings):
    """Authentication configuration."""
    
    jwt_secret_key: str = "your-jwt-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    class Config:
        env_prefix = ""
        case_sensitive = False
        extra = "ignore"
    
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
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False
        extra = "ignore"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "json"
    
    class Config:
        env_prefix = "LOG_"
        case_sensitive = False
        extra = "ignore"


class CurrencyConfig(BaseSettings):
    """Currency configuration settings."""
    
    default_symbol: str = "₹"
    default_code: str = "INR"
    decimal_places: int = 2
    thousands_separator: str = ","
    decimal_separator: str = "."
    
    class Config:
        env_prefix = "CURRENCY_"
        case_sensitive = False
        extra = "ignore"
    
    @validator("default_symbol")
    def validate_symbol(cls, v: str) -> str:
        """Validate currency symbol."""
        if not v or len(v) > 3:
            raise ValueError("Currency symbol must be 1-3 characters")
        return v
    
    @validator("decimal_places")
    def validate_decimal_places(cls, v: int) -> int:
        """Validate decimal places."""
        if v < 0 or v > 4:
            raise ValueError("Decimal places must be between 0 and 4")
        return v


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
    custom_jwt: CustomJWTConfig
    currency: CurrencyConfig
    
    def __init__(self, **kwargs):
        # Initialize sub-configurations
        supabase_config = SupabaseConfig()
        database_config = DatabaseConfig()
        auth_config = AuthConfig()
        api_config = APIConfig()
        logging_config = LoggingConfig()
        custom_jwt_config = CustomJWTConfig()
        currency_config = CurrencyConfig()
        
        super().__init__(
            supabase=supabase_config,
            database=database_config,
            auth=auth_config,
            api=api_config,
            logging=logging_config,
            custom_jwt=custom_jwt_config,
            currency=currency_config,
            **kwargs
        )
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
    
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