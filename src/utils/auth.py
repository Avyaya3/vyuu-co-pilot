"""
Authentication utilities for Supabase integration.

This module provides JWT handling, user authentication, session management,
and security utilities for the Supabase authentication system.
"""

import logging
from typing import Optional, Dict, Any, Union
from uuid import UUID
import asyncio
from datetime import datetime, timezone, timedelta

from supabase import Client
from gotrue.errors import AuthApiError, AuthRetryableError
import jwt

try:
    from ..config import get_config
    from .database import get_db_client, DatabaseConnectionError
except ImportError:
    from config import get_config
    from utils.database import get_db_client, DatabaseConnectionError

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    pass


class TokenValidationError(Exception):
    """Raised when token validation fails."""
    pass


class SupabaseAuth:
    """
    Supabase authentication manager with JWT handling and user management.
    """
    
    def __init__(self):
        self.config = get_config()
        self.db_client = get_db_client()
    
    @property
    def client(self) -> Client:
        """Get the Supabase client."""
        return self.db_client.client
    
    def verify_supabase_jwt(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a Supabase JWT token.
        
        Args:
            token: Supabase JWT token to verify.
            
        Returns:
            Decoded token payload with user information.
            
        Raises:
            TokenValidationError: If token verification fails.
        """
        try:
            # Use Supabase client to verify the JWT token
            response = self.client.auth.get_user(token)
            
            if response.user:
                logger.debug(f"Supabase JWT verified for user {response.user.id}")
                return {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "role": response.user.role if hasattr(response.user, 'role') else "authenticated",
                    "user_metadata": response.user.user_metadata or {},
                    "app_metadata": response.user.app_metadata or {},
                    "aud": response.user.aud,
                    "exp": response.user.confirmation_sent_at,  # Token expiration handled by Supabase
                }
            else:
                raise TokenValidationError("Invalid token: No user found")
                
        except Exception as e:
            logger.error(f"Supabase JWT verification failed: {e}")
            raise TokenValidationError(f"Token verification failed: {e}")
    
    def create_custom_jwt_token(
        self,
        user_id: str,
        email: str = None,
        secret: str = None,
        issuer: str = None,
        audience: str = None,
        expires_in_hours: int = 24,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new custom JWT token.
        
        Args:
            user_id: User ID to include in token
            email: User email to include in token
            secret: Secret key for signing (uses config if not provided)
            issuer: Token issuer (uses config if not provided)
            audience: Token audience (uses config if not provided)
            expires_in_hours: Token expiration time in hours
            user_metadata: Additional user metadata
            
        Returns:
            Signed JWT token string
        """
        # Use config values if not provided
        if not secret:
            secret = self.config.custom_jwt.secret
        if not issuer:
            issuer = self.config.custom_jwt.issuer
        if not audience:
            audience = self.config.custom_jwt.audience
        
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "email": email or "",
            "iat": now,
            "exp": now + timedelta(hours=expires_in_hours),
            "iss": issuer,
            "aud": audience,
            "user_id": user_id,
            "role": "authenticated",
            "user_metadata": user_metadata or {},
            "token_type": "custom_jwt"
        }
        
        token = jwt.encode(payload, secret, algorithm="HS256")
        logger.info(f"Created new custom JWT token for user {user_id}")
        return token
    
    def verify_custom_jwt(
        self, 
        token: str, 
        secret: str, 
        issuer: Optional[str] = None,
        audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify and decode a custom JWT token from NextJS.
        
        Args:
            token: Custom JWT token to verify
            secret: Secret key used to sign the token
            issuer: Expected token issuer (optional)
            audience: Expected token audience (optional)
            
        Returns:
            Decoded token payload with user information
            
        Raises:
            TokenValidationError: If token verification fails
        """
        try:
            # Decode the JWT token
            payload = jwt.decode(
                token, 
                secret, 
                algorithms=["HS256"],
                options={
                    "verify_exp": True, 
                    "verify_iat": True,
                    "verify_aud": False,  # We'll validate audience manually
                    "verify_iss": False   # We'll validate issuer manually
                }
            )
            
            # Validate issuer if provided
            if issuer and payload.get("iss") != issuer:
                raise TokenValidationError(f"Invalid issuer. Expected: {issuer}, Got: {payload.get('iss')}")
            
            # Validate audience if provided
            if audience and payload.get("aud") != audience:
                raise TokenValidationError(f"Invalid audience. Expected: {audience}, Got: {payload.get('aud')}")
            
            # Extract user information
            user_id = payload.get("sub") or payload.get("user_id")
            if not user_id:
                raise TokenValidationError("Token missing user ID (sub or user_id claim)")
            
            logger.debug(f"Custom JWT verified for user {user_id}")
            
            return {
                "user_id": user_id,
                "email": payload.get("email"),
                "role": payload.get("role", "authenticated"),
                "user_metadata": payload.get("user_metadata", {}),
                "app_metadata": payload.get("app_metadata", {}),
                "aud": payload.get("aud"),
                "iss": payload.get("iss"),
                "exp": payload.get("exp"),
                "iat": payload.get("iat"),
                "token_type": "custom_jwt"
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Custom JWT token has expired")
            raise TokenValidationError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid custom JWT token: {e}")
            raise TokenValidationError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Custom JWT verification failed: {e}")
            raise TokenValidationError(f"Token verification failed: {e}")
    
    def create_supabase_jwt_token(
        self,
        user_id: str,
        expires_in_hours: int = 1,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a Supabase-compatible JWT token for MCP calls.
        
        This token is signed with Supabase's JWT secret and can be used
        to authenticate with Supabase MCP tools while maintaining RLS.
        
        Args:
            user_id: User ID to include in the token
            expires_in_hours: Token expiration time in hours (default: 1 hour for security)
            additional_claims: Additional claims to include in the token
            
        Returns:
            Supabase-compatible JWT token string
            
        Raises:
            TokenValidationError: If Supabase configuration is not available
        """
        try:
            # Get Supabase JWT secret from configuration
            supabase_jwt_secret = self.config.supabase.service_role_key
            
            if not supabase_jwt_secret:
                raise TokenValidationError("Supabase JWT secret not configured")
            
            now = datetime.now(timezone.utc)
            payload = {
                "sub": user_id,
                "role": "authenticated",
                "iat": now,
                "exp": now + timedelta(hours=expires_in_hours),
                "aud": "authenticated",
                "iss": f"https://{self.config.supabase.url.split('//')[1].split('.')[0]}.supabase.co/auth/v1"
            }
            
            # Add any additional claims
            if additional_claims:
                payload.update(additional_claims)
            
            # Sign with Supabase JWT secret
            token = jwt.encode(payload, supabase_jwt_secret, algorithm="HS256")
            
            logger.info(f"Created Supabase JWT token for user {user_id} (expires in {expires_in_hours}h)")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create Supabase JWT token: {e}")
            raise TokenValidationError(f"Supabase JWT token creation failed: {e}")
    
    def refresh_custom_jwt_token(
        self,
        old_token: str,
        secret: str = None,
        issuer: str = None,
        audience: str = None,
        expires_in_hours: int = 24
    ) -> str:
        """
        Refresh a custom JWT token by creating a new one with the same user data.
        
        Args:
            old_token: The expired or soon-to-expire token
            secret: Secret key for signing (uses config if not provided)
            issuer: Token issuer (uses config if not provided)
            audience: Token audience (uses config if not provided)
            expires_in_hours: New token expiration time in hours
            
        Returns:
            New signed JWT token string
            
        Raises:
            TokenValidationError: If old token is invalid or cannot be decoded
        """
        try:
            # Decode the old token without verification to extract user data
            # We'll use a more lenient approach for refresh
            old_payload = jwt.decode(
                old_token, 
                options={"verify_signature": False, "verify_exp": False}
            )
            
            # Extract user information
            user_id = old_payload.get("sub") or old_payload.get("user_id")
            if not user_id:
                raise TokenValidationError("Old token missing user ID")
            
            email = old_payload.get("email", "")
            user_metadata = old_payload.get("user_metadata", {})
            
            # Create new token with same user data
            new_token = self.create_custom_jwt_token(
                user_id=user_id,
                email=email,
                secret=secret,
                issuer=issuer,
                audience=audience,
                expires_in_hours=expires_in_hours,
                user_metadata=user_metadata
            )
            
            logger.info(f"Refreshed JWT token for user {user_id}")
            return new_token
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token for refresh: {e}")
            raise TokenValidationError(f"Cannot refresh invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise TokenValidationError(f"Token refresh failed: {e}")
    
    def get_supabase_jwt_for_mcp(self, user_id: str) -> str:
        """
        Get a Supabase JWT token for MCP tool calls.
        
        This is a convenience method that creates a short-lived Supabase JWT
        token specifically for MCP tool authentication.
        
        Args:
            user_id: User ID to include in the token
            
        Returns:
            Supabase JWT token for MCP calls
        """
        return self.create_supabase_jwt_token(
            user_id=user_id,
            expires_in_hours=1,  # Short expiration for security
            additional_claims={
                "purpose": "mcp_tool_call",
                "created_by": "fastapi_server"
            }
        )
    
    def extract_token_from_header(self, authorization_header: str) -> str:
        """
        Extract JWT token from Authorization header.
        
        Args:
            authorization_header: HTTP Authorization header value.
            
        Returns:
            Extracted JWT token.
            
        Raises:
            TokenValidationError: If header format is invalid.
        """
        if not authorization_header:
            raise TokenValidationError("Missing Authorization header")
        
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise TokenValidationError("Invalid Authorization header format. Expected: Bearer <token>")
        
        return parts[1]
    
    async def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate a user with email and password.
        
        Args:
            email: User's email address.
            password: User's password.
            
        Returns:
            Authentication result with user data and session.
            
        Raises:
            AuthenticationError: If authentication fails.
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user and response.session:
                logger.info(f"User authenticated successfully: {email}")
                return {
                    "user": response.user,
                    "session": response.session,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                }
            else:
                raise AuthenticationError("Authentication failed: Invalid credentials")
                
        except AuthApiError as e:
            logger.warning(f"Authentication failed for {email}: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"Authentication error: {e}")
    
    async def register_user(
        self, 
        email: str, 
        password: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            email: User's email address.
            password: User's password.
            metadata: Additional user metadata.
            
        Returns:
            Registration result with user data.
            
        Raises:
            AuthenticationError: If registration fails.
        """
        try:
            user_data = {
                "email": email,
                "password": password,
            }
            
            if metadata:
                user_data["data"] = metadata
            
            response = self.client.auth.sign_up(user_data)
            
            if response.user:
                logger.info(f"User registered successfully: {email}")
                return {
                    "user": response.user,
                    "session": response.session,
                }
            else:
                raise AuthenticationError("Registration failed")
                
        except AuthApiError as e:
            logger.warning(f"Registration failed for {email}: {e}")
            raise AuthenticationError(f"Registration failed: {e}")
        except Exception as e:
            logger.error(f"Registration error: {e}")
            raise AuthenticationError(f"Registration error: {e}")
    
    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh a user session using a refresh token.
        
        Args:
            refresh_token: The refresh token.
            
        Returns:
            New session data.
            
        Raises:
            AuthenticationError: If session refresh fails.
        """
        try:
            response = self.client.auth.refresh_session(refresh_token)
            
            if response.session:
                logger.info("Session refreshed successfully")
                return {
                    "session": response.session,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                }
            else:
                raise AuthenticationError("Session refresh failed")
                
        except AuthApiError as e:
            logger.warning(f"Session refresh failed: {e}")
            raise AuthenticationError(f"Session refresh failed: {e}")
        except Exception as e:
            logger.error(f"Session refresh error: {e}")
            raise AuthenticationError(f"Session refresh error: {e}")
    
    async def sign_out(self, access_token: str) -> bool:
        """
        Sign out a user and invalidate their session.
        
        Args:
            access_token: The user's access token.
            
        Returns:
            True if sign out was successful.
            
        Raises:
            AuthenticationError: If sign out fails.
        """
        try:
            self.client.auth.sign_out()
            logger.info("User signed out successfully")
            return True
            
        except Exception as e:
            logger.error(f"Sign out error: {e}")
            raise AuthenticationError(f"Sign out failed: {e}")
    
    async def get_user_profile(self, user_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """
        Get user profile data from the database.
        
        Args:
            user_id: User's unique identifier.
            
        Returns:
            User profile data or None if not found.
            
        Raises:
            DatabaseConnectionError: If database query fails.
        """
        try:
            query = """
                SELECT id, email, created_at, updated_at, metadata, role
                FROM auth.users 
                WHERE id = $1
            """
            
            result = await self.db_client.execute_query(
                query, 
                str(user_id), 
                fetch_one=True
            )
            
            if result:
                return dict(result)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            raise DatabaseConnectionError(f"Failed to get user profile: {e}")
    
    async def update_user_metadata(
        self, 
        user_id: Union[str, UUID], 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update user metadata in the database.
        
        Args:
            user_id: User's unique identifier.
            metadata: Metadata to update.
            
        Returns:
            True if update was successful.
            
        Raises:
            DatabaseConnectionError: If database update fails.
        """
        try:
            query = """
                UPDATE auth.users 
                SET raw_user_meta_data = raw_user_meta_data || $2,
                    updated_at = NOW()
                WHERE id = $1
            """
            
            await self.db_client.execute_query(
                query, 
                str(user_id), 
                metadata,
                fetch_all=False
            )
            
            logger.info(f"User metadata updated for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user metadata: {e}")
            raise DatabaseConnectionError(f"Failed to update user metadata: {e}")
    
    def extract_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Extract user information from a Supabase JWT token.
        
        Args:
            token: Supabase JWT token to extract user from.
            
        Returns:
            User information or None if extraction fails.
        """
        try:
            user_data = self.verify_supabase_jwt(token)
            return user_data
        except TokenValidationError:
            logger.debug("Failed to extract user from token")
            return None


# Global authentication instance
auth_manager: Optional[SupabaseAuth] = None


def get_auth_manager() -> SupabaseAuth:
    """
    Get the global authentication manager instance.
    
    Returns:
        SupabaseAuth: The authentication manager instance.
    """
    global auth_manager
    if auth_manager is None:
        auth_manager = SupabaseAuth()
    return auth_manager


# FastAPI Authentication Dependencies
async def verify_jwt_token(authorization: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to verify custom JWT token from Authorization header.
    
    Only supports custom JWT tokens from NextJS (no Supabase JWT support).
    
    Args:
        authorization: Authorization header value (Bearer <token>)
        
    Returns:
        User data if token is valid, None if no token provided
        
    Raises:
        HTTPException: If token is invalid
    """
    from fastapi import HTTPException, status
    
    if not authorization:
        return None
    
    try:
        auth_manager = get_auth_manager()
        token = auth_manager.extract_token_from_header(authorization)
        
        # Get custom JWT configuration
        config = auth_manager.config
        if not hasattr(config, 'custom_jwt') or not config.custom_jwt or not config.custom_jwt.secret:
            raise TokenValidationError("Custom JWT configuration not available")
        
        # Verify custom JWT token
        user_data = auth_manager.verify_custom_jwt(
            token=token,
            secret=config.custom_jwt.secret,
            issuer=config.custom_jwt.issuer,
            audience=config.custom_jwt.audience
        )
        user_data["token_type"] = "custom_jwt"
        return user_data
                
    except TokenValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during JWT verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )


async def get_current_user(authorization: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to get current authenticated user.
    
    This is an optional dependency - endpoints can work with or without authentication.
    
    Args:
        authorization: Authorization header value (Bearer <token>)
        
    Returns:
        User data if authenticated, None if no authentication provided
    """
    try:
        return await verify_jwt_token(authorization)
    except HTTPException:
        # Re-raise HTTP exceptions (invalid tokens)
        raise
    except Exception:
        # For other exceptions, return None (no authentication)
        return None


async def require_authentication(authorization: Optional[str] = None) -> Dict[str, Any]:
    """
    FastAPI dependency that requires authentication.
    
    Args:
        authorization: Authorization header value (Bearer <token>)
        
    Returns:
        User data (guaranteed to be valid)
        
    Raises:
        HTTPException: If no token or invalid token
    """
    from fastapi import HTTPException, status
    
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    user_data = await verify_jwt_token(authorization)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return user_data 