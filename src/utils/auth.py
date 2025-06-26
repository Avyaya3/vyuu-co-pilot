"""
Authentication utilities for Supabase integration.

This module provides JWT handling, user authentication, session management,
and security utilities for the Supabase authentication system.
"""

import logging
from typing import Optional, Dict, Any, Union
from uuid import UUID
import asyncio

from supabase import Client
from gotrue.errors import AuthApiError, AuthRetryableError

from ..config import get_config
from .database import get_db_client, DatabaseConnectionError

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