"""
FastAPI Application for Vyuu Copilot v2 - LangGraph Intent Orchestration System.

This module provides REST API endpoints for chatbot UI integration.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

from vyuu_copilot_v2.orchestrator import MainOrchestrator
from vyuu_copilot_v2.config.settings import AppConfig, get_config
from vyuu_copilot_v2.utils.auth import verify_jwt_token, get_current_user, get_auth_manager, TokenValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Optional[MainOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global orchestrator
    
    # Startup
    logger.info("Starting Vyuu Copilot v2 API server...")
    try:
        config = get_config()
        orchestrator = MainOrchestrator(use_database=False)
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Vyuu Copilot v2 API server...")


# Create FastAPI application
app = FastAPI(
    title="Vyuu Copilot v2 API",
    description="LangGraph-based intent orchestration system for financial assistance",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins if hasattr(config.api, 'cors_origins') else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Cache-Control",
        "Connection",
        "Last-Event-ID",
        "X-Accel-Buffering"  # For SSE compatibility
    ],
    expose_headers=[
        "Content-Type",
        "Cache-Control",
        "Connection",
        "X-Accel-Buffering"
    ]
)


# Pydantic Models for API
class ChatMessage(BaseModel):
    """Single chat message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list, 
        description="Previous conversation messages"
    )
    user_id: Optional[str] = Field(None, description="User ID for personalization", alias="userId")
    financial_data: Optional[Dict[str, Any]] = Field(
        None, 
        description="Financial data from NextJS for schema-based extraction",
        alias="financialData"
    )
    
    @validator("message")
    def validate_message(cls, v: str) -> str:
        """Validate message content."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID")
    status: str = Field(..., description="Response status: 'success' or 'error'")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list, 
        description="Updated conversation history"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional response metadata"
    )
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    orchestrator_status: str = Field(..., description="Orchestrator status")


class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""
    token: str = Field(..., description="Current JWT token to refresh")
    
    @validator("token")
    def validate_token(cls, v: str) -> str:
        """Validate token format."""
        if not v or len(v) < 10:
            raise ValueError("Token must be at least 10 characters long")
        return v.strip()


class TokenRefreshResponse(BaseModel):
    """Token refresh response model."""
    new_token: str = Field(..., description="New JWT token")
    expires_in_hours: int = Field(..., description="Token expiration time in hours")
    user_id: str = Field(..., description="User ID from the token")
    status: str = Field(..., description="Response status")


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Vyuu Copilot v2 API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global orchestrator
    
    orchestrator_status = "healthy" if orchestrator is not None else "unhealthy"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="0.1.0",
        orchestrator_status=orchestrator_status
    )


@app.post("/auth/refresh", response_model=TokenRefreshResponse)
async def refresh_token(request: TokenRefreshRequest):
    """
    Refresh JWT token endpoint.
    
    This endpoint allows clients to refresh expired or soon-to-expire JWT tokens
    by providing the current token and receiving a new one with extended expiration.
    """
    try:
        auth_manager = get_auth_manager()
        
        # Refresh the token
        new_token = auth_manager.refresh_custom_jwt_token(
            old_token=request.token,
            expires_in_hours=24  # Default to 24 hours
        )
        
        # Decode the new token to get user information
        import jwt
        from vyuu_copilot_v2.config.settings import get_config
        
        config = get_config()
        payload = jwt.decode(
            new_token, 
            config.custom_jwt.secret, 
            algorithms=["HS256"],
            options={"verify_exp": False}  # Don't verify expiration for this check
        )
        
        user_id = payload.get("sub") or payload.get("user_id")
        
        logger.info(f"Token refreshed successfully for user {user_id}")
        
        return TokenRefreshResponse(
            new_token=new_token,
            expires_in_hours=24,
            user_id=user_id,
            status="success"
        )
        
    except TokenValidationError as e:
        logger.warning(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during token refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Main chat endpoint for chatbot UI integration.
    
    This endpoint processes user messages through the LangGraph orchestrator
    and returns structured responses suitable for chatbot interfaces.
    """
    global orchestrator
    
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    
    start_time = time.time()
    
    try:
        # Extract user ID from JWT token if available
        user_id = current_user.get("user_id") if current_user else request.user_id
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid4())
        
        # Convert conversation history to the format expected by orchestrator
        conversation_history = []
        if request.conversation_history:
            for msg in request.conversation_history:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                })
        
        logger.info(f"Processing chat request for session {session_id[:8]}...")
        
        # Log the incoming request for debugging
        logger.info(f"Request details - Message: {request.message[:100]}...")
        logger.info(f"Request details - Session ID: {request.session_id}")
        logger.info(f"Request details - User ID: {user_id}")
        logger.info(f"Request details - Financial data present: {request.financial_data is not None}")
        if request.financial_data:
            logger.info(f"Request details - Financial data keys: {list(request.financial_data.keys())}")
        else:
            logger.warning("Request details - No financial data provided in request!")
        
        # Process the message through the orchestrator
        result = await orchestrator.process_user_message(
            user_input=request.message,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history,
            financial_data=request.financial_data
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Convert conversation history back to ChatMessage format
        response_history = []
        if result.get("conversation_history"):
            for msg in result["conversation_history"]:
                timestamp = msg.get("timestamp")
                # Convert datetime to ISO string if needed
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                
                response_history.append(ChatMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    timestamp=timestamp
                ))
        
        # Add the assistant's response to history
        response_history.append(ChatMessage(
            role="assistant",
            content=result.get("response", ""),
            timestamp=datetime.now(timezone.utc).isoformat()
        ))
        
        return ChatResponse(
            response=result.get("response", "I processed your request but didn't get a specific response."),
            session_id=result.get("session_id", session_id),
            status=result.get("status", "success"),
            conversation_history=response_history,
            metadata={
                **result.get("metadata", {}),
                "api_version": "0.1.0",
                "processing_time_ms": processing_time
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        processing_time = (time.time() - start_time) * 1000
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "error_code": "CHAT_PROCESSING_ERROR",
                "details": {
                    "message": str(e),
                    "processing_time_ms": processing_time
                }
            }
        )


@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Streaming chat endpoint for real-time response delivery.
    
    This endpoint provides Server-Sent Events (SSE) streaming for chatbot responses,
    allowing the frontend to receive incremental updates as the LangGraph processes
    the user's request. This solves timeout issues with long-running operations.
    """
    global orchestrator
    
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    
    async def generate_stream():
        """Generate Server-Sent Events stream."""
        try:
            # Extract user ID from JWT token if available
            user_id = current_user.get("user_id") if current_user else request.user_id
            
            # Generate session ID if not provided
            session_id = request.session_id or str(uuid4())
            
            # Convert conversation history to the format expected by orchestrator
            conversation_history = []
            if request.conversation_history:
                for msg in request.conversation_history:
                    conversation_history.append({
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp
                    })
            
            logger.info(f"Starting streaming chat for session {session_id[:8]}...")
            
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connection', 'session_id': session_id, 'status': 'connected'})}\n\n"
            
            # Stream the response using the orchestrator
            async for event in orchestrator.process_user_message_stream(
                user_input=request.message,
                user_id=user_id,
                session_id=session_id,
                conversation_history=conversation_history,
                financial_data=request.financial_data
            ):
                yield f"data: {json.dumps(event)}\n\n"
            
            # Send stream completion event
            yield f"data: {json.dumps({'type': 'stream_complete', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            # Send error event
            error_event = {
                'type': 'error',
                'message': f"Streaming error: {str(e)}",
                'error_code': 'STREAM_ERROR',
                'details': {'error': str(e)}
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Expose-Headers": "*"
        }
    )


class SimpleChatRequest(BaseModel):
    """Simple chat request model."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    
    @validator("message")
    def validate_message(cls, v: str) -> str:
        """Validate message content."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


@app.post("/chat/simple", response_model=Dict[str, Any])
async def simple_chat_endpoint(
    request: SimpleChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Simplified chat endpoint that accepts just a message string.
    
    This endpoint is useful for simple integrations that don't need
    full conversation history management.
    """
    # Create a ChatRequest from the simple parameters
    chat_request = ChatRequest(
        message=request.message,
        session_id=request.session_id,
        user_id=current_user.get("user_id") if current_user else None
    )
    
    # Use the main chat endpoint
    response = await chat_endpoint(chat_request, current_user)
    
    # Return simplified response
    return {
        "response": response.response,
        "session_id": response.session_id,
        "status": response.status
    }


@app.get("/sessions/{session_id}/history", response_model=List[ChatMessage])
async def get_session_history(
    session_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Get conversation history for a specific session.
    
    This endpoint allows the UI to retrieve previous conversation
    history for a given session.
    """
    global orchestrator
    
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    
    try:
        # Get session state from orchestrator
        session_state = orchestrator.get_session_state(session_id)
        
        if not session_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Convert to ChatMessage format
        history = []
        if session_state.get("conversation_history"):
            for msg in session_state["conversation_history"]:
                timestamp = msg.get("timestamp")
                # Convert datetime to ISO string if needed
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                
                history.append(ChatMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    timestamp=timestamp
                ))
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history"
        )


@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Clear conversation history for a specific session.
    
    This endpoint allows the UI to reset a conversation session.
    """
    global orchestrator
    
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    
    try:
        success = orchestrator.clear_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {"message": "Session cleared successfully", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear session"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail.get("error", "Unknown error") if isinstance(exc.detail, dict) else str(exc.detail),
            error_code=exc.detail.get("error_code", "HTTP_ERROR") if isinstance(exc.detail, dict) else "HTTP_ERROR",
            details=exc.detail.get("details") if isinstance(exc.detail, dict) else None
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured error responses."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"message": str(exc)}
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    from datetime import datetime, timezone
    
    # Get configuration
    config = get_config()
    
    # Run the server
    uvicorn.run(
        "src.api:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )
