"""
Node Execution Logger for LangGraph Intent Orchestration System.

This module provides comprehensive logging utilities for tracking node execution
with timestamps, execution time, and structured metadata for performance monitoring
and debugging.

Features:
- Automatic execution time tracking
- Structured logging with timestamps
- Node entry/exit logging
- Error tracking with context
- Performance metrics collection
- Session and correlation ID tracking
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union, Callable, Awaitable
from functools import wraps
import uuid

logger = logging.getLogger(__name__)


class NodeExecutionMetrics:
    """Container for node execution metrics and timing information."""
    
    def __init__(self, node_name: str, session_id: str):
        self.node_name = node_name
        self.session_id = session_id
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_time_seconds: float = 0.0
        self.start_timestamp: Optional[float] = None
        self.end_timestamp: Optional[float] = None
        self.correlation_id: str = str(uuid.uuid4())[:8]
        self.success: bool = False
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def start_execution(self):
        """Mark the start of node execution."""
        self.start_time = datetime.now(timezone.utc)
        self.start_timestamp = time.time()
        
        logger.info(
            f"🚀 NODE_START: {self.node_name}",
            extra={
                "node_name": self.node_name,
                "session_id": self.session_id,
                "correlation_id": self.correlation_id,
                "start_time": self.start_time.isoformat(),
                "start_timestamp": self.start_timestamp,
                "event_type": "node_start"
            }
        )
    
    def end_execution(self, success: bool = True, error: Optional[str] = None, 
                     error_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Mark the end of node execution with results."""
        self.end_time = datetime.now(timezone.utc)
        self.end_timestamp = time.time()
        self.success = success
        self.error = error
        self.error_type = error_type
        
        if metadata:
            self.metadata.update(metadata)
        
        # Calculate execution time
        if self.start_timestamp is not None:
            self.execution_time_seconds = self.end_timestamp - self.start_timestamp
        
        # Log completion
        log_level = logging.INFO if success else logging.ERROR
        status_emoji = "✅" if success else "❌"
        
        logger.log(
            log_level,
            f"{status_emoji} NODE_END: {self.node_name} | "
            f"Time: {self.execution_time_seconds:.3f}s | "
            f"Status: {'SUCCESS' if success else 'ERROR'}",
            extra={
                "node_name": self.node_name,
                "session_id": self.session_id,
                "correlation_id": self.correlation_id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat(),
                "execution_time_seconds": self.execution_time_seconds,
                "success": success,
                "error": error,
                "error_type": error_type,
                "metadata": self.metadata,
                "event_type": "node_end"
            }
        )
        
        # Log detailed performance metrics
        logger.debug(
            f"📊 NODE_METRICS: {self.node_name} | "
            f"Execution Time: {self.execution_time_seconds:.3f}s | "
            f"Correlation ID: {self.correlation_id} | "
            f"Metadata: {self.metadata}",
            extra={
                "node_name": self.node_name,
                "session_id": self.session_id,
                "correlation_id": self.correlation_id,
                "execution_time_seconds": self.execution_time_seconds,
                "performance_metrics": {
                    "execution_time_seconds": self.execution_time_seconds,
                    "start_timestamp": self.start_timestamp,
                    "end_timestamp": self.end_timestamp,
                    "success": success
                },
                "event_type": "node_metrics"
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for state storage."""
        return {
            "node_name": self.node_name,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time_seconds": self.execution_time_seconds,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "success": self.success,
            "error": self.error,
            "error_type": self.error_type,
            "metadata": self.metadata
        }


class NodeExecutionLogger:
    """Main logger class for tracking node execution with comprehensive metrics."""
    
    def __init__(self, node_name: str, session_id: str):
        self.metrics = NodeExecutionMetrics(node_name, session_id)
    
    def start(self):
        """Start tracking node execution."""
        self.metrics.start_execution()
    
    def end(self, success: bool = True, error: Optional[str] = None, 
            error_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """End tracking and log results."""
        self.metrics.end_execution(success, error, error_type, metadata)
        return self.metrics.to_dict()
    
    def log_step(self, step_name: str, step_data: Optional[Dict[str, Any]] = None):
        """Log an intermediate step within the node execution."""
        step_metadata = {
            "step_name": step_name,
            "step_timestamp": time.time(),
            "step_time": datetime.now(timezone.utc).isoformat()
        }
        
        if step_data:
            step_metadata.update(step_data)
        
        logger.debug(
            f"🔧 NODE_STEP: {self.metrics.node_name} | Step: {step_name}",
            extra={
                "node_name": self.metrics.node_name,
                "session_id": self.metrics.session_id,
                "correlation_id": self.metrics.correlation_id,
                "step_name": step_name,
                "step_metadata": step_metadata,
                "event_type": "node_step"
            }
        )
        
        # Add step to metrics metadata
        if "steps" not in self.metrics.metadata:
            self.metrics.metadata["steps"] = []
        self.metrics.metadata["steps"].append(step_metadata)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context."""
        error_metadata = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_timestamp": time.time(),
            "error_time": datetime.now(timezone.utc).isoformat()
        }
        
        if context:
            error_metadata.update(context)
        
        logger.error(
            f"💥 NODE_ERROR: {self.metrics.node_name} | "
            f"Error: {type(error).__name__} | "
            f"Message: {str(error)}",
            extra={
                "node_name": self.metrics.node_name,
                "session_id": self.metrics.session_id,
                "correlation_id": self.metrics.correlation_id,
                "error_metadata": error_metadata,
                "event_type": "node_error"
            }
        )
        
        # Add error to metrics metadata
        if "errors" not in self.metrics.metadata:
            self.metrics.metadata["errors"] = []
        self.metrics.metadata["errors"].append(error_metadata)


@asynccontextmanager
async def track_node_execution(node_name: str, session_id: str):
    """
    Async context manager for tracking node execution with automatic timing.
    
    Usage:
        async with track_node_execution("my_node", session_id) as logger:
            # Node execution code here
            logger.log_step("processing_data")
            result = await some_operation()
            logger.log_step("completed_processing", {"result_count": len(result)})
    """
    execution_logger = NodeExecutionLogger(node_name, session_id)
    execution_logger.start()
    
    try:
        yield execution_logger
        execution_logger.end(success=True)
    except Exception as e:
        execution_logger.log_error(e)
        execution_logger.end(success=False, error=str(e), error_type=type(e).__name__)
        raise


def log_node_execution(node_name: str, session_id: str):
    """
    Decorator for automatically tracking node execution with timing and logging.
    
    Usage:
        @log_node_execution("my_node", "session_123")
        async def my_node_function(state: MainState) -> MainState:
            # Node implementation
            return updated_state
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract session_id from state if not provided
            actual_session_id = session_id
            if not actual_session_id and args:
                state = args[0]
                if hasattr(state, 'session_id'):
                    actual_session_id = state.session_id
            
            async with track_node_execution(node_name, actual_session_id) as exec_logger:
                try:
                    # Log function entry
                    exec_logger.log_step("function_entry", {
                        "function_name": func.__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    })
                    
                    # Execute the function
                    result = await func(*args, **kwargs)
                    
                    # Log function exit
                    exec_logger.log_step("function_exit", {
                        "function_name": func.__name__,
                        "result_type": type(result).__name__
                    })
                    
                    return result
                    
                except Exception as e:
                    exec_logger.log_error(e, {
                        "function_name": func.__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    })
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions
            actual_session_id = session_id
            if not actual_session_id and args:
                state = args[0]
                if hasattr(state, 'session_id'):
                    actual_session_id = state.session_id
            
            execution_logger = NodeExecutionLogger(node_name, actual_session_id)
            execution_logger.start()
            
            try:
                execution_logger.log_step("function_entry", {
                    "function_name": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })
                
                result = func(*args, **kwargs)
                
                execution_logger.log_step("function_exit", {
                    "function_name": func.__name__,
                    "result_type": type(result).__name__
                })
                
                execution_logger.end(success=True)
                return result
                
            except Exception as e:
                execution_logger.log_error(e, {
                    "function_name": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })
                execution_logger.end(success=False, error=str(e), error_type=type(e).__name__)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def create_execution_logger(node_name: str, session_id: str) -> NodeExecutionLogger:
    """
    Create a new execution logger instance for manual tracking.
    
    Usage:
        logger = create_execution_logger("my_node", session_id)
        logger.start()
        try:
            # Node execution code
            logger.log_step("processing")
            result = await some_operation()
            logger.end(success=True, metadata={"result_count": len(result)})
        except Exception as e:
            logger.end(success=False, error=str(e), error_type=type(e).__name__)
    """
    return NodeExecutionLogger(node_name, session_id)


def add_execution_metrics_to_state(state: Any, execution_metrics: Dict[str, Any]) -> Any:
    """
    Add execution metrics to state metadata.
    
    Args:
        state: The state object to update
        execution_metrics: Metrics from NodeExecutionLogger
        
    Returns:
        Updated state with execution metrics in metadata
    """
    if hasattr(state, 'model_copy'):
        # Pydantic model
        return state.model_copy(update={
            "metadata": {
                **state.metadata,
                "execution_metrics": execution_metrics
            }
        })
    elif hasattr(state, 'metadata'):
        # Dictionary-like state
        state.metadata = {
            **state.metadata,
            "execution_metrics": execution_metrics
        }
        return state
    else:
        # Fallback - try to add to existing metadata
        if not hasattr(state, 'metadata'):
            state.metadata = {}
        state.metadata["execution_metrics"] = execution_metrics
        return state
