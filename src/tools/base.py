"""
Base interfaces and protocols for database tools.
"""

from typing import Protocol, Dict, Any, Type
from pydantic import BaseModel


class ToolInterface(Protocol):
    """
    Protocol defining the interface for all database tools.
    
    All tools must implement:
    - name: Unique tool identifier
    - schema: Pydantic model for parameter validation
    - invoke: Async method to execute the tool with parameters
    """
    
    name: str
    schema: Type[BaseModel]
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Args:
            params: Dictionary of parameters matching the tool's schema
            
        Returns:
            Standardized response with success, data, and error fields
        """
        ...


class ToolResponse(BaseModel):
    """
    Standardized response format for all tools.
    """
    
    success: bool
    data: Any = None
    error: str | None = None
    tool_name: str
    execution_time_ms: float | None = None
    
    class Config:
        arbitrary_types_allowed = True 