"""
Central tool registry for database operations.

Provides a unified interface for accessing all available tools
and their schemas for the execution planning node.
"""

from typing import Dict, Type
from .base import ToolInterface
from .data_fetch import DataFetchTool
from .database_operations_tool import database_operations_tool
from .advice_tool import advice_tool


# Initialize tool instances
_data_fetch_tool = DataFetchTool()


# Tool registry mapping tool names to tool instances
TOOL_REGISTRY: Dict[str, ToolInterface] = {
    "data_fetch": _data_fetch_tool,
    "database_operations": database_operations_tool,
    "advice": advice_tool,
}


# Schema registry for quick access to tool schemas
TOOL_SCHEMAS: Dict[str, Type] = {
    "data_fetch": _data_fetch_tool.schema,
    "database_operations": database_operations_tool.schema,
    "advice": advice_tool.schema,
}


def get_tool(tool_name: str) -> ToolInterface:
    """
    Get a tool instance by name.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        Tool instance implementing ToolInterface
        
    Raises:
        ValueError: If tool name is not found in registry
    """
    if tool_name not in TOOL_REGISTRY:
        available_tools = list(TOOL_REGISTRY.keys())
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
    
    return TOOL_REGISTRY[tool_name]


def get_tool_schema(tool_name: str) -> Type:
    """
    Get a tool's parameter schema by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Pydantic model class for the tool's parameters
        
    Raises:
        ValueError: If tool name is not found in registry
    """
    if tool_name not in TOOL_SCHEMAS:
        available_tools = list(TOOL_SCHEMAS.keys())
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
    
    return TOOL_SCHEMAS[tool_name]


def list_available_tools() -> list[str]:
    """
    Get a list of all available tool names.
    
    Returns:
        List of tool names available in the registry
    """
    return list(TOOL_REGISTRY.keys())


def get_tool_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about all available tools.
    
    Returns:
        Dictionary with tool names as keys and info dictionaries as values
    """
    return {
        "data_fetch": {
            "name": "data_fetch",
            "description": "Schema-based data extraction for financial data from request payload",
            "operations": [
                # Basic data retrieval operations
                "get_assets", "get_liabilities", "get_savings", "get_income", 
                "get_expenses", "get_stocks", "get_insurance", "get_goals",
                "get_net_worth", "get_dashboard_metrics", "get_user_info"
            ]
        },
        "database_operations": {
            "name": "database_operations",
            "description": "Database operations for creating, updating, deleting, and transferring financial entities",
            "operations": [
                "create",
                "update", 
                "delete",
                "transfer"
            ]
        },
        "advice": {
            "name": "advice",
            "description": "Provide personalized financial advice based on user query and financial context",
            "operations": [
                "generate_advice"
            ]
        }
    } 