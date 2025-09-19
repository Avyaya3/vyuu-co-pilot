"""
Central tool registry for database operations.

Provides a unified interface for accessing all available tools
and their schemas for the execution planning node.
"""

from typing import Dict, Type
from .base import ToolInterface
from .data_fetch import DataFetchTool
from .db_aggregate import DbAggregateTool
from .db_action import DbActionTool


# Initialize tool instances
_data_fetch_tool = DataFetchTool()
_db_aggregate_tool = DbAggregateTool()
_db_action_tool = DbActionTool()


# Tool registry mapping tool names to tool instances
TOOL_REGISTRY: Dict[str, ToolInterface] = {
    "data_fetch": _data_fetch_tool,
    "db_aggregate": _db_aggregate_tool,
    "db_action": _db_action_tool,
}


# Schema registry for quick access to tool schemas
TOOL_SCHEMAS: Dict[str, Type] = {
    "data_fetch": _data_fetch_tool.schema,
    "db_aggregate": _db_aggregate_tool.schema,
    "db_action": _db_action_tool.schema,
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
        "db_aggregate": {
            "name": "db_aggregate",
            "description": "Analytics and aggregation operations (spending analysis, summaries)",
            "operations": [
                "spending_by_category",
                "monthly_summary",
                "account_totals",
                "budget_analysis",
                "transaction_trends",
                "income_vs_expense"
            ]
        },
        "db_action": {
            "name": "db_action",
            "description": "Create, update, and transfer operations (transactions, accounts)",
            "operations": [
                "create_transaction",
                "transfer_money",
                "create_account",
                "update_transaction",
                "delete_transaction"
            ]
        }
    } 