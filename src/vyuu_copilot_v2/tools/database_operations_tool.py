"""
Database operations tool for create, update, delete operations using Supabase MCP.

This tool handles all database modification operations:
- Create new financial entities (assets, liabilities, savings, etc.)
- Update existing entities
- Delete entities
- Transfer operations between entities
"""

import time
import logging
from typing import Dict, Any, Literal, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from .base import ToolResponse, ToolInterface

logger = logging.getLogger(__name__)


class DatabaseOperationsParams(BaseModel):
    """
    Parameters for database operations (create, update, delete).
    """
    
    action_type: Literal[
        "create",
        "update", 
        "delete",
        "transfer"
    ] = Field(description="Type of database operation to perform")
    
    entity_type: Literal[
        "asset",
        "liability",
        "savings",
        "income",
        "expense",
        "stock",
        "insurance",
        "goal"
    ] = Field(description="Type of financial entity")
    
    # Entity identification
    entity_id: Optional[str] = Field(None, description="Entity ID for update/delete operations")
    
    # Common entity fields
    name: Optional[str] = Field(None, description="Name of the entity")
    description: Optional[str] = Field(None, description="Description of the entity")
    category: Optional[str] = Field(None, description="Category of the entity")
    subcategory: Optional[str] = Field(None, description="Subcategory of the entity")
    type: Optional[str] = Field(None, description="Type of the entity")
    provider: Optional[str] = Field(None, description="Provider of the entity")
    
    # Financial fields
    amount: Optional[int] = Field(None, description="Amount in cents", ge=0)
    current_value: Optional[int] = Field(None, description="Current value in cents", ge=0)
    purchase_value: Optional[int] = Field(None, description="Purchase value in cents", ge=0)
    target_amount: Optional[int] = Field(None, description="Target amount in cents", ge=0)
    current_balance: Optional[int] = Field(None, description="Current balance in cents", ge=0)
    premium: Optional[int] = Field(None, description="Premium amount in cents", ge=0)
    coverage: Optional[int] = Field(None, description="Coverage amount in cents", ge=0)
    emi: Optional[int] = Field(None, description="EMI amount in cents", ge=0)
    monthly_contribution: Optional[int] = Field(None, description="Monthly contribution in cents", ge=0)
    
    # Rate fields
    interest_rate: Optional[float] = Field(None, description="Interest rate percentage")
    returns: Optional[float] = Field(None, description="Returns percentage")
    
    # Date fields
    date: Optional[datetime] = Field(None, description="Date of the entity")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    target_date: Optional[datetime] = Field(None, description="Target date")
    purchase_date: Optional[datetime] = Field(None, description="Purchase date")
    maturity_date: Optional[datetime] = Field(None, description="Maturity date")
    
    # Specific fields
    source: Optional[str] = Field(None, description="Income source")
    frequency: Optional[str] = Field(None, description="Frequency (monthly, quarterly, yearly)")
    payment_method: Optional[str] = Field(None, description="Payment method")
    priority: Optional[str] = Field(None, description="Priority level")
    policy_number: Optional[str] = Field(None, description="Insurance policy number")
    
    # Transfer specific fields
    from_entity_id: Optional[str] = Field(None, description="Source entity ID for transfers")
    to_entity_id: Optional[str] = Field(None, description="Destination entity ID for transfers")
    transfer_amount: Optional[int] = Field(None, description="Transfer amount in cents", ge=0)
    
    # User context
    user_id: Optional[str] = Field(None, description="User ID for the operation")


class DatabaseOperationsTool:
    """
    Tool for performing database operations using Supabase MCP.
    """
    
    name: str = "database_operations"
    schema = DatabaseOperationsParams
    
    def __init__(self):
        """Initialize the database operations tool."""
        self.logger = logging.getLogger(__name__)
    
    async def invoke(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Execute database operation with the given parameters.
        
        Args:
            params: Dictionary of parameters matching DatabaseOperationsParams schema
            
        Returns:
            ToolResponse with operation result
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            validated_params = DatabaseOperationsParams(**params)
            self.logger.info(f"Executing {validated_params.action_type} operation on {validated_params.entity_type}")
            
            # Execute the appropriate operation
            if validated_params.action_type == "create":
                result = await self._create_entity(validated_params)
            elif validated_params.action_type == "update":
                result = await self._update_entity(validated_params)
            elif validated_params.action_type == "delete":
                result = await self._delete_entity(validated_params)
            elif validated_params.action_type == "transfer":
                result = await self._transfer_entity(validated_params)
            else:
                raise ValueError(f"Unsupported action type: {validated_params.action_type}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResponse(
                success=True,
                data=result,
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Database operation failed: {e}")
            
            return ToolResponse(
                success=False,
                error=str(e),
                tool_name=self.name,
                execution_time_ms=execution_time
            )
    
    async def _create_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Create a new entity in the database."""
        # TODO: Implement Supabase MCP create operation
        # This will use the Supabase MCP to insert a new record
        # based on the entity_type and provided parameters
        
        entity_data = self._build_entity_data(params)
        
        # Placeholder for actual MCP call
        self.logger.info(f"Creating {params.entity_type} entity: {entity_data}")
        
        return {
            "action": "create",
            "entity_type": params.entity_type,
            "entity_data": entity_data,
            "message": f"Successfully created {params.entity_type} entity"
        }
    
    async def _update_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Update an existing entity in the database."""
        if not params.entity_id:
            raise ValueError("entity_id is required for update operations")
        
        # TODO: Implement Supabase MCP update operation
        entity_data = self._build_entity_data(params)
        
        # Placeholder for actual MCP call
        self.logger.info(f"Updating {params.entity_type} entity {params.entity_id}: {entity_data}")
        
        return {
            "action": "update",
            "entity_type": params.entity_type,
            "entity_id": params.entity_id,
            "entity_data": entity_data,
            "message": f"Successfully updated {params.entity_type} entity {params.entity_id}"
        }
    
    async def _delete_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Delete an entity from the database."""
        if not params.entity_id:
            raise ValueError("entity_id is required for delete operations")
        
        # TODO: Implement Supabase MCP delete operation
        self.logger.info(f"Deleting {params.entity_type} entity {params.entity_id}")
        
        return {
            "action": "delete",
            "entity_type": params.entity_type,
            "entity_id": params.entity_id,
            "message": f"Successfully deleted {params.entity_type} entity {params.entity_id}"
        }
    
    async def _transfer_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Transfer amount between entities."""
        if not params.from_entity_id or not params.to_entity_id or not params.transfer_amount:
            raise ValueError("from_entity_id, to_entity_id, and transfer_amount are required for transfer operations")
        
        # TODO: Implement Supabase MCP transfer operation
        self.logger.info(f"Transferring {params.transfer_amount} from {params.from_entity_id} to {params.to_entity_id}")
        
        return {
            "action": "transfer",
            "from_entity_id": params.from_entity_id,
            "to_entity_id": params.to_entity_id,
            "transfer_amount": params.transfer_amount,
            "message": f"Successfully transferred {params.transfer_amount} cents from {params.from_entity_id} to {params.to_entity_id}"
        }
    
    def _build_entity_data(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Build entity data dictionary from parameters."""
        entity_data = {}
        
        # Add all non-None fields
        for field_name, field_value in params.model_dump(exclude_none=True).items():
            if field_name not in ['action_type', 'entity_type', 'entity_id', 'from_entity_id', 'to_entity_id', 'transfer_amount']:
                entity_data[field_name] = field_value
        
        return entity_data


# Create tool instance
database_operations_tool = DatabaseOperationsTool()
