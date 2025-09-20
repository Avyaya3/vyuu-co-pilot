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
from vyuu_copilot_v2.utils.database import get_db_client

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
        self.db_client = get_db_client()
    
    async def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute database operation with the given parameters.
        
        Args:
            params: Dictionary of parameters matching DatabaseOperationsParams schema
            
        Returns:
            Dictionary with operation result
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
            
            return {
                "success": True,
                "data": result,
                "tool_name": self.name,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Database operation failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.name,
                "execution_time_ms": execution_time
            }
    
    async def _create_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Create a new entity in the database."""
        entity_data = self._build_entity_data(params)
        
        # Generate a unique ID
        import uuid
        entity_data["id"] = str(uuid.uuid4())
        
        # Add user_id to entity data
        if params.user_id:
            entity_data["userId"] = params.user_id
        
        # Add timestamps
        now = datetime.now()
        entity_data["createdAt"] = now
        entity_data["updatedAt"] = now
        
        # Build INSERT query
        table_name = self._get_table_name(params.entity_type)
        columns = list(entity_data.keys())
        # Quote column names to preserve camelCase
        quoted_columns = [f'"{col}"' for col in columns]
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = list(entity_data.values())
        
        query = f"""
            INSERT INTO {table_name} ({', '.join(quoted_columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id, "createdAt"
        """
        
        try:
            # Execute the insert query
            result = await self.db_client.execute_query(query, *values, fetch_one=True)
            
            if result:
                self.logger.info(f"Successfully created {params.entity_type} entity with ID: {result['id']}")
                return {
                    "action": "create",
                    "entity_type": params.entity_type,
                    "entity_id": result["id"],
                    "entity_data": entity_data,
                    "created_at": result["createdAt"],
                    "message": f"Successfully created {params.entity_type} entity with ID {result['id']}"
                }
            else:
                raise Exception("Insert operation returned no result")
                
        except Exception as e:
            self.logger.error(f"Failed to create {params.entity_type} entity: {e}")
            raise Exception(f"Failed to create {params.entity_type} entity: {str(e)}")
    
    async def _update_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Update an existing entity in the database."""
        if not params.entity_id:
            raise ValueError("entity_id is required for update operations")
        
        entity_data = self._build_entity_data(params)
        
        # Add updated timestamp
        entity_data["updatedAt"] = datetime.now()
        
        # Build UPDATE query
        table_name = self._get_table_name(params.entity_type)
        set_clauses = []
        values = []
        
        for i, (key, value) in enumerate(entity_data.items()):
            set_clauses.append(f'"{key}" = ${i+1}')
            values.append(value)
        
        # Add entity_id as the last parameter
        values.append(params.entity_id)
        
        query = f"""
            UPDATE {table_name}
            SET {', '.join(set_clauses)}
            WHERE id = ${len(values)}
            RETURNING id, "updatedAt"
        """
        
        try:
            # Execute the update query
            result = await self.db_client.execute_query(query, *values, fetch_one=True)
            
            if result:
                self.logger.info(f"Successfully updated {params.entity_type} entity {params.entity_id}")
                return {
                    "action": "update",
                    "entity_type": params.entity_type,
                    "entity_id": params.entity_id,
                    "entity_data": entity_data,
                    "updated_at": result["updatedAt"],
                    "message": f"Successfully updated {params.entity_type} entity {params.entity_id}"
                }
            else:
                raise Exception(f"Entity with ID {params.entity_id} not found or no changes made")
                
        except Exception as e:
            self.logger.error(f"Failed to update {params.entity_type} entity {params.entity_id}: {e}")
            raise Exception(f"Failed to update {params.entity_type} entity {params.entity_id}: {str(e)}")
    
    async def _delete_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Delete an entity from the database."""
        if not params.entity_id:
            raise ValueError("entity_id is required for delete operations")
        
        table_name = self._get_table_name(params.entity_type)
        
        query = f"""
            DELETE FROM {table_name}
            WHERE id = $1
            RETURNING id
        """
        
        try:
            # Execute the delete query
            result = await self.db_client.execute_query(query, params.entity_id, fetch_one=True)
            
            if result:
                self.logger.info(f"Successfully deleted {params.entity_type} entity {params.entity_id}")
                return {
                    "action": "delete",
                    "entity_type": params.entity_type,
                    "entity_id": params.entity_id,
                    "message": f"Successfully deleted {params.entity_type} entity {params.entity_id}"
                }
            else:
                raise Exception(f"Entity with ID {params.entity_id} not found")
                
        except Exception as e:
            self.logger.error(f"Failed to delete {params.entity_type} entity {params.entity_id}: {e}")
            raise Exception(f"Failed to delete {params.entity_type} entity {params.entity_id}: {str(e)}")
    
    async def _transfer_entity(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Transfer amount between entities."""
        if not params.from_entity_id or not params.to_entity_id or not params.transfer_amount:
            raise ValueError("from_entity_id, to_entity_id, and transfer_amount are required for transfer operations")
        
        # This is a complex operation that would require:
        # 1. Check if source entity has sufficient balance
        # 2. Update source entity (decrease balance)
        # 3. Update destination entity (increase balance)
        # 4. Create a transaction record
        
        # For now, we'll implement a simple version that assumes both entities are savings accounts
        # In a real implementation, this would be more sophisticated
        
        try:
            # Start a transaction-like operation
            # First, get current balances
            from_query = f"""
                SELECT "currentBalance" FROM savings WHERE id = $1
            """
            to_query = f"""
                SELECT "currentBalance" FROM savings WHERE id = $1
            """
            
            from_balance = await self.db_client.execute_query(from_query, params.from_entity_id, fetch_one=True)
            to_balance = await self.db_client.execute_query(to_query, params.to_entity_id, fetch_one=True)
            
            if not from_balance or not to_balance:
                raise Exception("One or both entities not found")
            
            current_from_balance = from_balance["currentBalance"]
            current_to_balance = to_balance["currentBalance"]
            
            if current_from_balance < params.transfer_amount:
                raise Exception(f"Insufficient balance. Available: {current_from_balance}, Required: {params.transfer_amount}")
            
            # Update both entities
            new_from_balance = current_from_balance - params.transfer_amount
            new_to_balance = current_to_balance + params.transfer_amount
            
            update_from_query = f"""
                UPDATE savings 
                SET "currentBalance" = $1, "updatedAt" = $2
                WHERE id = $3
                RETURNING id
            """
            
            update_to_query = f"""
                UPDATE savings 
                SET "currentBalance" = $1, "updatedAt" = $2
                WHERE id = $3
                RETURNING id
            """
            
            now = datetime.now().isoformat()
            
            # Update source entity
            await self.db_client.execute_query(update_from_query, new_from_balance, now, params.from_entity_id, fetch_one=True)
            
            # Update destination entity
            await self.db_client.execute_query(update_to_query, new_to_balance, now, params.to_entity_id, fetch_one=True)
            
            self.logger.info(f"Successfully transferred {params.transfer_amount} from {params.from_entity_id} to {params.to_entity_id}")
            
            return {
                "action": "transfer",
                "from_entity_id": params.from_entity_id,
                "to_entity_id": params.to_entity_id,
                "transfer_amount": params.transfer_amount,
                "from_new_balance": new_from_balance,
                "to_new_balance": new_to_balance,
                "message": f"Successfully transferred {params.transfer_amount} cents from {params.from_entity_id} to {params.to_entity_id}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to transfer {params.transfer_amount} from {params.from_entity_id} to {params.to_entity_id}: {e}")
            raise Exception(f"Transfer failed: {str(e)}")
    
    def _get_table_name(self, entity_type: str) -> str:
        """Get the correct database table name for an entity type."""
        # Map entity types to their database table names
        table_mapping = {
            'asset': 'assets',
            'assets': 'assets',
            'savings': 'savings',
            'liability': 'liabilities',
            'liabilities': 'liabilities',
            'income': 'incomes',
            'incomes': 'incomes',
            'expense': 'expenses',
            'expenses': 'expenses',
            'goal': 'goals',
            'goals': 'goals',
            'stock': 'stocks',
            'stocks': 'stocks',
            'insurance': 'insurances',
            'insurances': 'insurances'
        }
        
        return table_mapping.get(entity_type.lower(), entity_type)
    
    def _build_entity_data(self, params: DatabaseOperationsParams) -> Dict[str, Any]:
        """Build entity data dictionary from parameters."""
        entity_data = {}

        # Field mapping from snake_case to camelCase for database columns
        field_mapping = {
            'target_amount': 'targetAmount',
            'current_balance': 'currentBalance',
            'monthly_contribution': 'monthlyContribution',
            'interest_rate': 'interestRate',
            'maturity_date': 'maturityDate',
            'purchase_value': 'purchaseValue',
            'current_value': 'currentValue',
            'purchase_date': 'purchaseDate',
            'start_date': 'startDate',
            'end_date': 'endDate',
            'target_date': 'targetDate',
            'policy_number': 'policyNumber',
            'payment_method': 'paymentMethod'
        }

        # Date fields that need to be converted to datetime objects
        date_fields = {'purchase_date', 'start_date', 'end_date', 'target_date', 'maturity_date'}

        # Add all non-None fields with proper column name mapping
        for field_name, field_value in params.model_dump(exclude_none=True).items():
            if field_name not in ['action_type', 'entity_type', 'entity_id', 'from_entity_id', 'to_entity_id', 'transfer_amount', 'user_id']:
                # Map snake_case to camelCase if needed
                db_column_name = field_mapping.get(field_name, field_name)
                
                # Convert date strings to datetime objects
                if field_name in date_fields and isinstance(field_value, str):
                    try:
                        from datetime import datetime
                        entity_data[db_column_name] = datetime.fromisoformat(field_value.replace('Z', '+00:00'))
                    except ValueError:
                        # If parsing fails, keep the original value
                        entity_data[db_column_name] = field_value
                else:
                    entity_data[db_column_name] = field_value

        return entity_data


# Create tool instance
database_operations_tool = DatabaseOperationsTool()
