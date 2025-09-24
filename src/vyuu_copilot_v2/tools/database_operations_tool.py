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
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import ToolResponse, ToolInterface
from vyuu_copilot_v2.utils.database import get_db_client
from vyuu_copilot_v2.schemas.generated_intent_schemas import DatabaseOperationsParams

logger = logging.getLogger(__name__)


# DatabaseOperationsParams is now imported from generated_intent_schemas


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
            
            # Check for required parameters before execution
            missing_params = self._validate_required_parameters(validated_params)
            if missing_params:
                execution_time = (time.time() - start_time) * 1000
                return {
                    "success": False,
                    "error": f"More data required to do this operation. Missing required parameters: {', '.join(missing_params)}",
                    "tool_name": self.name,
                    "execution_time_ms": execution_time,
                    "missing_parameters": missing_params
                }
            
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
        
        # Add timestamps (timezone-naive for database compatibility)
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
        
        # Add updated timestamp (timezone-naive for database compatibility)
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
            
            now = datetime.now()
            
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
            'stock': 'investments',  # Fixed: stocks table doesn't exist, use investments
            'stocks': 'investments',  # Fixed: stocks table doesn't exist, use investments
            'investment': 'investments',  # Added: direct mapping for investments
            'investments': 'investments',  # Added: direct mapping for investments
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

        # Entity-specific field mappings (override general mappings)
        entity_specific_mappings = {
            'investment': {
                'start_date': 'purchaseDate',  # For investments, start_date maps to purchaseDate
                'end_date': None,  # For investments, end_date is not used (exclude from insert)
            },
            'stock': {
                'start_date': 'purchaseDate',  # For stocks, start_date maps to purchaseDate
                'end_date': None,  # For stocks, end_date is not used (exclude from insert)
            }
        }

        # Date fields that need to be converted to datetime objects
        date_fields = {'purchase_date', 'start_date', 'end_date', 'target_date', 'maturity_date'}

        # Add all non-None fields with proper column name mapping
        for field_name, field_value in params.model_dump(exclude_none=True).items():
            if field_name not in ['action_type', 'entity_type', 'entity_id', 'from_entity_id', 'to_entity_id', 'transfer_amount', 'user_id']:
                # Get entity-specific mapping first, then fall back to general mapping
                db_column_name = field_name
                
                # Check entity-specific mappings
                if params.entity_type in entity_specific_mappings:
                    entity_mapping = entity_specific_mappings[params.entity_type]
                    if field_name in entity_mapping:
                        db_column_name = entity_mapping[field_name]
                        # If mapped to None, skip this field entirely
                        if db_column_name is None:
                            self.logger.info(f"Skipping {field_name} (mapped to None)")
                            continue
                    elif field_name in field_mapping:
                        db_column_name = field_mapping[field_name]
                else:
                    # Use general mapping
                    db_column_name = field_mapping.get(field_name, field_name)
                
                self.logger.info(f"Processing field {field_name} -> {db_column_name}, value: {field_value}, type: {type(field_value)}")
                
                # Convert date strings/datetime objects to timezone-aware datetime objects
                if field_name in date_fields:
                    try:
                        from datetime import datetime, timezone
                        
                        if isinstance(field_value, str):
                            # Parse the date string
                            if field_value.endswith('Z'):
                                # Handle UTC timezone
                                dt = datetime.fromisoformat(field_value.replace('Z', '+00:00'))
                            elif '+' in field_value or field_value.endswith('00:00'):
                                # Handle timezone-aware strings
                                dt = datetime.fromisoformat(field_value)
                            else:
                                # Handle timezone-naive strings (keep timezone-naive for database compatibility)
                                dt = datetime.fromisoformat(field_value)
                        elif isinstance(field_value, datetime):
                            # Handle datetime objects
                            if field_value.tzinfo is None:
                                # Keep timezone-naive datetime as is (database expects this)
                                dt = field_value
                            else:
                                # Convert timezone-aware to timezone-naive (database expects this)
                                dt = field_value.replace(tzinfo=None)
                        else:
                            # Not a string or datetime, keep as is
                            dt = field_value
                        
                        self.logger.info(f"Converted {field_name} to timezone-aware datetime: {dt}")
                        entity_data[db_column_name] = dt
                    except (ValueError, TypeError) as e:
                        # If parsing fails, keep the original value
                        self.logger.warning(f"Failed to parse date {field_name}: {field_value}, error: {e}")
                        entity_data[db_column_name] = field_value
                else:
                    self.logger.info(f"Not a date field or not a string: {field_name} in date_fields: {field_name in date_fields}, is string: {isinstance(field_value, str)}")
                    entity_data[db_column_name] = field_value

        # Calculate returns for investments if not provided but amount and current_value are available
        if params.entity_type in ['investment', 'stock'] and 'returns' not in entity_data:
            if 'amount' in entity_data and 'currentValue' in entity_data:
                amount = entity_data['amount']
                current_value = entity_data['currentValue']
                if amount > 0:
                    returns_percentage = ((current_value - amount) / amount) * 100
                    entity_data['returns'] = round(returns_percentage, 2)
                    self.logger.info(f"Calculated returns: {returns_percentage:.2f}% for investment")
        
        return entity_data
    
    def _validate_required_parameters(self, params: DatabaseOperationsParams) -> List[str]:
        """
        Validate that all required parameters are present for the given operation.
        
        Args:
            params: Validated parameters
            
        Returns:
            List of missing required parameter names
        """
        missing_params = []
        
        # Define required parameters for each entity type and operation
        required_params = {
            "savings": {
                "create": ["name", "type", "current_balance", "interest_rate", "monthly_contribution"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            },
            "assets": {
                "create": ["name", "category", "current_value"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            },
            "liabilities": {
                "create": ["name", "type", "amount"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            },
            "income": {
                "create": ["name", "source", "amount"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            },
            "expense": {
                "create": ["name", "category", "amount"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            },
            "stock": {
                "create": ["name", "current_value"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            },
            "insurance": {
                "create": ["name", "type", "premium", "coverage"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            },
            "goal": {
                "create": ["name", "target_amount", "target_date"],
                "update": ["entity_id"],
                "delete": ["entity_id"],
                "transfer": ["from_entity_id", "to_entity_id", "transfer_amount"]
            }
        }
        
        # Get required parameters for this entity type and action
        entity_requirements = required_params.get(params.entity_type, {})
        action_requirements = entity_requirements.get(params.action_type, [])
        
        # Check each required parameter
        for param_name in action_requirements:
            param_value = getattr(params, param_name, None)
            if param_value is None or (isinstance(param_value, str) and param_value.strip() == ""):
                missing_params.append(param_name)
        
        # Special validation for user_id - it should be provided in most cases
        if not params.user_id and params.action_type in ["create", "update"]:
            missing_params.append("user_id")
        
        return missing_params


# Create tool instance
database_operations_tool = DatabaseOperationsTool()
