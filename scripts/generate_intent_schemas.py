#!/usr/bin/env python3
"""
Auto-generate Pydantic schemas from YAML intent configuration.

This script reads the intent_parameters.yaml file and generates:
1. Individual parameter schemas (DataFetchParams, AggregateParams, etc.)
2. Updated IntentClassificationResult with all intent param fields
3. Constants and mappings for easy imports

Usage:
    python scripts/generate_intent_schemas.py

Output:
    src/schemas/generated_intent_schemas.py
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class SchemaGenerator:
    """Generates Pydantic schemas from YAML intent configuration."""
    
    def __init__(self, yaml_path: str = None):
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / "config" / "intent_parameters.yaml"
        
        self.yaml_path = Path(yaml_path)
        self.config = self._load_yaml()
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _python_type_from_yaml(self, type_str: str) -> str:
        """Convert YAML type string to Python type annotation."""
        type_mapping = {
            "str": "str",
            "int": "int", 
            "float": "float",
            "bool": "bool",
            "List[str]": "List[str]",
            "List[int]": "List[int]",
            "List[float]": "List[float]",
            "Dict[str, Any]": "Dict[str, Any]",
            "datetime": "datetime",
        }
        return f"Optional[{type_mapping.get(type_str, 'Any')}]"
    
    def _generate_field_definition(self, param: Dict[str, Any]) -> str:
        """Generate Pydantic Field definition for a parameter."""
        name = param["name"]
        type_hint = self._python_type_from_yaml(param.get("type", "str"))
        description = param.get("description", f"Parameter {name}")
        
        field_args = [f'None']  # Default value
        field_kwargs = [f'description="{description}"']
        
        # Add constraints if present
        if "constraints" in param:
            constraints = param["constraints"]
            for key, value in constraints.items():
                if key in ["ge", "le", "gt", "lt"]:
                    field_kwargs.append(f"{key}={value}")
                elif key == "min_length":
                    field_kwargs.append(f"min_length={value}")
                elif key == "max_length":
                    field_kwargs.append(f"max_length={value}")
        
        field_kwargs_str = ", ".join(field_kwargs)
        return f'    {name}: {type_hint} = Field({", ".join(field_args)}, {field_kwargs_str})'
    
    def _generate_param_class(self, intent: str, params_config: Dict[str, Any]) -> str:
        """Generate a Pydantic parameter class for an intent."""
        class_name = f"{intent.title().replace('_', '')}Params"
        
        fields = []
        
        # Add critical parameters
        critical_params = params_config.get("critical", [])
        if critical_params:
            fields.append("    # Critical parameters")
            for param in critical_params:
                if isinstance(param, str):
                    # Simple string format - create basic field
                    param_dict = {"name": param, "type": "str", "description": f"Critical parameter: {param}"}
                else:
                    # Enhanced format with type info
                    param_dict = param
                fields.append(self._generate_field_definition(param_dict))
        
        # Add optional parameters  
        optional_params = params_config.get("optional", [])
        if optional_params:
            if critical_params:
                fields.append("")
            fields.append("    # Optional parameters")
            for param in optional_params:
                if isinstance(param, str):
                    # Simple string format - create basic field
                    param_dict = {"name": param, "type": "str", "description": f"Optional parameter: {param}"}
                else:
                    # Enhanced format with type info
                    param_dict = param
                fields.append(self._generate_field_definition(param_dict))
        
        if not fields:
            fields = ["    pass  # No parameters defined"]
        
        description = params_config.get("description", f"Parameters for {intent} intent")
        
        return f'''class {class_name}(BaseModel):
    """
    {description}
    
    Auto-generated from intent_parameters.yaml
    """
    
{chr(10).join(fields)}
'''
    
    def _generate_classification_result_fields(self, intents: List[str]) -> str:
        """Generate fields for IntentClassificationResult class."""
        fields = []
        
        for intent in intents:
            class_name = f"{intent.title().replace('_', '')}Params"
            field_name = f"{intent}_params"
            
            fields.append(f'''    {field_name}: Optional[{class_name}] = Field(
        None,
        description="Parameters extracted for {intent} intents"
    )''')
        
        return "\n".join(fields)
    
    def _generate_additional_classes(self) -> str:
        """Generate exception classes and other non-parameter classes."""
        return '''# Exception and Utility Classes

class IntentClassificationError(Exception):
    """Custom exception for intent classification errors."""
    
    def __init__(self, message: str, error_type: str = "classification_error", user_input: str = ""):
        self.message = message
        self.error_type = error_type
        self.user_input = user_input
        super().__init__(message)


class FallbackIntentResult(BaseModel):
    """Fallback result when intent classification fails."""
    
    intent: IntentCategory = Field(
        default=IntentCategory.UNKNOWN,
        description="Fallback intent type"
    )
    confidence: float = Field(
        default=0.0,
        description="Low confidence for fallback"
    )
    reasoning: str = Field(
        default="Unable to classify intent from user input",
        description="Explanation for fallback"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error that caused fallback"
    )
    user_input_analysis: str = Field(
        default="Could not analyze user input",
        description="Fallback analysis"
    )
    missing_params: List[str] = Field(
        default_factory=lambda: ["intent_clarification"],
        description="Requires intent clarification"
    )
    clarification_needed: bool = Field(
        default=True,
        description="Always needs clarification for unknown intents"
    )
    
    @classmethod
    def from_error(cls, error: Exception, user_input: str = "") -> "FallbackIntentResult":
        """Create fallback result from an error."""
        return cls(
            error_message=str(error),
            user_input_analysis=f"Failed to analyze: '{user_input[:50]}...'" if user_input else "No input provided",
            reasoning=f"Classification failed due to: {type(error).__name__}"
        )
    
    def to_classification_result(self) -> "IntentClassificationResult":
        """Convert to standard IntentClassificationResult."""
        return IntentClassificationResult(
            intent=self.intent,
            confidence=self.confidence,
            reasoning=self.reasoning,
            user_input_analysis=self.user_input_analysis,
            missing_params=self.missing_params,
            clarification_needed=self.clarification_needed
        )
'''
    
    def _generate_constants(self, intents: List[str]) -> str:
        """Generate constants and mappings."""
        # Generate INTENT_PARAM_MODELS mapping
        mapping_lines = []
        for intent in intents:
            class_name = f"{intent.title().replace('_', '')}Params"
            enum_name = f"IntentCategory.{intent.upper()}"
            mapping_lines.append(f"    {enum_name}: {class_name},")
        
        return f'''# Auto-generated mappings
INTENT_PARAM_MODELS = {{
{chr(10).join(mapping_lines)}
}}

# Intent type validation
SUPPORTED_INTENTS = {intents}
'''
    
    def generate_schema_file(self, output_path: str = None) -> str:
        """Generate complete schema file."""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "src" / "vyuu_copilot_v2" / "schemas" / "generated_intent_schemas.py"
        
        intent_params = self.config.get("intent_parameters", {})
        intents = list(intent_params.keys())
        
        # Generate file header
        header = f'''"""
Auto-generated Intent Schemas from YAML Configuration.

Generated on: {datetime.now().isoformat()}
Source: {self.yaml_path}

DO NOT EDIT MANUALLY - Run scripts/generate_intent_schemas.py to regenerate.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class IntentCategory(str, Enum):
    """Supported intent categories for user requests."""
{chr(10).join([f'    {intent.upper()} = "{intent}"' for intent in intents])}


class ConfidenceLevel(str, Enum):
    """Confidence level categories for intent classification."""
    HIGH = "high"      # 0.8 - 1.0
    MEDIUM = "medium"  # 0.5 - 0.79
    LOW = "low"        # 0.0 - 0.49


'''
        
        # Generate individual parameter classes
        param_classes = []
        for intent, config in intent_params.items():
            param_classes.append(self._generate_param_class(intent, config))
        
        # Generate IntentClassificationResult
        classification_result = f'''class IntentClassificationResult(BaseModel):
    """
    Structured result from intent classification LLM call.
    
    This model represents the complete output from LLM intent classification,
    including the classified intent, confidence scoring, and extracted parameters.
    
    Auto-generated from intent_parameters.yaml
    """
    
    intent: IntentCategory = Field(
        ...,
        description="Classified intent category"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for intent classification (0.0-1.0)"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief explanation of why this intent was chosen"
    )
{self._generate_classification_result_fields(intents)}
    missing_params: List[str] = Field(
        default_factory=list,
        description="List of required parameters that could not be extracted"
    )
    clarification_needed: bool = Field(
        False,
        description="Whether clarification is needed for missing parameters"
    )
    user_input_analysis: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Brief analysis of the user input"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v):
        """Validate confidence is within expected range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get categorical confidence level."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    @property
    def requires_clarification(self) -> bool:
        """Check if this classification requires clarification."""
        return (
            self.clarification_needed or 
            len(self.missing_params) > 0 or 
            self.confidence < 0.7 or
            self.intent == IntentCategory.UNKNOWN
        )
    
    @property
    def extracted_parameters(self) -> Dict[str, Any]:
        """Get all extracted parameters as a single dictionary."""
        params = {{}}
        
{chr(10).join([f'        if self.{intent}_params:' + chr(10) + f'            params.update(self.{intent}_params.model_dump(exclude_none=True))' for intent in intents])}
        
        return params


'''
        
        # Generate constants
        constants = self._generate_constants(intents)
        
        # Generate additional classes
        additional_classes = self._generate_additional_classes()
        
        # Combine all parts
        full_content = header + "\n\n".join(param_classes) + "\n\n" + additional_classes + "\n\n" + classification_result + "\n\n" + constants
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(full_content)
        
        print(f"✅ Generated schemas: {output_path}")
        print(f"📊 Generated {len(intents)} intent schemas: {', '.join(intents)}")
        
        return str(output_path)


def main():
    """Main entry point for schema generation."""
    generator = SchemaGenerator()
    generator.generate_schema_file()


if __name__ == "__main__":
    main() 