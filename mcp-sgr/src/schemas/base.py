"""Base schema definitions and validators for SGR."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pydantic import BaseModel, Field, validator
import jsonschema


class ConfidenceLevel(str, Enum):
    """Confidence levels for reasoning."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BudgetDepth(str, Enum):
    """Reasoning budget depth."""
    NONE = "none"
    LITE = "lite"
    FULL = "full"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SchemaField:
    """Definition of a schema field."""
    name: str
    type: str
    required: bool = True
    description: str = ""
    default: Any = None
    enum: Optional[List[Any]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None


class BaseReasoningModel(BaseModel):
    """Base model for all reasoning outputs."""
    
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "forbid"


class BaseSchema(ABC):
    """Base class for all SGR schemas."""
    
    def __init__(self):
        self.schema_id = self.__class__.__name__.lower().replace("schema", "")
        self._json_schema = None
    
    @abstractmethod
    def get_fields(self) -> List[SchemaField]:
        """Get list of schema fields."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get schema description."""
        pass
    
    @abstractmethod
    def get_examples(self) -> List[Dict[str, Any]]:
        """Get example instances of the schema."""
        pass
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        if self._json_schema is None:
            self._json_schema = self._build_json_schema()
        return self._json_schema
    
    def _build_json_schema(self) -> Dict[str, Any]:
        """Build JSON Schema from field definitions."""
        properties = {}
        required = []
        
        for field in self.get_fields():
            prop = {"type": field.type}
            
            if field.description:
                prop["description"] = field.description
            
            if field.enum:
                prop["enum"] = field.enum
            
            if field.type == "string":
                if field.min_length:
                    prop["minLength"] = field.min_length
                if field.max_length:
                    prop["maxLength"] = field.max_length
                if field.pattern:
                    prop["pattern"] = field.pattern
            
            if field.default is not None:
                prop["default"] = field.default
            
            properties[field.name] = prop
            
            if field.required:
                required.append(field.name)
        
        return {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": self.__class__.__name__,
            "description": self.get_description(),
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against the schema."""
        try:
            jsonschema.validate(instance=data, schema=self.to_json_schema())
            
            # Additional semantic validation
            warnings = self._semantic_validation(data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, warnings)
            
            return ValidationResult(
                valid=True,
                warnings=warnings,
                confidence=confidence
            )
        except jsonschema.ValidationError as e:
            return ValidationResult(
                valid=False,
                errors=[str(e)],
                confidence=0.0
            )
    
    def _semantic_validation(self, data: Dict[str, Any]) -> List[str]:
        """Perform semantic validation beyond JSON Schema."""
        warnings = []
        
        # Check for empty arrays that shouldn't be empty
        for field in self.get_fields():
            if field.type == "array" and field.required:
                value = data.get(field.name, [])
                if isinstance(value, list) and len(value) == 0:
                    warnings.append(f"Field '{field.name}' is empty but should contain items")
        
        return warnings
    
    def _calculate_confidence(self, data: Dict[str, Any], warnings: List[str]) -> float:
        """Calculate confidence score for the reasoning."""
        # Base confidence
        confidence = 1.0
        
        # Reduce for warnings
        confidence -= len(warnings) * 0.1
        
        # Reduce for missing optional fields
        all_fields = {f.name for f in self.get_fields()}
        provided_fields = set(data.keys())
        missing_optional = all_fields - provided_fields
        confidence -= len(missing_optional) * 0.05
        
        # Reduce for very short text fields
        for field in self.get_fields():
            if field.type == "string" and field.name in data:
                value = data[field.name]
                if isinstance(value, str) and len(value) < 10:
                    confidence -= 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def generate_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a prompt for filling this schema."""
        prompt = f"""You are analyzing the following task using structured reasoning.

Task: {task}

{f"Context: {json.dumps(context, indent=2)}" if context else ""}

Please provide your reasoning in the following JSON structure:
{json.dumps(self.to_json_schema(), indent=2)}

Focus on being thorough and specific. Each field should contain meaningful analysis.
Return only valid JSON that matches the schema exactly."""
        
        return prompt