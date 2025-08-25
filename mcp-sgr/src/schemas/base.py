"""Base schema definitions and validators for SGR."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import jsonschema
from pydantic import BaseModel, Field


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
        # Convert class name like CodeGenerationSchema -> code_generation
        import re

        class_name = self.__class__.__name__
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        if snake.endswith("_schema"):
            snake = snake[:-7]
        self.schema_id = snake
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

        for schema_field in self.get_fields():
            prop: Dict[str, Any] = {"type": schema_field.type}

            if schema_field.description:
                prop["description"] = schema_field.description

            if schema_field.enum:
                prop["enum"] = schema_field.enum

            if schema_field.type == "string":
                if schema_field.min_length:
                    prop["minLength"] = schema_field.min_length
                if schema_field.max_length:
                    prop["maxLength"] = schema_field.max_length
                if schema_field.pattern:
                    prop["pattern"] = schema_field.pattern

            if schema_field.default is not None:
                prop["default"] = schema_field.default

            properties[schema_field.name] = prop

            if schema_field.required:
                required.append(schema_field.name)

        return {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": self.__class__.__name__,
            "description": self.get_description(),
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against the schema."""
        try:
            jsonschema.validate(instance=data, schema=self.to_json_schema())

            # Additional semantic validation
            warnings = self._semantic_validation(data)

            # Calculate confidence
            confidence = self._calculate_confidence(data, warnings)

            return ValidationResult(valid=True, warnings=warnings, confidence=confidence)
        except jsonschema.ValidationError as e:
            return ValidationResult(valid=False, errors=[str(e)], confidence=0.0)

    def _semantic_validation(self, data: Dict[str, Any]) -> List[str]:
        """Perform semantic validation beyond JSON Schema."""
        warnings = []

        def get_nested_value(obj: Dict[str, Any], dotted_key: str) -> Any:
            current: Any = obj
            for part in dotted_key.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current

        # Check for empty arrays that shouldn't be empty
        for schema_field in self.get_fields():
            if schema_field.type == "array" and schema_field.required:
                value = get_nested_value(data, schema_field.name)
                if isinstance(value, list) and len(value) == 0:
                    warnings.append(f"Field '{schema_field.name}' is empty but should contain items")

        return warnings

    def _calculate_confidence(self, data: Dict[str, Any], warnings: List[str]) -> float:
        """Calculate confidence score for the reasoning.

        Heuristic: start from 0.9, subtract 0.1 per warning (min 0), and small
        penalties for clearly short string fields when present. Do not penalize
        for missing optional fields.
        """
        confidence = 0.9
        confidence -= min(len(warnings) * 0.1, 0.5)

        # Reduce for very short top-level string fields when present
        for schema_field in self.get_fields():
            if schema_field.type == "string":
                # Support dotted notation
                parts = schema_field.name.split(".")
                current: Any = data
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                if isinstance(current, str) and len(current) < 10:
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
