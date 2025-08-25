"""Custom schema implementation."""

import logging
from typing import Any, Dict, List

import jsonschema

from .base import BaseSchema, SchemaField, ValidationResult

logger = logging.getLogger(__name__)


class CustomSchema(BaseSchema):
    """Dynamic schema created from user-provided definition."""

    def __init__(self, schema_definition: Dict[str, Any]):
        """Initialize custom schema from definition.

        Args:
            schema_definition: JSON Schema compatible definition
        """
        super().__init__()
        self._schema_def = schema_definition
        self._validate_schema_definition()
        self.schema_id = schema_definition.get("$id", "custom_schema").replace("schema://", "")

    def _validate_schema_definition(self):
        """Validate that the schema definition is valid JSON Schema."""
        try:
            # Check it's a valid JSON Schema
            jsonschema.Draft7Validator.check_schema(self._schema_def)
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid schema definition: {e}")

        # Ensure required fields
        if "type" not in self._schema_def or self._schema_def["type"] != "object":
            raise ValueError("Schema must have type 'object'")

        if "properties" not in self._schema_def:
            raise ValueError("Schema must have 'properties' defined")

    def get_fields(self) -> List[SchemaField]:
        """Extract fields from schema definition."""
        fields = []
        properties = self._schema_def.get("properties", {})
        required = self._schema_def.get("required", [])

        for name, prop in properties.items():
            field = SchemaField(
                name=name,
                type=prop.get("type", "string"),
                required=name in required,
                description=prop.get("description", ""),
                default=prop.get("default"),
                enum=prop.get("enum"),
                min_length=prop.get("minLength"),
                max_length=prop.get("maxLength"),
                pattern=prop.get("pattern"),
            )
            fields.append(field)

        return fields

    def get_description(self) -> str:
        """Get schema description."""
        return self._schema_def.get("description", "Custom reasoning schema")

    def get_examples(self) -> List[Dict[str, Any]]:
        """Get examples if provided in schema."""
        return self._schema_def.get("examples", [])

    def to_json_schema(self) -> Dict[str, Any]:
        """Return the original schema definition."""
        return self._schema_def.copy()

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against custom schema."""
        try:
            # Use Draft7Validator to collect all errors rather than fail-fast
            validator = jsonschema.Draft7Validator(self._schema_def)
            errors = [e.message for e in validator.iter_errors(data)]
            if errors:
                return ValidationResult(valid=False, errors=errors, confidence=0.0)

            # Calculate basic confidence
            confidence = self._calculate_basic_confidence(data)

            return ValidationResult(valid=True, warnings=[], confidence=confidence)
        except jsonschema.SchemaError as e:
            return ValidationResult(valid=False, errors=[str(e)], confidence=0.0)

    def _calculate_basic_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate basic confidence for custom schema."""
        if not self._schema_def.get("properties"):
            return 0.5

        total_properties = len(self._schema_def["properties"])
        provided_properties = len([k for k in data.keys() if k in self._schema_def["properties"]])

        # Base confidence on property coverage
        confidence = provided_properties / total_properties if total_properties > 0 else 0.5

        # Adjust for required fields
        required = self._schema_def.get("required", [])
        if required:
            required_provided = len([k for k in required if k in data])
            required_ratio = required_provided / len(required)
            confidence = (confidence + required_ratio) / 2

        return min(1.0, max(0.0, confidence))
