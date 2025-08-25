"""Custom schema support for user-defined reasoning structures."""

import json
from typing import Any, Dict, List, Optional

import jsonschema

from .base import BaseSchema, SchemaField, ValidationResult


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


class SchemaBuilder:
    """Helper class to build custom schemas programmatically."""

    def __init__(self, schema_id: str, description: str = ""):
        self.schema_id = schema_id
        self.description = description
        self.properties = {}
        self.required = []
        self.examples = []

    def add_field(
        self,
        name: str,
        field_type: str = "string",
        description: str = "",
        required: bool = False,
        enum: Optional[List[Any]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        default: Any = None,
    ) -> "SchemaBuilder":
        """Add a field to the schema."""
        prop = {"type": field_type}

        if description:
            prop["description"] = description
        if enum:
            prop["enum"] = enum
        if min_length is not None:
            prop["minLength"] = min_length
        if max_length is not None:
            prop["maxLength"] = max_length
        if pattern:
            prop["pattern"] = pattern
        if default is not None:
            prop["default"] = default

        self.properties[name] = prop

        if required:
            self.required.append(name)

        return self

    def add_object_field(
        self,
        name: str,
        properties: Dict[str, Any],
        required: bool = False,
        required_properties: Optional[List[str]] = None,
    ) -> "SchemaBuilder":
        """Add an object field with nested properties."""
        prop = {"type": "object", "properties": properties}

        if required_properties:
            prop["required"] = required_properties

        self.properties[name] = prop

        if required:
            self.required.append(name)

        return self

    def add_array_field(
        self,
        name: str,
        item_type: str = "string",
        description: str = "",
        required: bool = False,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> "SchemaBuilder":
        """Add an array field."""
        prop = {"type": "array", "items": {"type": item_type}}

        if description:
            prop["description"] = description
        if min_items is not None:
            prop["minItems"] = min_items
        if max_items is not None:
            prop["maxItems"] = max_items

        self.properties[name] = prop

        if required:
            self.required.append(name)

        return self

    def add_example(self, example: Dict[str, Any]) -> "SchemaBuilder":
        """Add an example to the schema."""
        self.examples.append(example)
        return self

    def build(self) -> CustomSchema:
        """Build the custom schema."""
        schema_def = {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "description": self.description,
            "properties": self.properties,
            "required": self.required,
            "additionalProperties": False,
        }

        if self.examples:
            schema_def["examples"] = self.examples

        return CustomSchema(schema_def)


# Example of creating a custom schema
def create_bug_report_schema() -> CustomSchema:
    """Example: Create a custom schema for bug report analysis."""
    builder = SchemaBuilder(
        "bug_report_analysis", "Schema for analyzing and structuring bug reports"
    )

    builder.add_object_field(
        "issue_identification",
        {
            "title": {"type": "string", "minLength": 10},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "category": {
                "type": "string",
                "enum": ["ui", "backend", "data", "performance", "security"],
            },
            "affected_components": {"type": "array", "items": {"type": "string"}},
        },
        required=True,
        required_properties=["title", "severity", "category"],
    )

    builder.add_object_field(
        "reproduction",
        {
            "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "environment": {"type": "object"},
            "frequency": {"type": "string", "enum": ["always", "often", "sometimes", "rarely"]},
            "prerequisites": {"type": "array", "items": {"type": "string"}},
        },
        required=True,
        required_properties=["steps", "frequency"],
    )

    builder.add_object_field(
        "impact_analysis",
        {
            "users_affected": {"type": "string"},
            "business_impact": {"type": "string"},
            "workaround_available": {"type": "boolean"},
            "workaround_description": {"type": "string"},
        },
        required=True,
    )

    builder.add_array_field(
        "potential_causes",
        item_type="string",
        description="Possible root causes",
        required=True,
        min_items=1,
    )

    builder.add_object_field(
        "recommendation",
        {
            "priority": {"type": "string", "enum": ["p0", "p1", "p2", "p3"]},
            "estimated_effort": {"type": "string"},
            "suggested_assignee": {"type": "string"},
            "fix_approach": {"type": "string"},
        },
        required=True,
    )

    return builder.build()
