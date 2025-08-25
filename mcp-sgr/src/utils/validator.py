"""Schema validator utilities."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft7Validator
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates data against schemas."""

    @staticmethod
    def validate_json_schema(
        data: Dict[str, Any], schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate data against JSON Schema.

        Args:
            data: Data to validate
            schema: JSON Schema

        Returns:
            Tuple of (valid, errors)
        """
        try:
            # Create validator
            validator = Draft7Validator(schema)

            # Collect errors
            errors = []
            for error in validator.iter_errors(data):
                error_path = " -> ".join(str(p) for p in error.path)
                error_msg = f"{error_path}: {error.message}" if error_path else error.message
                errors.append(error_msg)

            return len(errors) == 0, errors

        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False, [str(e)]

    @staticmethod
    def validate_pydantic(
        data: Dict[str, Any], model_class: type[BaseModel]
    ) -> Tuple[bool, List[str], Optional[BaseModel]]:
        """Validate data against Pydantic model.

        Args:
            data: Data to validate
            model_class: Pydantic model class

        Returns:
            Tuple of (valid, errors, model_instance)
        """
        try:
            # Validate and create instance
            instance = model_class(**data)
            return True, [], instance

        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = " -> ".join(str(loc_part) for loc_part in error["loc"])
                errors.append(f"{loc}: {error['msg']}")

            return False, errors, None

        except Exception as e:
            logger.error(f"Pydantic validation error: {e}")
            return False, [str(e)], None

    @staticmethod
    def extract_schema_from_json(json_str: str) -> Optional[Dict[str, Any]]:
        """Extract JSON Schema from a JSON string.

        Useful for parsing LLM responses that might contain schema.

        Args:
            json_str: JSON string that might contain schema

        Returns:
            Extracted schema or None
        """
        try:
            data = json.loads(json_str)

            # Check if it's already a schema
            if isinstance(data, dict) and "$schema" in data:
                return data

            # Try to find schema in common locations
            if isinstance(data, dict):
                if "schema" in data:
                    return data["schema"]
                elif "json_schema" in data:
                    return data["json_schema"]
                elif "properties" in data and "type" in data:
                    # Looks like a schema without $schema field
                    return data

            return None

        except json.JSONDecodeError:
            logger.error("Failed to parse JSON for schema extraction")
            return None

    @staticmethod
    def merge_schemas(
        base_schema: Dict[str, Any], override_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two JSON schemas.

        Args:
            base_schema: Base schema
            override_schema: Schema with overrides

        Returns:
            Merged schema
        """
        merged = base_schema.copy()

        # Merge properties
        if "properties" in override_schema:
            if "properties" not in merged:
                merged["properties"] = {}
            merged["properties"].update(override_schema["properties"])

        # Merge required fields
        if "required" in override_schema:
            if "required" in merged:
                # Combine and deduplicate
                required = list(set(merged["required"] + override_schema["required"]))
                merged["required"] = required
            else:
                merged["required"] = override_schema["required"]

        # Override other fields
        for key in ["title", "description", "$id", "$schema"]:
            if key in override_schema:
                merged[key] = override_schema[key]

        return merged

    @staticmethod
    def simplify_schema(schema: Dict[str, Any], max_depth: int = 3) -> Dict[str, Any]:
        """Simplify a complex schema by limiting depth.

        Args:
            schema: Original schema
            max_depth: Maximum nesting depth

        Returns:
            Simplified schema
        """

        def _simplify_recursive(obj: Any, depth: int) -> Any:
            if depth >= max_depth:
                if isinstance(obj, dict) and "type" in obj:
                    # Replace deep objects with simple type
                    return {"type": "object", "additionalProperties": True}
                return obj

            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if key == "properties" and isinstance(value, dict):
                        # Recurse into properties
                        result[key] = {
                            k: _simplify_recursive(v, depth + 1) for k, v in value.items()
                        }
                    else:
                        result[key] = _simplify_recursive(value, depth)
                return result
            elif isinstance(obj, list):
                return [_simplify_recursive(item, depth) for item in obj]
            else:
                return obj

        return _simplify_recursive(schema, 0)

    @staticmethod
    def generate_example(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example data that matches schema.

        Args:
            schema: JSON Schema

        Returns:
            Example data
        """

        def _generate_value(prop: Dict[str, Any]) -> Any:
            prop_type = prop.get("type", "string")

            if "enum" in prop:
                return prop["enum"][0]
            elif "default" in prop:
                return prop["default"]
            elif prop_type == "string":
                if "format" in prop:
                    format_examples = {
                        "email": "user@example.com",
                        "date-time": "2024-01-01T00:00:00Z",
                        "date": "2024-01-01",
                        "time": "00:00:00",
                        "uri": "https://example.com",
                        "uuid": "123e4567-e89b-12d3-a456-426614174000",
                    }
                    return format_examples.get(prop["format"], "example")
                return prop.get("description", "example")[:50]
            elif prop_type == "number":
                return 42.0
            elif prop_type == "integer":
                return 42
            elif prop_type == "boolean":
                return True
            elif prop_type == "array":
                item_schema = prop.get("items", {"type": "string"})
                return [_generate_value(item_schema)]
            elif prop_type == "object":
                if "properties" in prop:
                    return {
                        key: _generate_value(value) for key, value in prop["properties"].items()
                    }
                return {}
            else:
                return None

        if schema.get("type") != "object" or "properties" not in schema:
            return {}

        example = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Generate required fields first
        for field in required:
            if field in properties:
                example[field] = _generate_value(properties[field])

        # Optionally add some non-required fields
        for field, prop in properties.items():
            if field not in example and len(example) < len(required) + 2:
                example[field] = _generate_value(prop)

        return example
