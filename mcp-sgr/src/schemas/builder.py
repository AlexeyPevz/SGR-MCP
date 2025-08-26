"""Schema builder for creating custom schemas programmatically."""

from typing import Any, Dict, List, Optional


class SchemaBuilder:
    """Builder for creating custom JSON schemas."""

    def __init__(self, schema_id: str, description: str = ""):
        """Initialize schema builder.

        Args:
            schema_id: Unique identifier for the schema
            description: Human-readable description
        """
        self.schema_id = schema_id
        self.schema = {
            "$id": f"schema://{schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "description": description,
            "properties": {},
            "required": [],
        }

    def add_field(
        self,
        name: str,
        field_type: str,
        description: str = "",
        required: bool = True,
        enum: Optional[List[Any]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> "SchemaBuilder":
        """Add a simple field to the schema.

        Args:
            name: Field name
            field_type: JSON type (string, number, boolean, etc.)
            description: Field description
            required: Whether field is required
            enum: Allowed values
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern for strings
            minimum: Minimum value for numbers
            maximum: Maximum value for numbers

        Returns:
            Self for chaining
        """
        field_def = {"type": field_type}

        if description:
            field_def["description"] = description
        if enum:
            field_def["enum"] = enum
        if min_length is not None:
            field_def["minLength"] = min_length
        if max_length is not None:
            field_def["maxLength"] = max_length
        if pattern:
            field_def["pattern"] = pattern
        if minimum is not None:
            field_def["minimum"] = minimum
        if maximum is not None:
            field_def["maximum"] = maximum

        self.schema["properties"][name] = field_def

        if required:
            self.schema["required"].append(name)

        return self

    def add_object_field(
        self,
        name: str,
        properties: Dict[str, Dict[str, Any]],
        required: bool = True,
        required_properties: Optional[List[str]] = None,
    ) -> "SchemaBuilder":
        """Add an object field with nested properties.

        Args:
            name: Field name
            properties: Nested property definitions
            required: Whether field is required
            required_properties: Which nested properties are required

        Returns:
            Self for chaining
        """
        field_def = {"type": "object", "properties": properties}

        if required_properties:
            field_def["required"] = required_properties

        self.schema["properties"][name] = field_def

        if required:
            self.schema["required"].append(name)

        return self

    def add_array_field(
        self,
        name: str,
        item_type: str = "string",
        description: str = "",
        required: bool = True,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        unique_items: bool = False,
    ) -> "SchemaBuilder":
        """Add an array field.

        Args:
            name: Field name
            item_type: Type of array items
            description: Field description
            required: Whether field is required
            min_items: Minimum number of items
            max_items: Maximum number of items
            unique_items: Whether items must be unique

        Returns:
            Self for chaining
        """
        field_def = {"type": "array", "items": {"type": item_type}}

        if description:
            field_def["description"] = description
        if min_items is not None:
            field_def["minItems"] = min_items
        if max_items is not None:
            field_def["maxItems"] = max_items
        if unique_items:
            field_def["uniqueItems"] = unique_items

        self.schema["properties"][name] = field_def

        if required:
            self.schema["required"].append(name)

        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the final schema.

        Returns:
            Complete JSON schema
        """
        # Remove empty required array
        if not self.schema["required"]:
            del self.schema["required"]

        return self.schema

    def to_json_schema(self) -> Dict[str, Any]:
        """Alias for build() for compatibility.

        Returns:
            Complete JSON schema
        """
        return self.build()
