"""Tests for SGR schemas."""

import json

import pytest

from src.schemas import (
    SCHEMA_REGISTRY,
    AnalysisSchema,
    CodeGenerationSchema,
    DecisionSchema,
    PlanningSchema,
    SummarizationSchema,
)
from src.schemas.custom import CustomSchema, SchemaBuilder


class TestSchemaRegistry:
    """Test schema registry."""

    def test_registry_contains_all_schemas(self):
        """Test that all schemas are in registry."""
        expected_schemas = {"analysis", "planning", "decision", "code_generation", "summarization"}
        assert set(SCHEMA_REGISTRY.keys()) == expected_schemas

    def test_registry_schemas_instantiable(self):
        """Test that all registry schemas can be instantiated."""
        for name, schema_class in SCHEMA_REGISTRY.items():
            schema = schema_class()
            assert schema is not None
            assert schema.schema_id == name


class TestAnalysisSchema:
    """Test analysis schema."""

    def test_schema_structure(self):
        """Test schema has correct structure."""
        schema = AnalysisSchema()
        json_schema = schema.to_json_schema()

        assert json_schema["type"] == "object"
        assert "properties" in json_schema
        assert "required" in json_schema

        # Check required fields
        required_fields = {"understanding", "goals", "constraints", "risks"}
        assert set(json_schema["required"]) == required_fields

    def test_valid_data_validation(self):
        """Test validation of valid data."""
        schema = AnalysisSchema()

        valid_data = {
            "understanding": {
                "task_summary": "Build a REST API for user management",
                "key_aspects": ["Authentication", "Authorization", "CRUD operations"],
            },
            "goals": {
                "primary": "Create secure user management system",
                "success_criteria": ["All endpoints secured", "Proper validation"],
            },
            "constraints": [{"type": "technical", "description": "Must use Python FastAPI"}],
            "risks": [
                {
                    "risk": "Security vulnerabilities",
                    "likelihood": "medium",
                    "impact": "high",
                    "mitigation": "Follow OWASP guidelines",
                }
            ],
        }

        result = schema.validate(valid_data)
        assert result.valid
        assert result.confidence > 0.5
        assert len(result.errors) == 0

    def test_invalid_data_validation(self):
        """Test validation of invalid data."""
        schema = AnalysisSchema()

        # Missing required field
        invalid_data = {
            "understanding": {
                "task_summary": "Test task"
                # Missing key_aspects
            },
            "goals": {
                "primary": "Test goal"
                # Missing success_criteria
            },
            # Missing constraints and risks
        }

        result = schema.validate(invalid_data)
        assert not result.valid
        assert len(result.errors) > 0
        assert result.confidence == 0.0


class TestCustomSchema:
    """Test custom schema functionality."""

    def test_schema_builder(self):
        """Test building custom schema."""
        builder = SchemaBuilder("test_schema", "Test schema description")

        builder.add_field(name="test_field", field_type="string", required=True, min_length=5)

        builder.add_array_field(name="items", item_type="string", required=False, min_items=1)

        schema = builder.build()
        json_schema = schema.to_json_schema()

        assert json_schema["$id"] == "schema://test_schema"
        assert json_schema["description"] == "Test schema description"
        assert "test_field" in json_schema["properties"]
        assert "items" in json_schema["properties"]
        assert "test_field" in json_schema["required"]
        assert "items" not in json_schema["required"]

    def test_custom_schema_validation(self):
        """Test custom schema validation."""
        # Define custom schema
        schema_def = {
            "$id": "schema://custom_test",
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "format": "email"},
            },
            "required": ["name", "age"],
        }

        schema = CustomSchema(schema_def)

        # Valid data
        valid_data = {"name": "John Doe", "age": 30, "email": "john@example.com"}

        result = schema.validate(valid_data)
        assert result.valid

        # Invalid data
        invalid_data = {"name": "Jo", "age": 200}  # Too short  # Too high

        result = schema.validate(invalid_data)
        assert not result.valid
        assert len(result.errors) >= 2


class TestSchemaExamples:
    """Test that schema examples are valid."""

    @pytest.mark.parametrize("schema_name", SCHEMA_REGISTRY.keys())
    def test_schema_examples_valid(self, schema_name):
        """Test that all schema examples validate correctly."""
        schema_class = SCHEMA_REGISTRY[schema_name]
        schema = schema_class()

        examples = schema.get_examples()
        assert len(examples) > 0, f"Schema {schema_name} has no examples"

        for i, example in enumerate(examples):
            result = schema.validate(example)
            assert result.valid, f"Example {i} of {schema_name} is invalid: {result.errors}"


class TestSchemaGeneration:
    """Test schema prompt generation."""

    def test_prompt_generation(self):
        """Test that schemas can generate prompts."""
        schema = AnalysisSchema()

        prompt = schema.generate_prompt(
            task="Analyze the security of a login system",
            context={"framework": "Django", "auth_type": "JWT"},
        )

        assert "Analyze the security of a login system" in prompt
        assert "Django" in prompt
        assert "JSON" in prompt
        assert "$schema" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
