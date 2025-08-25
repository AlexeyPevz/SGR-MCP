"""Tests for utility modules."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.utils.redact import PIIRedactor
from src.utils.router import ModelRouter, RoutingRule, TaskType
from src.utils.validator import SchemaValidator


class TestModelRouter:
    """Test ModelRouter class."""

    def test_routing_rule_matching(self):
        """Test routing rule condition matching."""
        rule = RoutingRule(condition='task_type == "code_generation"', backend="ollama")

        assert rule.matches({"task_type": "code_generation"})
        assert not rule.matches({"task_type": "analysis"})

    def test_routing_rule_in_operator(self):
        """Test routing rule with 'in' operator."""
        rule = RoutingRule(condition='task_type in ["analysis", "planning"]', backend="ollama")

        assert rule.matches({"task_type": "analysis"})
        assert rule.matches({"task_type": "planning"})
        assert not rule.matches({"task_type": "code_generation"})

    def test_routing_rule_comparison(self):
        """Test routing rule with numeric comparison."""
        rule = RoutingRule(condition="tokens > 1000", backend="openrouter")

        assert rule.matches({"tokens": 1500})
        assert not rule.matches({"tokens": 500})

    def test_router_backend_selection(self):
        """Test router backend selection."""
        router = ModelRouter()

        # Test code generation routing
        context = router.create_routing_context(
            task="Write a Python function", schema_type="code_generation"
        )
        selection = router.select_backend(context)

        assert selection["backend"] in ["ollama", "openrouter"]
        assert "model" in selection
        assert "retry" in selection

    def test_task_type_detection(self):
        """Test task type detection."""
        router = ModelRouter()

        assert router.detect_task_type("Analyze this code") == TaskType.ANALYSIS
        assert router.detect_task_type("Create a plan for deployment") == TaskType.PLANNING
        assert (
            router.detect_task_type("Write a function to sort arrays") == TaskType.CODE_GENERATION
        )
        assert router.detect_task_type("Choose between MongoDB and PostgreSQL") == TaskType.DECISION
        assert router.detect_task_type("Summarize this document") == TaskType.SUMMARIZATION


class TestSchemaValidator:
    """Test SchemaValidator class."""

    def test_json_schema_validation(self):
        """Test JSON schema validation."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
            "required": ["name"],
        }

        # Valid data
        valid, errors = SchemaValidator.validate_json_schema({"name": "John", "age": 30}, schema)
        assert valid
        assert len(errors) == 0

        # Invalid data - missing required field
        valid, errors = SchemaValidator.validate_json_schema({"age": 30}, schema)
        assert not valid
        assert len(errors) > 0

        # Invalid data - wrong type
        valid, errors = SchemaValidator.validate_json_schema(
            {"name": "John", "age": "thirty"}, schema
        )
        assert not valid
        assert len(errors) > 0

    def test_schema_generation(self):
        """Test example generation from schema."""
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["email", "count"],
        }

        example = SchemaValidator.generate_example(schema)

        assert "email" in example
        assert "@" in example["email"]
        assert "count" in example
        assert isinstance(example["count"], int)


class TestPIIRedactor:
    """Test PIIRedactor class."""

    def test_redact_email(self):
        """Test email redaction."""
        redactor = PIIRedactor()

        text = "Contact me at john.doe@example.com for details"
        redacted, counts = redactor.redact_text(text)

        assert "john.doe@example.com" not in redacted
        assert "[EMAIL]" in redacted
        assert counts.get("email", 0) == 1

    def test_redact_phone(self):
        """Test phone number redaction."""
        redactor = PIIRedactor()

        text = "Call me at 555-123-4567 or (555) 987-6543"
        redacted, counts = redactor.redact_text(text)

        assert "555-123-4567" not in redacted
        assert "(555) 987-6543" not in redacted
        assert redacted.count("[PHONE]") == 2
        assert counts.get("phone", 0) == 2

    def test_redact_credit_card(self):
        """Test credit card redaction."""
        redactor = PIIRedactor()

        text = "Payment with card 4111111111111111"
        redacted, counts = redactor.redact_text(text)

        assert "4111111111111111" not in redacted
        assert "[CREDIT_CARD]" in redacted
        assert counts.get("credit_card", 0) == 1

    def test_redact_dict(self):
        """Test dictionary redaction."""
        redactor = PIIRedactor()

        data = {
            "user": {"email": "user@example.com", "phone": "555-1234", "notes": "No PII here"},
            "count": 42,
        }

        redacted_data, counts = redactor.redact_dict(data)

        assert redacted_data["user"]["email"] != "user@example.com"
        assert "[EMAIL]" in redacted_data["user"]["email"]
        assert "[PHONE]" in redacted_data["user"]["phone"]
        assert redacted_data["user"]["notes"] == "No PII here"
        assert redacted_data["count"] == 42

    def test_custom_pattern(self):
        """Test adding custom redaction pattern."""
        redactor = PIIRedactor()

        # Add custom pattern for employee IDs
        redactor.add_custom_pattern(
            name="employee_id", pattern=r"EMP\d{6}", replacement="[EMPLOYEE_ID]"
        )

        text = "Employee EMP123456 has access"
        redacted, counts = redactor.redact_text(text)

        assert "EMP123456" not in redacted
        assert "[EMPLOYEE_ID]" in redacted
        assert counts.get("employee_id", 0) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
