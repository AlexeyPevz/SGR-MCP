"""Analysis schema for understanding tasks and problems."""

from typing import Any, Dict, List

from .base import BaseSchema, SchemaField


class AnalysisSchema(BaseSchema):
    """Schema for task analysis and understanding."""

    def get_description(self) -> str:
        return "Comprehensive analysis of a task including understanding, goals, constraints, and risks"

    def get_fields(self) -> List[SchemaField]:
        return [
            # Understanding section
            SchemaField(
                name="understanding",
                type="object",
                description="Deep understanding of the task",
                required=True,
            ),
            SchemaField(
                name="understanding.task_summary",
                type="string",
                description="Clear summary of what needs to be done",
                min_length=10,
                required=True,
            ),
            SchemaField(
                name="understanding.key_aspects",
                type="array",
                description="Key aspects and components of the task",
                required=True,
            ),
            SchemaField(
                name="understanding.ambiguities",
                type="array",
                description="Unclear or ambiguous parts that need clarification",
                required=False,
            ),
            SchemaField(
                name="understanding.assumptions",
                type="array",
                description="Assumptions being made",
                required=False,
            ),
            # Goals section
            SchemaField(
                name="goals", type="object", description="Goals and success criteria", required=True
            ),
            SchemaField(
                name="goals.primary",
                type="string",
                description="Primary goal to achieve",
                required=True,
            ),
            SchemaField(
                name="goals.secondary",
                type="array",
                description="Secondary goals or nice-to-haves",
                required=False,
            ),
            SchemaField(
                name="goals.success_criteria",
                type="array",
                description="Measurable criteria for success",
                required=True,
            ),
            # Constraints section
            SchemaField(
                name="constraints",
                type="array",
                description="Limitations and constraints",
                required=True,
            ),
            # Risks section
            SchemaField(
                name="risks",
                type="array",
                description="Potential risks and mitigation strategies",
                required=True,
            ),
            # Data gaps
            SchemaField(
                name="data_gaps",
                type="array",
                description="Missing information or data needed",
                required=False,
            ),
            # Dependencies
            SchemaField(
                name="dependencies",
                type="array",
                description="External dependencies or prerequisites",
                required=False,
            ),
        ]

    def get_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "understanding": {
                    "task_summary": "Build a REST API for user authentication with JWT tokens",
                    "key_aspects": [
                        "User registration with email/password",
                        "Login endpoint returning JWT tokens",
                        "Token refresh mechanism",
                        "Password reset functionality",
                    ],
                    "ambiguities": [
                        "Specific JWT expiration times not specified",
                        "Password complexity requirements unclear",
                    ],
                    "assumptions": [
                        "Using standard JWT format",
                        "Email verification is required",
                        "PostgreSQL as database",
                    ],
                },
                "goals": {
                    "primary": "Create secure authentication system with JWT",
                    "secondary": ["Support social login providers", "Implement rate limiting"],
                    "success_criteria": [
                        "All endpoints return proper HTTP status codes",
                        "Tokens expire and refresh correctly",
                        "Passwords are properly hashed",
                        "API handles edge cases gracefully",
                    ],
                },
                "constraints": [
                    {"type": "technical", "description": "Must use Python FastAPI framework"},
                    {"type": "security", "description": "Must follow OWASP guidelines"},
                    {"type": "time", "description": "Complete within 2 weeks"},
                ],
                "risks": [
                    {
                        "risk": "Token hijacking",
                        "likelihood": "medium",
                        "impact": "high",
                        "mitigation": "Implement token rotation and IP validation",
                    },
                    {
                        "risk": "Brute force attacks",
                        "likelihood": "high",
                        "impact": "medium",
                        "mitigation": "Add rate limiting and account lockout",
                    },
                ],
                "data_gaps": [
                    "Email service provider details",
                    "Specific security requirements",
                    "Expected user load",
                ],
                "dependencies": [
                    "Email service for verification",
                    "Redis for session storage",
                    "PostgreSQL database",
                ],
            }
        ]

    def _build_json_schema(self) -> Dict[str, Any]:
        """Override to build nested schema properly."""
        return {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "AnalysisSchema",
            "description": self.get_description(),
            "required": ["understanding", "goals", "constraints", "risks"],
            "properties": {
                "understanding": {
                    "type": "object",
                    "required": ["task_summary", "key_aspects"],
                    "properties": {
                        "task_summary": {
                            "type": "string",
                            "minLength": 10,
                            "description": "Clear summary of what needs to be done",
                        },
                        "key_aspects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Key aspects and components of the task",
                        },
                        "ambiguities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Unclear or ambiguous parts",
                        },
                        "assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Assumptions being made",
                        },
                    },
                },
                "goals": {
                    "type": "object",
                    "required": ["primary", "success_criteria"],
                    "properties": {
                        "primary": {"type": "string", "description": "Primary goal to achieve"},
                        "secondary": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Secondary goals",
                        },
                        "success_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Measurable criteria for success",
                        },
                    },
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["technical", "business", "resource", "time", "security"],
                            },
                            "description": {"type": "string"},
                        },
                        "required": ["type", "description"],
                    },
                    "description": "Limitations and constraints",
                },
                "risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk": {"type": "string"},
                            "likelihood": {"type": "string", "enum": ["low", "medium", "high"]},
                            "impact": {"type": "string", "enum": ["low", "medium", "high"]},
                            "mitigation": {"type": "string"},
                        },
                        "required": ["risk", "likelihood", "impact", "mitigation"],
                    },
                    "description": "Potential risks and mitigation",
                },
                "data_gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Missing information needed",
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "External dependencies",
                },
            },
            "additionalProperties": False,
        }
