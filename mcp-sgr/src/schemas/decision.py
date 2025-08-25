"""Decision schema for structured decision-making."""

from typing import Any, Dict, List

from .base import BaseSchema, SchemaField


class DecisionSchema(BaseSchema):
    """Schema for decision-making with alternatives and validation."""

    def get_description(self) -> str:
        return "Structured decision-making including options analysis, selection rationale, and validation"

    def get_fields(self) -> List[SchemaField]:
        return [
            SchemaField(
                name="context",
                type="object",
                description="Decision context and requirements",
                required=True,
            ),
            SchemaField(
                name="options",
                type="array",
                description="Available options to choose from",
                required=True,
            ),
            SchemaField(
                name="evaluation_criteria",
                type="array",
                description="Criteria for evaluating options",
                required=True,
            ),
            SchemaField(
                name="analysis",
                type="object",
                description="Comparative analysis of options",
                required=True,
            ),
            SchemaField(
                name="decision",
                type="object",
                description="Final decision and rationale",
                required=True,
            ),
            SchemaField(
                name="validation",
                type="object",
                description="Decision validation and confidence",
                required=True,
            ),
            SchemaField(
                name="fallback_plan",
                type="object",
                description="Plan B if decision fails",
                required=False,
            ),
        ]

    def get_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "context": {
                    "problem": "Choose database for new microservice",
                    "requirements": [
                        "Handle 10K requests per second",
                        "Support complex queries",
                        "ACID compliance required",
                        "Easy horizontal scaling",
                    ],
                    "constraints": [
                        "Team has PostgreSQL experience",
                        "Budget limited to open-source solutions",
                        "Must integrate with existing monitoring",
                    ],
                },
                "options": [
                    {
                        "id": "postgres",
                        "name": "PostgreSQL",
                        "description": "Traditional RDBMS with strong consistency",
                    },
                    {
                        "id": "mongo",
                        "name": "MongoDB",
                        "description": "Document database with flexible schema",
                    },
                    {
                        "id": "cassandra",
                        "name": "Cassandra",
                        "description": "Wide-column store for high scalability",
                    },
                ],
                "evaluation_criteria": [
                    {
                        "name": "Performance",
                        "weight": 0.3,
                        "description": "Query performance and throughput",
                    },
                    {
                        "name": "Scalability",
                        "weight": 0.25,
                        "description": "Ease of horizontal scaling",
                    },
                    {
                        "name": "Team Experience",
                        "weight": 0.25,
                        "description": "Existing team knowledge",
                    },
                    {
                        "name": "Operational Complexity",
                        "weight": 0.2,
                        "description": "Ease of deployment and maintenance",
                    },
                ],
                "analysis": {
                    "scoring": [
                        {
                            "option": "postgres",
                            "scores": {
                                "Performance": 8,
                                "Scalability": 6,
                                "Team Experience": 10,
                                "Operational Complexity": 8,
                            },
                            "weighted_total": 7.7,
                        },
                        {
                            "option": "mongo",
                            "scores": {
                                "Performance": 7,
                                "Scalability": 9,
                                "Team Experience": 4,
                                "Operational Complexity": 7,
                            },
                            "weighted_total": 6.8,
                        },
                        {
                            "option": "cassandra",
                            "scores": {
                                "Performance": 9,
                                "Scalability": 10,
                                "Team Experience": 2,
                                "Operational Complexity": 5,
                            },
                            "weighted_total": 6.7,
                        },
                    ],
                    "tradeoffs": [
                        "PostgreSQL offers best team fit but limited scalability",
                        "Cassandra has best performance but steep learning curve",
                        "MongoDB middle ground but lacks ACID guarantees",
                    ],
                },
                "decision": {
                    "selected_option": "postgres",
                    "rationale": [
                        "Highest weighted score (7.7)",
                        "Team can be immediately productive",
                        "ACID compliance out of the box",
                        "Can use read replicas for initial scaling",
                    ],
                    "key_factors": [
                        "Team experience reduces time to market",
                        "Proven solution for similar workloads",
                        "Lower operational risk",
                    ],
                },
                "validation": {
                    "confidence_score": 0.85,
                    "supporting_evidence": [
                        "Similar services using PostgreSQL handle 15K RPS",
                        "Team delivered 3 PostgreSQL projects successfully",
                        "Monitoring integration already exists",
                    ],
                    "assumptions_validated": [
                        "10K RPS achievable with proper indexing",
                        "Read replicas sufficient for 2-year growth",
                    ],
                    "remaining_risks": [
                        "May need to migrate if scale exceeds projections",
                        "Complex sharding if write scaling needed",
                    ],
                },
                "fallback_plan": {
                    "trigger": "If PostgreSQL can't meet performance after optimization",
                    "approach": "Implement caching layer with Redis",
                    "alternative": "Migrate hot data to Cassandra, keep PostgreSQL for complex queries",
                    "timeline": "Can be implemented in 4 weeks if needed",
                },
            }
        ]

    def _build_json_schema(self) -> Dict[str, Any]:
        """Override to build nested schema properly."""
        return {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "DecisionSchema",
            "description": self.get_description(),
            "required": [
                "context",
                "options",
                "evaluation_criteria",
                "analysis",
                "decision",
                "validation",
            ],
            "properties": {
                "context": {
                    "type": "object",
                    "required": ["problem", "requirements"],
                    "properties": {
                        "problem": {"type": "string"},
                        "requirements": {"type": "array", "items": {"type": "string"}},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "description"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "evaluation_criteria": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["name", "weight"],
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "description": {"type": "string"},
                        },
                    },
                },
                "analysis": {
                    "type": "object",
                    "required": ["scoring"],
                    "properties": {
                        "scoring": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "option": {"type": "string"},
                                    "scores": {"type": "object"},
                                    "weighted_total": {"type": "number"},
                                },
                            },
                        },
                        "tradeoffs": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "decision": {
                    "type": "object",
                    "required": ["selected_option", "rationale"],
                    "properties": {
                        "selected_option": {"type": "string"},
                        "rationale": {"type": "array", "items": {"type": "string"}},
                        "key_factors": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "validation": {
                    "type": "object",
                    "required": ["confidence_score"],
                    "properties": {
                        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                        "assumptions_validated": {"type": "array", "items": {"type": "string"}},
                        "remaining_risks": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "fallback_plan": {
                    "type": "object",
                    "properties": {
                        "trigger": {"type": "string"},
                        "approach": {"type": "string"},
                        "alternative": {"type": "string"},
                        "timeline": {"type": "string"},
                    },
                },
            },
            "additionalProperties": False,
        }
