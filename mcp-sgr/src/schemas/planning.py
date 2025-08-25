"""Planning schema for creating structured action plans."""

from typing import Any, Dict, List
from .base import BaseSchema, SchemaField


class PlanningSchema(BaseSchema):
    """Schema for structured planning and approach design."""
    
    def get_description(self) -> str:
        return "Structured planning including approach, steps, alternatives, and resource requirements"
    
    def get_fields(self) -> List[SchemaField]:
        return [
            SchemaField(
                name="approach",
                type="object",
                description="Overall approach and strategy",
                required=True
            ),
            SchemaField(
                name="steps",
                type="array",
                description="Detailed implementation steps",
                required=True
            ),
            SchemaField(
                name="alternatives",
                type="array",
                description="Alternative approaches considered",
                required=False
            ),
            SchemaField(
                name="timeline",
                type="object",
                description="Timeline and milestones",
                required=False
            ),
            SchemaField(
                name="resources",
                type="object",
                description="Required resources",
                required=True
            ),
            SchemaField(
                name="success_metrics",
                type="array",
                description="How to measure success",
                required=True
            )
        ]
    
    def get_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "approach": {
                    "strategy": "Incremental migration with backward compatibility",
                    "rationale": "Minimizes risk and allows rollback at any stage",
                    "key_principles": [
                        "Zero downtime during migration",
                        "Maintain data integrity throughout",
                        "Provide clear rollback procedures"
                    ]
                },
                "steps": [
                    {
                        "id": 1,
                        "name": "Setup parallel infrastructure",
                        "description": "Create new database schema alongside existing",
                        "duration": "2 days",
                        "dependencies": [],
                        "deliverables": ["New schema created", "Migration scripts ready"]
                    },
                    {
                        "id": 2,
                        "name": "Implement dual-write logic",
                        "description": "Write to both old and new systems",
                        "duration": "3 days",
                        "dependencies": [1],
                        "deliverables": ["Dual-write code deployed", "Monitoring in place"]
                    },
                    {
                        "id": 3,
                        "name": "Migrate historical data",
                        "description": "Batch migrate existing records",
                        "duration": "1 day",
                        "dependencies": [2],
                        "deliverables": ["All data migrated", "Verification complete"]
                    }
                ],
                "alternatives": [
                    {
                        "approach": "Big bang migration",
                        "pros": ["Faster completion", "Simpler implementation"],
                        "cons": ["Higher risk", "Difficult rollback", "Requires downtime"],
                        "rejection_reason": "Unacceptable downtime for production system"
                    }
                ],
                "timeline": {
                    "total_duration": "2 weeks",
                    "milestones": [
                        {"name": "Infrastructure ready", "date": "Day 3"},
                        {"name": "Dual-write active", "date": "Day 7"},
                        {"name": "Migration complete", "date": "Day 14"}
                    ],
                    "buffer": "20% contingency included"
                },
                "resources": {
                    "team": ["2 backend engineers", "1 DBA", "1 DevOps"],
                    "tools": ["Migration framework", "Data validation tools"],
                    "infrastructure": ["Staging environment", "Additional database capacity"]
                },
                "success_metrics": [
                    "Zero data loss verified by checksums",
                    "No increase in error rates during migration",
                    "Performance metrics remain within SLA",
                    "Successful rollback test completed"
                ]
            }
        ]
    
    def _build_json_schema(self) -> Dict[str, Any]:
        """Override to build nested schema properly."""
        return {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "PlanningSchema",
            "description": self.get_description(),
            "required": ["approach", "steps", "resources", "success_metrics"],
            "properties": {
                "approach": {
                    "type": "object",
                    "required": ["strategy", "rationale"],
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "description": "High-level strategy"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this approach"
                        },
                        "key_principles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Guiding principles"
                        }
                    }
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "description"],
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "duration": {"type": "string"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "deliverables": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "approach": {"type": "string"},
                            "pros": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "cons": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "rejection_reason": {"type": "string"}
                        }
                    }
                },
                "timeline": {
                    "type": "object",
                    "properties": {
                        "total_duration": {"type": "string"},
                        "milestones": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "date": {"type": "string"}
                                }
                            }
                        },
                        "buffer": {"type": "string"}
                    }
                },
                "resources": {
                    "type": "object",
                    "properties": {
                        "team": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "infrastructure": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "success_metrics": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                    "description": "Measurable success criteria"
                }
            },
            "additionalProperties": False
        }