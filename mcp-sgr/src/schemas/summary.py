"""Summarization schema for structured content summarization."""

from typing import Any, Dict, List
from .base import BaseSchema, SchemaField


class SummarizationSchema(BaseSchema):
    """Schema for structured summarization with context and validation."""
    
    def get_description(self) -> str:
        return "Structured summarization including purpose, key points, and validation"
    
    def get_fields(self) -> List[SchemaField]:
        return [
            SchemaField(
                name="purpose",
                type="object",
                description="Purpose and context of the summary",
                required=True
            ),
            SchemaField(
                name="content_analysis",
                type="object",
                description="Analysis of source content",
                required=True
            ),
            SchemaField(
                name="key_points",
                type="array",
                description="Main points extracted",
                required=True
            ),
            SchemaField(
                name="summary",
                type="object",
                description="The actual summary",
                required=True
            ),
            SchemaField(
                name="validation",
                type="object",
                description="Summary validation and quality checks",
                required=True
            )
        ]
    
    def get_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "purpose": {
                    "goal": "Summarize technical RFC for engineering team",
                    "audience": "Senior engineers and architects",
                    "desired_length": "500 words",
                    "focus_areas": [
                        "Technical implementation details",
                        "Breaking changes",
                        "Migration strategy"
                    ],
                    "constraints": [
                        "Maintain technical accuracy",
                        "Include all critical warnings",
                        "Preserve specific version numbers"
                    ]
                },
                "content_analysis": {
                    "source_type": "Technical RFC document",
                    "length": "8,500 words",
                    "complexity": "high",
                    "main_sections": [
                        "Problem Statement",
                        "Proposed Solution",
                        "Implementation Details",
                        "Breaking Changes",
                        "Migration Guide",
                        "Security Considerations"
                    ],
                    "key_themes": [
                        "API versioning strategy",
                        "Backward compatibility",
                        "Performance improvements",
                        "Security enhancements"
                    ],
                    "technical_depth": "Implementation-level details with code examples"
                },
                "key_points": [
                    {
                        "point": "New API versioning scheme using headers instead of URL paths",
                        "importance": "critical",
                        "section": "Proposed Solution",
                        "supporting_details": [
                            "Version specified via 'API-Version' header",
                            "Default version strategy for unspecified requests",
                            "Deprecation timeline of 6 months"
                        ]
                    },
                    {
                        "point": "Breaking changes in authentication flow",
                        "importance": "critical",
                        "section": "Breaking Changes",
                        "supporting_details": [
                            "OAuth 2.0 replacing API keys",
                            "New token refresh mechanism",
                            "Existing tokens valid for 30 days"
                        ]
                    },
                    {
                        "point": "Performance improvements through caching layer",
                        "importance": "high",
                        "section": "Implementation Details",
                        "supporting_details": [
                            "Redis-based response caching",
                            "30% reduction in response time",
                            "Configurable TTL per endpoint"
                        ]
                    },
                    {
                        "point": "Phased migration approach",
                        "importance": "high",
                        "section": "Migration Guide",
                        "supporting_details": [
                            "Phase 1: Dual support (3 months)",
                            "Phase 2: Deprecation warnings (2 months)",
                            "Phase 3: Old API sunset (1 month)"
                        ]
                    }
                ],
                "summary": {
                    "executive_summary": "The RFC proposes a major API redesign focusing on versioning, authentication, and performance. The new system moves from URL-based to header-based versioning, replaces API keys with OAuth 2.0, and introduces a Redis caching layer for improved performance.",
                    "detailed_summary": "This RFC outlines a comprehensive API v2.0 upgrade addressing three critical areas:\n\n1. **Versioning Strategy**: The proposal shifts from URL path versioning (/v1/, /v2/) to header-based versioning using 'API-Version' headers. This change allows cleaner URLs and better version negotiation. Unspecified versions will default to v1.0 until deprecation.\n\n2. **Authentication Overhaul**: API keys will be replaced with OAuth 2.0 authentication. This provides better security through token rotation and scope-based permissions. Existing API keys remain valid for 30 days post-launch.\n\n3. **Performance Enhancements**: A new Redis-based caching layer will reduce response times by approximately 30%. Cache TTL is configurable per endpoint, with intelligent invalidation based on resource updates.\n\n**Migration Plan**: A 6-month phased approach ensures smooth transition:\n- Months 1-3: Both APIs run in parallel\n- Months 4-5: Deprecation warnings on v1 endpoints\n- Month 6: v1 API sunset\n\n**Critical Actions Required**:\n- Update client libraries to support header-based versioning\n- Implement OAuth 2.0 flow before API key expiration\n- Review caching implications for real-time data endpoints",
                    "technical_notes": [
                        "Header format: 'API-Version: 2.0' (semantic versioning)",
                        "OAuth scopes: read, write, admin (granular per resource)",
                        "Cache key format: {version}:{endpoint}:{params_hash}",
                        "Rate limits increase from 100 to 500 requests/minute"
                    ]
                },
                "validation": {
                    "completeness_check": {
                        "all_sections_covered": true,
                        "key_points_preserved": true,
                        "critical_details_included": true
                    },
                    "accuracy_verification": {
                        "technical_terms_correct": true,
                        "numbers_preserved": true,
                        "timeline_accurate": true
                    },
                    "length_compliance": {
                        "target_length": 500,
                        "actual_length": 487,
                        "within_tolerance": true
                    },
                    "audience_appropriateness": {
                        "technical_level": "appropriate",
                        "assumed_knowledge": "API design, OAuth, caching",
                        "jargon_explained": false
                    },
                    "quality_score": 0.92,
                    "potential_improvements": [
                        "Add specific code examples for header implementation",
                        "Include performance benchmarks data",
                        "Clarify Redis configuration requirements"
                    ]
                }
            }
        ]
    
    def _build_json_schema(self) -> Dict[str, Any]:
        """Override to build nested schema properly."""
        return {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "SummarizationSchema",
            "description": self.get_description(),
            "required": ["purpose", "content_analysis", "key_points", "summary", "validation"],
            "properties": {
                "purpose": {
                    "type": "object",
                    "required": ["goal", "audience"],
                    "properties": {
                        "goal": {"type": "string"},
                        "audience": {"type": "string"},
                        "desired_length": {"type": "string"},
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "content_analysis": {
                    "type": "object",
                    "required": ["source_type", "key_themes"],
                    "properties": {
                        "source_type": {"type": "string"},
                        "length": {"type": "string"},
                        "complexity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"]
                        },
                        "main_sections": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "key_themes": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "technical_depth": {"type": "string"}
                    }
                },
                "key_points": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["point", "importance"],
                        "properties": {
                            "point": {"type": "string"},
                            "importance": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"]
                            },
                            "section": {"type": "string"},
                            "supporting_details": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "summary": {
                    "type": "object",
                    "required": ["executive_summary", "detailed_summary"],
                    "properties": {
                        "executive_summary": {
                            "type": "string",
                            "minLength": 50,
                            "maxLength": 300
                        },
                        "detailed_summary": {
                            "type": "string",
                            "minLength": 200
                        },
                        "technical_notes": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "validation": {
                    "type": "object",
                    "required": ["completeness_check", "quality_score"],
                    "properties": {
                        "completeness_check": {
                            "type": "object",
                            "properties": {
                                "all_sections_covered": {"type": "boolean"},
                                "key_points_preserved": {"type": "boolean"},
                                "critical_details_included": {"type": "boolean"}
                            }
                        },
                        "accuracy_verification": {
                            "type": "object",
                            "properties": {
                                "technical_terms_correct": {"type": "boolean"},
                                "numbers_preserved": {"type": "boolean"},
                                "timeline_accurate": {"type": "boolean"}
                            }
                        },
                        "length_compliance": {
                            "type": "object",
                            "properties": {
                                "target_length": {"type": "integer"},
                                "actual_length": {"type": "integer"},
                                "within_tolerance": {"type": "boolean"}
                            }
                        },
                        "audience_appropriateness": {
                            "type": "object",
                            "properties": {
                                "technical_level": {"type": "string"},
                                "assumed_knowledge": {"type": "string"},
                                "jargon_explained": {"type": "boolean"}
                            }
                        },
                        "quality_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "potential_improvements": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            },
            "additionalProperties": False
        }