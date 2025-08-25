"""Code generation schema for structured code development."""

from typing import Any, Dict, List
from .base import BaseSchema, SchemaField


class CodeGenerationSchema(BaseSchema):
    """Schema for code generation with design, implementation, and validation."""
    
    def get_description(self) -> str:
        return "Structured code generation including requirements, design, implementation, and testing"
    
    def get_fields(self) -> List[SchemaField]:
        return [
            SchemaField(
                name="understanding",
                type="object",
                description="Understanding of the coding task",
                required=True
            ),
            SchemaField(
                name="design",
                type="object",
                description="Technical design and architecture",
                required=True
            ),
            SchemaField(
                name="implementation",
                type="object",
                description="Code implementation details",
                required=True
            ),
            SchemaField(
                name="validation",
                type="object",
                description="Testing and validation approach",
                required=True
            ),
            SchemaField(
                name="documentation",
                type="object",
                description="Documentation plan",
                required=False
            )
        ]
    
    def get_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "understanding": {
                    "goal": "Create a rate limiter middleware for FastAPI",
                    "requirements": [
                        "Support multiple rate limiting strategies",
                        "Redis-based distributed rate limiting",
                        "Configurable per endpoint",
                        "Return proper 429 status with headers"
                    ],
                    "constraints": [
                        "Must be async compatible",
                        "Minimal performance overhead",
                        "Support both IP and user-based limiting"
                    ],
                    "assumptions": [
                        "Redis is available and configured",
                        "FastAPI 0.100+ is used",
                        "Python 3.11+ environment"
                    ]
                },
                "design": {
                    "approach": "Decorator-based middleware with pluggable backends",
                    "architecture": {
                        "components": [
                            "RateLimiter base class",
                            "Redis backend implementation",
                            "In-memory backend for testing",
                            "FastAPI dependency injection"
                        ],
                        "patterns": [
                            "Strategy pattern for different algorithms",
                            "Decorator pattern for endpoint configuration",
                            "Factory pattern for backend selection"
                        ]
                    },
                    "interfaces": [
                        {
                            "name": "RateLimiter",
                            "methods": ["check_rate_limit", "increment_counter", "get_ttl"]
                        },
                        {
                            "name": "RateLimitBackend",
                            "methods": ["get", "set", "incr", "expire"]
                        }
                    ],
                    "data_flow": [
                        "Request arrives at endpoint",
                        "Middleware extracts identifier (IP/user)",
                        "Check current count in backend",
                        "Accept or reject based on limit",
                        "Increment counter if accepted"
                    ]
                },
                "implementation": {
                    "language": "python",
                    "framework": "FastAPI",
                    "dependencies": [
                        "redis>=4.5.0",
                        "fastapi>=0.100.0",
                        "pydantic>=2.0"
                    ],
                    "structure": {
                        "files": [
                            "rate_limiter/__init__.py",
                            "rate_limiter/core.py",
                            "rate_limiter/backends.py",
                            "rate_limiter/middleware.py",
                            "rate_limiter/decorators.py"
                        ],
                        "classes": [
                            "RateLimiter",
                            "RedisBackend",
                            "InMemoryBackend",
                            "RateLimitMiddleware"
                        ]
                    },
                    "key_algorithms": [
                        {
                            "name": "Token bucket",
                            "description": "Allows burst traffic up to bucket size"
                        },
                        {
                            "name": "Fixed window",
                            "description": "Simple counter reset every time window"
                        },
                        {
                            "name": "Sliding window",
                            "description": "Smooth rate limiting without reset spikes"
                        }
                    ],
                    "code_snippets": {
                        "decorator_example": "@rate_limit(calls=100, period=60)",
                        "middleware_setup": "app.add_middleware(RateLimitMiddleware, backend=redis_backend)"
                    }
                },
                "validation": {
                    "test_strategy": "Unit tests for algorithms, integration tests for middleware",
                    "test_cases": [
                        "Single request within limit",
                        "Burst requests exceeding limit",
                        "Rate limit reset after time window",
                        "Concurrent requests handling",
                        "Different identifiers (IP vs user)",
                        "Redis connection failure fallback"
                    ],
                    "performance_tests": [
                        "Measure overhead per request",
                        "Load test with 10K concurrent requests",
                        "Redis operation latency impact"
                    ],
                    "edge_cases": [
                        "Redis unavailable",
                        "Clock skew between servers",
                        "IP behind proxy/load balancer",
                        "User without authentication"
                    ],
                    "validation_criteria": [
                        "All tests pass",
                        "< 1ms overhead per request",
                        "Proper HTTP headers returned",
                        "Graceful degradation on backend failure"
                    ]
                },
                "documentation": {
                    "api_docs": "OpenAPI schema with rate limit annotations",
                    "usage_examples": [
                        "Basic setup with Redis",
                        "Custom rate limit per endpoint",
                        "User-based rate limiting",
                        "Whitelist certain IPs"
                    ],
                    "configuration": {
                        "RATE_LIMIT_ENABLED": "Toggle rate limiting",
                        "RATE_LIMIT_BACKEND": "redis|memory",
                        "RATE_LIMIT_REDIS_URL": "Redis connection string",
                        "RATE_LIMIT_DEFAULT": "Default limit if not specified"
                    }
                }
            }
        ]
    
    def _build_json_schema(self) -> Dict[str, Any]:
        """Override to build nested schema properly."""
        return {
            "$id": f"schema://{self.schema_id}",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "CodeGenerationSchema",
            "description": self.get_description(),
            "required": ["understanding", "design", "implementation", "validation"],
            "properties": {
                "understanding": {
                    "type": "object",
                    "required": ["goal", "requirements"],
                    "properties": {
                        "goal": {
                            "type": "string",
                            "minLength": 10,
                            "description": "Main goal of the code"
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "assumptions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "design": {
                    "type": "object",
                    "required": ["approach"],
                    "properties": {
                        "approach": {
                            "type": "string",
                            "description": "High-level technical approach"
                        },
                        "architecture": {
                            "type": "object",
                            "properties": {
                                "components": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "patterns": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "interfaces": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "methods": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "data_flow": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "implementation": {
                    "type": "object",
                    "required": ["language"],
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript", "typescript", "go", "java", "rust", "other"]
                        },
                        "framework": {"type": "string"},
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "structure": {
                            "type": "object",
                            "properties": {
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "classes": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "key_algorithms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            }
                        },
                        "code_snippets": {
                            "type": "object",
                            "additionalProperties": {"type": "string"}
                        }
                    }
                },
                "validation": {
                    "type": "object",
                    "required": ["test_strategy", "test_cases"],
                    "properties": {
                        "test_strategy": {"type": "string"},
                        "test_cases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "performance_tests": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "edge_cases": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "validation_criteria": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "documentation": {
                    "type": "object",
                    "properties": {
                        "api_docs": {"type": "string"},
                        "usage_examples": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "configuration": {
                            "type": "object",
                            "additionalProperties": {"type": "string"}
                        }
                    }
                }
            },
            "additionalProperties": False
        }