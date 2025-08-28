"""SGR Schema definitions."""

from typing import Dict, Type

from .analysis import AnalysisSchema
from .base import BaseSchema, SchemaField, ValidationResult
from .code import CodeGenerationSchema
from .decision import DecisionSchema
from .planning import PlanningSchema
from .rag import RAGAnalysisSchema, RAGValidationSchema
from .story_generation import StoryGenerationSchema, StoryResponse
from .summary import SummarizationSchema

SCHEMA_REGISTRY: Dict[str, Type[BaseSchema]] = {
    "analysis": AnalysisSchema,
    "planning": PlanningSchema,
    "decision": DecisionSchema,
    "code_generation": CodeGenerationSchema,
    "story_generation": StoryGenerationSchema,
    "summarization": SummarizationSchema,
}

__all__ = [
    "BaseSchema",
    "SchemaField",
    "ValidationResult",
    "AnalysisSchema",
    "PlanningSchema",
    "DecisionSchema",
    "CodeGenerationSchema",
    "StoryGenerationSchema",
    "StoryResponse",
    "SummarizationSchema",
    "RAGAnalysisSchema",
    "RAGValidationSchema",
    "SCHEMA_REGISTRY",
]
