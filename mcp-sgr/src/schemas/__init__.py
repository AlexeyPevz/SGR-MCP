"""SGR Schema definitions."""

from typing import Dict, Type

from .analysis import AnalysisSchema
from .base import BaseSchema, SchemaField, ValidationResult
from .code import CodeGenerationSchema
from .decision import DecisionSchema
from .planning import PlanningSchema
from .summary import SummarizationSchema

SCHEMA_REGISTRY: Dict[str, Type[BaseSchema]] = {
    "analysis": AnalysisSchema,
    "planning": PlanningSchema,
    "decision": DecisionSchema,
    "code_generation": CodeGenerationSchema,
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
    "SummarizationSchema",
    "SCHEMA_REGISTRY",
]
