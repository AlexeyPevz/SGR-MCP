"""SGR Schema definitions."""

from .base import BaseSchema, SchemaField, ValidationResult
from .analysis import AnalysisSchema
from .planning import PlanningSchema
from .decision import DecisionSchema
from .code import CodeGenerationSchema
from .summary import SummarizationSchema

SCHEMA_REGISTRY = {
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