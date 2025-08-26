"""Improved Apply SGR tool implementation v2."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib

from ..schemas import SCHEMA_REGISTRY
from ..schemas.base import BaseSchema, ValidationResult
from ..schemas.custom import CustomSchema
from ..utils.cache import CacheManager
from ..utils.llm_client import LLMClient
from ..utils.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


# Модели с подтвержденной поддержкой structured output
STRUCTURED_OUTPUT_MODELS = {
    "claude-3.5": ["haiku", "sonnet"],
    "gpt-4": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "gemini": ["gemini-pro", "gemini-flash", "gemini-1.5"],
}


def supports_structured_output(model: str) -> bool:
    """Check if model supports structured output."""
    model_lower = model.lower()
    
    # Check known good models
    for family, variants in STRUCTURED_OUTPUT_MODELS.items():
        if family in model_lower:
            return any(variant in model_lower for variant in variants)
    
    # Default to false for unknown models
    return False


async def apply_sgr_tool_v2(
    arguments: Dict[str, Any],
    llm_client: LLMClient,
    cache_manager: CacheManager,
    telemetry: TelemetryManager,
) -> Dict[str, Any]:
    """Apply SGR schema to a task - improved version.
    
    Key improvements:
    1. Better model compatibility checking
    2. Simplified schemas for better compatibility
    3. Improved error handling and fallbacks
    4. More efficient prompt structure
    """
    
    # Start telemetry
    span_id = await telemetry.start_span("apply_sgr_v2", arguments)
    
    try:
        # Extract arguments
        task = arguments["task"]
        context = arguments.get("context", {})
        schema_type = arguments.get("schema_type", "auto")
        custom_schema_def = arguments.get("custom_schema")
        budget = arguments.get("budget", "lite")
        backend = arguments.get("backend")
        
        # Get current model
        import os
        current_model = os.getenv("OPENROUTER_DEFAULT_MODEL", "unknown")
        
        # Check model compatibility
        if not supports_structured_output(current_model):
            logger.warning(f"Model {current_model} does not support structured output. Results may be poor.")
            # Force simpler approach for incompatible models
            if budget == "full":
                budget = "lite"
        
        # Generate cache key
        cache_key = generate_cache_key(task, context, schema_type, budget, backend, current_model)
        
        # Check cache
        if budget != "full":  # Don't cache full mode
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {cache_key}")
                await telemetry.end_span(span_id, {"cache_hit": True})
                return cached_result
        
        # Auto-detect schema type if needed
        if schema_type == "auto":
            schema_type = await detect_schema_type(task, context, llm_client)
            logger.info(f"Auto-detected schema type: {schema_type}")
        
        # Get or create schema
        schema = get_schema(schema_type, custom_schema_def)
        
        # Generate reasoning with improved approach
        result = await generate_reasoning_v2(
            task, context, schema, budget, llm_client, 
            backend, supports_structured_output(current_model)
        )
        
        # Cache result if valid
        if result.get("metadata", {}).get("validation", {}).get("valid", False):
            await cache_manager.set(cache_key, result, ttl=3600)
        
        await telemetry.end_span(span_id, {"success": True})
        return result
        
    except Exception as e:
        logger.error(f"Error in apply_sgr_v2: {e}")
        await telemetry.end_span(span_id, {"error": str(e)})
        
        # Return structured error response
        return {
            "reasoning": {"error": str(e)},
            "confidence": 0.0,
            "suggested_actions": [],
            "metadata": {
                "validation": {"valid": False, "errors": [str(e)]},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "budget": budget,
                "error": True
            }
        }


def generate_cache_key(task: str, context: Dict, schema_type: str, 
                      budget: str, backend: Optional[str], model: str) -> str:
    """Generate stable cache key."""
    payload = {
        "task": task,
        "context": context,
        "schema_type": schema_type,
        "budget": budget,
        "backend": backend or "auto",
        "model": model,
        "version": "v2"
    }
    key_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"sgr:v2:{hashlib.sha256(key_str.encode('utf-8')).hexdigest()}"


async def detect_schema_type(task: str, context: Dict, llm_client: LLMClient) -> str:
    """Auto-detect appropriate schema type - simplified."""
    
    # Simple keyword-based detection
    task_lower = task.lower()
    
    if any(word in task_lower for word in ["analyze", "review", "check", "evaluate"]):
        return "analysis"
    elif any(word in task_lower for word in ["plan", "design", "architect", "roadmap"]):
        return "planning"
    elif any(word in task_lower for word in ["decide", "choose", "select", "compare"]):
        return "decision"
    elif any(word in task_lower for word in ["code", "implement", "function", "class"]):
        return "code_generation"
    elif any(word in task_lower for word in ["summarize", "summary", "brief", "overview"]):
        return "summarization"
    
    # Default to analysis
    return "analysis"


def get_schema(schema_type: str, custom_schema_def: Optional[Dict]) -> BaseSchema:
    """Get or create schema instance."""
    if schema_type == "custom" and custom_schema_def:
        return CustomSchema(custom_schema_def)
    
    schema_factory = SCHEMA_REGISTRY.get(schema_type)
    if not schema_factory:
        # Fallback to analysis
        logger.warning(f"Unknown schema type: {schema_type}, falling back to analysis")
        schema_factory = SCHEMA_REGISTRY.get("analysis")
    
    return schema_factory()


async def generate_reasoning_v2(
    task: str,
    context: Dict,
    schema: BaseSchema,
    budget: str,
    llm_client: LLMClient,
    backend: Optional[str],
    supports_structured: bool
) -> Dict[str, Any]:
    """Generate reasoning with improved approach."""
    
    # Get simplified schema for better compatibility
    if not supports_structured:
        schema_json = simplify_schema(schema.to_json_schema())
    else:
        schema_json = schema.to_json_schema()
    
    # Generate prompt based on budget
    if budget == "lite":
        prompt = generate_lite_prompt(task, context, schema_json)
        max_tokens = 1000
    else:  # full
        prompt = generate_full_prompt(task, context, schema_json, schema)
        max_tokens = 2000
    
    # System prompt - simplified for better compatibility
    system_prompt = (
        "You are a helpful assistant that provides structured JSON responses. "
        "Always respond with valid JSON matching the provided schema."
    )
    
    # Try to generate response
    try:
        # Configure response format
        response_format = None
        if supports_structured:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sgr_response",
                    "schema": schema_json,
                    "strict": False  # Less strict for better compatibility
                }
            }
        
        response = await llm_client.generate(
            prompt,
            temperature=0.1,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            backend=backend,
            response_format=response_format
        )
        
        # Parse response
        reasoning = parse_llm_response(response)
        
        # Validate
        validation_result = schema.validate(reasoning)
        
        # If invalid in lite mode, try to fix common issues
        if budget == "lite" and not validation_result.valid:
            reasoning = fix_common_issues(reasoning, schema_json)
            validation_result = schema.validate(reasoning)
        
        # Extract actions
        actions = extract_actions(reasoning, schema)
        
        # Calculate confidence
        confidence = calculate_confidence(reasoning, validation_result, budget)
        
        return {
            "reasoning": reasoning,
            "confidence": confidence,
            "suggested_actions": actions,
            "metadata": {
                "validation": {
                    "valid": validation_result.valid,
                    "errors": validation_result.errors
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "budget": budget,
                "schema_type": getattr(schema, "schema_id", "unknown"),
                "model_supports_structured": supports_structured
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate reasoning: {e}")
        
        # Return minimal valid response
        return {
            "reasoning": create_minimal_reasoning(schema_json),
            "confidence": 0.1,
            "suggested_actions": ["Review the task manually due to processing error"],
            "metadata": {
                "validation": {"valid": False, "errors": [str(e)]},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "budget": budget,
                "error": True
            }
        }


def simplify_schema(schema: Dict) -> Dict:
    """Simplify schema for models without structured output support."""
    # Remove complex validations
    simplified = {
        "type": "object",
        "properties": {}
    }
    
    for key, prop in schema.get("properties", {}).items():
        # Simplify property definitions
        if prop.get("type") == "array":
            simplified["properties"][key] = {
                "type": "array",
                "items": {"type": "string"}  # Simplify to string array
            }
        elif prop.get("type") == "object":
            # Flatten nested objects
            simplified["properties"][key] = {"type": "object"}
        else:
            # Keep simple types
            simplified["properties"][key] = {"type": prop.get("type", "string")}
    
    # Don't enforce required fields for non-structured models
    return simplified


def generate_lite_prompt(task: str, context: Dict, schema: Dict) -> str:
    """Generate simplified prompt for lite mode."""
    return f"""Task: {task}

Context: {json.dumps(context, indent=2) if context else "None"}

Analyze this task and provide a JSON response with these fields:
{json.dumps(schema.get("properties", {}), indent=2)}

Be concise but complete. Include a confidence score (0-1) if applicable."""


def generate_full_prompt(task: str, context: Dict, schema: Dict, schema_obj: BaseSchema) -> str:
    """Generate comprehensive prompt for full mode."""
    examples = schema_obj.get_examples() if hasattr(schema_obj, 'get_examples') else []
    example_text = ""
    if examples:
        example_text = f"\n\nExample response:\n{json.dumps(examples[0], indent=2)}"
    
    return f"""Task: {task}

Context: {json.dumps(context, indent=2) if context else "None"}

Provide a comprehensive analysis following this structure:
{json.dumps(schema.get("properties", {}), indent=2)}

Requirements:
1. Be thorough and detailed
2. Consider multiple perspectives
3. Provide specific, actionable insights
4. Include confidence assessment
5. Think step-by-step
{example_text}

Generate a complete JSON response."""


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response with better error handling."""
    try:
        # Try direct JSON parse
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return json.loads(response[start:end].strip())
        
        # Try to find JSON object
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except:
                pass
        
        # Return empty dict as last resort
        logger.warning("Failed to parse LLM response as JSON")
        return {}


def fix_common_issues(reasoning: Dict, schema: Dict) -> Dict:
    """Fix common validation issues."""
    if not reasoning:
        return {}
    
    # Ensure required fields exist
    for key in schema.get("required", []):
        if key not in reasoning:
            prop_type = schema.get("properties", {}).get(key, {}).get("type", "string")
            if prop_type == "array":
                reasoning[key] = []
            elif prop_type == "object":
                reasoning[key] = {}
            elif prop_type == "number":
                reasoning[key] = 0.5
            else:
                reasoning[key] = "Not specified"
    
    # Fix empty arrays (add at least one item)
    for key, value in reasoning.items():
        if isinstance(value, list) and len(value) == 0:
            reasoning[key] = ["No specific items identified"]
    
    return reasoning


def extract_actions(reasoning: Dict, schema: BaseSchema) -> List[str]:
    """Extract suggested actions from reasoning."""
    actions = []
    
    # Common action fields
    action_fields = ["suggested_actions", "recommendations", "next_steps", "improvements"]
    
    for field in action_fields:
        if field in reasoning and isinstance(reasoning[field], list):
            actions.extend(reasoning[field])
    
    # If no actions found, generate basic ones
    if not actions:
        if "issues" in reasoning and isinstance(reasoning["issues"], list):
            for issue in reasoning["issues"]:
                if isinstance(issue, dict):
                    actions.append(f"Address: {issue.get('description', 'Issue')}")
                elif isinstance(issue, str):
                    actions.append(f"Address: {issue}")
    
    return actions[:5]  # Limit to 5 actions


def calculate_confidence(reasoning: Dict, validation: ValidationResult, budget: str) -> float:
    """Calculate confidence score."""
    base_confidence = 0.5
    
    # Use validation confidence if available
    if validation.confidence > 0:
        base_confidence = validation.confidence
    
    # Use reasoning confidence if present
    if "confidence" in reasoning and isinstance(reasoning["confidence"], (int, float)):
        return float(reasoning["confidence"])
    
    # Adjust based on completeness
    if validation.valid:
        base_confidence += 0.2
    
    # Adjust based on budget
    if budget == "full":
        base_confidence += 0.1
    elif budget == "lite":
        base_confidence = min(base_confidence, 0.7)  # Cap lite mode confidence
    
    return min(base_confidence, 1.0)


def create_minimal_reasoning(schema: Dict) -> Dict:
    """Create minimal valid reasoning structure."""
    reasoning = {}
    
    for key, prop in schema.get("properties", {}).items():
        prop_type = prop.get("type", "string")
        if prop_type == "array":
            reasoning[key] = ["Unable to process - manual review needed"]
        elif prop_type == "object":
            reasoning[key] = {}
        elif prop_type == "number":
            reasoning[key] = 0.0
        elif prop_type == "boolean":
            reasoning[key] = False
        else:
            reasoning[key] = "Processing error - manual review needed"
    
    return reasoning