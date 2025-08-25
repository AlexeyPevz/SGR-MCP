"""Apply SGR tool implementation."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..schemas import SCHEMA_REGISTRY
from ..schemas.base import BudgetDepth
from ..schemas.custom import CustomSchema
from ..utils.cache import CacheManager
from ..utils.llm_client import LLMClient
from ..utils.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


async def apply_sgr_tool(
    arguments: Dict[str, Any],
    llm_client: LLMClient,
    cache_manager: CacheManager,
    telemetry: TelemetryManager,
) -> Dict[str, Any]:
    """Apply SGR schema to a task.

    Args:
        arguments: Tool arguments containing task, context, schema_type, etc.
        llm_client: LLM client for reasoning generation
        cache_manager: Cache manager for storing results
        telemetry: Telemetry manager for tracking

    Returns:
        Dictionary containing reasoning, confidence, actions, and metadata
    """
    # Start telemetry span
    span_id = await telemetry.start_span("apply_sgr", arguments)

    try:
        # Extract arguments
        task = arguments["task"]
        context = arguments.get("context", {})
        schema_type = arguments.get("schema_type", "auto")
        custom_schema_def = arguments.get("custom_schema")
        budget = arguments.get("budget", "lite")

        # Check cache first (stable key)
        import hashlib

        payload_for_key = {
            "task": task,
            "context": context,
            "schema_type": schema_type,
            "budget": budget,
        }
        key_str = json.dumps(payload_for_key, sort_keys=True, ensure_ascii=False)
        cache_key = f"sgr:{hashlib.sha256(key_str.encode('utf-8')).hexdigest()}"
        cached_result = await cache_manager.get(cache_key)
        if cached_result and budget != "full":
            logger.info(f"Cache hit for {cache_key}")
            await telemetry.end_span(span_id, {"cache_hit": True})
            return cached_result

        # Auto-detect schema type if needed
        if schema_type == "auto":
            schema_type = await _detect_schema_type(task, context, llm_client)
            logger.info(f"Auto-detected schema type: {schema_type}")

        # Get or create schema
        if schema_type == "custom" and custom_schema_def:
            schema = CustomSchema(custom_schema_def)
        elif schema_type in SCHEMA_REGISTRY:
            schema = SCHEMA_REGISTRY[schema_type]()
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

        # Generate reasoning based on budget
        reasoning = await _generate_reasoning(task, context, schema, budget, llm_client)

        # Validate reasoning
        validation_result = schema.validate(reasoning)

        # Extract suggested actions
        actions = _extract_actions(reasoning, schema_type)

        # Prepare result
        result = {
            "reasoning": reasoning,
            "confidence": validation_result.confidence,
            "suggested_actions": actions,
            "metadata": {
                "schema_type": schema_type,
                "budget": budget,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validation": {
                    "valid": validation_result.valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                },
            },
        }

        # Cache result
        await cache_manager.set(cache_key, result, ttl=3600)

        # End telemetry span
        await telemetry.end_span(
            span_id,
            {
                "confidence": validation_result.confidence,
                "schema_type": schema_type,
                "cache_hit": False,
            },
        )

        return result

    except Exception as e:
        logger.error(f"Error in apply_sgr: {e}", exc_info=True)
        await telemetry.end_span(span_id, {"error": str(e)})
        raise


async def _detect_schema_type(task: str, context: Dict[str, Any], llm_client: LLMClient) -> str:
    """Auto-detect the best schema type for a task."""

    prompt = f"""Analyze this task and determine the best reasoning schema type.

Task: {task}
Context: {json.dumps(context, indent=2) if context else "None"}

Available schema types:
- analysis: For understanding problems, identifying constraints and risks
- planning: For creating step-by-step plans and approaches
- decision: For comparing options and making choices
- code_generation: For designing and implementing code
- summarization: For condensing and structuring information

Return only the schema type name that best fits this task."""

    response = await llm_client.generate(prompt, temperature=0.1)
    schema_type = response.strip().lower()

    # Validate response
    if schema_type not in SCHEMA_REGISTRY:
        logger.warning(f"Invalid auto-detected schema: {schema_type}, using 'analysis'")
        return "analysis"

    return schema_type


async def _generate_reasoning(
    task: str, context: Dict[str, Any], schema: Any, budget: str, llm_client: LLMClient
) -> Dict[str, Any]:
    """Generate reasoning based on schema and budget."""

    # Adjust prompt based on budget
    if budget == "none":
        # Minimal reasoning - just basic structure
        prompt = f"""Task: {task}

Provide a minimal JSON response following this structure:
{json.dumps(schema.to_json_schema()["properties"], indent=2)}

Be extremely concise - one line per field maximum."""

    elif budget == "lite":
        # Standard reasoning
        prompt = schema.generate_prompt(task, context)

    else:  # full
        # Comprehensive reasoning with examples
        examples = schema.get_examples()
        example_text = ""
        if examples:
            example_text = f"\n\nExample of good reasoning:\n{json.dumps(examples[0], indent=2)}"

        prompt = f"""{schema.generate_prompt(task, context)}

{example_text}

Requirements for FULL analysis:
1. Be extremely thorough and detailed
2. Consider edge cases and alternatives
3. Provide specific, actionable insights
4. Include confidence assessments where relevant
5. Think step-by-step through each section

Take your time to provide comprehensive reasoning."""

    # Generate reasoning
    try:
        response = await llm_client.generate(
            prompt,
            temperature=0.3 if budget == "full" else 0.1,
            max_tokens=4000 if budget == "full" else 2000,
        )
    except Exception as e:
        logger.error(f"LLM generation failed, falling back to minimal structure: {e}")
        # Fallback to minimal reasoning structure matching the schema shape
        schema_json = schema.to_json_schema()
        fallback: Dict[str, Any] = {}
        props = schema_json.get("properties", {})
        for key, prop in props.items():
            t = prop.get("type")
            if t == "object":
                fallback[key] = {}
            elif t == "array":
                fallback[key] = []
            elif t == "number" or t == "integer":
                fallback[key] = 0
            elif t == "boolean":
                fallback[key] = False
            else:
                fallback[key] = ""
        return fallback

    # Parse JSON response
    try:
        # Clean response if needed
        raw = response.strip()
        # Strip code fences
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        # Try direct load
        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse reasoning JSON: {e}")
        logger.debug(f"Raw response: {response}")

        # Try to balance braces and extract best-effort JSON
        import re

        candidate = raw
        # Find first '{' and last '}'
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception as inner_err:
                logger.debug(f"Snippet JSON parse failed: {inner_err}")

        # Regex fallback
        json_match = re.search(r"\{[\s\S]*\}", candidate)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception as regex_err:
                logger.debug(f"Regex JSON parse failed: {regex_err}")

        # Fallback - return minimal structure
        return {"error": "Failed to parse reasoning", "raw_response": response}


def _extract_actions(reasoning: Dict[str, Any], schema_type: str) -> List[str]:
    """Extract suggested actions from reasoning."""
    actions = []

    if schema_type == "analysis":
        # Extract from risks and data gaps
        risks = reasoning.get("risks", [])
        for risk in risks:
            if isinstance(risk, dict) and "mitigation" in risk:
                actions.append(f"Mitigate risk: {risk['mitigation']}")

        gaps = reasoning.get("data_gaps", [])
        for gap in gaps:
            actions.append(f"Gather information: {gap}")

    elif schema_type == "planning":
        # Extract from steps
        steps = reasoning.get("steps", [])
        for step in steps:
            if isinstance(step, dict):
                actions.append(f"Step {step.get('id', '?')}: {step.get('name', 'Unknown')}")
            else:
                actions.append(str(step))

    elif schema_type == "decision":
        # Extract selected option and validation
        decision = reasoning.get("decision", {})
        if "selected_option" in decision:
            actions.append(f"Implement option: {decision['selected_option']}")

        fallback = reasoning.get("fallback_plan", {})
        if "trigger" in fallback:
            actions.append(f"Monitor for: {fallback['trigger']}")

    elif schema_type == "code_generation":
        # Extract implementation steps
        impl = reasoning.get("implementation", {})
        if "structure" in impl and "files" in impl["structure"]:
            for file in impl["structure"]["files"]:
                actions.append(f"Create file: {file}")

        validation = reasoning.get("validation", {})
        for test in validation.get("test_cases", []):
            actions.append(f"Test: {test}")

    elif schema_type == "summarization":
        # Extract key points
        for point in reasoning.get("key_points", []):
            if isinstance(point, dict):
                actions.append(f"Highlight: {point.get('point', 'Unknown')}")

    return actions[:10]  # Limit to 10 actions
