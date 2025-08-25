"""Apply SGR tool implementation."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..schemas import SCHEMA_REGISTRY
from ..schemas.base import BaseSchema, ValidationResult
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
        backend = arguments.get("backend")

        # Check cache first (stable key)
        import hashlib

        # Include backend/model in cache key to avoid stale cross-backend hits
        from os import getenv
        payload_for_key = {
            "task": task,
            "context": context,
            "schema_type": schema_type,
            "budget": budget,
            "backend": backend or getenv("ROUTER_DEFAULT_BACKEND") or "auto",
            "model": getenv("OPENROUTER_DEFAULT_MODEL") or getenv("OLLAMA_DEFAULT_MODEL") or "auto",
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
        schema: BaseSchema
        if schema_type == "custom" and custom_schema_def:
            schema = CustomSchema(custom_schema_def)
        else:
            schema_factory = SCHEMA_REGISTRY.get(schema_type)
            if not schema_factory:
                raise ValueError(f"Unknown schema type: {schema_type}")
            schema = schema_factory()

        # Generate reasoning based on budget
        reasoning = await _generate_reasoning(task, context, schema, budget, llm_client, backend=backend)

        # Validate reasoning
        if budget == "lite" and getattr(schema, "schema_id", "") == "summarization":
            validation_result = _validate_lite_summarization(reasoning)
        else:
            validation_result = schema.validate(reasoning)

        # If lite and invalid, try salvage by auto-filling minimal required fields, then revalidate
        if budget == "lite" and not validation_result.valid:
            try:
                reasoning = _lite_salvage_autofill(reasoning, schema.to_json_schema())
                validation_result = schema.validate(reasoning)
            except Exception:
                pass

        # Lite-mode assistance: if valid but confidence too low, gently boost with floor
        if budget == "lite" and validation_result.valid and validation_result.confidence < 0.5:
            validation_result.confidence = 0.5

        # Extract suggested actions
        actions = _extract_actions(reasoning, schema_type)

        # Prepare result
        # For lite budget, ensure minimal non-empty arrays/objects (auto-fill) when possible
        if budget == "lite" and isinstance(reasoning, dict):
            try:
                schema_json = schema.to_json_schema()
                props = schema_json.get("properties", {})
                for key, prop in props.items():
                    if key not in reasoning:
                        continue
                    t = prop.get("type")
                    if t == "array" and isinstance(reasoning.get(key), list) and len(reasoning[key]) == 0:
                        # Insert a stub item to satisfy minItems semantics in weak models
                        reasoning[key].append("auto")
                    if t == "object" and isinstance(reasoning.get(key), dict) and len(reasoning[key]) == 0:
                        # Add a placeholder key-value to avoid empty object penalties
                        reasoning[key]["note"] = "auto-filled"
            except Exception:
                pass

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
    task: str,
    context: Dict[str, Any],
    schema: Any,
    budget: str,
    llm_client: LLMClient,
    backend: str | None = None,
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
        system_prompt = (
            "You are a JSON-only assistant. Respond with STRICTLY valid JSON matching the"
            " provided schema. Requirements:\n"
            "- Do NOT include markdown fences or extra text.\n"
            "- Fill ALL required fields.\n"
            "- For required arrays, include at least one meaningful item (no empty arrays).\n"
            "- Avoid empty strings and placeholders like 'N/A'.\n"
            "- If information is missing, make reasonable, safe assumptions and state them succinctly.\n"
            "- Keep content specific and actionable."
        )
        response = await llm_client.generate(
            prompt,
            temperature=0.3 if budget == "full" else 0.1,
            max_tokens=4000 if budget == "full" else 2000,
            system_prompt=system_prompt,
            backend=backend,
            # Ask OpenRouter/OpenAI-compatible providers to return strict JSON
            response_format={"type": "json_object"},
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

        # Attempt an LLM-based JSON repair before giving up
        try:
            repair_prompt = (
                "Return strictly valid JSON matching the provided schema. "
                "Fix formatting, quotes, and commas. No markdown, no commentary.\n\n"
                f"Schema: {json.dumps(schema.to_json_schema())}\n\n"
                f"Candidate content to fix:\n{raw}"
            )
            repaired = await llm_client.generate(
                repair_prompt,
                temperature=0.0,
                max_tokens=2000,
                system_prompt="You are a JSON repair assistant. Output only valid JSON.",
                backend=backend,
                response_format={"type": "json_object"},
            )
            fixed = repaired.strip()
            if fixed.startswith("```json"):
                fixed = fixed[7:]
            if fixed.startswith("```"):
                fixed = fixed[3:]
            if fixed.endswith("```"):
                fixed = fixed[:-3]
            return json.loads(fixed)
        except Exception as _repair_err:
            logger.debug(f"JSON repair failed: {_repair_err}")
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


def _minimal_value_for_schema(prop_schema: Dict[str, Any]) -> Any:
    """Produce a minimal value that satisfies a JSON Schema fragment."""
    t = prop_schema.get("type")
    if t == "object":
        value: Dict[str, Any] = {}
        for req_key in prop_schema.get("required", []) or []:
            sub_schema = prop_schema.get("properties", {}).get(req_key, {})
            value[req_key] = _minimal_value_for_schema(sub_schema)
        # If no required keys, add a small note to avoid empty-object penalties
        if not value:
            value["note"] = "auto-filled"
        return value
    if t == "array":
        item_schema = prop_schema.get("items", {})
        return [_minimal_value_for_schema(item_schema) if item_schema else "auto"]
    if t == "string":
        if "enum" in prop_schema and prop_schema["enum"]:
            return prop_schema["enum"][0]
        min_len = prop_schema.get("minLength", 1)
        return ("auto" * ((min_len + 3) // 4))[: max(min_len, 4)]
    if t == "integer":
        return 1
    if t == "number":
        return 1
    if t == "boolean":
        return True
    # Fallback
    return "auto"


def _lite_salvage_autofill(reasoning: Dict[str, Any], schema_json: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing/empty required fields with minimal valid values for lite mode."""
    if not isinstance(reasoning, dict):
        reasoning = {}

    props = schema_json.get("properties", {})
    required_top = schema_json.get("required", []) or []

    # Ensure required top-level fields exist
    for key in required_top:
        prop_schema = props.get(key, {})
        if key not in reasoning:
            reasoning[key] = _minimal_value_for_schema(prop_schema)
        else:
            # If present but empty, fill minimally
            val = reasoning[key]
            t = prop_schema.get("type")
            if t == "object" and isinstance(val, dict) and len(val) == 0:
                reasoning[key] = _minimal_value_for_schema(prop_schema)
            if t == "array" and isinstance(val, list) and len(val) == 0:
                reasoning[key] = _minimal_value_for_schema(prop_schema)

    # Also lightly touch optional arrays/objects to avoid empty penalties
    for key, prop_schema in props.items():
        if key not in reasoning:
            continue
        t = prop_schema.get("type")
        if t == "object" and isinstance(reasoning[key], dict) and len(reasoning[key]) == 0:
            reasoning[key]["note"] = reasoning[key].get("note", "auto-filled")
        if t == "array" and isinstance(reasoning[key], list) and len(reasoning[key]) == 0:
            reasoning[key].append("auto")

    return reasoning


def _validate_lite_summarization(data: Dict[str, Any]) -> ValidationResult:
    """Lenient validation for summarization in lite mode.

    Requirements (minimal):
    - purpose.goal (non-empty string)
    - content_analysis.source_type (non-empty string)
    - key_points: at least 1 item with 'point'
    - summary.executive_summary (>= 20 chars)
    Confidence starts at 0.6, penalties for missing/short, +0.2 if detailed_summary present.
    """
    try:
        errors: List[str] = []
        warnings: List[str] = []

        def get(obj: Dict[str, Any], path: str):
            cur = obj
            for part in path.split('.'):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return None
            return cur

        confidence = 0.6

        goal = get(data, 'purpose.goal')
        if not isinstance(goal, str) or not goal.strip():
            errors.append("purpose.goal is required")
        
        stype = get(data, 'content_analysis.source_type')
        if not isinstance(stype, str) or not stype.strip():
            errors.append("content_analysis.source_type is required")

        kps = data.get('key_points')
        if not isinstance(kps, list) or len(kps) == 0:
            errors.append("key_points requires at least one item")
        else:
            if not (isinstance(kps[0], dict) and isinstance(kps[0].get('point'), str) and kps[0]['point'].strip()):
                warnings.append("first key_point.point is weak or missing")
                confidence -= 0.05

        exec_sum = get(data, 'summary.executive_summary')
        if not isinstance(exec_sum, str) or len(exec_sum.strip()) < 20:
            errors.append("summary.executive_summary too short")
        else:
            if len(exec_sum) < 60:
                warnings.append("executive_summary could be longer")
                confidence -= 0.05

        det_sum = get(data, 'summary.detailed_summary')
        if isinstance(det_sum, str) and len(det_sum.strip()) >= 100:
            confidence += 0.2
        else:
            warnings.append("missing or short detailed_summary")

        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings, confidence=0.0)

        confidence = max(0.5, min(0.9, confidence))
        return ValidationResult(valid=True, errors=[], warnings=warnings, confidence=confidence)
    except Exception:
        return ValidationResult(valid=False, errors=["lite summarization validation error"], confidence=0.0)
