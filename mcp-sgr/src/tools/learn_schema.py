"""Learn schema from examples tool implementation."""

import json
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from ..schemas.custom import SchemaBuilder
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


async def learn_schema_tool(arguments: Dict[str, Any], llm_client: LLMClient) -> Dict[str, Any]:
    """Learn a new SGR schema from provided examples.

    Analyzes examples to extract patterns and creates a custom schema.

    Args:
        arguments: Tool arguments containing examples and task_type
        llm_client: LLM client for analysis

    Returns:
        Dictionary with learned schema and metadata
    """
    try:
        # Extract arguments
        examples = arguments["examples"]
        task_type = arguments["task_type"]
        description = arguments.get("description", f"Learned schema for {task_type}")

        # Validate examples
        if len(examples) < 3:
            raise ValueError("At least 3 examples are required to learn a schema")

        # Analyze examples to extract common patterns
        pattern_analysis = await _analyze_example_patterns(examples, llm_client)

        # Build schema from patterns
        schema_builder = SchemaBuilder(
            schema_id=task_type.lower().replace(" ", "_"), description=description
        )

        # Add fields based on analysis
        for field_info in pattern_analysis["fields"]:
            if field_info["type"] == "object":
                # Handle nested objects
                schema_builder.add_object_field(
                    name=field_info["name"],
                    properties=field_info.get("properties", {}),
                    required=field_info["required"],
                    required_properties=field_info.get("required_properties", []),
                )
            elif field_info["type"] == "array":
                # Handle arrays
                schema_builder.add_array_field(
                    name=field_info["name"],
                    item_type=field_info.get("item_type", "string"),
                    description=field_info.get("description", ""),
                    required=field_info["required"],
                    min_items=field_info.get("min_items"),
                )
            else:
                # Handle simple fields
                schema_builder.add_field(
                    name=field_info["name"],
                    field_type=field_info["type"],
                    description=field_info.get("description", ""),
                    required=field_info["required"],
                    enum=field_info.get("enum"),
                    min_length=field_info.get("min_length"),
                    pattern=field_info.get("pattern"),
                )

        # Build the schema
        custom_schema = schema_builder.build()

        # Validate schema against examples
        validation_results: List[bool] = []
        for example in examples:
            if "expected_reasoning" in example:
                validation = custom_schema.validate(example["expected_reasoning"])
                validation_results.append(validation.valid)

        validation_rate = (
            sum(validation_results) / len(validation_results) if validation_results else 0
        )

        # Generate usage guide
        usage_guide = await _generate_usage_guide(
            task_type, pattern_analysis, custom_schema, llm_client
        )

        # Prepare result
        result = {
            "status": "success",
            "schema": custom_schema.to_json_schema(),
            "metadata": {
                "task_type": task_type,
                "learned_from": len(examples),
                "validation_rate": validation_rate,
                "common_patterns": pattern_analysis["patterns"],
                "field_count": len(pattern_analysis["fields"]),
            },
            "usage_guide": usage_guide,
            "example_prompt": custom_schema.generate_prompt(
                f"Example task for {task_type}", {"example_context": "true"}
            ),
        }

        logger.info(f"Successfully learned schema for {task_type} from {len(examples)} examples")

        return result

    except Exception as e:
        logger.error(f"Error in learn_schema: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to learn schema from examples",
        }


async def _analyze_example_patterns(
    examples: List[Dict[str, Any]], llm_client: LLMClient
) -> Dict[str, Any]:
    """Analyze examples to extract common patterns and field definitions."""

    # Extract all expected reasoning examples
    reasoning_examples = []
    for ex in examples:
        if "expected_reasoning" in ex:
            reasoning_examples.append(ex["expected_reasoning"])

    # Manual pattern extraction (simplified)
    field_occurrences: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "types": set(), "values": []}
    )

    for reasoning in reasoning_examples:
        _extract_fields_recursive(reasoning, field_occurrences)

    # Convert to field definitions
    total_examples = len(reasoning_examples)
    fields: List[Dict[str, Any]] = []

    for field_name, info in field_occurrences.items():
        # Determine if required (appears in most examples)
        required = info["count"] >= (total_examples * 0.8)

        # Determine type (most common)
        types_list = list(info["types"])
        field_type = types_list[0] if len(types_list) == 1 else "string"

        # Check for enums (repeated values)
        unique_values = list(set(info["values"]))
        enum = unique_values if len(unique_values) <= 5 and len(info["values"]) > 5 else None

        field_def = {
            "name": field_name,
            "type": field_type,
            "required": required,
            "occurrences": info["count"],
        }

        if enum and field_type == "string":
            field_def["enum"] = enum

        fields.append(field_def)

    # Use LLM to identify high-level patterns
    pattern_prompt = f"""Analyze these reasoning examples to identify common patterns:

{json.dumps(reasoning_examples[:3], indent=2)}

Identify:
1. Common structural patterns
2. Typical reasoning flow
3. Key sections that appear consistently

Return a JSON with:
{{
    "patterns": ["list of identified patterns"],
    "reasoning_flow": ["typical step sequence"],
    "key_sections": ["main sections"]
}}"""

    llm_response = await llm_client.generate(pattern_prompt, temperature=0.1, max_tokens=1000)

    try:
        llm_patterns = json.loads(llm_response.strip().strip("```json").strip("```"))
    except Exception:
        llm_patterns = {
            "patterns": ["structured analysis", "step-by-step reasoning"],
            "reasoning_flow": ["understand", "analyze", "conclude"],
            "key_sections": ["overview", "details", "summary"],
        }

    return {
        "fields": fields,
        "patterns": llm_patterns.get("patterns", []),
        "reasoning_flow": llm_patterns.get("reasoning_flow", []),
        "key_sections": llm_patterns.get("key_sections", []),
    }


def _extract_fields_recursive(
    obj: Any, field_occurrences: Dict[str, Dict], prefix: str = ""
) -> None:
    """Recursively extract field information from objects."""

    if isinstance(obj, dict):
        for key, value in obj.items():
            field_path = f"{prefix}.{key}" if prefix else key
            field_occurrences[field_path]["count"] += 1

            if isinstance(value, dict):
                field_occurrences[field_path]["types"].add("object")
                _extract_fields_recursive(value, field_occurrences, field_path)
            elif isinstance(value, list):
                field_occurrences[field_path]["types"].add("array")
                if value and isinstance(value[0], dict):
                    _extract_fields_recursive(value[0], field_occurrences, field_path + "[0]")
            elif isinstance(value, str):
                field_occurrences[field_path]["types"].add("string")
                field_occurrences[field_path]["values"].append(value)
            elif isinstance(value, bool):
                field_occurrences[field_path]["types"].add("boolean")
                field_occurrences[field_path]["values"].append(value)
            elif isinstance(value, (int, float)):
                field_occurrences[field_path]["types"].add("number")
                field_occurrences[field_path]["values"].append(value)


async def _generate_usage_guide(
    task_type: str, pattern_analysis: Dict[str, Any], schema: Any, llm_client: LLMClient
) -> str:
    """Generate a usage guide for the learned schema."""

    guide = f"""# Usage Guide for {task_type} Schema

## Overview
This schema was learned from examples and captures the following patterns:
{chr(10).join(f'- {p}' for p in pattern_analysis['patterns'][:5])}

## Typical Reasoning Flow
{' → '.join(pattern_analysis['reasoning_flow'])}

## Key Sections
{chr(10).join(f'- **{section}**: Important part of the reasoning' for section in pattern_analysis['key_sections'])}

## Field Definitions
"""

    # Add field descriptions
    for field in pattern_analysis["fields"][:10]:  # Top 10 fields
        guide += f"\n### {field['name']}\n"
        guide += f"- Type: `{field['type']}`\n"
        guide += f"- Required: {'Yes' if field['required'] else 'No'}\n"
        if "enum" in field:
            guide += f"- Allowed values: {', '.join(field['enum'])}\n"
        guide += f"- Appears in {field['occurrences']}/{len(pattern_analysis['fields'])} examples\n"

    guide += """
## Best Practices
1. Ensure all required fields are populated
2. Follow the typical reasoning flow identified above
3. Be specific and detailed in your responses
4. Validate your output against the schema

## Example Usage
```python
result = await sgr.apply_sgr(
    task="Your specific task here",
    schema_type="custom",
    custom_schema=<learned_schema>
)
```
"""

    return guide
