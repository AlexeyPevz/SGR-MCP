"""Enhance prompt tool implementation."""

import json
import logging
from typing import Any, Dict

from ..schemas import SCHEMA_REGISTRY
from ..utils.cache import CacheManager
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


async def enhance_prompt_tool(
    arguments: Dict[str, Any], llm_client: LLMClient, cache_manager: CacheManager
) -> Dict[str, Any]:
    """Transform a simple prompt into a structured prompt with SGR guidance.

    Args:
        arguments: Tool arguments containing original_prompt, target_model, etc.
        llm_client: LLM client for enhancement
        cache_manager: Cache manager

    Returns:
        Dictionary with enhanced prompt and metadata
    """
    try:
        # Extract arguments
        original_prompt = arguments["original_prompt"]
        target_model = arguments.get("target_model", "default")
        enhancement_level = arguments.get("enhancement_level", "standard")

        # Check cache with stable key
        import hashlib

        key_payload = {
            "original_prompt": original_prompt,
            "target_model": target_model,
            "enhancement_level": enhancement_level,
        }
        key_str = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
        cache_key = f"enhance:{hashlib.sha256(key_str.encode('utf-8')).hexdigest()}"
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            logger.info("Cache hit for prompt enhancement")
            return cached_result

        # Analyze the prompt to determine best schema
        prompt_analysis = await _analyze_prompt(original_prompt, llm_client)

        # Get appropriate schema
        schema_type = prompt_analysis["detected_type"]
        schema_factory = SCHEMA_REGISTRY.get(schema_type) or SCHEMA_REGISTRY.get("analysis")
        if not schema_factory:
            raise ValueError("No schema available")
        schema = schema_factory()

        # Generate enhanced prompt based on level
        if enhancement_level == "minimal":
            enhanced_prompt = await _minimal_enhancement(original_prompt, prompt_analysis, schema)
        elif enhancement_level == "comprehensive":
            enhanced_prompt = await _comprehensive_enhancement(
                original_prompt, prompt_analysis, schema, llm_client
            )
        else:  # standard
            enhanced_prompt = await _standard_enhancement(original_prompt, prompt_analysis, schema)

        # Prepare result
        result = {
            "enhanced_prompt": enhanced_prompt,
            "original_prompt": original_prompt,
            "metadata": {
                "detected_intent": prompt_analysis["intent"],
                "suggested_schema": schema_type,
                "enhancement_level": enhancement_level,
                "target_model": target_model,
                "improvements": prompt_analysis["improvements"],
            },
        }

        # Cache result
        await cache_manager.set(cache_key, result, ttl=3600)

        return result

    except Exception as e:
        logger.error(f"Error in enhance_prompt: {e}", exc_info=True)
        raise


async def _analyze_prompt(prompt: str, llm_client: LLMClient) -> Dict[str, Any]:
    """Analyze the original prompt to understand intent and structure needs."""

    analysis_prompt = f"""Analyze this prompt and determine:
1. The primary intent/task type
2. What schema type would be most helpful
3. Key improvements needed

Prompt: "{prompt}"

Available schema types:
- analysis: For understanding problems
- planning: For creating plans
- decision: For making choices
- code_generation: For writing code
- summarization: For condensing information

Respond in JSON format:
{{
    "intent": "brief description of what user wants",
    "detected_type": "most appropriate schema type",
    "key_elements": ["list", "of", "key", "elements"],
    "improvements": ["list", "of", "suggested", "improvements"]
}}"""

    response = await llm_client.generate(analysis_prompt, temperature=0.1, max_tokens=500)

    try:
        # Parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]

        analysis = json.loads(response)

        # Validate detected type
        if analysis["detected_type"] not in SCHEMA_REGISTRY:
            analysis["detected_type"] = "analysis"

        return analysis

    except Exception:
        # Fallback analysis
        return {
            "intent": "General task execution",
            "detected_type": "analysis",
            "key_elements": ["task", "execution"],
            "improvements": ["Add more context", "Clarify objectives"],
        }


async def _minimal_enhancement(original: str, analysis: Dict[str, Any], schema: Any) -> str:
    """Minimal enhancement - just add structure hints."""

    structure_hints = []

    # Add key questions based on schema
    if analysis["detected_type"] == "analysis":
        structure_hints = [
            "What is the main problem or task?",
            "What are the constraints?",
            "What are the success criteria?",
        ]
    elif analysis["detected_type"] == "planning":
        structure_hints = [
            "What are the main steps?",
            "What resources are needed?",
            "What is the timeline?",
        ]
    elif analysis["detected_type"] == "decision":
        structure_hints = [
            "What are the options?",
            "What are the evaluation criteria?",
            "What are the tradeoffs?",
        ]
    elif analysis["detected_type"] == "code_generation":
        structure_hints = [
            "What should the code do?",
            "What are the technical requirements?",
            "What language/framework?",
        ]

    enhanced = f"""{original}

Please address these key points:
{chr(10).join(f'- {hint}' for hint in structure_hints)}"""

    return enhanced


async def _standard_enhancement(original: str, analysis: Dict[str, Any], schema: Any) -> str:
    """Standard enhancement - add schema structure."""

    # Get schema fields
    fields = schema.get_fields()
    _ = [f for f in fields if f.required]

    # Build structured prompt
    enhanced = f"""{original}

Please provide a structured response covering:

"""

    # Add main sections from schema
    if analysis["detected_type"] == "analysis":
        enhanced += """1. **Understanding**
   - Clear summary of the task
   - Key aspects to consider
   - Any ambiguities or assumptions

2. **Goals and Success Criteria**
   - Primary objective
   - How to measure success

3. **Constraints and Limitations**
   - Technical, resource, or time constraints

4. **Risks and Mitigation**
   - Potential issues
   - How to address them"""

    elif analysis["detected_type"] == "planning":
        enhanced += """1. **Approach**
   - Overall strategy
   - Key principles

2. **Step-by-Step Plan**
   - Detailed steps with dependencies
   - Timeline and milestones

3. **Resources Required**
   - Team, tools, infrastructure

4. **Success Metrics**
   - How to measure progress"""

    elif analysis["detected_type"] == "decision":
        enhanced += """1. **Context and Requirements**
   - What decision needs to be made
   - Key requirements and constraints

2. **Options Analysis**
   - Available options
   - Pros and cons of each

3. **Evaluation**
   - Criteria for evaluation
   - Scoring or comparison

4. **Recommendation**
   - Selected option and rationale"""

    elif analysis["detected_type"] == "code_generation":
        enhanced += """1. **Requirements Understanding**
   - What the code should accomplish
   - Technical constraints

2. **Design Approach**
   - Architecture and patterns
   - Key components

3. **Implementation Plan**
   - Language and frameworks
   - File structure

4. **Testing Strategy**
   - Test cases to cover
   - Validation approach"""

    else:  # summarization
        enhanced += """1. **Purpose and Audience**
   - Why this summary is needed
   - Who will read it

2. **Key Points**
   - Main takeaways
   - Critical information

3. **Summary**
   - Concise overview
   - Detailed points as needed"""

    enhanced += "\n\nBe specific and thorough in your response."

    return enhanced


async def _comprehensive_enhancement(
    original: str, analysis: Dict[str, Any], schema: Any, llm_client: LLMClient
) -> str:
    """Comprehensive enhancement - full schema integration with examples."""

    # Get schema details
    schema_json = json.dumps(schema.to_json_schema(), indent=2)
    examples = schema.get_examples()

    # Build comprehensive prompt
    enhanced = f"""Task: {original}

You need to provide a comprehensive, structured response following this exact schema:

```json
{schema_json}
```

"""

    # Add example if available
    if examples:
        enhanced += f"""Here's an example of a high-quality response:

```json
{json.dumps(examples[0], indent=2)}
```

"""

    # Add specific guidance based on detected improvements
    if analysis.get("improvements"):
        enhanced += "Key areas to focus on:\n"
        for improvement in analysis["improvements"]:
            enhanced += f"- {improvement}\n"
        enhanced += "\n"

    # Add quality criteria
    enhanced += """Quality criteria for your response:
1. **Completeness**: Address all required fields thoroughly
2. **Specificity**: Provide concrete details, not generalizations
3. **Actionability**: Include specific next steps
4. **Validation**: Consider edge cases and risks
5. **Clarity**: Use clear, unambiguous language

Provide your response in valid JSON format matching the schema exactly."""

    return enhanced
