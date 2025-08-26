"""
Apply SGR v3 - Production-ready version based on extensive testing.

Key improvements:
1. Default to unstructured mode (structured output is unreliable)
2. Robust JSON parsing with markdown handling
3. Model-specific optimizations
4. Fallback chain for reliability
5. Better prompts based on testing
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from ..schemas.base import BaseReasoning
from ..utils.llm_client import LLMClient
from ..utils.exceptions import SGRError

logger = logging.getLogger(__name__)

# Model capabilities based on testing
MODEL_CAPABILITIES = {
    # Best models for unstructured JSON
    "qwen/qwen-2.5-72b-instruct": {
        "supports_structured": False,
        "json_quality": "excellent",
        "recommended": True,
        "cost_efficiency": "excellent"
    },
    "openai/gpt-4o-mini": {
        "supports_structured": False,  # Broken even on direct API
        "json_quality": "excellent",
        "recommended": True,
        "cost_efficiency": "good"
    },
    "mistralai/mistral-7b-instruct": {
        "supports_structured": False,
        "json_quality": "good",
        "recommended": True,
        "cost_efficiency": "excellent"
    },
    # Only model with working structured output
    "google/gemini-flash-1.5": {
        "supports_structured": True,
        "json_quality": "good",
        "recommended": True,
        "cost_efficiency": "good"
    },
    "google/gemini-pro-1.5": {
        "supports_structured": True,
        "json_quality": "excellent",
        "recommended": True,
        "cost_efficiency": "fair"
    },
    # Models with issues
    "anthropic/claude-3.5-haiku": {
        "supports_structured": False,  # Broken via OpenRouter
        "json_quality": "poor",  # Returns markdown, not JSON
        "recommended": False,
        "cost_efficiency": "good"
    },
    "deepseek/deepseek-chat": {
        "supports_structured": False,
        "json_quality": "poor",
        "recommended": False,
        "cost_efficiency": "excellent"
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "supports_structured": False,
        "json_quality": "poor",
        "recommended": False,
        "cost_efficiency": "good"
    }
}

# Optimized prompts based on testing
PROMPTS = {
    "lite": {
        "system": "You are a JSON-only assistant. Respond with STRICTLY valid JSON matching the provided structure.",
        "user": """Task: {task}

Provide a JSON response following this structure:
{schema}

Be concise but complete."""
    },
    "full": {
        "system": "You are an analytical assistant. Always respond in valid JSON format.",
        "user": """Task: {task}

Analyze this step by step:
1. Understand the task requirements
2. Break down the problem
3. Consider multiple approaches
4. Provide structured reasoning

Format your response as JSON:
{schema}

Include confidence scores and detailed analysis."""
    },
    "universal": {
        "system": "Provide structured JSON responses for all analyses.",
        "user": """Task: {task}

Analyze and respond with a JSON object containing:
- summary: brief overview (string)
- key_points: main findings (array of strings)  
- recommendations: suggested actions (array of strings)
- confidence: 0-1 score (number)

Example format:
{{
  "summary": "...",
  "key_points": ["point 1", "point 2"],
  "recommendations": ["action 1", "action 2"],
  "confidence": 0.85
}}

Your analysis:"""
    }
}

# Simplified schemas for better compatibility
SCHEMAS = {
    "minimal": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["analysis"]
    },
    "lite": {
        "type": "object", 
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["summary", "key_points", "recommendations"]
    },
    "full": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "context": {"type": "string"},
                    "assumptions": {"type": "array", "items": {"type": "string"}}
                }
            },
            "reasoning": {
                "type": "object",
                "properties": {
                    "approach": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "alternatives": {"type": "array", "items": {"type": "string"}}
                }
            },
            "conclusions": {
                "type": "object",
                "properties": {
                    "findings": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "limitations": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "required": ["analysis", "reasoning", "conclusions"]
    }
}


def robust_json_parse(content: str) -> Dict[str, Any]:
    """Parse JSON with robust handling of common issues."""
    
    # Remove markdown code blocks
    if "```" in content:
        # Find JSON content between backticks
        parts = content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Inside code block
                if part.startswith("json\n"):
                    part = part[5:]
                elif part.startswith("json "):
                    part = part[5:]
                try:
                    return json.loads(part.strip())
                except:
                    continue
    
    # Try direct parsing
    content = content.strip()
    
    # Handle common prefixes
    prefixes = ["Here's the JSON:", "JSON:", "Response:", "Analysis:"]
    for prefix in prefixes:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
    
    # Find JSON object
    start_idx = content.find("{")
    if start_idx != -1:
        # Find matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(content)):
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
    
    # Last resort - try to parse the whole thing
    return json.loads(content)


def get_model_info(model: str) -> Dict[str, Any]:
    """Get model capabilities and recommendations."""
    
    # Check exact match
    if model in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model]
    
    # Check prefix match
    for model_key, info in MODEL_CAPABILITIES.items():
        if model.startswith(model_key.split("/")[0]):
            return info
    
    # Default capabilities
    return {
        "supports_structured": False,
        "json_quality": "unknown",
        "recommended": False,
        "cost_efficiency": "unknown"
    }


def select_best_model(available_models: List[str], task_type: str = "general") -> str:
    """Select the best model from available options."""
    
    # Priority order based on testing
    priority_models = [
        "qwen/qwen-2.5-72b-instruct",
        "openai/gpt-4o-mini",
        "mistralai/mistral-7b-instruct",
        "google/gemini-flash-1.5",
    ]
    
    # Check if any priority model is available
    for model in priority_models:
        if model in available_models:
            return model
    
    # Return first available model
    return available_models[0] if available_models else "qwen/qwen-2.5-72b-instruct"


async def apply_sgr_v3(
    task: str,
    mode: str = "lite",
    model: Optional[str] = None,
    use_structured: Optional[bool] = None,
    max_retries: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Schema-Guided Reasoning v3 with improved model handling.
    
    Args:
        task: The task to analyze
        mode: SGR mode - "minimal", "lite", or "full"
        model: Specific model to use (optional)
        use_structured: Force structured output mode (not recommended)
        max_retries: Maximum retry attempts for JSON parsing
        
    Returns:
        Dict containing reasoning results
    """
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    # Select model if not specified
    if not model:
        available_models = llm_client.list_models()
        model = select_best_model(available_models)
    
    # Get model capabilities
    model_info = get_model_info(model)
    
    # Determine if we should use structured output
    if use_structured is None:
        use_structured = model_info["supports_structured"] and model_info["json_quality"] != "poor"
    
    # Log configuration
    logger.info(f"Using model: {model} (structured: {use_structured})")
    
    # Select schema and prompt
    schema = SCHEMAS.get(mode, SCHEMAS["lite"])
    prompt_template = PROMPTS.get(mode, PROMPTS["universal"])
    
    # Prepare messages
    messages = [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": prompt_template["user"].format(
            task=task,
            schema=json.dumps(schema, indent=2)
        )}
    ]
    
    # Try to get response
    last_error = None
    for attempt in range(max_retries):
        try:
            if use_structured and model_info["supports_structured"]:
                # Use structured output (only for Gemini)
                response = await llm_client.complete(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "sgr_reasoning",
                            "schema": schema,
                            "strict": True
                        }
                    },
                    temperature=0.1,
                    **kwargs
                )
            else:
                # Use unstructured with JSON instructions
                response = await llm_client.complete(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    **kwargs
                )
            
            # Parse response
            if isinstance(response, dict):
                reasoning = response
            else:
                content = response.get("content", "") if isinstance(response, dict) else str(response)
                reasoning = robust_json_parse(content)
            
            # Validate basic structure
            if mode == "minimal" and "analysis" not in reasoning:
                raise ValueError("Missing required field: analysis")
            elif mode == "lite" and not all(field in reasoning for field in ["summary", "key_points"]):
                raise ValueError("Missing required fields for lite mode")
            elif mode == "full" and "analysis" not in reasoning:
                raise ValueError("Missing analysis section for full mode")
            
            # Add metadata
            reasoning["_metadata"] = {
                "model": model,
                "mode": mode,
                "structured_output": use_structured,
                "attempt": attempt + 1
            }
            
            return {
                "success": True,
                "reasoning": reasoning,
                "model": model,
                "mode": mode
            }
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            # Try with a simpler schema on retry
            if attempt == 0 and mode == "full":
                mode = "lite"
                schema = SCHEMAS["lite"]
            elif attempt == 1:
                mode = "minimal"
                schema = SCHEMAS["minimal"]
    
    # All attempts failed
    return {
        "success": False,
        "error": str(last_error),
        "model": model,
        "mode": mode,
        "fallback": create_fallback_reasoning(task)
    }


def create_fallback_reasoning(task: str) -> Dict[str, Any]:
    """Create a minimal fallback reasoning when all else fails."""
    return {
        "analysis": f"Unable to complete detailed analysis for: {task}",
        "confidence": 0.1,
        "_metadata": {
            "fallback": True,
            "reason": "All parsing attempts failed"
        }
    }


# Convenience functions for different modes
async def apply_sgr_minimal(task: str, **kwargs) -> Dict[str, Any]:
    """Apply minimal SGR - fastest, least detailed."""
    return await apply_sgr_v3(task, mode="minimal", **kwargs)


async def apply_sgr_lite(task: str, **kwargs) -> Dict[str, Any]:
    """Apply lite SGR - balanced speed and detail."""
    return await apply_sgr_v3(task, mode="lite", **kwargs)


async def apply_sgr_full(task: str, **kwargs) -> Dict[str, Any]:
    """Apply full SGR - most detailed analysis."""
    return await apply_sgr_v3(task, mode="full", **kwargs)


# Export main function
apply_sgr = apply_sgr_v3