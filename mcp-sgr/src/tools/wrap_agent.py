"""Wrap agent call tool implementation."""

import json
import asyncio
import aiohttp
from typing import Any, Dict, Optional
from datetime import datetime
import logging

from .apply_sgr import apply_sgr_tool
from ..utils.llm_client import LLMClient
from ..utils.cache import CacheManager
from ..utils.telemetry import TelemetryManager

logger = logging.getLogger(__name__)


async def wrap_agent_call_tool(
    arguments: Dict[str, Any],
    llm_client: LLMClient,
    cache_manager: CacheManager,
    telemetry: TelemetryManager
) -> Dict[str, Any]:
    """Wrap an agent call with pre/post SGR analysis.
    
    Args:
        arguments: Tool arguments containing agent_endpoint, request, config
        llm_client: LLM client for reasoning
        cache_manager: Cache manager
        telemetry: Telemetry manager
        
    Returns:
        Dictionary with original response, reasoning chain, and quality metrics
    """
    # Start telemetry span
    span_id = await telemetry.start_span("wrap_agent_call", arguments)
    
    try:
        # Extract arguments
        agent_endpoint = arguments["agent_endpoint"]
        agent_request = arguments["agent_request"]
        sgr_config = arguments.get("sgr_config", {})
        
        # Default SGR config
        schema_type = sgr_config.get("schema_type", "auto")
        budget = sgr_config.get("budget", "lite")
        pre_analysis = sgr_config.get("pre_analysis", True)
        post_analysis = sgr_config.get("post_analysis", True)
        include_alternatives = sgr_config.get("include_alternatives", False)
        
        result = {
            "original_response": None,
            "reasoning_chain": {},
            "quality_metrics": {},
            "suggestions": []
        }
        
        # Pre-analysis
        if pre_analysis:
            logger.info("Performing pre-analysis...")
            pre_reasoning = await _perform_pre_analysis(
                agent_endpoint,
                agent_request,
                schema_type,
                budget,
                llm_client,
                cache_manager,
                telemetry
            )
            result["reasoning_chain"]["pre"] = pre_reasoning
            
            # Enhance request based on pre-analysis
            enhanced_request = await _enhance_request(
                agent_request,
                pre_reasoning,
                llm_client
            )
        else:
            enhanced_request = agent_request
        
        # Call the actual agent
        logger.info(f"Calling agent at {agent_endpoint}")
        agent_response = await _call_agent(agent_endpoint, enhanced_request)
        result["original_response"] = agent_response
        
        # Post-analysis
        if post_analysis:
            logger.info("Performing post-analysis...")
            post_reasoning = await _perform_post_analysis(
                agent_request,
                agent_response,
                pre_reasoning if pre_analysis else None,
                schema_type,
                budget,
                llm_client,
                cache_manager,
                telemetry
            )
            result["reasoning_chain"]["post"] = post_reasoning
            
            # Calculate quality metrics
            quality_metrics = await _calculate_quality_metrics(
                agent_request,
                agent_response,
                post_reasoning,
                llm_client
            )
            result["quality_metrics"] = quality_metrics
            
            # Generate suggestions
            if quality_metrics.get("confidence", 1.0) < 0.7 or include_alternatives:
                suggestions = await _generate_suggestions(
                    agent_request,
                    agent_response,
                    post_reasoning,
                    quality_metrics,
                    llm_client
                )
                result["suggestions"] = suggestions
        
        # End telemetry span
        await telemetry.end_span(span_id, {
            "pre_analysis": pre_analysis,
            "post_analysis": post_analysis,
            "confidence": result.get("quality_metrics", {}).get("confidence", 0)
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in wrap_agent_call: {e}", exc_info=True)
        await telemetry.end_span(span_id, {"error": str(e)})
        raise


async def _perform_pre_analysis(
    agent_endpoint: str,
    agent_request: Dict[str, Any],
    schema_type: str,
    budget: str,
    llm_client: LLMClient,
    cache_manager: CacheManager,
    telemetry: TelemetryManager
) -> Dict[str, Any]:
    """Perform pre-call analysis of the request."""
    
    # Construct task for SGR
    task = f"""Analyze this agent request before execution:

Agent Endpoint: {agent_endpoint}
Request: {json.dumps(agent_request, indent=2)}

Provide structured analysis to understand:
1. What the request is trying to achieve
2. Potential issues or ambiguities
3. Expected response characteristics
4. Risk factors"""
    
    # Use apply_sgr for analysis
    result = await apply_sgr_tool(
        {
            "task": task,
            "context": {
                "agent_type": agent_endpoint.split("/")[-1],
                "request_size": len(json.dumps(agent_request))
            },
            "schema_type": schema_type if schema_type != "auto" else "analysis",
            "budget": budget
        },
        llm_client,
        cache_manager,
        telemetry
    )
    
    return result


async def _enhance_request(
    original_request: Dict[str, Any],
    pre_reasoning: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Enhance request based on pre-analysis insights."""
    
    # Extract insights from pre-reasoning
    reasoning = pre_reasoning.get("reasoning", {})
    
    # For now, return original request
    # In future, could add clarifications, constraints, etc.
    enhanced = original_request.copy()
    
    # Add any identified constraints or clarifications
    if "understanding" in reasoning:
        ambiguities = reasoning["understanding"].get("ambiguities", [])
        if ambiguities:
            # Could add clarifications to prompt
            logger.info(f"Identified ambiguities: {ambiguities}")
    
    return enhanced


async def _call_agent(endpoint: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """Call the actual agent endpoint."""
    
    # Check if it's a callable (local function)
    if callable(endpoint):
        logger.info("Calling local agent function")
        if asyncio.iscoroutinefunction(endpoint):
            return await endpoint(**request)
        else:
            return endpoint(**request)
    
    # Otherwise treat as HTTP endpoint
    if endpoint.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=request) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Agent call failed: {response.status} - {error_text}")
    
    # Mock response for testing
    logger.warning(f"Mock response for endpoint: {endpoint}")
    return {
        "status": "success",
        "message": f"Mock response from {endpoint}",
        "data": {
            "processed": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


async def _perform_post_analysis(
    original_request: Dict[str, Any],
    agent_response: Dict[str, Any],
    pre_reasoning: Optional[Dict[str, Any]],
    schema_type: str,
    budget: str,
    llm_client: LLMClient,
    cache_manager: CacheManager,
    telemetry: TelemetryManager
) -> Dict[str, Any]:
    """Perform post-call analysis of the response."""
    
    # Construct task for SGR
    task = f"""Analyze this agent response:

Original Request: {json.dumps(original_request, indent=2)}
Agent Response: {json.dumps(agent_response, indent=2)}

{f"Pre-analysis insights: {json.dumps(pre_reasoning['reasoning'], indent=2)}" if pre_reasoning else ""}

Evaluate:
1. Whether the response adequately addresses the request
2. Quality and completeness of the response
3. Any issues or inconsistencies
4. Actionable next steps"""
    
    # Use apply_sgr for analysis
    result = await apply_sgr_tool(
        {
            "task": task,
            "context": {
                "has_pre_analysis": pre_reasoning is not None,
                "response_size": len(json.dumps(agent_response))
            },
            "schema_type": schema_type if schema_type != "auto" else "analysis",
            "budget": budget
        },
        llm_client,
        cache_manager,
        telemetry
    )
    
    return result


async def _calculate_quality_metrics(
    request: Dict[str, Any],
    response: Dict[str, Any],
    post_reasoning: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Calculate quality metrics for the agent response."""
    
    metrics = {
        "confidence": post_reasoning.get("confidence", 0.5),
        "completeness": 0.0,
        "consistency": 0.0,
        "clarity": 0.0,
        "actionability": 0.0
    }
    
    # Extract from post-reasoning
    reasoning = post_reasoning.get("reasoning", {})
    
    # Simple heuristics for now
    # In future, could use more sophisticated analysis
    
    # Completeness: check if response has expected fields
    if response and isinstance(response, dict):
        expected_fields = ["status", "data", "message"]
        present_fields = sum(1 for f in expected_fields if f in response)
        metrics["completeness"] = present_fields / len(expected_fields)
    
    # Consistency: check if pre/post analysis align
    if "goals" in reasoning:
        success_criteria = reasoning["goals"].get("success_criteria", [])
        if success_criteria:
            metrics["consistency"] = 0.8  # Placeholder
    
    # Clarity: based on response structure
    if response and "error" not in response:
        metrics["clarity"] = 0.9
    else:
        metrics["clarity"] = 0.4
    
    # Actionability: based on suggested actions
    actions = post_reasoning.get("suggested_actions", [])
    if actions:
        metrics["actionability"] = min(1.0, len(actions) / 5)
    
    # Overall quality score
    metrics["overall"] = sum(metrics.values()) / len(metrics)
    
    return metrics


async def _generate_suggestions(
    request: Dict[str, Any],
    response: Dict[str, Any],
    post_reasoning: Dict[str, Any],
    quality_metrics: Dict[str, Any],
    llm_client: LLMClient
) -> List[str]:
    """Generate improvement suggestions."""
    
    suggestions = []
    
    # Based on quality metrics
    if quality_metrics.get("confidence", 1.0) < 0.6:
        suggestions.append("Consider reformulating the request for better clarity")
    
    if quality_metrics.get("completeness", 1.0) < 0.7:
        suggestions.append("Request additional information to fill gaps")
    
    # From post-reasoning
    reasoning = post_reasoning.get("reasoning", {})
    
    # Extract risks that need mitigation
    risks = reasoning.get("risks", [])
    for risk in risks[:3]:  # Top 3 risks
        if isinstance(risk, dict) and "mitigation" in risk:
            suggestions.append(f"Risk mitigation: {risk['mitigation']}")
    
    # Extract data gaps
    gaps = reasoning.get("data_gaps", [])
    for gap in gaps[:2]:  # Top 2 gaps
        suggestions.append(f"Address data gap: {gap}")
    
    # From suggested actions
    actions = post_reasoning.get("suggested_actions", [])
    suggestions.extend(actions[:3])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)
    
    return unique_suggestions[:5]  # Return top 5 suggestions