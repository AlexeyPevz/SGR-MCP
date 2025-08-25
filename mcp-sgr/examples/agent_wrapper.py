"""Example of wrapping agent calls with SGR."""

import asyncio
import json
from typing import Any, Dict
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_client import LLMClient
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager
from src.tools import wrap_agent_call_tool


# Mock agent functions for demonstration
async def mock_coding_agent(prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """Mock coding agent that generates code."""
    # In real scenario, this would call actual agent
    return {
        "status": "success",
        "code": """def calculate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib""",
        "language": "python",
        "explanation": "Iterative implementation of Fibonacci sequence generator"
    }


async def mock_analysis_agent(text: str, analysis_type: str = "sentiment") -> Dict[str, Any]:
    """Mock analysis agent."""
    return {
        "status": "success",
        "analysis": {
            "type": analysis_type,
            "results": {
                "sentiment": "neutral",
                "confidence": 0.85,
                "key_points": ["Technical content", "Instructional tone"]
            }
        },
        "processed_tokens": len(text.split())
    }


async def example_wrap_coding_agent():
    """Example of wrapping a coding agent with SGR."""
    print("\n=== Example: Wrap Coding Agent ===\n")
    
    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()
    
    await cache_manager.initialize()
    await telemetry.initialize()
    
    # Agent request
    agent_request = {
        "prompt": "Write a Python function to calculate Fibonacci numbers",
        "max_tokens": 1000
    }
    
    print("Original Request:")
    print(json.dumps(agent_request, indent=2))
    
    # Wrap the agent call
    result = await wrap_agent_call_tool(
        arguments={
            "agent_endpoint": mock_coding_agent,  # Can be URL or callable
            "agent_request": agent_request,
            "sgr_config": {
                "schema_type": "code_generation",
                "budget": "full",
                "pre_analysis": True,
                "post_analysis": True,
                "include_alternatives": True
            }
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry
    )
    
    print("\n--- Results ---")
    
    # Show original response
    print("\nOriginal Agent Response:")
    print(f"Status: {result['original_response']['status']}")
    print(f"Code Preview: {result['original_response']['code'][:100]}...")
    
    # Show pre-analysis insights
    if "pre" in result["reasoning_chain"]:
        pre_analysis = result["reasoning_chain"]["pre"]
        print(f"\nPre-Analysis Confidence: {pre_analysis['confidence']:.2f}")
        
        if "reasoning" in pre_analysis and "understanding" in pre_analysis["reasoning"]:
            understanding = pre_analysis["reasoning"]["understanding"]
            print(f"Task Understanding: {understanding.get('task_summary', 'N/A')}")
    
    # Show post-analysis
    if "post" in result["reasoning_chain"]:
        post_analysis = result["reasoning_chain"]["post"]
        print(f"\nPost-Analysis Confidence: {post_analysis['confidence']:.2f}")
        
        if "suggested_actions" in post_analysis:
            print("\nSuggested Improvements:")
            for action in post_analysis["suggested_actions"][:3]:
                print(f"- {action}")
    
    # Show quality metrics
    print("\nQuality Metrics:")
    for metric, value in result["quality_metrics"].items():
        if isinstance(value, float):
            print(f"- {metric}: {value:.2f}")
        else:
            print(f"- {metric}: {value}")
    
    # Show suggestions
    if result["suggestions"]:
        print("\nSuggestions for Better Results:")
        for suggestion in result["suggestions"]:
            print(f"- {suggestion}")
    
    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def example_wrap_with_retry():
    """Example of wrapping an agent that might fail."""
    print("\n=== Example: Wrap Agent with Error Handling ===\n")
    
    # Failing agent for demonstration
    async def failing_agent(**kwargs):
        raise Exception("Agent temporarily unavailable")
    
    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()
    
    await cache_manager.initialize()
    await telemetry.initialize()
    
    try:
        result = await wrap_agent_call_tool(
            arguments={
                "agent_endpoint": failing_agent,
                "agent_request": {"task": "test"},
                "sgr_config": {
                    "schema_type": "analysis",
                    "budget": "lite",
                    "pre_analysis": True,
                    "post_analysis": False  # Skip post since agent fails
                }
            },
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry
        )
    except Exception as e:
        print(f"Agent call failed as expected: {e}")
        print("\nBut we still got pre-analysis insights!")
        
        # In real scenario, pre-analysis could help reformulate request
    
    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def example_wrap_http_agent():
    """Example of wrapping an HTTP-based agent."""
    print("\n=== Example: Wrap HTTP Agent ===\n")
    
    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()
    
    await cache_manager.initialize()
    await telemetry.initialize()
    
    # HTTP endpoint (would be real in production)
    agent_endpoint = "http://localhost:8000/api/generate"
    
    agent_request = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms"}
        ],
        "temperature": 0.7
    }
    
    print("HTTP Agent Endpoint:", agent_endpoint)
    print("Request:", json.dumps(agent_request, indent=2))
    
    # Wrap the HTTP agent call
    result = await wrap_agent_call_tool(
        arguments={
            "agent_endpoint": agent_endpoint,
            "agent_request": agent_request,
            "sgr_config": {
                "schema_type": "summarization",
                "budget": "lite",
                "pre_analysis": True,
                "post_analysis": True
            }
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry
    )
    
    print("\n--- Wrapped Results ---")
    
    # Show analysis insights
    if "pre" in result["reasoning_chain"]:
        print("\nPre-call Analysis:")
        pre_reasoning = result["reasoning_chain"]["pre"]["reasoning"]
        if "goals" in pre_reasoning:
            print(f"Identified Goal: {pre_reasoning['goals'].get('primary', 'N/A')}")
    
    print("\nNote: In production, this would make actual HTTP calls")
    print("The wrapper adds reasoning transparency to any agent!")
    
    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def example_batch_agent_calls():
    """Example of wrapping multiple agent calls."""
    print("\n=== Example: Batch Agent Wrapping ===\n")
    
    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()
    
    await cache_manager.initialize()
    await telemetry.initialize()
    
    # Multiple tasks
    tasks = [
        {
            "agent": mock_coding_agent,
            "request": {"prompt": "Sort a list in Python", "max_tokens": 500},
            "schema": "code_generation"
        },
        {
            "agent": mock_analysis_agent,
            "request": {"text": "The product is amazing!", "analysis_type": "sentiment"},
            "schema": "analysis"
        }
    ]
    
    print("Processing multiple agent calls with SGR wrapping...\n")
    
    results = []
    for i, task in enumerate(tasks):
        print(f"Task {i+1}: {task['request']}")
        
        result = await wrap_agent_call_tool(
            arguments={
                "agent_endpoint": task["agent"],
                "agent_request": task["request"],
                "sgr_config": {
                    "schema_type": task["schema"],
                    "budget": "lite",
                    "pre_analysis": True,
                    "post_analysis": True
                }
            },
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry
        )
        
        results.append(result)
        
        # Show summary
        confidence = result.get("quality_metrics", {}).get("confidence", 0)
        print(f"Confidence: {confidence:.2f}")
        print(f"Quality Score: {result.get('quality_metrics', {}).get('overall', 0):.2f}\n")
    
    # Aggregate insights
    print("\n--- Batch Summary ---")
    avg_confidence = sum(r.get("quality_metrics", {}).get("confidence", 0) for r in results) / len(results)
    print(f"Average Confidence: {avg_confidence:.2f}")
    
    total_suggestions = sum(len(r.get("suggestions", [])) for r in results)
    print(f"Total Improvement Suggestions: {total_suggestions}")
    
    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def main():
    """Run all agent wrapper examples."""
    print("MCP-SGR Agent Wrapper Examples")
    print("=" * 50)
    
    await example_wrap_coding_agent()
    await example_wrap_with_retry()
    await example_wrap_http_agent()
    await example_batch_agent_calls()
    
    print("\n" + "=" * 50)
    print("Agent wrapper examples completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()