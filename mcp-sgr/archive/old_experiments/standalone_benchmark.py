#!/usr/bin/env python3
"""Standalone benchmark script for MCP-SGR with minimal dependencies."""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Check for required modules
try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call(["pip", "install", "--user", "aiohttp"])
    import aiohttp

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: Please set OPENROUTER_API_KEY environment variable")
    exit(1)

# Models to test with structured output support
MODELS_TO_TEST = {
    "ultra_cheap": [
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-thinking-exp:free",
    ],
    "cheap": [
        "google/gemini-flash-1.5",
        "openai/gpt-4o-mini",
        "qwen/qwen-2.5-7b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
    ],
    "medium": [
        "anthropic/claude-3.5-haiku",
        "openai/gpt-4o-mini-2024-07-18",
        "google/gemini-flash-1.5-8b",
    ],
    "strong": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-pro-1.5",
    ]
}

# Simple test tasks
TASKS = {
    "simple_analysis": {
        "prompt": "Analyze this code for issues: def add(a, b): return a + b",
        "schema": {
            "type": "object",
            "properties": {
                "issues": {"type": "array", "items": {"type": "string"}},
                "suggestions": {"type": "array", "items": {"type": "string"}},
                "severity": {"type": "string", "enum": ["low", "medium", "high"]}
            },
            "required": ["issues", "suggestions", "severity"]
        }
    },
    "planning": {
        "prompt": "Create a plan to build a simple todo app",
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "duration": {"type": "string"}
                        },
                        "required": ["title", "description"]
                    }
                },
                "total_duration": {"type": "string"},
                "technologies": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["steps", "technologies"]
        }
    }
}


async def call_openrouter(model: str, prompt: str, schema: Dict, with_structured: bool = True) -> Dict[str, Any]:
    """Call OpenRouter API with or without structured output."""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mcp-sgr/benchmarks",
        "X-Title": "MCP-SGR Benchmarks"
    }
    
    # Prepare the request
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides structured JSON responses."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nProvide response as JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    # Add structured output if requested and model supports it
    if with_structured and model not in ["qwen/qwen-2.5-7b-instruct", "meta-llama/llama-3.1-8b-instruct"]:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True
            }
        }
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    return {
                        "error": f"API error: {result.get('error', {}).get('message', 'Unknown error')}",
                        "status": response.status,
                        "latency": int((time.time() - start_time) * 1000)
                    }
                
                # Extract response
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse JSON
                try:
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    parsed = json.loads(content.strip())
                    
                    # Validate against schema (basic check)
                    valid = all(key in parsed for key in schema.get("required", []))
                    
                    return {
                        "success": True,
                        "response": parsed,
                        "valid": valid,
                        "latency": int((time.time() - start_time) * 1000),
                        "usage": result.get("usage", {})
                    }
                    
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Invalid JSON response",
                        "raw_response": content[:200],
                        "latency": int((time.time() - start_time) * 1000)
                    }
                    
        except asyncio.TimeoutError:
            return {
                "error": "Request timeout",
                "latency": int((time.time() - start_time) * 1000)
            }
        except Exception as e:
            return {
                "error": f"Request failed: {str(e)}",
                "latency": int((time.time() - start_time) * 1000)
            }


async def benchmark_model(model: str, tier: str) -> Dict[str, Any]:
    """Benchmark a single model."""
    print(f"\nTesting {model} ({tier})...")
    
    results = {
        "model": model,
        "tier": tier,
        "tasks": {}
    }
    
    for task_name, task_config in TASKS.items():
        print(f"  - Task: {task_name}")
        
        # Test with structured output
        structured_result = await call_openrouter(
            model, 
            task_config["prompt"], 
            task_config["schema"],
            with_structured=True
        )
        
        # Test without structured output
        unstructured_result = await call_openrouter(
            model,
            task_config["prompt"],
            task_config["schema"], 
            with_structured=False
        )
        
        results["tasks"][task_name] = {
            "with_structured": structured_result,
            "without_structured": unstructured_result
        }
        
        # Quick summary
        if structured_result.get("success"):
            print(f"    ✓ Structured: {structured_result['latency']}ms, Valid: {structured_result['valid']}")
        else:
            print(f"    ✗ Structured: {structured_result.get('error', 'Failed')}")
            
        if unstructured_result.get("success"):
            print(f"    ✓ Unstructured: {unstructured_result['latency']}ms, Valid: {unstructured_result['valid']}")
        else:
            print(f"    ✗ Unstructured: {unstructured_result.get('error', 'Failed')}")
    
    return results


async def run_benchmarks(tiers: List[str]) -> Dict[str, Any]:
    """Run benchmarks for specified tiers."""
    print(f"Starting benchmarks at {datetime.now().isoformat()}")
    print(f"Testing tiers: {', '.join(tiers)}")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tiers": tiers,
        "models": []
    }
    
    for tier in tiers:
        if tier not in MODELS_TO_TEST:
            print(f"Warning: Unknown tier {tier}")
            continue
            
        for model in MODELS_TO_TEST[tier]:
            result = await benchmark_model(model, tier)
            all_results["models"].append(result)
            
            # Small delay between models to avoid rate limits
            await asyncio.sleep(1)
    
    return all_results


def analyze_results(results: Dict[str, Any]) -> None:
    """Analyze and print summary of results."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Calculate success rates
    structured_success = 0
    structured_total = 0
    unstructured_success = 0
    unstructured_total = 0
    
    # Latency stats
    structured_latencies = []
    unstructured_latencies = []
    
    print("\nDetailed Results by Model:")
    print("-" * 60)
    
    for model_result in results["models"]:
        print(f"\n{model_result['model']} ({model_result['tier']})")
        
        for task_name, task_results in model_result["tasks"].items():
            print(f"  {task_name}:")
            
            # Structured
            s_result = task_results["with_structured"]
            if s_result.get("success"):
                structured_success += 1
                structured_latencies.append(s_result["latency"])
                print(f"    Structured:   ✓ {s_result['latency']:4d}ms Valid: {s_result['valid']}")
            else:
                print(f"    Structured:   ✗ {s_result.get('error', 'Failed')}")
            structured_total += 1
            
            # Unstructured
            u_result = task_results["without_structured"]
            if u_result.get("success"):
                unstructured_success += 1
                unstructured_latencies.append(u_result["latency"])
                print(f"    Unstructured: ✓ {u_result['latency']:4d}ms Valid: {u_result['valid']}")
            else:
                print(f"    Unstructured: ✗ {u_result.get('error', 'Failed')}")
            unstructured_total += 1
    
    # Summary statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    
    if structured_total > 0:
        structured_rate = (structured_success / structured_total) * 100
        print(f"\nStructured Output:")
        print(f"  Success Rate: {structured_rate:.1f}% ({structured_success}/{structured_total})")
        if structured_latencies:
            avg_latency = sum(structured_latencies) / len(structured_latencies)
            print(f"  Avg Latency:  {avg_latency:.0f}ms")
    
    if unstructured_total > 0:
        unstructured_rate = (unstructured_success / unstructured_total) * 100
        print(f"\nUnstructured Output:")
        print(f"  Success Rate: {unstructured_rate:.1f}% ({unstructured_success}/{unstructured_total})")
        if unstructured_latencies:
            avg_latency = sum(unstructured_latencies) / len(unstructured_latencies)
            print(f"  Avg Latency:  {avg_latency:.0f}ms")
    
    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_file}")


async def main():
    """Main entry point."""
    # Default to testing cheap models
    tiers = ["ultra_cheap", "cheap"]
    
    # Check command line args
    import sys
    if len(sys.argv) > 1:
        tiers = sys.argv[1].split(",")
    
    print("MCP-SGR Structured Output Benchmark")
    print("===================================")
    print(f"OpenRouter API Key: {'*' * 20}{OPENROUTER_API_KEY[-10:]}")
    
    results = await run_benchmarks(tiers)
    analyze_results(results)


if __name__ == "__main__":
    asyncio.run(main())