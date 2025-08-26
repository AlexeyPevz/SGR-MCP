#!/usr/bin/env python3
"""Benchmark comparing performance with and without SGR reasoning."""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: Please set OPENROUTER_API_KEY environment variable")
    exit(1)

# Models to test
MODELS = [
    {"name": "google/gemini-2.0-flash-exp:free", "tier": "ultra_cheap", "supports_structured": True},
    {"name": "openai/gpt-4o-mini", "tier": "cheap", "supports_structured": True},
    {"name": "anthropic/claude-3.5-haiku", "tier": "medium", "supports_structured": True},
    {"name": "meta-llama/llama-3.1-8b-instruct", "tier": "cheap", "supports_structured": False},
]

# Test scenarios with different complexities
TEST_SCENARIOS = {
    "code_analysis": {
        "task": """Analyze this Python code for potential issues:
```python
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
```""",
        "sgr_schema": {
            "type": "object",
            "properties": {
                "understanding": {
                    "type": "object",
                    "properties": {
                        "purpose": {"type": "string"},
                        "inputs": {"type": "string"},
                        "outputs": {"type": "string"}
                    }
                },
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high"]}
                        }
                    }
                },
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["understanding", "issues", "improvements", "confidence"]
        }
    },
    "planning": {
        "task": "Create a detailed plan for implementing a caching system for a REST API",
        "sgr_schema": {
            "type": "object",
            "properties": {
                "understanding": {
                    "type": "object",
                    "properties": {
                        "requirements": {"type": "array", "items": {"type": "string"}},
                        "constraints": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "plan": {
                    "type": "object",
                    "properties": {
                        "phases": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "tasks": {"type": "array", "items": {"type": "string"}},
                                    "duration": {"type": "string"}
                                }
                            }
                        },
                        "technologies": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk": {"type": "string"},
                            "mitigation": {"type": "string"}
                        }
                    }
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["understanding", "plan", "risks", "confidence"]
        }
    }
}


def call_openrouter(model: str, messages: List[Dict], response_format: Optional[Dict] = None, max_tokens: int = 1000) -> Dict[str, Any]:
    """Call OpenRouter API."""
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens
    }
    
    if response_format:
        data["response_format"] = response_format
    
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(data).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mcp-sgr/benchmarks",
            "X-Title": "MCP-SGR Benchmarks"
        }
    )
    
    start_time = time.time()
    
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            latency = int((time.time() - start_time) * 1000)
            
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            
            return {
                "success": True,
                "content": content,
                "latency": latency,
                "usage": usage
            }
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get('error', {}).get('message', 'Unknown error')
        except:
            error_msg = f"HTTP {e.code}"
        
        return {
            "success": False,
            "error": f"API error: {error_msg}",
            "status": e.code,
            "latency": int((time.time() - start_time) * 1000)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}",
            "latency": int((time.time() - start_time) * 1000)
        }


def test_baseline(model: str, task: str, schema: Dict) -> Dict[str, Any]:
    """Test without SGR - just ask for JSON output."""
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Provide responses in JSON format."
        },
        {
            "role": "user",
            "content": f"{task}\n\nProvide your response as JSON with this structure:\n{json.dumps(schema, indent=2)}"
        }
    ]
    
    return call_openrouter(model, messages)


def test_sgr_lite(model: str, task: str, schema: Dict, supports_structured: bool) -> Dict[str, Any]:
    """Test with SGR lite approach - guided reasoning."""
    
    messages = [
        {
            "role": "system",
            "content": """You are an expert assistant that uses structured reasoning.
Follow this process:
1. First, understand the task and identify key aspects
2. Then, analyze systematically 
3. Finally, provide a comprehensive response with confidence level"""
        },
        {
            "role": "user",
            "content": f"""{task}

Please reason through this step by step:
1. Understanding: What is being asked and what are the key requirements?
2. Analysis: Examine the details systematically
3. Response: Provide a structured response

Format your final response as JSON matching this schema:
{json.dumps(schema, indent=2)}"""
        }
    ]
    
    # Use structured output if supported
    response_format = None
    if supports_structured:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True
            }
        }
    
    return call_openrouter(model, messages, response_format)


def test_sgr_full(model: str, task: str, schema: Dict, supports_structured: bool) -> Dict[str, Any]:
    """Test with SGR full approach - multi-step reasoning."""
    
    # Step 1: Analysis
    analysis_messages = [
        {
            "role": "system",
            "content": "You are an expert analyst. Break down the task into its components."
        },
        {
            "role": "user",
            "content": f"Analyze this task and identify key aspects, requirements, and challenges:\n\n{task}"
        }
    ]
    
    analysis_result = call_openrouter(model, analysis_messages, max_tokens=500)
    if not analysis_result.get("success"):
        return analysis_result
    
    # Step 2: Structured response with context
    messages = [
        {
            "role": "system",
            "content": """You are an expert assistant. Use the analysis to provide a comprehensive structured response."""
        },
        {
            "role": "user",
            "content": f"""Task: {task}

Analysis from previous step:
{analysis_result['content']}

Based on this analysis, provide a detailed response in JSON format matching this schema:
{json.dumps(schema, indent=2)}

Ensure your response is comprehensive and includes a confidence score."""
        }
    ]
    
    response_format = None
    if supports_structured:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True
            }
        }
    
    final_result = call_openrouter(model, messages, response_format, max_tokens=1500)
    
    # Combine latencies
    if final_result.get("success"):
        final_result["latency"] = analysis_result["latency"] + final_result["latency"]
        final_result["sgr_steps"] = 2
    
    return final_result


def parse_json_response(content: str) -> tuple[Dict, bool]:
    """Try to parse JSON from response content."""
    try:
        # Clean up markdown if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end]
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end]
        
        parsed = json.loads(content.strip())
        return parsed, True
    except:
        return {}, False


def calculate_quality_score(response: Dict, schema: Dict) -> float:
    """Calculate quality score based on response completeness and structure."""
    if not response:
        return 0.0
    
    score = 0.0
    required_fields = schema.get("required", [])
    
    # Check required fields
    fields_present = sum(1 for field in required_fields if field in response)
    if required_fields:
        score += (fields_present / len(required_fields)) * 0.5
    
    # Check confidence if present
    if "confidence" in response and isinstance(response["confidence"], (int, float)):
        score += response["confidence"] * 0.3
    
    # Check depth (number of non-empty fields)
    non_empty_fields = sum(1 for v in response.values() if v)
    if non_empty_fields > 0:
        score += min(non_empty_fields / 10, 0.2)
    
    return min(score, 1.0)


def main():
    """Run comprehensive benchmark."""
    print("MCP-SGR Comprehensive Benchmark")
    print("===============================")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Models: {len(MODELS)}")
    print(f"Scenarios: {len(TEST_SCENARIOS)}")
    print()
    
    results = []
    
    for model_info in MODELS:
        model = model_info["name"]
        print(f"\n{'='*60}")
        print(f"Testing: {model} ({model_info['tier']})")
        print(f"Structured Output Support: {model_info['supports_structured']}")
        print('='*60)
        
        for scenario_name, scenario in TEST_SCENARIOS.items():
            print(f"\nScenario: {scenario_name}")
            print("-" * 40)
            
            # Test 1: Baseline (no SGR)
            print("  Baseline: ", end="", flush=True)
            baseline_result = test_baseline(model, scenario["task"], scenario["sgr_schema"])
            
            baseline_score = 0
            if baseline_result.get("success"):
                parsed, valid = parse_json_response(baseline_result["content"])
                baseline_score = calculate_quality_score(parsed, scenario["sgr_schema"])
                print(f"✓ {baseline_result['latency']}ms, Quality: {baseline_score:.2f}")
            else:
                print(f"✗ {baseline_result.get('error', 'Failed')}")
            
            time.sleep(2)  # Rate limit protection
            
            # Test 2: SGR Lite
            print("  SGR Lite: ", end="", flush=True)
            lite_result = test_sgr_lite(model, scenario["task"], scenario["sgr_schema"], model_info["supports_structured"])
            
            lite_score = 0
            if lite_result.get("success"):
                parsed, valid = parse_json_response(lite_result["content"])
                lite_score = calculate_quality_score(parsed, scenario["sgr_schema"])
                print(f"✓ {lite_result['latency']}ms, Quality: {lite_score:.2f}")
            else:
                print(f"✗ {lite_result.get('error', 'Failed')}")
            
            time.sleep(2)
            
            # Test 3: SGR Full
            print("  SGR Full: ", end="", flush=True)
            full_result = test_sgr_full(model, scenario["task"], scenario["sgr_schema"], model_info["supports_structured"])
            
            full_score = 0
            if full_result.get("success"):
                parsed, valid = parse_json_response(full_result["content"])
                full_score = calculate_quality_score(parsed, scenario["sgr_schema"])
                steps = full_result.get("sgr_steps", 1)
                print(f"✓ {full_result['latency']}ms, Quality: {full_score:.2f}, Steps: {steps}")
            else:
                print(f"✗ {full_result.get('error', 'Failed')}")
            
            # Store results
            results.append({
                "model": model,
                "tier": model_info["tier"],
                "scenario": scenario_name,
                "baseline": {
                    "success": baseline_result.get("success", False),
                    "latency": baseline_result.get("latency", 0),
                    "quality_score": baseline_score
                },
                "sgr_lite": {
                    "success": lite_result.get("success", False),
                    "latency": lite_result.get("latency", 0),
                    "quality_score": lite_score
                },
                "sgr_full": {
                    "success": full_result.get("success", False),
                    "latency": full_result.get("latency", 0),
                    "quality_score": full_score,
                    "steps": full_result.get("sgr_steps", 1)
                }
            })
            
            time.sleep(3)  # Longer delay between scenarios
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Calculate aggregated statistics
    baseline_scores = [r["baseline"]["quality_score"] for r in results if r["baseline"]["success"]]
    lite_scores = [r["sgr_lite"]["quality_score"] for r in results if r["sgr_lite"]["success"]]
    full_scores = [r["sgr_full"]["quality_score"] for r in results if r["sgr_full"]["success"]]
    
    baseline_latencies = [r["baseline"]["latency"] for r in results if r["baseline"]["success"]]
    lite_latencies = [r["sgr_lite"]["latency"] for r in results if r["sgr_lite"]["success"]]
    full_latencies = [r["sgr_full"]["latency"] for r in results if r["sgr_full"]["success"]]
    
    print("\nQuality Scores (0-1):")
    if baseline_scores:
        print(f"  Baseline: {sum(baseline_scores)/len(baseline_scores):.3f} (n={len(baseline_scores)})")
    if lite_scores:
        print(f"  SGR Lite: {sum(lite_scores)/len(lite_scores):.3f} (n={len(lite_scores)})")
    if full_scores:
        print(f"  SGR Full: {sum(full_scores)/len(full_scores):.3f} (n={len(full_scores)})")
    
    print("\nAverage Latencies:")
    if baseline_latencies:
        print(f"  Baseline: {sum(baseline_latencies)/len(baseline_latencies):.0f}ms")
    if lite_latencies:
        print(f"  SGR Lite: {sum(lite_latencies)/len(lite_latencies):.0f}ms")
    if full_latencies:
        print(f"  SGR Full: {sum(full_latencies)/len(full_latencies):.0f}ms")
    
    # Quality improvement
    if baseline_scores and lite_scores:
        lite_improvement = ((sum(lite_scores)/len(lite_scores)) - (sum(baseline_scores)/len(baseline_scores))) / (sum(baseline_scores)/len(baseline_scores)) * 100
        print(f"\nSGR Lite Quality Improvement: {lite_improvement:+.1f}%")
    
    if baseline_scores and full_scores:
        full_improvement = ((sum(full_scores)/len(full_scores)) - (sum(baseline_scores)/len(baseline_scores))) / (sum(baseline_scores)/len(baseline_scores)) * 100
        print(f"SGR Full Quality Improvement: {full_improvement:+.1f}%")
    
    # Save detailed results
    output_file = f"sgr_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "baseline_avg_quality": sum(baseline_scores)/len(baseline_scores) if baseline_scores else 0,
                "sgr_lite_avg_quality": sum(lite_scores)/len(lite_scores) if lite_scores else 0,
                "sgr_full_avg_quality": sum(full_scores)/len(full_scores) if full_scores else 0,
                "baseline_avg_latency": sum(baseline_latencies)/len(baseline_latencies) if baseline_latencies else 0,
                "sgr_lite_avg_latency": sum(lite_latencies)/len(lite_latencies) if lite_latencies else 0,
                "sgr_full_avg_latency": sum(full_latencies)/len(full_latencies) if full_latencies else 0
            },
            "detailed_results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()