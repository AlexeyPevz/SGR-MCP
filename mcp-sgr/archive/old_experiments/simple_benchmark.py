#!/usr/bin/env python3
"""Simple benchmark script using only standard library."""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Any

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: Please set OPENROUTER_API_KEY environment variable")
    exit(1)

# Models to test - focusing on those with structured output support
MODELS_TO_TEST = {
    "ultra_cheap": [
        "google/gemini-2.0-flash-exp:free",
    ],
    "cheap": [
        "google/gemini-flash-1.5",
        "openai/gpt-4o-mini",
    ],
    "medium": [
        "anthropic/claude-3.5-haiku",
    ]
}

# Simple test task
TEST_TASK = {
    "prompt": "Analyze this Python function and provide feedback: def calculate_sum(numbers): return sum(numbers)",
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "improvements": {"type": "array", "items": {"type": "string"}},
            "rating": {"type": "number", "minimum": 1, "maximum": 10}
        },
        "required": ["analysis", "improvements", "rating"]
    }
}


def call_openrouter(model: str, prompt: str, schema: Dict, with_structured: bool = True) -> Dict[str, Any]:
    """Call OpenRouter API with or without structured output."""
    
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
        "max_tokens": 500
    }
    
    # Add structured output if requested
    if with_structured:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True
            }
        }
    
    # Create request
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
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            
            # Extract response
            content = result["choices"][0]["message"]["content"]
            
            # Try to parse JSON
            try:
                # Clean up potential markdown
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                
                parsed = json.loads(content.strip())
                
                # Basic validation
                valid = all(key in parsed for key in schema.get("required", []))
                
                return {
                    "success": True,
                    "response": parsed,
                    "valid": valid,
                    "latency": int((time.time() - start_time) * 1000),
                    "tokens": result.get("usage", {})
                }
                
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Invalid JSON response",
                    "raw_response": content[:200],
                    "latency": int((time.time() - start_time) * 1000)
                }
                
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get('error', {}).get('message', 'Unknown error')
        except:
            error_msg = f"HTTP {e.code}"
        
        return {
            "error": f"API error: {error_msg}",
            "status": e.code,
            "latency": int((time.time() - start_time) * 1000)
        }
    except Exception as e:
        return {
            "error": f"Request failed: {str(e)}",
            "latency": int((time.time() - start_time) * 1000)
        }


def main():
    """Run simple benchmark."""
    print("MCP-SGR Simple Structured Output Benchmark")
    print("==========================================")
    print(f"API Key: {'*' * 30}{OPENROUTER_API_KEY[-10:]}")
    print(f"Time: {datetime.now().isoformat()}\n")
    
    results = []
    
    for tier, models in MODELS_TO_TEST.items():
        for model in models:
            print(f"\nTesting {model} ({tier}):")
            print("-" * 40)
            
            # Test WITH structured output
            print("  With structured output: ", end="", flush=True)
            structured_result = call_openrouter(model, TEST_TASK["prompt"], TEST_TASK["schema"], with_structured=True)
            
            if structured_result.get("success"):
                print(f"✓ {structured_result['latency']}ms, Valid: {structured_result['valid']}")
                if structured_result.get("response"):
                    rating = structured_result["response"].get("rating", "N/A")
                    print(f"    Rating: {rating}/10")
            else:
                print(f"✗ {structured_result.get('error', 'Failed')}")
            
            # Small delay
            time.sleep(1)
            
            # Test WITHOUT structured output
            print("  Without structured output: ", end="", flush=True)
            unstructured_result = call_openrouter(model, TEST_TASK["prompt"], TEST_TASK["schema"], with_structured=False)
            
            if unstructured_result.get("success"):
                print(f"✓ {unstructured_result['latency']}ms, Valid: {unstructured_result['valid']}")
                if unstructured_result.get("response"):
                    rating = unstructured_result["response"].get("rating", "N/A")
                    print(f"    Rating: {rating}/10")
            else:
                print(f"✗ {unstructured_result.get('error', 'Failed')}")
            
            # Store results
            results.append({
                "model": model,
                "tier": tier,
                "structured": structured_result,
                "unstructured": unstructured_result
            })
            
            # Delay between models
            time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    structured_success = sum(1 for r in results if r["structured"].get("success"))
    unstructured_success = sum(1 for r in results if r["unstructured"].get("success"))
    total = len(results)
    
    print(f"\nStructured Output Success Rate: {structured_success}/{total} ({structured_success/total*100:.0f}%)")
    print(f"Unstructured Output Success Rate: {unstructured_success}/{total} ({unstructured_success/total*100:.0f}%)")
    
    # Calculate average latencies
    structured_latencies = [r["structured"]["latency"] for r in results if r["structured"].get("success")]
    unstructured_latencies = [r["unstructured"]["latency"] for r in results if r["unstructured"].get("success")]
    
    if structured_latencies:
        avg_structured = sum(structured_latencies) / len(structured_latencies)
        print(f"\nAverage Structured Latency: {avg_structured:.0f}ms")
    
    if unstructured_latencies:
        avg_unstructured = sum(unstructured_latencies) / len(unstructured_latencies)
        print(f"Average Unstructured Latency: {avg_unstructured:.0f}ms")
    
    # Save results
    output_file = f"simple_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()