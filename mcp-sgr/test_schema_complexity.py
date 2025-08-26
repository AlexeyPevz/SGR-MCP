#!/usr/bin/env python3
"""Test different schema complexity levels to find optimal balance."""

import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Any, Tuple

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Test task
TEST_TASK = "Analyze the security implications of storing passwords in plain text"

# Different complexity levels of schemas
SCHEMAS = {
    "ultra_simple": {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "score": {"type": "number"}
        }
    },
    
    "simple": {
        "type": "object", 
        "properties": {
            "summary": {"type": "string"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "score": {"type": "number"}
        },
        "required": ["summary", "issues"]
    },
    
    "moderate": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5
            },
            "recommendations": {
                "type": "array", 
                "items": {"type": "string"}
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["summary", "issues", "severity"]
    },
    
    "complex": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["summary"]
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "impact": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"]
                        }
                    },
                    "required": ["issue", "severity"]
                }
            },
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "priority": {"type": "number"},
                        "effort": {"type": "string"}
                    }
                }
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "confidence": {"type": "number"},
                    "analysis_type": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            }
        },
        "required": ["analysis", "findings", "recommendations"]
    }
}

# Models to test
TEST_MODELS = [
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o-mini", 
    "google/gemini-flash-1.5",
    "openai/gpt-4o"
]


def test_schema(model: str, schema_name: str, schema: Dict, use_structured: bool = True) -> Dict[str, Any]:
    """Test a schema with a model."""
    
    prompt = f"""Task: {TEST_TASK}

Provide your analysis in JSON format following this structure:
{json.dumps(schema, indent=2)}"""
    
    messages = [
        {"role": "system", "content": "You are a security analyst. Provide detailed analysis in JSON format."},
        {"role": "user", "content": prompt}
    ]
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    if use_structured:
        data["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis",
                "schema": schema,
                "strict": True
            }
        }
    
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(data).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
    )
    
    start_time = time.time()
    
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            elapsed = time.time() - start_time
            
            content = result["choices"][0]["message"]["content"]
            
            # Try to parse JSON
            try:
                parsed = json.loads(content)
                
                # Check if all required fields are present
                missing_required = []
                for field in schema.get("required", []):
                    if field not in parsed:
                        missing_required.append(field)
                
                return {
                    "success": True,
                    "latency": elapsed,
                    "response": parsed,
                    "missing_required": missing_required,
                    "response_size": len(content)
                }
                
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"JSON parse error: {str(e)}",
                    "latency": elapsed,
                    "raw_response": content[:200]
                }
                
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        error_body = e.read().decode('utf-8')
        error_details = "Unknown error"
        
        try:
            error_json = json.loads(error_body)
            error_details = error_json.get('error', {}).get('message', str(e))
        except:
            error_details = f"HTTP {e.code}"
            
        return {
            "success": False,
            "error": error_details,
            "http_code": e.code,
            "latency": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "latency": elapsed
        }


def main():
    """Run schema complexity tests."""
    
    print("🔍 Schema Complexity Testing")
    print("=" * 80)
    print(f"Task: {TEST_TASK}")
    print("=" * 80)
    
    results = {}
    
    for model in TEST_MODELS:
        print(f"\n📊 Testing model: {model}")
        print("-" * 60)
        
        model_results = {}
        
        for schema_name, schema in SCHEMAS.items():
            print(f"\n{schema_name} schema:", end=" ", flush=True)
            
            # Test with structured output
            result = test_schema(model, schema_name, schema, use_structured=True)
            
            if result["success"]:
                print(f"✓ {result['latency']:.2f}s, {result['response_size']} chars")
                if result["missing_required"]:
                    print(f"  ⚠️  Missing required fields: {result['missing_required']}")
            else:
                print(f"✗ {result.get('error', 'Failed')}")
                if result.get('http_code') == 400:
                    print("  → Schema rejected by model")
            
            model_results[schema_name] = result
            time.sleep(2)  # Rate limiting
        
        # Test without structured output for comparison
        print(f"\nWithout structured output (moderate schema):", end=" ", flush=True)
        unstructured_result = test_schema(model, "moderate_unstructured", SCHEMAS["moderate"], use_structured=False)
        
        if unstructured_result["success"]:
            print(f"✓ {unstructured_result['latency']:.2f}s")
        else:
            print(f"✗ {unstructured_result.get('error', 'Failed')}")
        
        model_results["unstructured"] = unstructured_result
        results[model] = model_results
    
    # Analysis
    print("\n" + "="*80)
    print("COMPLEXITY ANALYSIS")
    print("="*80)
    
    # Success rate by complexity
    complexity_success = {schema_name: [] for schema_name in SCHEMAS.keys()}
    
    for model, model_results in results.items():
        for schema_name in SCHEMAS.keys():
            if schema_name in model_results:
                complexity_success[schema_name].append(model_results[schema_name]["success"])
    
    print("\nSuccess rate by schema complexity:")
    for schema_name, successes in complexity_success.items():
        if successes:
            rate = sum(successes) / len(successes) * 100
            print(f"  {schema_name}: {rate:.0f}% ({sum(successes)}/{len(successes)})")
    
    # Model compatibility
    print("\nModel compatibility:")
    for model in TEST_MODELS:
        if model in results:
            successes = sum(1 for r in results[model].values() if r["success"])
            total = len(results[model])
            print(f"  {model}: {successes}/{total} schemas work")
            
            # Find max complexity that works
            working_schemas = []
            for schema_name in ["ultra_simple", "simple", "moderate", "complex"]:
                if schema_name in results[model] and results[model][schema_name]["success"]:
                    working_schemas.append(schema_name)
            
            if working_schemas:
                print(f"    Max working complexity: {working_schemas[-1]}")
    
    # Latency analysis
    print("\nAverage latency by complexity:")
    for schema_name in SCHEMAS.keys():
        latencies = []
        for model, model_results in results.items():
            if schema_name in model_results and model_results[schema_name]["success"]:
                latencies.append(model_results[schema_name]["latency"])
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"  {schema_name}: {avg_latency:.2f}s")
    
    # Recommendations
    print("\n" + "-"*60)
    print("RECOMMENDATIONS:")
    print("-"*60)
    
    # Find best schema for each model
    for model in TEST_MODELS:
        if model in results:
            print(f"\n{model}:")
            
            # Find most complex working schema
            best_schema = None
            for schema_name in reversed(["ultra_simple", "simple", "moderate", "complex"]):
                if schema_name in results[model] and results[model][schema_name]["success"]:
                    best_schema = schema_name
                    break
            
            if best_schema:
                print(f"  ✅ Use '{best_schema}' schema")
            else:
                print(f"  ❌ Structured output not working properly")
            
            # Check if unstructured is better
            if "unstructured" in results[model] and results[model]["unstructured"]["success"]:
                if not best_schema or best_schema == "ultra_simple":
                    print(f"  💡 Consider using unstructured mode instead")
    
    # Save results
    filename = f"schema_complexity_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump({
            "task": TEST_TASK,
            "schemas": SCHEMAS,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {filename}")


if __name__ == "__main__":
    main()