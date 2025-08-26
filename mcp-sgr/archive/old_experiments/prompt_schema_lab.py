#!/usr/bin/env python3
"""Interactive prompt and schema testing laboratory for MCP-SGR."""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: Please set OPENROUTER_API_KEY environment variable")
    exit(1)

# Test configurations
TEST_MODELS = {
    "1": {"name": "anthropic/claude-3.5-haiku", "supports_structured": True},
    "2": {"name": "openai/gpt-4o-mini", "supports_structured": True},
    "3": {"name": "google/gemini-2.0-flash-exp:free", "supports_structured": True},
}

# Preset schemas for testing
PRESET_SCHEMAS = {
    "simple": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["summary", "key_points", "confidence"]
    },
    "analysis": {
        "type": "object",
        "properties": {
            "understanding": {"type": "string"},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "point": {"type": "string"},
                        "importance": {"type": "string", "enum": ["low", "medium", "high"]}
                    }
                }
            },
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"}
        },
        "required": ["understanding", "findings", "recommendations", "confidence"]
    },
    "minimal": {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "score": {"type": "number"}
        },
        "required": ["result"]
    }
}

# Preset prompts
PRESET_PROMPTS = {
    "baseline": """Task: {task}

Provide a JSON response with the following structure:
{schema}""",
    
    "structured": """Task: {task}

Please analyze this systematically and provide a structured response.

Requirements:
1. Be specific and actionable
2. Consider multiple aspects
3. Rate your confidence

Response format:
{schema}""",
    
    "cot": """Task: {task}

Let's think through this step by step:
1. First, understand what's being asked
2. Then, analyze the key aspects
3. Finally, provide recommendations

Structure your response as JSON:
{schema}""",
    
    "expert": """You are an expert analyst. 

Task: {task}

Provide a comprehensive analysis following best practices in your field.
Be thorough but concise. Include confidence assessment.

JSON structure required:
{schema}"""
}


def call_model(model: str, messages: List[Dict], schema: Dict, use_structured: bool = True) -> Tuple[Dict, float]:
    """Call model and return result with timing."""
    
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
                "name": "test_response",
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
            
            # Parse JSON
            try:
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                parsed = json.loads(content.strip())
                return {"success": True, "response": parsed, "raw": content}, elapsed
                
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"JSON parse error: {e}", "raw": content}, elapsed
                
    except Exception as e:
        elapsed = time.time() - start_time
        return {"success": False, "error": str(e)}, elapsed


def test_prompt_schema_combination(
    task: str,
    prompt_template: str,
    schema: Dict,
    model_info: Dict,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Test a specific prompt/schema combination."""
    
    # Format prompt
    schema_str = json.dumps(schema, indent=2)
    prompt = prompt_template.format(task=task, schema=schema_str)
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Test with structured output
    print(f"\n{'='*60}")
    print(f"Model: {model_info['name']}")
    print(f"Structured Output: {model_info['supports_structured']}")
    print(f"{'='*60}")
    
    if model_info['supports_structured']:
        print("\nWith structured output:")
        result_structured, time_structured = call_model(
            model_info['name'], messages, schema, use_structured=True
        )
        
        if result_structured['success']:
            print(f"✓ Success ({time_structured:.2f}s)")
            print(f"Response: {json.dumps(result_structured['response'], indent=2)}")
        else:
            print(f"✗ Failed: {result_structured['error']}")
            if 'raw' in result_structured:
                print(f"Raw: {result_structured['raw'][:200]}...")
    
    # Test without structured output
    print("\nWithout structured output:")
    result_unstructured, time_unstructured = call_model(
        model_info['name'], messages, schema, use_structured=False
    )
    
    if result_unstructured['success']:
        print(f"✓ Success ({time_unstructured:.2f}s)")
        print(f"Response: {json.dumps(result_unstructured['response'], indent=2)}")
    else:
        print(f"✗ Failed: {result_unstructured['error']}")
        if 'raw' in result_unstructured:
            print(f"Raw: {result_unstructured['raw'][:200]}...")
    
    # Analyze results
    analysis = {
        "model": model_info['name'],
        "structured_success": result_structured.get('success', False) if model_info['supports_structured'] else None,
        "unstructured_success": result_unstructured.get('success', False),
        "structured_time": time_structured if model_info['supports_structured'] else None,
        "unstructured_time": time_unstructured,
    }
    
    # Check schema compliance
    if result_structured.get('success') and model_info['supports_structured']:
        response = result_structured['response']
        analysis['has_all_required'] = all(field in response for field in schema.get('required', []))
        analysis['confidence_present'] = 'confidence' in response
        analysis['confidence_value'] = response.get('confidence', 0)
    
    return analysis


def interactive_mode():
    """Interactive prompt/schema testing mode."""
    
    print("\n🔬 MCP-SGR Prompt & Schema Laboratory")
    print("=" * 50)
    
    # Select task
    print("\nEnter your test task (or press Enter for default):")
    default_task = "Analyze the pros and cons of using microservices architecture"
    task = input(f"[{default_task}]: ").strip() or default_task
    
    # Select model
    print("\nSelect model to test:")
    for key, model in TEST_MODELS.items():
        print(f"{key}. {model['name']}")
    model_choice = input("Choice [1]: ").strip() or "1"
    model_info = TEST_MODELS.get(model_choice, TEST_MODELS["1"])
    
    # Select or create schema
    print("\nSelect schema:")
    print("1. Simple (summary + key points)")
    print("2. Analysis (detailed structure)")
    print("3. Minimal (just result)")
    print("4. Custom (enter your own)")
    schema_choice = input("Choice [1]: ").strip() or "1"
    
    if schema_choice == "4":
        print("\nEnter custom schema (JSON format):")
        custom_schema = input()
        try:
            schema = json.loads(custom_schema)
        except:
            print("Invalid JSON, using simple schema")
            schema = PRESET_SCHEMAS["simple"]
    else:
        schema_map = {"1": "simple", "2": "analysis", "3": "minimal"}
        schema = PRESET_SCHEMAS[schema_map.get(schema_choice, "simple")]
    
    # Select or create prompt template
    print("\nSelect prompt style:")
    print("1. Baseline (minimal)")
    print("2. Structured (clear requirements)")
    print("3. Chain-of-thought")
    print("4. Expert (role-based)")
    print("5. Custom")
    prompt_choice = input("Choice [2]: ").strip() or "2"
    
    if prompt_choice == "5":
        print("\nEnter custom prompt template (use {task} and {schema} placeholders):")
        prompt_template = input()
    else:
        prompt_map = {"1": "baseline", "2": "structured", "3": "cot", "4": "expert"}
        prompt_template = PRESET_PROMPTS[prompt_map.get(prompt_choice, "structured")]
    
    # Optional system prompt
    print("\nEnter system prompt (or press Enter to skip):")
    system_prompt = input().strip() or None
    
    # Run test
    print("\n🚀 Running test...")
    analysis = test_prompt_schema_combination(task, prompt_template, schema, model_info, system_prompt)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {analysis['model']}")
    if analysis['structured_success'] is not None:
        print(f"Structured Output: {'✓' if analysis['structured_success'] else '✗'} ({analysis['structured_time']:.2f}s)")
    print(f"Unstructured Output: {'✓' if analysis['unstructured_success'] else '✗'} ({analysis['unstructured_time']:.2f}s)")
    
    if analysis.get('has_all_required') is not None:
        print(f"Schema Compliance: {'✓' if analysis['has_all_required'] else '✗'}")
    if analysis.get('confidence_present'):
        print(f"Confidence: {analysis['confidence_value']:.2f}")
    
    # Save results option
    print("\nSave results? (y/n) [n]: ", end="")
    if input().strip().lower() == 'y':
        filename = f"prompt_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump({
                "task": task,
                "model": model_info,
                "schema": schema,
                "prompt_template": prompt_template,
                "system_prompt": system_prompt,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        print(f"Results saved to {filename}")
    
    print("\nTest another? (y/n) [y]: ", end="")
    if input().strip().lower() != 'n':
        interactive_mode()


def batch_test_mode():
    """Test multiple prompt/schema combinations."""
    
    print("\n📊 Batch Testing Mode")
    print("=" * 50)
    
    task = "Create a plan for implementing a caching system in a web application"
    
    results = []
    
    # Test all combinations
    for model_key, model_info in TEST_MODELS.items():
        for schema_name, schema in PRESET_SCHEMAS.items():
            for prompt_name, prompt_template in PRESET_PROMPTS.items():
                print(f"\nTesting: {model_info['name']} / {schema_name} / {prompt_name}")
                
                try:
                    analysis = test_prompt_schema_combination(
                        task, prompt_template, schema, model_info
                    )
                    analysis['schema_name'] = schema_name
                    analysis['prompt_name'] = prompt_name
                    results.append(analysis)
                except Exception as e:
                    print(f"Error: {e}")
                    results.append({
                        "model": model_info['name'],
                        "schema_name": schema_name,
                        "prompt_name": prompt_name,
                        "error": str(e)
                    })
                
                time.sleep(2)  # Rate limiting
    
    # Summary table
    print("\n" + "="*80)
    print("BATCH TEST RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'Schema':<10} {'Prompt':<10} {'Struct':<8} {'Unstruct':<8}")
    print("-"*80)
    
    for r in results:
        if 'error' not in r:
            struct = "✓" if r.get('structured_success') else "✗" if r.get('structured_success') is not None else "N/A"
            unstruct = "✓" if r.get('unstructured_success') else "✗"
            print(f"{r['model']:<30} {r['schema_name']:<10} {r['prompt_name']:<10} {struct:<8} {unstruct:<8}")
    
    # Save results
    filename = f"batch_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump({
            "task": task,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nDetailed results saved to {filename}")


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_test_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()