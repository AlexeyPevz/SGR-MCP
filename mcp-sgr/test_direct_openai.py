#!/usr/bin/env python3
"""Test direct OpenAI API and various models including budget alternatives."""

import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Any, Tuple

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Test configuration
TEST_TASK = "Analyze this code and suggest improvements: def process_data(items): result = []; for item in items: if item > 0: result.append(item * 2); return result"

# Simple schema for testing
TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 10}
    },
    "required": ["summary", "issues", "improvements", "quality_score"]
}

# Models to test
TEST_CONFIGS = [
    # Direct OpenAI API
    {
        "name": "gpt-4o (direct)",
        "model": "gpt-4o",
        "api_type": "openai",
        "supports_structured": True
    },
    {
        "name": "gpt-4o-mini (direct)",
        "model": "gpt-4o-mini",
        "api_type": "openai",
        "supports_structured": True
    },
    {
        "name": "gpt-3.5-turbo (direct)",
        "model": "gpt-3.5-turbo",
        "api_type": "openai",
        "supports_structured": True
    },
    
    # OpenRouter models - existing
    {
        "name": "claude-3.5-haiku",
        "model": "anthropic/claude-3.5-haiku",
        "api_type": "openrouter",
        "supports_structured": True
    },
    {
        "name": "claude-3.5-sonnet",
        "model": "anthropic/claude-3.5-sonnet",
        "api_type": "openrouter",
        "supports_structured": True
    },
    
    # Budget alternatives via OpenRouter
    {
        "name": "deepseek-chat",
        "model": "deepseek/deepseek-chat",
        "api_type": "openrouter",
        "supports_structured": False  # Test both
    },
    {
        "name": "qwen-2.5-72b",
        "model": "qwen/qwen-2.5-72b-instruct",
        "api_type": "openrouter",
        "supports_structured": False
    },
    {
        "name": "mistral-7b",
        "model": "mistralai/mistral-7b-instruct",
        "api_type": "openrouter",
        "supports_structured": False
    },
    {
        "name": "mixtral-8x7b",
        "model": "mistralai/mixtral-8x7b-instruct",
        "api_type": "openrouter",
        "supports_structured": False
    },
    {
        "name": "llama-3.1-70b",
        "model": "meta-llama/llama-3.1-70b-instruct",
        "api_type": "openrouter",
        "supports_structured": False
    },
    {
        "name": "gemini-flash-1.5",
        "model": "google/gemini-flash-1.5",
        "api_type": "openrouter",
        "supports_structured": True
    }
]


def call_openai_direct(model: str, messages: List[Dict], schema: Dict, use_structured: bool = True) -> Tuple[Dict, float]:
    """Call OpenAI API directly."""
    
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
    else:
        # Add instruction for JSON output
        messages[-1]["content"] += f"\n\nRespond with JSON in this exact format:\n{json.dumps(schema, indent=2)}"
    
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(data).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
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
                parsed = json.loads(content)
                return {
                    "success": True,
                    "response": parsed,
                    "latency": elapsed,
                    "tokens": result.get("usage", {})
                }, elapsed
                
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"JSON parse error: {e}",
                    "raw": content[:200],
                    "latency": elapsed
                }, elapsed
                
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "latency": elapsed
        }, elapsed


def call_openrouter(model: str, messages: List[Dict], schema: Dict, use_structured: bool = True) -> Tuple[Dict, float]:
    """Call model via OpenRouter."""
    
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
    else:
        # Add instruction for JSON output
        messages[-1]["content"] += f"\n\nRespond with JSON in this exact format:\n{json.dumps(schema, indent=2)}"
    
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
            
            # Parse JSON - handle markdown wrapped JSON
            if content.strip().startswith("```"):
                lines = content.strip().split('\n')
                json_start = 1 if lines[0] == "```json" else 0
                json_end = len(lines) - 1 if lines[-1] == "```" else len(lines)
                content = '\n'.join(lines[json_start:json_end])
            
            try:
                parsed = json.loads(content)
                return {
                    "success": True,
                    "response": parsed,
                    "latency": elapsed,
                    "tokens": result.get("usage", {})
                }, elapsed
                
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"JSON parse error: {e}",
                    "raw": content[:200],
                    "latency": elapsed
                }, elapsed
                
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        error_body = e.read().decode('utf-8')
        error_msg = f"HTTP {e.code}"
        
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get('error', {}).get('message', error_msg)
        except:
            pass
            
        return {
            "success": False,
            "error": error_msg,
            "http_code": e.code,
            "latency": elapsed
        }, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "latency": elapsed
        }, elapsed


def evaluate_response_quality(response: Dict) -> float:
    """Evaluate the quality of model response."""
    score = 0.0
    
    # Check summary
    summary = response.get("summary", "")
    if len(summary) > 20:
        score += 2.0
    
    # Check issues
    issues = response.get("issues", [])
    if len(issues) >= 1:
        score += 1.5
    if len(issues) >= 2:
        score += 1.0
    # Bonus for specific issues
    issues_text = " ".join(str(i).lower() for i in issues)
    if any(keyword in issues_text for keyword in ["semicolon", "style", "naming", "efficiency"]):
        score += 1.5
    
    # Check improvements
    improvements = response.get("improvements", [])
    if len(improvements) >= 1:
        score += 1.5
    if len(improvements) >= 2:
        score += 1.0
    
    # Check quality score
    quality_score = response.get("quality_score", 0)
    if 3 <= quality_score <= 8:  # Reasonable range
        score += 1.5
    
    return min(score, 10.0)


def test_model(config: Dict) -> Dict[str, Any]:
    """Test a single model configuration."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"Model: {config['model']}")
    print(f"API: {config['api_type']}")
    print(f"{'='*60}")
    
    messages = [
        {"role": "system", "content": "You are a code review expert. Analyze code and provide structured feedback."},
        {"role": "user", "content": f"Task: {TEST_TASK}"}
    ]
    
    results = {}
    
    # Test with structured output if supported
    if config["supports_structured"]:
        print("\nWith structured output:", end=" ", flush=True)
        
        if config["api_type"] == "openai":
            result, latency = call_openai_direct(config["model"], messages.copy(), TEST_SCHEMA, use_structured=True)
        else:
            result, latency = call_openrouter(config["model"], messages.copy(), TEST_SCHEMA, use_structured=True)
        
        if result["success"]:
            quality = evaluate_response_quality(result["response"])
            print(f"✓ Success ({latency:.2f}s, Quality: {quality:.1f}/10)")
            print(f"  Summary: {result['response'].get('summary', '')[:60]}...")
            print(f"  Issues: {len(result['response'].get('issues', []))}")
            print(f"  Improvements: {len(result['response'].get('improvements', []))}")
        else:
            print(f"✗ Failed: {result['error']}")
            if "raw" in result:
                print(f"  Raw response: {result['raw']}...")
        
        results["structured"] = result
    
    # Test without structured output
    print("\nWithout structured output:", end=" ", flush=True)
    
    if config["api_type"] == "openai":
        result, latency = call_openai_direct(config["model"], messages.copy(), TEST_SCHEMA, use_structured=False)
    else:
        result, latency = call_openrouter(config["model"], messages.copy(), TEST_SCHEMA, use_structured=False)
    
    if result["success"]:
        quality = evaluate_response_quality(result["response"])
        print(f"✓ Success ({latency:.2f}s, Quality: {quality:.1f}/10)")
        
        # Check schema compliance
        missing = []
        for field in TEST_SCHEMA["required"]:
            if field not in result["response"]:
                missing.append(field)
        
        if missing:
            print(f"  ⚠️  Missing required fields: {missing}")
    else:
        print(f"✗ Failed: {result['error']}")
        if "raw" in result:
            print(f"  Raw response: {result['raw']}...")
    
    results["unstructured"] = result
    
    return results


def main():
    """Run comprehensive model testing."""
    
    print("🚀 Comprehensive Model Testing")
    print("=" * 80)
    print(f"Task: {TEST_TASK}")
    print("=" * 80)
    
    # Check API keys
    if not OPENAI_API_KEY:
        print("⚠️  Warning: OPENAI_API_KEY not set, skipping direct OpenAI tests")
    if not OPENROUTER_API_KEY:
        print("⚠️  Warning: OPENROUTER_API_KEY not set, skipping OpenRouter tests")
    
    all_results = {}
    
    for config in TEST_CONFIGS:
        # Skip if API key not available
        if config["api_type"] == "openai" and not OPENAI_API_KEY:
            continue
        if config["api_type"] == "openrouter" and not OPENROUTER_API_KEY:
            continue
        
        try:
            results = test_model(config)
            all_results[config["name"]] = {
                "config": config,
                "results": results
            }
        except Exception as e:
            print(f"\n❌ Error testing {config['name']}: {e}")
            all_results[config["name"]] = {
                "config": config,
                "error": str(e)
            }
        
        time.sleep(2)  # Rate limiting
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Success rates
    print("\n📊 Success Rates:")
    for name, data in all_results.items():
        if "results" in data:
            structured_success = data["results"].get("structured", {}).get("success", False)
            unstructured_success = data["results"].get("unstructured", {}).get("success", False)
            
            status = []
            if data["config"]["supports_structured"]:
                status.append(f"Structured: {'✓' if structured_success else '✗'}")
            status.append(f"Unstructured: {'✓' if unstructured_success else '✗'}")
            
            print(f"{name:20} | {' | '.join(status)}")
    
    # Quality comparison
    print("\n🏆 Quality Rankings (Unstructured):")
    quality_scores = []
    
    for name, data in all_results.items():
        if "results" in data and data["results"].get("unstructured", {}).get("success"):
            response = data["results"]["unstructured"]["response"]
            quality = evaluate_response_quality(response)
            latency = data["results"]["unstructured"]["latency"]
            quality_scores.append((name, quality, latency))
    
    quality_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, quality, latency) in enumerate(quality_scores[:10]):
        print(f"{i+1}. {name:20} | Quality: {quality:.1f}/10 | Latency: {latency:.2f}s")
    
    # Cost efficiency (quality per dollar estimate)
    print("\n💰 Budget Options (Quality > 6.0):")
    budget_models = ["deepseek", "qwen", "mistral", "mixtral", "llama"]
    
    for name, quality, latency in quality_scores:
        if any(budget in name.lower() for budget in budget_models) and quality >= 6.0:
            print(f"  {name}: Quality {quality:.1f}/10, {latency:.2f}s")
    
    # Save detailed results
    filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump({
            "task": TEST_TASK,
            "schema": TEST_SCHEMA,
            "results": all_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to {filename}")


if __name__ == "__main__":
    main()