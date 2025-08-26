#!/usr/bin/env python3
"""Optimize SGR Lite mode by testing different prompt strategies."""

import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Any, Tuple

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: Please set OPENROUTER_API_KEY")
    exit(1)

# Test task for optimization
TEST_TASK = "Analyze this Python code for issues and suggest improvements:\ndef calculate_average(numbers):\n    return sum(numbers) / len(numbers)"

# Target schema (simplified for Lite mode)
LITE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"}
    },
    "required": ["summary", "issues", "improvements", "confidence"]
}

# Different prompt strategies for SGR Lite
LITE_STRATEGIES = {
    "v1_current": {
        "system": "You are a JSON-only assistant. Respond with STRICTLY valid JSON matching the provided schema.",
        "prompt": """Task: {task}

Provide a JSON response following this structure:
{schema}

Be concise but complete."""
    },
    
    "v2_guided": {
        "system": "You are an analytical assistant. Always respond in valid JSON format.",
        "prompt": """Task: {task}

Analyze this step by step:
1. Understand the task
2. Identify key issues or points
3. Suggest improvements

Format your response as JSON:
{schema}

Include a confidence score (0-1) based on your analysis completeness."""
    },
    
    "v3_example": {
        "system": "You are a JSON assistant. Follow the example format exactly.",
        "prompt": """Task: {task}

Respond with JSON like this example:
{{
  "summary": "Brief overview of the analysis",
  "issues": ["First issue", "Second issue"],
  "improvements": ["First suggestion", "Second suggestion"],
  "confidence": 0.8
}}

Your JSON response:"""
    },
    
    "v4_structured": {
        "system": "Provide structured JSON responses for all analyses.",
        "prompt": """Task: {task}

Required JSON structure:
{schema}

Guidelines:
- Summary: One sentence overview
- Issues: List 2-3 specific problems (if any)
- Improvements: List 2-3 actionable suggestions
- Confidence: Rate 0-1 based on clarity of the task

Response:"""
    },
    
    "v5_minimal": {
        "system": "JSON responses only. Be direct and structured.",
        "prompt": """Task: {task}

JSON format: {{"summary": "...", "issues": [...], "improvements": [...], "confidence": 0.X}}

Analyze and respond:"""
    },
    
    "v6_role": {
        "system": "You are a code review expert. Communicate only in JSON.",
        "prompt": """As an expert, analyze this task:
{task}

Provide your expert analysis in this JSON format:
{schema}

Focus on practical, actionable insights."""
    },
    
    "v7_template": {
        "system": "Fill in JSON templates with your analysis.",
        "prompt": """Task: {task}

Fill this template with your analysis:
{{
  "summary": "[Your 1-sentence summary here]",
  "issues": [
    "[Issue 1 if any]",
    "[Issue 2 if any]"
  ],
  "improvements": [
    "[Improvement 1]",
    "[Improvement 2]"
  ],
  "confidence": [0.0-1.0]
}}"""
    },
    
    "v8_direct": {
        "system": "Expert analyst providing JSON-formatted insights.",
        "prompt": """{task}

Respond with JSON containing: summary, issues array, improvements array, confidence number."""
    }
}


def test_strategy(model: str, strategy_name: str, strategy: Dict) -> Tuple[Dict, float]:
    """Test a specific prompt strategy."""
    
    # Format prompt
    prompt = strategy["prompt"].format(
        task=TEST_TASK,
        schema=json.dumps(LITE_SCHEMA, indent=2)
    )
    
    messages = [
        {"role": "system", "content": strategy["system"]},
        {"role": "user", "content": prompt}
    ]
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 800,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis",
                "schema": LITE_SCHEMA,
                "strict": True
            }
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
            parsed = json.loads(content)
            
            # Evaluate quality
            quality_score = evaluate_response_quality(parsed)
            
            return {
                "success": True,
                "response": parsed,
                "latency": elapsed,
                "tokens": result.get("usage", {}),
                "quality_score": quality_score
            }, elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "latency": elapsed
        }, elapsed


def evaluate_response_quality(response: Dict) -> float:
    """Evaluate the quality of the response."""
    score = 0.0
    
    # Check completeness
    if response.get("summary") and len(response["summary"]) > 10:
        score += 0.25
    
    # Check issues quality
    issues = response.get("issues", [])
    if len(issues) >= 1:
        score += 0.15
    if len(issues) >= 2:
        score += 0.1
    # Bonus for specific issues
    if any("division by zero" in str(i).lower() for i in issues):
        score += 0.1
    
    # Check improvements quality
    improvements = response.get("improvements", [])
    if len(improvements) >= 1:
        score += 0.15
    if len(improvements) >= 2:
        score += 0.1
    
    # Check confidence
    confidence = response.get("confidence", 0)
    if 0.5 <= confidence <= 0.9:  # Reasonable confidence
        score += 0.1
    
    # Check for generic/placeholder content
    generic_terms = ["n/a", "none", "no issues", "looks good"]
    response_str = json.dumps(response).lower()
    if not any(term in response_str for term in generic_terms):
        score += 0.05
    
    return min(score, 1.0)


def run_optimization_test():
    """Run optimization test for all strategies."""
    
    print("🔧 SGR Lite Optimization Test")
    print("=" * 60)
    print(f"Task: {TEST_TASK}")
    print("=" * 60)
    
    models = [
        "anthropic/claude-3.5-haiku",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-exp:free"
    ]
    
    results = {}
    
    for model in models:
        print(f"\n📊 Testing model: {model}")
        print("-" * 60)
        
        model_results = {}
        
        for strategy_name, strategy in LITE_STRATEGIES.items():
            print(f"\n{strategy_name}:", end=" ", flush=True)
            
            result, latency = test_strategy(model, strategy_name, strategy)
            
            if result["success"]:
                print(f"✓ {latency:.2f}s, Quality: {result['quality_score']:.2f}")
                print(f"  Summary: {result['response']['summary'][:60]}...")
                print(f"  Issues: {len(result['response'].get('issues', []))}")
                print(f"  Improvements: {len(result['response'].get('improvements', []))}")
                print(f"  Confidence: {result['response'].get('confidence', 0):.2f}")
            else:
                print(f"✗ Failed: {result['error']}")
            
            model_results[strategy_name] = result
            time.sleep(2)  # Rate limiting
        
        results[model] = model_results
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    # Find best strategies per model
    for model, model_results in results.items():
        print(f"\n{model}:")
        
        # Sort by quality score
        successful = [(name, r) for name, r in model_results.items() if r["success"]]
        if successful:
            sorted_strategies = sorted(successful, key=lambda x: x[1]["quality_score"], reverse=True)
            
            print(f"  Best strategy: {sorted_strategies[0][0]} (Quality: {sorted_strategies[0][1]['quality_score']:.2f})")
            print(f"  Success rate: {len(successful)}/{len(model_results)} ({len(successful)/len(model_results)*100:.0f}%)")
            
            # Top 3 strategies
            print("  Top 3 strategies:")
            for i, (name, result) in enumerate(sorted_strategies[:3]):
                print(f"    {i+1}. {name}: {result['quality_score']:.2f} quality, {result['latency']:.2f}s")
    
    # Overall best strategy
    print("\n" + "-"*60)
    print("OVERALL BEST STRATEGIES:")
    
    all_results = []
    for model, model_results in results.items():
        for strategy_name, result in model_results.items():
            if result["success"]:
                all_results.append({
                    "model": model,
                    "strategy": strategy_name,
                    "quality": result["quality_score"],
                    "latency": result["latency"]
                })
    
    # Sort by quality
    all_results.sort(key=lambda x: x["quality"], reverse=True)
    
    print("\nTop 5 combinations:")
    for i, r in enumerate(all_results[:5]):
        print(f"{i+1}. {r['strategy']} on {r['model'].split('/')[-1]}: {r['quality']:.2f} quality, {r['latency']:.2f}s")
    
    # Save results
    filename = f"sgr_lite_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump({
            "task": TEST_TASK,
            "schema": LITE_SCHEMA,
            "strategies": LITE_STRATEGIES,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {filename}")
    
    # Generate improved SGR Lite prompt
    if all_results:
        best = all_results[0]
        print(f"\n💡 RECOMMENDED SGR LITE CONFIGURATION:")
        print(f"Strategy: {best['strategy']}")
        print(f"Best with: {best['model']}")
        print("\nPrompt template:")
        print("-" * 40)
        print(f"System: {LITE_STRATEGIES[best['strategy']]['system']}")
        print(f"\nUser: {LITE_STRATEGIES[best['strategy']]['prompt']}")


if __name__ == "__main__":
    run_optimization_test()