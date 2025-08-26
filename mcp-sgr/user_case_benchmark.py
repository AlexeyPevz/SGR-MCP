#!/usr/bin/env python3
"""
User Case Benchmark - Test SGR with custom user scenarios

This script allows testing specific user-provided cases to demonstrate
SGR effectiveness on real-world tasks.
"""

import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Quick model configs for user testing
QUICK_TEST_MODELS = {
    "budget": {
        "name": "Qwen-2.5-72B",
        "id": "qwen/qwen-2.5-72b-instruct",
        "api": "openrouter",
        "cost_per_1k": 0.0003
    },
    "premium": {
        "name": "GPT-4o-mini",
        "id": "gpt-4o-mini",
        "api": "openai", 
        "cost_per_1k": 0.00015
    }
}

# SGR prompts optimized based on testing
SGR_PROMPTS = {
    "analysis": """You are an expert analyst. Analyze this task systematically.

Task: {task}

Provide a structured analysis in JSON format:
{{
  "understanding": "Clear summary of what needs to be done",
  "key_requirements": ["req1", "req2", ...],
  "approach": {{
    "methodology": "How to approach this",
    "steps": ["step1", "step2", ...],
    "considerations": ["consideration1", ...]
  }},
  "solution": {{
    "main_solution": "Primary approach",
    "alternatives": ["alt1", "alt2", ...],
    "rationale": "Why this approach"
  }},
  "quality_check": {{
    "completeness": "Does this fully address the task?",
    "accuracy": "Technical correctness assessment",
    "practicality": "Real-world applicability"
  }},
  "confidence": 0.85
}}""",
    
    "refinement": """Based on the analysis below, provide a comprehensive, high-quality response to the original task.

Analysis:
{analysis}

Original Task:
{task}

Provide a detailed, practical, and actionable response that fully addresses all requirements."""
}


def call_api(model_info: Dict, messages: List[Dict]) -> Tuple[Optional[str], float, int, float]:
    """Call model API and return response, time, tokens, and cost."""
    
    start_time = time.time()
    
    if model_info["api"] == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    
    headers["Content-Type"] = "application/json"
    
    data = {
        "model": model_info["id"],
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2500
    }
    
    request = urllib.request.Request(url, json.dumps(data).encode('utf-8'), headers)
    
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            elapsed = time.time() - start_time
            
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            cost = (tokens / 1000) * model_info["cost_per_1k"]
            
            return content, elapsed, tokens, cost
            
    except Exception as e:
        print(f"\nError: {e}")
        return None, time.time() - start_time, 0, 0


def apply_sgr_to_task(model_info: Dict, task: str) -> Dict[str, Any]:
    """Apply SGR to enhance model response."""
    
    print("  → Phase 1: Structured analysis...", end="", flush=True)
    
    # Phase 1: Analysis
    messages = [
        {"role": "system", "content": "You are an expert analyst. Always respond in valid JSON."},
        {"role": "user", "content": SGR_PROMPTS["analysis"].format(task=task)}
    ]
    
    analysis_response, time1, tokens1, cost1 = call_api(model_info, messages)
    
    if not analysis_response:
        print(" ✗")
        return {"success": False, "error": "Analysis failed"}
    
    print(f" ✓ ({time1:.1f}s)")
    
    # Parse analysis
    try:
        # Extract JSON from response
        if "```json" in analysis_response:
            json_str = analysis_response.split("```json")[1].split("```")[0]
        elif "```" in analysis_response:
            json_str = analysis_response.split("```")[1].split("```")[0]
        else:
            start = analysis_response.find("{")
            end = analysis_response.rfind("}") + 1
            json_str = analysis_response[start:end] if start >= 0 else analysis_response
        
        analysis = json.loads(json_str)
    except Exception as e:
        print(f"  ⚠️  JSON parse error: {e}")
        analysis = {"raw": analysis_response}
    
    # Phase 2: Refinement
    print("  → Phase 2: Generating refined response...", end="", flush=True)
    
    messages = [
        {"role": "system", "content": "You are an expert providing detailed, practical solutions."},
        {"role": "user", "content": SGR_PROMPTS["refinement"].format(
            analysis=json.dumps(analysis, indent=2) if isinstance(analysis, dict) else str(analysis),
            task=task
        )}
    ]
    
    final_response, time2, tokens2, cost2 = call_api(model_info, messages)
    
    if not final_response:
        print(" ✗")
        return {"success": False, "error": "Refinement failed"}
    
    print(f" ✓ ({time2:.1f}s)")
    
    return {
        "success": True,
        "response": final_response,
        "analysis": analysis,
        "total_time": time1 + time2,
        "total_tokens": tokens1 + tokens2,
        "total_cost": cost1 + cost2,
        "phases": {
            "analysis": {"time": time1, "tokens": tokens1, "cost": cost1},
            "refinement": {"time": time2, "tokens": tokens2, "cost": cost2}
        }
    }


def run_baseline(model_info: Dict, task: str) -> Dict[str, Any]:
    """Run model without SGR for comparison."""
    
    print("  → Generating response...", end="", flush=True)
    
    messages = [
        {"role": "system", "content": "You are an expert assistant. Provide detailed, high-quality responses."},
        {"role": "user", "content": task}
    ]
    
    response, elapsed, tokens, cost = call_api(model_info, messages)
    
    if not response:
        print(" ✗")
        return {"success": False, "error": "API call failed"}
    
    print(f" ✓ ({elapsed:.1f}s)")
    
    return {
        "success": True,
        "response": response,
        "total_time": elapsed,
        "total_tokens": tokens,
        "total_cost": cost
    }


def evaluate_responses(task: str, responses: Dict[str, Dict]) -> Dict[str, Any]:
    """Basic evaluation of response quality."""
    
    evaluation = {}
    
    for model_name, result in responses.items():
        if not result["success"]:
            evaluation[model_name] = {"score": 0, "notes": "Failed to generate response"}
            continue
        
        response = result["response"]
        
        # Basic quality metrics
        metrics = {
            "length": len(response),
            "structured": sum(1 for marker in ["1.", "2.", "•", "-", "#", "##"] if marker in response),
            "code_blocks": response.count("```"),
            "technical_depth": sum(1 for term in [
                "implement", "architecture", "performance", "optimize", "algorithm",
                "design", "pattern", "best practice", "security", "scalability",
                "error handling", "edge case", "complexity", "efficiency"
            ] if term.lower() in response.lower()),
            "actionable": sum(1 for word in [
                "should", "recommend", "suggest", "must", "need to", "consider",
                "ensure", "verify", "implement", "create", "add", "modify"
            ] if word.lower() in response.lower())
        }
        
        # Simple scoring (0-10)
        score = min(10, (
            min(metrics["length"] / 200, 3) +  # Length score (max 3)
            min(metrics["structured"], 2) +      # Structure score (max 2)
            min(metrics["technical_depth"] / 3, 2) +  # Technical score (max 2)
            min(metrics["actionable"] / 3, 2) +  # Actionability score (max 2)
            (1 if metrics["code_blocks"] > 0 else 0)  # Code presence (max 1)
        ))
        
        evaluation[model_name] = {
            "score": round(score, 2),
            "metrics": metrics,
            "cost": result["total_cost"],
            "time": result["total_time"],
            "value_score": score / result["total_cost"] if result["total_cost"] > 0 else 0
        }
    
    return evaluation


def interactive_mode():
    """Interactive mode for testing custom user cases."""
    
    print("\n🔬 SGR User Case Testing")
    print("=" * 60)
    print("Test your own scenarios to see how SGR improves responses")
    print("=" * 60)
    
    while True:
        print("\nEnter your task/question (or 'quit' to exit):")
        print("Example: 'Explain how to implement a rate limiter in Python'")
        print("> ", end="")
        
        task = input().strip()
        
        if task.lower() in ['quit', 'exit', 'q']:
            break
        
        if not task:
            print("Please enter a task.")
            continue
        
        print(f"\n📋 Task: {task[:100]}{'...' if len(task) > 100 else ''}")
        print("-" * 60)
        
        results = {}
        
        # Test budget model with SGR
        print("\n💰 Testing Qwen-2.5-72B with SGR:")
        sgr_result = apply_sgr_to_task(QUICK_TEST_MODELS["budget"], task)
        results["Qwen-72B + SGR"] = sgr_result
        
        # Test budget model without SGR
        print("\n💰 Testing Qwen-2.5-72B without SGR:")
        baseline_budget = run_baseline(QUICK_TEST_MODELS["budget"], task)
        results["Qwen-72B Baseline"] = baseline_budget
        
        # Test premium model without SGR (if API key available)
        if OPENAI_API_KEY:
            print("\n💎 Testing GPT-4o-mini without SGR:")
            baseline_premium = run_baseline(QUICK_TEST_MODELS["premium"], task)
            results["GPT-4o-mini Baseline"] = baseline_premium
        
        # Evaluate results
        print("\n" + "="*60)
        print("📊 RESULTS COMPARISON")
        print("="*60)
        
        evaluation = evaluate_responses(task, results)
        
        # Print comparison table
        print(f"\n{'Model':<25} {'Score':<8} {'Cost':<10} {'Time':<10} {'Value':<10}")
        print("-" * 65)
        
        for model_name, eval_data in evaluation.items():
            if eval_data["score"] > 0:
                print(f"{model_name:<25} {eval_data['score']:<8.1f} "
                      f"${eval_data['cost']:<9.4f} {eval_data['time']:<10.1f} "
                      f"{eval_data['value_score']:<10.0f}")
        
        # Show responses
        print("\n" + "="*60)
        print("📝 RESPONSE SAMPLES")
        print("="*60)
        
        for model_name, result in results.items():
            if result["success"]:
                print(f"\n### {model_name}")
                print("-" * 40)
                print(result["response"][:500] + "..." if len(result["response"]) > 500 else result["response"])
        
        # Key findings
        sgr_score = evaluation.get("Qwen-72B + SGR", {}).get("score", 0)
        baseline_score = evaluation.get("Qwen-72B Baseline", {}).get("score", 0)
        premium_score = evaluation.get("GPT-4o-mini Baseline", {}).get("score", 0)
        
        print("\n" + "="*60)
        print("🎯 KEY FINDINGS")
        print("="*60)
        
        if sgr_score > 0 and baseline_score > 0:
            improvement = ((sgr_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            print(f"\n✅ SGR improved Qwen-72B by {improvement:+.0f}%")
        
        if sgr_score > 0 and premium_score > 0:
            ratio = sgr_score / premium_score
            if ratio >= 0.9:
                print(f"🏆 Qwen-72B + SGR matches GPT-4o-mini quality ({ratio:.0%})!")
            else:
                print(f"📊 Qwen-72B + SGR achieves {ratio:.0%} of GPT-4o-mini quality")
        
        # Save option
        print("\nSave this test? (y/n): ", end="")
        if input().strip().lower() == 'y':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"user_case_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "task": task,
                    "results": results,
                    "evaluation": evaluation
                }, f, indent=2)
            
            print(f"✓ Saved to {filename}")


def batch_mode(test_file: str):
    """Run batch tests from a JSON file."""
    
    print(f"\n📁 Loading test cases from {test_file}")
    
    try:
        with open(test_file, 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    if not isinstance(test_cases, list):
        test_cases = [test_cases]
    
    print(f"Found {len(test_cases)} test cases")
    
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        if isinstance(test_case, str):
            task = test_case
            case_id = f"case_{i+1}"
        else:
            task = test_case.get("task", test_case.get("prompt", ""))
            case_id = test_case.get("id", f"case_{i+1}")
        
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}/{len(test_cases)}: {case_id}")
        print(f"{'='*60}")
        print(f"Task: {task[:100]}...")
        
        results = {}
        
        # Test with SGR
        print("\n💰 Qwen-72B + SGR:")
        results["sgr"] = apply_sgr_to_task(QUICK_TEST_MODELS["budget"], task)
        
        # Test without SGR
        print("\n💰 Qwen-72B Baseline:")
        results["baseline"] = run_baseline(QUICK_TEST_MODELS["budget"], task)
        
        # Evaluate
        evaluation = evaluate_responses(task, {
            "Qwen + SGR": results["sgr"],
            "Qwen Baseline": results["baseline"]
        })
        
        all_results.append({
            "case_id": case_id,
            "task": task,
            "results": results,
            "evaluation": evaluation
        })
        
        time.sleep(2)  # Rate limiting
    
    # Summary
    print("\n" + "="*60)
    print("📊 BATCH SUMMARY")
    print("="*60)
    
    sgr_scores = []
    baseline_scores = []
    improvements = []
    
    for result in all_results:
        sgr_score = result["evaluation"].get("Qwen + SGR", {}).get("score", 0)
        baseline_score = result["evaluation"].get("Qwen Baseline", {}).get("score", 0)
        
        if sgr_score > 0 and baseline_score > 0:
            sgr_scores.append(sgr_score)
            baseline_scores.append(baseline_score)
            improvements.append((sgr_score - baseline_score) / baseline_score * 100)
    
    if sgr_scores:
        avg_sgr = sum(sgr_scores) / len(sgr_scores)
        avg_baseline = sum(baseline_scores) / len(baseline_scores)
        avg_improvement = sum(improvements) / len(improvements)
        
        print(f"\nAverage Scores:")
        print(f"  Qwen + SGR: {avg_sgr:.1f}")
        print(f"  Qwen Baseline: {avg_baseline:.1f}")
        print(f"  Improvement: {avg_improvement:+.0f}%")
    
    # Save results
    filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to {filename}")


def main():
    """Main entry point."""
    
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        return
    
    if len(sys.argv) > 1:
        # Batch mode
        batch_mode(sys.argv[1])
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()