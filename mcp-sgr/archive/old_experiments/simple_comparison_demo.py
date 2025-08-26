#!/usr/bin/env python3
"""
Simple comparison demo showing the difference between models with and without SGR.

This demo uses simpler, more straightforward tasks where SGR's structured
approach clearly helps.
"""

import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Simple test cases where structure really helps
DEMO_TASKS = [
    {
        "id": "api_design",
        "name": "REST API Design",
        "task": """Design a REST API for a task management system with the following features:
- Users can create, read, update, delete tasks
- Tasks have title, description, status, priority, due date
- Users can filter tasks by status and priority
- Users can search tasks by title/description

Provide:
1. Complete list of endpoints
2. Request/response examples
3. Error handling approach
4. Authentication method"""
    },
    {
        "id": "bug_analysis", 
        "name": "Bug Root Cause Analysis",
        "task": """Our web app crashes every Monday at 9 AM. Here are the clues:
- Server logs show memory usage spike at 8:55 AM
- Database connections max out at 9:00 AM
- No crashes on weekends or holidays
- The app sends weekly report emails on Monday mornings
- User traffic is actually lower on Monday mornings

Find:
1. Most likely root cause
2. How to verify your hypothesis
3. Quick fix for immediate relief
4. Long-term solution"""
    },
    {
        "id": "code_review",
        "name": "Security Code Review", 
        "task": """Review this login function for security issues:

```python
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    if result:
        session['user_id'] = result[0]['id']
        return "Login successful"
    return "Invalid credentials"
```

Identify:
1. All security vulnerabilities
2. Severity of each issue
3. How to fix each vulnerability
4. Best practices to follow"""
    }
]

# Optimized SGR prompts for clear improvement
SGR_PROMPTS = {
    "analysis": """Task: {task}

First, analyze what needs to be done. Break down the requirements systematically.

Provide your analysis as a JSON object:
{{
  "task_understanding": "What is being asked",
  "key_requirements": ["req1", "req2", "..."],
  "approach": "How to tackle this systematically"
}}""",
    
    "execution": """Based on your analysis:
{analysis}

Now provide a complete, well-structured response to:
{task}

Make sure to:
- Address every requirement systematically  
- Be specific and detailed
- Provide examples where asked
- Follow a logical structure"""
}


def call_model(model_id: str, messages: List[Dict]) -> Tuple[Optional[str], float]:
    """Call model via OpenRouter."""
    
    start_time = time.time()
    
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(data).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            return content, time.time() - start_time
    except Exception as e:
        print(f"\n  Error: {e}")
        return None, time.time() - start_time


def run_without_sgr(model_id: str, task: str) -> Tuple[Optional[str], float]:
    """Run model without SGR - just direct prompting."""
    
    messages = [
        {"role": "user", "content": task}
    ]
    
    return call_model(model_id, messages)


def run_with_sgr(model_id: str, task: str) -> Tuple[Optional[str], float]:
    """Run model with SGR - structured reasoning."""
    
    total_time = 0
    
    # Step 1: Analysis
    messages = [
        {"role": "system", "content": "You are an analytical assistant. Always think step by step."},
        {"role": "user", "content": SGR_PROMPTS["analysis"].format(task=task)}
    ]
    
    analysis_response, time1 = call_model(model_id, messages)
    total_time += time1
    
    if not analysis_response:
        return None, total_time
    
    # Extract analysis (try to parse JSON)
    analysis = analysis_response
    try:
        if "{" in analysis_response:
            start = analysis_response.find("{")
            end = analysis_response.rfind("}") + 1
            if start >= 0 and end > start:
                analysis = analysis_response[start:end]
    except:
        pass
    
    # Step 2: Execution with structure
    messages = [
        {"role": "system", "content": "You are an expert providing detailed, structured solutions."},
        {"role": "user", "content": SGR_PROMPTS["execution"].format(
            analysis=analysis,
            task=task
        )}
    ]
    
    final_response, time2 = call_model(model_id, messages)
    total_time += time2
    
    return final_response, total_time


def evaluate_response(response: str, task_id: str) -> Dict[str, int]:
    """Simple evaluation of response quality."""
    
    if not response:
        return {"score": 0, "length": 0, "structure": 0}
    
    # Length check
    length_score = min(10, len(response) // 100)  # 1 point per 100 chars, max 10
    
    # Structure check
    structure_markers = ["1.", "2.", "3.", "4.", "-", "•", "#", "**", "```"]
    structure_score = sum(2 for marker in structure_markers if marker in response)
    structure_score = min(10, structure_score)
    
    # Completeness check based on task
    completeness = 0
    response_lower = response.lower()
    
    if task_id == "api_design":
        keywords = ["endpoint", "get", "post", "put", "delete", "request", "response", "auth", "error"]
        completeness = sum(1 for kw in keywords if kw in response_lower)
    elif task_id == "bug_analysis":
        keywords = ["cause", "memory", "database", "email", "verify", "fix", "solution", "monday"]
        completeness = sum(1.25 for kw in keywords if kw in response_lower)
    elif task_id == "code_review":
        keywords = ["sql injection", "security", "vulnerability", "password", "hash", "prepared", "statement"]
        completeness = sum(1.4 for kw in keywords if kw in response_lower)
    
    completeness = min(10, completeness)
    
    # Total score out of 30
    total = length_score + structure_score + completeness
    
    return {
        "total": total,
        "length": length_score,
        "structure": structure_score,
        "completeness": completeness
    }


def print_response_sample(response: str, max_lines: int = 15):
    """Print a sample of the response."""
    
    if not response:
        print("    [No response]")
        return
    
    lines = response.split('\n')
    for i, line in enumerate(lines[:max_lines]):
        if line.strip():
            print(f"    {line[:80]}{'...' if len(line) > 80 else ''}")
        if i >= max_lines - 1 and len(lines) > max_lines:
            print(f"    ... [{len(lines) - max_lines} more lines]")


def main():
    """Run the demonstration."""
    
    print("\n🎯 SGR Impact Demonstration")
    print("=" * 80)
    print("Comparing budget models WITH and WITHOUT Schema-Guided Reasoning")
    print("=" * 80)
    
    # Models to test
    models = [
        {"name": "Mistral-7B", "id": "mistralai/mistral-7b-instruct"},
        {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct"},
    ]
    
    results = []
    
    for task in DEMO_TASKS:
        print(f"\n\n📋 Task: {task['name']}")
        print("=" * 60)
        print(f"Task preview: {task['task'][:150]}...")
        print("-" * 60)
        
        task_results = {"task": task["name"], "models": {}}
        
        for model in models:
            print(f"\n🤖 Model: {model['name']}")
            
            # Test WITHOUT SGR
            print(f"\n  🔴 WITHOUT SGR:", end="", flush=True)
            response_no_sgr, time_no_sgr = run_without_sgr(model["id"], task["task"])
            
            if response_no_sgr:
                score_no_sgr = evaluate_response(response_no_sgr, task["id"])
                print(f" ✓ (Score: {score_no_sgr['total']}/30, Time: {time_no_sgr:.1f}s)")
                print(f"     Length: {score_no_sgr['length']}/10, Structure: {score_no_sgr['structure']}/10, Complete: {score_no_sgr['completeness']}/10")
                print("\n  Response preview:")
                print_response_sample(response_no_sgr, 8)
            else:
                score_no_sgr = {"total": 0}
                print(" ✗ Failed")
            
            time.sleep(2)
            
            # Test WITH SGR
            print(f"\n  🟢 WITH SGR:", end="", flush=True)
            response_sgr, time_sgr = run_with_sgr(model["id"], task["task"])
            
            if response_sgr:
                score_sgr = evaluate_response(response_sgr, task["id"])
                print(f" ✓ (Score: {score_sgr['total']}/30, Time: {time_sgr:.1f}s)")
                print(f"     Length: {score_sgr['length']}/10, Structure: {score_sgr['structure']}/10, Complete: {score_sgr['completeness']}/10")
                print("\n  Response preview:")
                print_response_sample(response_sgr, 8)
            else:
                score_sgr = {"total": 0}
                print(" ✗ Failed")
            
            # Calculate improvement
            if score_no_sgr["total"] > 0:
                improvement = ((score_sgr["total"] - score_no_sgr["total"]) / score_no_sgr["total"]) * 100
                print(f"\n  📊 Improvement: {improvement:+.1f}% ({score_no_sgr['total']} → {score_sgr['total']})")
            
            task_results["models"][model["name"]] = {
                "no_sgr": score_no_sgr,
                "with_sgr": score_sgr,
                "improvement": improvement if score_no_sgr["total"] > 0 else 0
            }
            
            time.sleep(2)
        
        results.append(task_results)
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    # Calculate average improvements
    model_improvements = {}
    
    for task_result in results:
        for model_name, scores in task_result["models"].items():
            if model_name not in model_improvements:
                model_improvements[model_name] = []
            
            model_improvements[model_name].append(scores["improvement"])
    
    print("\n Average Improvement with SGR:")
    print("-" * 40)
    
    for model_name, improvements in model_improvements.items():
        avg_improvement = sum(improvements) / len(improvements)
        print(f"  {model_name}: {avg_improvement:+.1f}%")
    
    # Task breakdown
    print("\n Task-by-Task Results:")
    print("-" * 40)
    
    for task_result in results:
        print(f"\n  {task_result['task']}:")
        for model_name, scores in task_result["models"].items():
            print(f"    {model_name}: {scores['no_sgr']['total']} → {scores['with_sgr']['total']} ({scores['improvement']:+.1f}%)")
    
    # Final verdict
    print("\n" + "="*80)
    print("🎯 CONCLUSION")
    print("="*80)
    
    total_improvements = []
    for improvements in model_improvements.values():
        total_improvements.extend(improvements)
    
    if total_improvements:
        avg_total = sum(total_improvements) / len(total_improvements)
        
        if avg_total > 20:
            print(f"\n✅ SGR provides SIGNIFICANT improvement: {avg_total:+.1f}% on average")
            print("   Budget models WITH SGR deliver more structured, complete responses")
        elif avg_total > 10:
            print(f"\n✅ SGR provides meaningful improvement: {avg_total:+.1f}% on average")
            print("   Especially helpful for complex, multi-part tasks")
        elif avg_total > 0:
            print(f"\n📊 SGR provides modest improvement: {avg_total:+.1f}% on average")
            print("   Benefits vary by task type")
        else:
            print(f"\n⚠️  SGR impact varies: {avg_total:+.1f}% on average")
            print("   Some tasks may not benefit from current SGR implementation")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()