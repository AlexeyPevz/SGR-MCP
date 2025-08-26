#!/usr/bin/env python3
"""
Comprehensive benchmark for SGR v4 - testing multiple models and task types
"""

import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
sys.path.append('/workspace/mcp-sgr/src/tools')
from apply_sgr_v4 import TASK_SCHEMAS, detect_task_type

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Расширенный список моделей для тестирования
TEST_MODELS = [
    # Большие модели
    {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct", "size": "72B"},
    {"name": "Llama-3.1-70B", "id": "meta-llama/llama-3.1-70b-instruct", "size": "70B"},
    {"name": "Mixtral-8x22B", "id": "mistralai/mixtral-8x22b-instruct", "size": "176B"},
    
    # Средние модели  
    {"name": "Qwen-2.5-32B", "id": "qwen/qwen-2.5-32b-instruct", "size": "32B"},
    {"name": "Gemini-Flash-1.5", "id": "google/gemini-flash-1.5", "size": "unknown"},
    {"name": "Claude-3-Haiku", "id": "anthropic/claude-3-haiku", "size": "unknown"},
    
    # Маленькие модели
    {"name": "Mistral-7B", "id": "mistralai/mistral-7b-instruct", "size": "7B"},
    {"name": "Llama-3.2-3B", "id": "meta-llama/llama-3.2-3b-instruct", "size": "3B"},
    {"name": "Gemma-2-9B", "id": "google/gemma-2-9b-it", "size": "9B"},
    
    # GPT модели
    {"name": "GPT-4o-mini", "id": "openai/gpt-4o-mini", "size": "unknown"},
    {"name": "GPT-3.5-turbo", "id": "openai/gpt-3.5-turbo", "size": "unknown"},
]

# Тестовые задачи разных типов
TEST_TASKS = {
    "code_review": {
        "name": "Security Code Review",
        "task": """Review this authentication function:

```python
def authenticate_user(username, password):
    # Check credentials
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    user = db.execute(query).fetchone()
    
    if user:
        # Create session
        session_id = username + "_" + str(time.time())
        sessions[session_id] = user
        return {"success": True, "session": session_id}
    
    return {"error": "Invalid credentials"}
```

Analyze for security vulnerabilities, performance issues, and best practices."""
    },
    
    "system_design": {
        "name": "Distributed Cache Design",
        "task": """Design a distributed caching system with these requirements:

1. Handle 1M requests per second
2. Sub-millisecond latency (p99 < 1ms)
3. Support for multiple data centers
4. Automatic failover and recovery
5. TTL and eviction policies
6. Consistency guarantees (eventual consistency is OK)

Consider data structures, protocols, and operational aspects."""
    },
    
    "debugging": {
        "name": "Production Issue Debug",
        "task": """Debug this production issue:

Our payment processing service is failing intermittently:
- Errors spike every day at 2 AM
- Error: "Connection pool exhausted"
- CPU and memory usage are normal
- Database connections look healthy
- The service processes payments fine the rest of the day
- Started happening after we added automatic daily reports
- Reports query the payment database directly

What's the root cause and how do we fix it?"""
    },
    
    "general_reasoning": {
        "name": "Technical Decision",
        "task": """Our startup needs to choose between PostgreSQL and MongoDB for our main database.

Context:
- B2B SaaS application
- Complex relational data (users, organizations, permissions, audit logs)
- Need ACID compliance for financial transactions
- Expecting 100k users in year 1
- Team has more SQL experience
- Some features need flexible schemas

What would you recommend and why?"""
    }
}


def call_model_with_sgr(
    model_id: str, 
    task: str, 
    task_type: str,
    schema: Dict
) -> Tuple[Optional[Dict], float, int, Optional[str]]:
    """Call model with SGR v4 approach and return parsed result."""
    
    start_time = time.time()
    
    system_prompt = """You are an expert assistant providing structured analysis.

Your response MUST follow the provided JSON schema exactly. The schema guides your reasoning - each field represents a critical aspect to analyze.

Think step-by-step through each schema section. Be specific and actionable."""
    
    user_prompt = f"""Task: {task}

Analyze this as a {task_type.replace('_', ' ')} task.

Structure your response as JSON matching this schema:
{json.dumps(schema, indent=2)}

Important:
- Fill ALL required fields
- Be specific and detailed
- Follow the schema's logical flow
- Respond with valid JSON only"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 3000
    }
    
    # Add response_format for models that support it
    if "gemini" in model_id.lower():
        data["response_format"] = {"type": "json_object"}
    
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(data).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            
            # Try to parse JSON
            parsed = None
            error = None
            
            try:
                # Try direct parsing
                if content.strip().startswith("{"):
                    parsed = json.loads(content)
                # Try extracting from markdown
                elif "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                    parsed = json.loads(json_str)
                # Try finding JSON in text
                else:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(content[start:end])
                    else:
                        error = "No JSON found in response"
                        
            except json.JSONDecodeError as e:
                error = f"JSON parse error: {str(e)[:50]}"
                
            return parsed, time.time() - start_time, tokens, error
            
    except Exception as e:
        error = f"API error: {str(e)[:50]}"
        return None, time.time() - start_time, 0, error


def evaluate_response(response: Dict, task_type: str, schema: Dict) -> Dict[str, float]:
    """Evaluate the quality and completeness of SGR response."""
    
    if not response:
        return {"total": 0, "completeness": 0, "quality": 0}
    
    scores = {}
    
    # Check schema compliance
    required_fields = schema.get("required", [])
    present_fields = sum(1 for field in required_fields if field in response)
    scores["schema_compliance"] = (present_fields / len(required_fields)) * 10 if required_fields else 10
    
    # Task-specific evaluation
    if task_type == "code_review":
        # Check if SQL injection found
        vulns = response.get("security_analysis", {}).get("vulnerabilities", [])
        sql_injection_found = any(
            "sql" in str(v).lower() and "injection" in str(v).lower() 
            for v in vulns
        )
        scores["critical_issue_found"] = 10 if sql_injection_found else 0
        
        # Check completeness
        scores["analysis_depth"] = min(10, len(vulns) * 3)
        scores["recommendations"] = 10 if response.get("recommendations", {}).get("must_fix") else 0
        
    elif task_type == "system_design":
        # Check key components
        scores["requirements"] = 10 if response.get("requirements_analysis") else 0
        scores["architecture"] = 10 if response.get("architecture", {}).get("components") else 0
        scores["decisions"] = min(10, len(response.get("design_decisions", [])) * 2)
        
    elif task_type == "debugging":
        # Check reasoning quality
        hypotheses = response.get("hypothesis", [])
        scores["hypothesis_count"] = min(10, len(hypotheses) * 3)
        scores["root_cause"] = 10 if response.get("root_cause", {}).get("identified") else 5
        scores["solution"] = 10 if response.get("solution", {}).get("immediate_fix") else 0
        
    else:  # general_reasoning
        scores["understanding"] = 10 if response.get("understanding", {}).get("core_question") else 0
        scores["reasoning"] = min(10, len(response.get("reasoning_steps", [])) * 3)
        scores["conclusion"] = 10 if response.get("answer", {}).get("main_response") else 0
    
    # Calculate total
    scores["total"] = sum(scores.values()) / len(scores) if scores else 0
    
    return scores


def main():
    """Run comprehensive SGR v4 benchmark."""
    
    print("\n🚀 Comprehensive SGR v4 Benchmark")
    print("=" * 80)
    print(f"Testing {len(TEST_MODELS)} models on {len(TEST_TASKS)} task types")
    print("=" * 80)
    
    results = []
    
    # Test each task type
    for task_key, task_info in TEST_TASKS.items():
        print(f"\n\n📋 Task Type: {task_info['name']}")
        print("=" * 80)
        print(f"Task preview: {task_info['task'][:150]}...")
        
        schema = TASK_SCHEMAS[task_key]
        print(f"Schema sections: {', '.join(schema['properties'].keys())}")
        
        task_results = []
        
        # Test each model
        for i, model in enumerate(TEST_MODELS):
            print(f"\n{i+1}/{len(TEST_MODELS)} Testing {model['name']} ({model['size']})...", end="", flush=True)
            
            # Call model with SGR
            response, duration, tokens, error = call_model_with_sgr(
                model["id"], 
                task_info["task"],
                task_key,
                schema
            )
            
            if response:
                scores = evaluate_response(response, task_key, schema)
                print(f" ✅ Score: {scores['total']:.1f}/10, Time: {duration:.1f}s")
                
                # Show some details for interesting responses
                if task_key == "code_review" and scores.get("critical_issue_found", 0) > 0:
                    print(f"     → Found SQL injection vulnerability!")
                elif task_key == "debugging" and response.get("root_cause", {}).get("identified"):
                    cause = response["root_cause"].get("description", "")[:60]
                    print(f"     → Root cause: {cause}...")
                    
            else:
                scores = {"total": 0}
                print(f" ❌ Failed: {error}")
            
            result = {
                "model": model["name"],
                "model_size": model["size"],
                "task": task_key,
                "success": response is not None,
                "scores": scores,
                "duration": duration,
                "tokens": tokens,
                "error": error
            }
            
            task_results.append(result)
            results.append(result)
            
            # Rate limiting
            time.sleep(1)
        
        # Task summary
        print(f"\n📊 {task_info['name']} Summary:")
        print("-" * 60)
        
        # Sort by score
        sorted_results = sorted(task_results, key=lambda x: x["scores"]["total"], reverse=True)
        
        print("Top 5 models:")
        for i, r in enumerate(sorted_results[:5]):
            status = "✅" if r["success"] else "❌"
            print(f"  {i+1}. {status} {r['model']}: {r['scores']['total']:.1f}/10")
    
    # Overall summary
    print("\n\n" + "="*80)
    print("📊 OVERALL BENCHMARK RESULTS")
    print("="*80)
    
    # Group by model
    model_scores = {}
    for r in results:
        model = r["model"]
        if model not in model_scores:
            model_scores[model] = []
        model_scores[model].append(r["scores"]["total"])
    
    # Calculate averages
    model_averages = []
    for model, scores in model_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        success_rate = sum(1 for r in results if r["model"] == model and r["success"]) / len(TEST_TASKS)
        model_size = next(m["size"] for m in TEST_MODELS if m["name"] == model)
        
        model_averages.append({
            "model": model,
            "size": model_size,
            "avg_score": avg_score,
            "success_rate": success_rate
        })
    
    # Sort by average score
    model_averages.sort(key=lambda x: x["avg_score"], reverse=True)
    
    print("\n🏆 Model Rankings (by average score):")
    print("-" * 70)
    print(f"{'Rank':<6}{'Model':<20}{'Size':<10}{'Avg Score':<12}{'Success Rate'}")
    print("-" * 70)
    
    for i, m in enumerate(model_averages):
        print(f"{i+1:<6}{m['model']:<20}{m['size']:<10}{m['avg_score']:<12.1f}{m['success_rate']:.0%}")
    
    # Size category analysis
    print("\n📈 Performance by Model Size:")
    print("-" * 50)
    
    size_categories = {"small": [], "medium": [], "large": []}
    for m in model_averages:
        if m["size"] in ["3B", "7B", "9B"]:
            size_categories["small"].append(m["avg_score"])
        elif m["size"] in ["32B", "unknown"]:
            size_categories["medium"].append(m["avg_score"])
        else:  # 70B+
            size_categories["large"].append(m["avg_score"])
    
    for category, scores in size_categories.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {category.capitalize()}: {avg:.1f}/10 average")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sgr_v4_benchmark_results_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "models_tested": len(TEST_MODELS),
            "tasks_tested": len(TEST_TASKS),
            "results": results,
            "model_rankings": model_averages
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    # Final insights
    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS")
    print("="*80)
    
    if model_averages:
        best_model = model_averages[0]
        print(f"\n1. Best Overall Model: {best_model['model']} ({best_model['avg_score']:.1f}/10)")
        
        # Best budget model
        budget_models = [m for m in model_averages if m["size"] in ["3B", "7B", "9B"]]
        if budget_models:
            best_budget = budget_models[0]
            print(f"2. Best Budget Model: {best_budget['model']} ({best_budget['avg_score']:.1f}/10)")
        
        # Success rate analysis
        high_success = [m for m in model_averages if m["success_rate"] >= 0.75]
        print(f"3. Models with >75% success rate: {len(high_success)}/{len(TEST_MODELS)}")
        
        # SGR effectiveness
        sgr_effective = [m for m in model_averages if m["avg_score"] >= 7.0]
        print(f"4. Models effective with SGR (>7/10): {len(sgr_effective)}/{len(TEST_MODELS)}")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()