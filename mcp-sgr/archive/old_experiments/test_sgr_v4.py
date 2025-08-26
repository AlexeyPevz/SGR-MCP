#!/usr/bin/env python3
"""
Test script for SGR v4 - однофазная реализация
"""

import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('/workspace/mcp-sgr/src/tools')
from apply_sgr_v4 import detect_task_type, TASK_SCHEMAS

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Тестовые задачи разных типов
TEST_TASKS = {
    "code_review": """Review this Python function:

```python
def process_payment(user_id, amount, card_number):
    # Check if user exists
    user = db.query(f"SELECT * FROM users WHERE id = {user_id}")
    if not user:
        return {"error": "User not found"}
    
    # Process payment
    charge = stripe.charge.create(
        amount=amount,
        currency="usd",
        source=card_number,
        description=f"Payment from user {user_id}"
    )
    
    # Log transaction
    db.execute(f"INSERT INTO transactions VALUES ({user_id}, {amount}, '{charge.id}')")
    
    return {"success": True, "charge_id": charge.id}
```

Analyze for security issues, performance, and best practices.""",

    "system_design": """Design a real-time collaboration system like Google Docs with these requirements:

1. Support 100+ concurrent users per document
2. Sub-second latency for text updates
3. Conflict resolution for simultaneous edits
4. Offline mode with sync when reconnected
5. Version history and rollback capability
6. Must scale to millions of documents

Consider the architecture, data structures, and algorithms needed.""",

    "debugging": """Our production API is experiencing issues:

Symptoms:
- Random 500 errors (about 5% of requests)
- Errors started after deploying new caching layer
- Only happens under load (>1000 req/s)
- Error logs show "connection pool exhausted"
- But monitoring shows DB connections are fine
- Redis memory usage is normal

The caching code looks fine in review. What could be the issue?""",

    "general": """Explain the tradeoffs between microservices and monolithic architectures for a startup."""
}


def call_sgr_v4(model: str, task: str, task_type: str, schema: Dict) -> Tuple[Optional[Dict], float, int]:
    """Call model with SGR v4 approach."""
    
    start_time = time.time()
    
    # Подготовка промпта в стиле SGR v4
    system_prompt = """You are an expert assistant providing structured analysis. 

Your response MUST follow the provided JSON schema exactly. The schema is designed to guide your reasoning process - each field represents a critical aspect you must consider.

Think systematically through each part of the schema. Be specific, concrete, and thorough."""
    
    user_prompt = f"""Task: {task}

Task Type: {task_type.replace('_', ' ').title()}

Provide your response following this exact JSON structure:
{json.dumps(schema, indent=2)}

Important:
1. Fill ALL required fields
2. Be specific and actionable
3. Show your reasoning explicitly
4. Follow the logical flow of the schema

Respond ONLY with valid JSON matching the schema above. No additional text."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 3000
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
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            
            # Parse JSON
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(content)
                
                return parsed, time.time() - start_time, tokens
                
            except json.JSONDecodeError as e:
                print(f"\n  JSON Parse Error: {e}")
                print(f"  Response preview: {content[:200]}...")
                return None, time.time() - start_time, tokens
                
    except Exception as e:
        print(f"\n  API Error: {e}")
        return None, time.time() - start_time, 0


def evaluate_sgr_response(response: Dict, task_type: str) -> Dict[str, float]:
    """Evaluate quality of SGR response."""
    
    if not response:
        return {"total": 0}
    
    scores = {}
    
    if task_type == "code_review":
        # Check if critical security issues found
        vulnerabilities = response.get("security_analysis", {}).get("vulnerabilities", [])
        sql_injection_found = any(v.get("type") == "sql_injection" for v in vulnerabilities)
        scores["security"] = 10 if sql_injection_found else 0
        
        # Check completeness
        scores["completeness"] = 0
        if response.get("initial_understanding", {}).get("purpose"):
            scores["completeness"] += 2.5
        if len(vulnerabilities) > 0:
            scores["completeness"] += 2.5
        if response.get("recommendations", {}).get("must_fix"):
            scores["completeness"] += 2.5
        if response.get("performance_analysis", {}).get("bottlenecks"):
            scores["completeness"] += 2.5
            
    elif task_type == "system_design":
        # Check for key components
        scores["requirements"] = 10 if response.get("requirements_analysis") else 0
        scores["decisions"] = len(response.get("design_decisions", [])) * 2  # 2 points per decision
        scores["architecture"] = 10 if response.get("architecture", {}).get("components") else 0
        scores["completeness"] = (scores["requirements"] + min(10, scores["decisions"]) + scores["architecture"]) / 3
        
    elif task_type == "debugging":
        # Check hypothesis quality
        hypotheses = response.get("hypothesis", [])
        scores["hypothesis_count"] = min(10, len(hypotheses) * 3)
        scores["solution"] = 10 if response.get("solution", {}).get("immediate_fix") else 0
        scores["completeness"] = (scores["hypothesis_count"] + scores["solution"]) / 2
        
    else:  # general
        scores["understanding"] = 10 if response.get("understanding", {}).get("core_question") else 0
        scores["reasoning"] = min(10, len(response.get("reasoning_steps", [])) * 3)
        scores["answer"] = 10 if response.get("answer", {}).get("main_response") else 0
        scores["completeness"] = (scores["understanding"] + scores["reasoning"] + scores["answer"]) / 3
    
    scores["total"] = scores.get("completeness", 0)
    return scores


def main():
    """Test SGR v4 implementation."""
    
    print("\n🚀 Testing SGR v4 - Single-Phase Implementation")
    print("=" * 80)
    
    # Test task type detection
    print("\n1️⃣ Testing Task Type Detection:")
    print("-" * 50)
    
    for task_name, task in TEST_TASKS.items():
        detected = detect_task_type(task)
        print(f"  {task_name}: detected as '{detected}'")
    
    # Test with different models
    models = [
        {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct"},
        {"name": "Mistral-7B", "id": "mistralai/mistral-7b-instruct"},
    ]
    
    print("\n2️⃣ Testing SGR Response Generation:")
    print("-" * 50)
    
    results = []
    
    # Test code review task with both models
    task_type = "code_review"
    task = TEST_TASKS[task_type]
    schema = TASK_SCHEMAS[task_type]
    
    print(f"\n📋 Task: Code Review")
    print(f"Schema has {len(schema['properties'])} main sections")
    
    for model in models:
        print(f"\n🤖 Testing {model['name']}...")
        
        response, duration, tokens = call_sgr_v4(model["id"], task, task_type, schema)
        
        if response:
            scores = evaluate_sgr_response(response, task_type)
            print(f"✅ Success! Time: {duration:.1f}s, Tokens: {tokens}")
            print(f"   Scores: {scores}")
            
            # Show some details
            if task_type == "code_review" and "security_analysis" in response:
                vulns = response["security_analysis"].get("vulnerabilities", [])
                if vulns:
                    print(f"   Found {len(vulns)} vulnerabilities:")
                    for v in vulns[:2]:  # Show first 2
                        print(f"     - {v.get('type', 'unknown')}: {v.get('description', '')[:60]}...")
        else:
            print(f"❌ Failed to parse response")
            scores = {"total": 0}
        
        results.append({
            "model": model["name"],
            "task": task_type,
            "success": response is not None,
            "scores": scores,
            "duration": duration,
            "tokens": tokens
        })
        
        time.sleep(2)  # Rate limiting
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"\n{status} {result['model']} - {result['task']}:")
        print(f"   Score: {result['scores'].get('total', 0):.1f}/10")
        print(f"   Time: {result['duration']:.1f}s")
        print(f"   Cost: ~${(result['tokens'] / 1000) * 0.0003:.4f}")
    
    # Comparison
    if len(results) >= 2:
        print("\n📈 Model Comparison:")
        print("-" * 50)
        
        qwen_score = next((r["scores"]["total"] for r in results if "Qwen" in r["model"]), 0)
        mistral_score = next((r["scores"]["total"] for r in results if "Mistral" in r["model"]), 0)
        
        improvement = ((qwen_score - mistral_score) / max(mistral_score, 0.1)) * 100 if mistral_score > 0 else 0
        
        print(f"Qwen-2.5-72B vs Mistral-7B: {improvement:+.1f}% improvement")
        print(f"Conclusion: {'Qwen significantly better for SGR' if improvement > 20 else 'Models perform similarly'}")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()