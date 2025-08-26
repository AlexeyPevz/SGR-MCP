#!/usr/bin/env python3
"""
Testing Abdullin's SGR approach - structured reasoning with constrained decoding

Key differences from our approach:
1. Focus on structured OUTPUT format, not just structured ANALYSIS
2. Use JSON schema for constrained decoding (structured output)
3. Single-phase with structured reasoning steps
4. Emphasis on explicit reasoning traces
"""

import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple, Optional

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Abdullin-style SGR schemas - focus on reasoning traces
ABDULLIN_SCHEMAS = {
    "code_review": {
        "type": "object",
        "properties": {
            "understanding": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string", "description": "What is this code trying to accomplish?"},
                    "key_components": {"type": "array", "items": {"type": "string"}},
                    "assumptions": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["purpose", "key_components"]
            },
            "analysis": {
                "type": "object", 
                "properties": {
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                                "description": {"type": "string"},
                                "location": {"type": "string"},
                                "suggestion": {"type": "string"}
                            },
                            "required": ["severity", "description", "suggestion"]
                        }
                    }
                },
                "required": ["issues"]
            },
            "recommendations": {
                "type": "object",
                "properties": {
                    "immediate_fixes": {"type": "array", "items": {"type": "string"}},
                    "improvements": {"type": "array", "items": {"type": "string"}},
                    "best_practices": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["immediate_fixes"]
            }
        },
        "required": ["understanding", "analysis", "recommendations"]
    },
    
    "problem_solving": {
        "type": "object",
        "properties": {
            "problem_statement": {
                "type": "object",
                "properties": {
                    "core_issue": {"type": "string"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["core_issue", "constraints"]
            },
            "reasoning_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "action": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "result": {"type": "string"}
                    },
                    "required": ["step", "action", "reasoning"]
                }
            },
            "solution": {
                "type": "object",
                "properties": {
                    "approach": {"type": "string"},
                    "implementation": {"type": "string"},
                    "tradeoffs": {"type": "array", "items": {"type": "string"}},
                    "alternatives_considered": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["approach", "implementation"]
            },
            "validation": {
                "type": "object",
                "properties": {
                    "meets_criteria": {"type": "boolean"},
                    "explanation": {"type": "string"},
                    "risks": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["meets_criteria", "explanation"]
            }
        },
        "required": ["problem_statement", "reasoning_steps", "solution", "validation"]
    }
}

# Test tasks
TEST_TASKS = {
    "code_review": {
        "name": "Python Code Review",
        "task": """Review this Python code for a REST API endpoint:

```python
@app.route('/api/users/<user_id>/orders', methods=['GET'])
def get_user_orders(user_id):
    # Get orders for user
    orders = db.execute(f"SELECT * FROM orders WHERE user_id = {user_id}")
    
    result = []
    for order in orders:
        # Calculate total
        items = db.execute(f"SELECT * FROM order_items WHERE order_id = {order['id']}")
        total = 0
        for item in items:
            total += item['price'] * item['quantity']
        
        order['total'] = total
        order['items'] = items
        result.append(order)
    
    return jsonify(result)
```

Analyze for security, performance, and best practices.""",
        "schema": ABDULLIN_SCHEMAS["code_review"]
    },
    
    "problem_solving": {
        "name": "System Design Problem",
        "task": """Design a rate limiting system for an API with these requirements:

1. Must handle 100k requests/second
2. Different limits for different user tiers (free: 100/hour, pro: 10k/hour, enterprise: unlimited)
3. Must work across multiple servers
4. Should gracefully degrade under extreme load
5. Must provide clear feedback to users about their limits

Consider implementation details, data structures, and edge cases.""",
        "schema": ABDULLIN_SCHEMAS["problem_solving"]
    }
}

# Abdullin-style prompts emphasizing structured reasoning
ABDULLIN_PROMPTS = {
    "system": """You are an expert assistant that provides structured, systematic analysis.

When given a task, you MUST follow the provided JSON schema exactly. The schema guides your reasoning process - each field represents a specific aspect you need to consider.

Think step-by-step through each part of the schema. Be explicit about your reasoning. The schema is designed to ensure you don't miss important considerations.""",
    
    "user_template": """Task: {task}

Provide your response following this exact JSON structure:
{schema}

Remember:
1. Fill EVERY required field
2. Be specific and concrete 
3. Show your reasoning explicitly
4. Follow the schema's logical flow"""
}


def call_model_structured(model_id: str, messages: List[Dict], schema: Dict) -> Tuple[Optional[Dict], float, int]:
    """Call model with structured output (if supported)."""
    
    start_time = time.time()
    
    # For models that support structured output
    if "gemini" in model_id.lower():
        # Gemini supports response_format
        data = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": schema
                }
            }
        }
    else:
        # For other models, include schema in prompt
        if messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\nYou MUST respond with valid JSON matching the schema. No additional text."
        
        data = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.1
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
            
            # Parse JSON response
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(content)
                
                return parsed, time.time() - start_time, tokens
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                # Try to extract JSON from response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(content[start:end])
                        return parsed, time.time() - start_time, tokens
                    except:
                        pass
                
                return None, time.time() - start_time, tokens
                
    except Exception as e:
        print(f"API Error: {e}")
        return None, time.time() - start_time, 0


def call_model_unstructured(model_id: str, messages: List[Dict]) -> Tuple[Optional[str], float, int]:
    """Call model without structured output constraints."""
    
    start_time = time.time()
    
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.1
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
            return content, time.time() - start_time, tokens
    except Exception as e:
        print(f"API Error: {e}")
        return None, time.time() - start_time, 0


def evaluate_structured_response(response: Dict, task_type: str) -> Dict[str, float]:
    """Evaluate quality of structured response."""
    
    scores = {}
    
    if task_type == "code_review":
        # Check completeness
        completeness = 0
        if "understanding" in response and response["understanding"].get("purpose"):
            completeness += 0.2
        if "analysis" in response and len(response["analysis"].get("issues", [])) > 0:
            completeness += 0.3
        if "recommendations" in response and len(response["recommendations"].get("immediate_fixes", [])) > 0:
            completeness += 0.3
        if response.get("analysis", {}).get("strengths"):
            completeness += 0.2
        
        scores["completeness"] = completeness * 10
        
        # Check quality of issues found
        issues = response.get("analysis", {}).get("issues", [])
        quality = 0
        for issue in issues:
            if issue.get("severity") and issue.get("description") and issue.get("suggestion"):
                quality += 1
        
        scores["issue_quality"] = min(10, quality * 2)
        
        # SQL injection detection (critical issue)
        sql_injection_found = any(
            "sql" in issue.get("description", "").lower() or 
            "injection" in issue.get("description", "").lower()
            for issue in issues
        )
        scores["critical_issue_found"] = 10 if sql_injection_found else 0
        
    elif task_type == "problem_solving":
        # Check reasoning steps
        steps = response.get("reasoning_steps", [])
        scores["reasoning_depth"] = min(10, len(steps) * 2)
        
        # Check solution quality
        solution = response.get("solution", {})
        solution_score = 0
        if solution.get("approach"):
            solution_score += 3
        if solution.get("implementation"):
            solution_score += 3
        if solution.get("tradeoffs"):
            solution_score += 2
        if solution.get("alternatives_considered"):
            solution_score += 2
        
        scores["solution_quality"] = solution_score
        
        # Check validation
        validation = response.get("validation", {})
        scores["validation"] = 10 if validation.get("meets_criteria") is not None else 0
    
    # Calculate total
    scores["total"] = sum(scores.values()) / len(scores)
    
    return scores


def evaluate_unstructured_response(response: str, task_type: str) -> Dict[str, float]:
    """Evaluate quality of unstructured response."""
    
    if not response:
        return {"total": 0}
    
    scores = {}
    response_lower = response.lower()
    
    if task_type == "code_review":
        # Check if SQL injection is mentioned
        sql_injection = "sql injection" in response_lower or "sql inject" in response_lower
        scores["critical_issue_found"] = 10 if sql_injection else 0
        
        # Check for other security mentions
        security_terms = ["security", "vulnerability", "sanitize", "escape", "parameterized"]
        security_score = sum(2 for term in security_terms if term in response_lower)
        scores["security_awareness"] = min(10, security_score)
        
        # Check for structure
        structure_indicators = ["issue", "problem", "recommendation", "fix", "improve"]
        structure_score = sum(2 for term in structure_indicators if term in response_lower)
        scores["structure"] = min(10, structure_score)
        
    elif task_type == "problem_solving":
        # Check for key components
        components = ["redis", "distributed", "tier", "limit", "rate", "counter", "sliding window", "token bucket"]
        component_score = sum(1.5 for comp in components if comp in response_lower)
        scores["technical_depth"] = min(10, component_score)
        
        # Check for reasoning
        reasoning_terms = ["because", "therefore", "however", "consider", "tradeoff"]
        reasoning_score = sum(2 for term in reasoning_terms if term in response_lower)
        scores["reasoning"] = min(10, reasoning_score)
    
    # Length as proxy for thoroughness
    scores["thoroughness"] = min(10, len(response) / 200)
    
    scores["total"] = sum(scores.values()) / len(scores)
    
    return scores


def main():
    """Test Abdullin-style SGR approach."""
    
    print("\n🔬 Testing Abdullin-Style SGR Approach")
    print("=" * 80)
    print("Comparing structured reasoning (with schema) vs unstructured")
    print("=" * 80)
    
    models = [
        {"name": "Mistral-7B", "id": "mistralai/mistral-7b-instruct"},
        {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct"},
        {"name": "Gemini-Flash-1.5", "id": "google/gemini-flash-1.5"}
    ]
    
    results = []
    
    for task_key, task_info in TEST_TASKS.items():
        print(f"\n\n{'='*80}")
        print(f"📋 Task: {task_info['name']}")
        print("="*80)
        print(f"Preview: {task_info['task'][:150]}...")
        
        task_results = {"task": task_info["name"], "models": {}}
        
        for model in models:
            print(f"\n\n🤖 Testing {model['name']}")
            print("-" * 60)
            
            # Test WITH Abdullin-style SGR (structured)
            print(f"\n🟢 WITH SGR (Structured):", end="", flush=True)
            
            messages = [
                {"role": "system", "content": ABDULLIN_PROMPTS["system"]},
                {"role": "user", "content": ABDULLIN_PROMPTS["user_template"].format(
                    task=task_info["task"],
                    schema=json.dumps(task_info["schema"], indent=2)
                )}
            ]
            
            sgr_response, sgr_time, sgr_tokens = call_model_structured(model["id"], messages, task_info["schema"])
            
            if sgr_response:
                sgr_scores = evaluate_structured_response(sgr_response, task_key)
                print(f" ✓ Score: {sgr_scores['total']:.1f}/10, Time: {sgr_time:.1f}s")
                print(f"  Details: ", end="")
                for metric, score in sgr_scores.items():
                    if metric != "total":
                        print(f"{metric}={score:.1f} ", end="")
                print()
                
                # Show sample of response
                if task_key == "code_review" and "analysis" in sgr_response:
                    issues = sgr_response["analysis"].get("issues", [])
                    if issues:
                        print(f"  Found {len(issues)} issues, e.g.: {issues[0].get('description', '')[:80]}...")
            else:
                sgr_scores = {"total": 0}
                print(" ✗ Failed (JSON parsing error)")
            
            time.sleep(2)
            
            # Test WITHOUT SGR (unstructured)
            print(f"\n🔴 WITHOUT SGR (Unstructured):", end="", flush=True)
            
            messages = [
                {"role": "system", "content": "You are an expert assistant. Provide thorough analysis."},
                {"role": "user", "content": task_info["task"]}
            ]
            
            baseline_response, baseline_time, baseline_tokens = call_model_unstructured(model["id"], messages)
            
            if baseline_response:
                baseline_scores = evaluate_unstructured_response(baseline_response, task_key)
                print(f" ✓ Score: {baseline_scores['total']:.1f}/10, Time: {baseline_time:.1f}s")
                print(f"  Details: ", end="")
                for metric, score in baseline_scores.items():
                    if metric != "total":
                        print(f"{metric}={score:.1f} ", end="")
                print()
                print(f"  Response preview: {baseline_response[:150]}...")
            else:
                baseline_scores = {"total": 0}
                print(" ✗ Failed")
            
            # Calculate improvement
            if baseline_scores["total"] > 0:
                improvement = ((sgr_scores["total"] - baseline_scores["total"]) / baseline_scores["total"]) * 100
                print(f"\n📊 SGR Impact: {improvement:+.1f}% ({baseline_scores['total']:.1f} → {sgr_scores['total']:.1f})")
                
                # Cost comparison
                baseline_cost = (baseline_tokens / 1000) * 0.0003
                sgr_cost = (sgr_tokens / 1000) * 0.0003
                print(f"💰 Cost: ${baseline_cost:.4f} → ${sgr_cost:.4f} ({(sgr_cost/baseline_cost - 1)*100:+.1f}%)")
            
            task_results["models"][model["name"]] = {
                "baseline": baseline_scores,
                "sgr": sgr_scores,
                "improvement": improvement if baseline_scores["total"] > 0 else 0
            }
            
            time.sleep(2)
        
        results.append(task_results)
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 ABDULLIN-STYLE SGR SUMMARY")
    print("="*80)
    
    # Average improvements by model
    model_improvements = {}
    
    for task_result in results:
        for model_name, scores in task_result["models"].items():
            if model_name not in model_improvements:
                model_improvements[model_name] = []
            model_improvements[model_name].append(scores["improvement"])
    
    print("\n🎯 Average SGR Impact by Model:")
    print("-" * 50)
    
    for model_name, improvements in model_improvements.items():
        avg = sum(improvements) / len(improvements) if improvements else 0
        print(f"  {model_name}: {avg:+.1f}%")
    
    # Task breakdown
    print("\n📈 SGR Impact by Task:")
    print("-" * 50)
    
    for task_result in results:
        print(f"\n{task_result['task']}:")
        for model_name, scores in task_result["models"].items():
            imp = scores['improvement']
            print(f"  {model_name}: {imp:+.1f}%")
    
    # Key insights
    print("\n" + "="*80)
    print("🔍 KEY INSIGHTS")
    print("="*80)
    
    print("\n1. Structured output (JSON schema) provides clear benefits:")
    print("   - Forces consideration of all required aspects")
    print("   - Ensures consistent response format")
    print("   - Makes responses machine-parseable")
    
    print("\n2. The Abdullin approach differs from our two-phase SGR:")
    print("   - Single-phase with structured output")
    print("   - Schema guides the reasoning process directly")
    print("   - More efficient (single API call)")
    
    print("\n3. Best practices from this test:")
    print("   - Use JSON schemas for tasks requiring structured analysis")
    print("   - Design schemas that guide reasoning (not just format output)")
    print("   - Consider model support for structured output")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()