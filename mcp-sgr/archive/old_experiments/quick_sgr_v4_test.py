#!/usr/bin/env python3
"""
Quick SGR v4 test - testing key models across different sizes
"""

import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('/workspace/mcp-sgr/src/tools')
from apply_sgr_v4 import TASK_SCHEMAS

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ключевые модели разных размеров
QUICK_TEST_MODELS = [
    # Large (70B+)
    {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct", "size": "72B", "cost": 0.0003},
    {"name": "Llama-3.1-70B", "id": "meta-llama/llama-3.1-70b-instruct", "size": "70B", "cost": 0.0003},
    
    # Medium (20-50B)
    {"name": "Qwen-2.5-32B", "id": "qwen/qwen-2.5-32b-instruct", "size": "32B", "cost": 0.0002},
    {"name": "Claude-3-Haiku", "id": "anthropic/claude-3-haiku", "size": "~20B", "cost": 0.00025},
    
    # Small (3-10B)
    {"name": "Mistral-7B", "id": "mistralai/mistral-7b-instruct", "size": "7B", "cost": 0.00007},
    {"name": "Gemma-2-9B", "id": "google/gemma-2-9b-it", "size": "9B", "cost": 0.0001},
    
    # GPT family
    {"name": "GPT-4o-mini", "id": "openai/gpt-4o-mini", "size": "unknown", "cost": 0.00015},
]

# Одна задача для быстрого теста
QUICK_TEST_TASK = {
    "type": "code_review",
    "task": """Review this Python code for security issues:

```python
def get_user_data(request):
    user_id = request.args.get('user_id')
    
    # Get user from database
    query = f"SELECT * FROM users WHERE id = {user_id}"
    user = db.execute(query).fetchone()
    
    if user:
        # Log access
        log_msg = f"User {user_id} accessed their data"
        os.system(f"echo '{log_msg}' >> /var/log/access.log")
        
        return {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "password": user["password"]  # For debugging
        }
    
    return {"error": "User not found"}
```

Identify ALL security vulnerabilities and suggest fixes."""
}


def test_model_sgr(model: Dict, task: str, task_type: str, schema: Dict) -> Dict:
    """Test a single model with SGR."""
    
    start_time = time.time()
    
    # Prepare prompts
    system_prompt = """You are a security expert analyzing code.

Your response MUST be valid JSON matching the provided schema. The schema guides your analysis - each field represents a critical security aspect."""
    
    user_prompt = f"""Task: {task}

Provide your security analysis as JSON matching this schema:
{json.dumps(schema, indent=2)}

Focus on finding ALL security issues. Be thorough and specific."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = {
        "model": model["id"],
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
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            
            # Parse response
            parsed = None
            error = None
            
            try:
                # Direct JSON
                if content.strip().startswith("{"):
                    parsed = json.loads(content)
                # Markdown wrapped
                elif "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                    parsed = json.loads(json_str)
                # Find JSON in text
                else:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(content[start:end])
                    else:
                        # Last resort - check if it's the schema itself
                        if '"type"' in content and '"properties"' in content:
                            error = "Model returned schema instead of data"
                        else:
                            error = "No JSON found in response"
                            
            except json.JSONDecodeError as e:
                error = f"JSON parse error: {str(e)[:100]}"
                
            # Evaluate response
            if parsed and not error:
                # Check for vulnerabilities found
                vulns = parsed.get("security_analysis", {}).get("vulnerabilities", [])
                
                # Expected vulnerabilities
                expected = {
                    "sql_injection": False,
                    "command_injection": False,
                    "password_exposure": False
                }
                
                for vuln in vulns:
                    vuln_str = str(vuln).lower()
                    if "sql" in vuln_str and "injection" in vuln_str:
                        expected["sql_injection"] = True
                    if "command" in vuln_str or "os.system" in vuln_str:
                        expected["command_injection"] = True
                    if "password" in vuln_str:
                        expected["password_exposure"] = True
                
                # Calculate score
                found_count = sum(expected.values())
                total_vulns = len(vulns)
                
                score = (found_count / 3) * 10  # 3 expected vulnerabilities
                
                return {
                    "success": True,
                    "duration": time.time() - start_time,
                    "tokens": tokens,
                    "cost": tokens * model["cost"] / 1000,
                    "score": score,
                    "vulnerabilities_found": total_vulns,
                    "expected_found": expected,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "duration": time.time() - start_time,
                    "tokens": tokens,
                    "cost": tokens * model["cost"] / 1000,
                    "score": 0,
                    "vulnerabilities_found": 0,
                    "expected_found": {},
                    "error": error or "Unknown error"
                }
                
    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "tokens": 0,
            "cost": 0,
            "score": 0,
            "vulnerabilities_found": 0,
            "expected_found": {},
            "error": f"API error: {str(e)[:100]}"
        }


def main():
    """Run quick SGR v4 test."""
    
    print("\n🚀 Quick SGR v4 Test")
    print("=" * 80)
    print(f"Testing {len(QUICK_TEST_MODELS)} models on code security review task")
    print("\nExpected vulnerabilities:")
    print("  1. SQL Injection (user_id in query)")
    print("  2. Command Injection (os.system with user input)")
    print("  3. Password Exposure (returning password field)")
    print("=" * 80)
    
    schema = TASK_SCHEMAS["code_review"]
    results = []
    
    # Test each model
    for i, model in enumerate(QUICK_TEST_MODELS):
        print(f"\n{i+1}/{len(QUICK_TEST_MODELS)} Testing {model['name']} ({model['size']})...", end="", flush=True)
        
        result = test_model_sgr(model, QUICK_TEST_TASK["task"], QUICK_TEST_TASK["type"], schema)
        result["model"] = model["name"]
        result["size"] = model["size"]
        results.append(result)
        
        if result["success"]:
            print(f" ✅ Score: {result['score']:.1f}/10")
            print(f"     Found {result['vulnerabilities_found']} vulnerabilities")
            found = result["expected_found"]
            checks = []
            for vuln, found_it in found.items():
                checks.append(f"{'✓' if found_it else '✗'} {vuln}")
            print(f"     Checks: {', '.join(checks)}")
            print(f"     Cost: ${result['cost']:.4f}, Time: {result['duration']:.1f}s")
        else:
            print(f" ❌ Failed: {result['error']}")
        
        time.sleep(1)  # Rate limiting
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    print("\n🏆 Model Rankings:")
    print("-" * 70)
    print(f"{'Rank':<6}{'Model':<20}{'Size':<10}{'Score':<10}{'Vulns':<10}{'Cost'}")
    print("-" * 70)
    
    for i, r in enumerate(sorted_results):
        status = "✅" if r["success"] else "❌"
        print(f"{i+1:<6}{r['model']:<20}{r['size']:<10}{r['score']:<10.1f}{r['vulnerabilities_found']:<10}${r['cost']:.4f}")
    
    # Success rate by size
    print("\n📈 Success Rate by Model Size:")
    print("-" * 50)
    
    size_groups = {}
    for r in results:
        size = r["size"]
        if "B" in size:
            # Handle ~20B format
            size_num = size.replace("B", "").replace("~", "")
            try:
                size_int = int(size_num)
                if size_int <= 10:
                    group = "Small (≤10B)"
                elif size_int <= 50:
                    group = "Medium (10-50B)"
                else:
                    group = "Large (>50B)"
            except:
                group = "Unknown"
        else:
            group = "Unknown"
        
        if group not in size_groups:
            size_groups[group] = {"success": 0, "total": 0}
        
        size_groups[group]["total"] += 1
        if r["success"]:
            size_groups[group]["success"] += 1
    
    for group, stats in sorted(size_groups.items()):
        rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  {group}: {rate:.0f}% ({stats['success']}/{stats['total']})")
    
    # Cost-effectiveness
    print("\n💰 Cost-Effectiveness (Score/Dollar):")
    print("-" * 50)
    
    cost_effective = []
    for r in results:
        if r["success"] and r["cost"] > 0:
            effectiveness = r["score"] / r["cost"]
            cost_effective.append({
                "model": r["model"],
                "effectiveness": effectiveness,
                "score": r["score"],
                "cost": r["cost"]
            })
    
    cost_effective.sort(key=lambda x: x["effectiveness"], reverse=True)
    
    for i, ce in enumerate(cost_effective[:5]):
        print(f"  {i+1}. {ce['model']}: {ce['effectiveness']:.0f} points/$ (score: {ce['score']:.1f}, cost: ${ce['cost']:.4f})")
    
    # Key findings
    print("\n" + "="*80)
    print("🔍 KEY FINDINGS")
    print("="*80)
    
    successful = [r for r in results if r["success"]]
    if successful:
        # Best overall
        best = sorted_results[0]
        print(f"\n1. Best Overall: {best['model']} (Score: {best['score']:.1f}/10)")
        
        # Most thorough
        most_thorough = max(successful, key=lambda x: x["vulnerabilities_found"])
        print(f"2. Most Thorough: {most_thorough['model']} ({most_thorough['vulnerabilities_found']} vulnerabilities found)")
        
        # Best value
        if cost_effective:
            print(f"3. Best Value: {cost_effective[0]['model']} ({cost_effective[0]['effectiveness']:.0f} points/$)")
        
        # Reliability
        success_rate = (len(successful) / len(results)) * 100
        print(f"4. Overall Success Rate: {success_rate:.0f}% ({len(successful)}/{len(results)})")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()