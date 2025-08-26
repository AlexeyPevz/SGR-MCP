#!/usr/bin/env python3
"""
Fair SGR Evaluation - comparing structured vs unstructured outputs fairly
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

# Test models
TEST_MODELS = [
    {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct", "size": "72B"},
    {"name": "DeepSeek-V2.5", "id": "deepseek/deepseek-chat", "size": "236B"},
    {"name": "Gemma-2-9B", "id": "google/gemma-2-9b-it", "size": "9B"},
]

# Single comprehensive test task
EVALUATION_TASK = {
    "type": "code_review",
    "task": """Analyze this e-commerce API endpoint for ALL issues:

```python
@app.route('/api/purchase', methods=['POST'])
def purchase_item():
    user_id = request.json.get('user_id')
    item_id = request.json.get('item_id') 
    quantity = int(request.json.get('quantity', 1))
    
    # Check item availability
    item = db.execute(f"SELECT * FROM items WHERE id = {item_id}").fetchone()
    if not item or item['stock'] < quantity:
        return {"error": "Item not available"}, 400
    
    # Calculate price
    total_price = item['price'] * quantity
    
    # Process payment (simplified)
    payment_result = process_payment(user_id, total_price)
    if not payment_result['success']:
        return {"error": "Payment failed"}, 400
    
    # Update inventory
    db.execute(f"UPDATE items SET stock = stock - {quantity} WHERE id = {item_id}")
    
    # Create order
    order_id = str(uuid.uuid4())
    db.execute(f"INSERT INTO orders VALUES ('{order_id}', {user_id}, {item_id}, {quantity}, {total_price})")
    
    # Send email
    os.system(f"echo 'Order {order_id} confirmed' | mail -s 'Order' user_{user_id}@example.com")
    
    return {"order_id": order_id, "total": total_price}
```

Find ALL security vulnerabilities, race conditions, and other issues.""",
    
    "evaluation_criteria": {
        "security_issues": [
            {"name": "SQL Injection - item_id", "critical": True},
            {"name": "SQL Injection - quantity", "critical": True},
            {"name": "SQL Injection - order insert", "critical": True},
            {"name": "Command Injection - email", "critical": True},
            {"name": "No input validation", "critical": False},
            {"name": "Integer overflow - quantity", "critical": False}
        ],
        "race_conditions": [
            {"name": "TOCTOU - stock check vs update", "critical": True},
            {"name": "No transaction isolation", "critical": True}
        ],
        "best_practices": [
            {"name": "No error handling", "critical": False},
            {"name": "No authentication check", "critical": True},
            {"name": "Sensitive data in response", "critical": False},
            {"name": "No rate limiting", "critical": False}
        ]
    }
}


def call_baseline(model: Dict, task: str) -> Dict:
    """Call model without SGR."""
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": "You are a security expert. Be thorough and systematic."},
        {"role": "user", "content": task}
    ]
    
    data = {
        "model": model["id"],
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
            
            return {
                "success": True,
                "content": content,
                "duration": time.time() - start_time
            }
    except Exception as e:
        return {
            "success": False,
            "content": None,
            "duration": time.time() - start_time,
            "error": str(e)
        }


def call_sgr(model: Dict, task: str, schema: Dict) -> Dict:
    """Call model with SGR."""
    start_time = time.time()
    
    system_prompt = """You are a security expert providing structured analysis.

Your response MUST be valid JSON matching the provided schema. Be systematic and thorough - the schema guides you to cover all important aspects."""
    
    user_prompt = f"""{task}

Provide your analysis as JSON matching this schema:
{json.dumps(schema, indent=2)}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = {
        "model": model["id"],
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
            
            # Parse JSON
            parsed = None
            if content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                except:
                    pass
            elif "```json" in content:
                try:
                    json_str = content.split("```json")[1].split("```")[0]
                    parsed = json.loads(json_str)
                except:
                    pass
            
            return {
                "success": parsed is not None,
                "content": content,
                "parsed": parsed,
                "duration": time.time() - start_time
            }
    except Exception as e:
        return {
            "success": False,
            "content": None,
            "parsed": None,
            "duration": time.time() - start_time,
            "error": str(e)
        }


def evaluate_coverage(content: str, criteria: Dict) -> Dict:
    """Evaluate how many issues were found."""
    if not content:
        return {"found": [], "missed": [], "coverage": 0, "critical_coverage": 0}
    
    content_lower = content.lower()
    found = []
    missed = []
    critical_found = 0
    critical_total = 0
    
    # Check all categories
    for category, issues in criteria.items():
        for issue in issues:
            if issue["critical"]:
                critical_total += 1
                
            # Check various ways the issue might be mentioned
            issue_terms = issue["name"].lower().split(" - ")
            issue_found = False
            
            for term in issue_terms:
                if term in content_lower:
                    issue_found = True
                    break
            
            # Also check common variations
            if "sql injection" in issue["name"].lower() and "sql" in content_lower and "inject" in content_lower:
                issue_found = True
            elif "command injection" in issue["name"].lower() and ("command" in content_lower or "os.system" in content_lower):
                issue_found = True
            elif "race" in issue["name"].lower() and ("race" in content_lower or "concurrent" in content_lower or "toctou" in content_lower):
                issue_found = True
            elif "authentication" in issue["name"].lower() and ("auth" in content_lower or "permission" in content_lower):
                issue_found = True
                
            if issue_found:
                found.append(issue["name"])
                if issue["critical"]:
                    critical_found += 1
            else:
                missed.append(issue["name"])
    
    total_issues = sum(len(issues) for issues in criteria.values())
    coverage = len(found) / total_issues if total_issues > 0 else 0
    critical_coverage = critical_found / critical_total if critical_total > 0 else 0
    
    return {
        "found": found,
        "missed": missed,
        "coverage": coverage,
        "critical_coverage": critical_coverage,
        "total_found": len(found),
        "total_issues": total_issues,
        "critical_found": critical_found,
        "critical_total": critical_total
    }


def evaluate_structure(content: str, parsed: Optional[Dict]) -> Dict:
    """Evaluate response structure and organization."""
    scores = {
        "has_structure": 0,
        "is_parseable": 0,
        "is_complete": 0,
        "is_organized": 0
    }
    
    if parsed:
        scores["is_parseable"] = 1
        
        # Check if major sections exist
        if "security_analysis" in parsed or "vulnerabilities" in parsed:
            scores["has_structure"] = 1
            
        # Check completeness
        if isinstance(parsed, dict) and len(parsed) >= 3:
            scores["is_complete"] = 1
            
        # Check organization (nested structure)
        if any(isinstance(v, dict) or isinstance(v, list) for v in parsed.values()):
            scores["is_organized"] = 1
    else:
        # For baseline, check text structure
        if any(marker in content for marker in ["1.", "2.", "##", "**", "- "]):
            scores["has_structure"] = 0.5
        
        if len(content) > 500:
            scores["is_complete"] = 0.5
            
        if content.count("\n") > 10:
            scores["is_organized"] = 0.5
    
    return scores


def main():
    """Run fair evaluation."""
    print("\n⚖️  Fair SGR Evaluation")
    print("=" * 80)
    print("Comparing structured (SGR) vs unstructured outputs")
    print("Evaluating: Coverage, Critical Issues, Structure, and Usability")
    print("=" * 80)
    
    schema = TASK_SCHEMAS["code_review"]
    criteria = EVALUATION_TASK["evaluation_criteria"]
    
    results = []
    
    for model in TEST_MODELS:
        print(f"\n\n🤖 Testing {model['name']} ({model['size']})")
        print("-" * 60)
        
        # Baseline test
        print("\n🔴 BASELINE (No SGR):")
        baseline = call_baseline(model, EVALUATION_TASK["task"])
        
        if baseline["success"]:
            coverage = evaluate_coverage(baseline["content"], criteria)
            structure = evaluate_structure(baseline["content"], None)
            
            print(f"✓ Response received ({len(baseline['content'])} chars)")
            print(f"  Coverage: {coverage['total_found']}/{coverage['total_issues']} issues ({coverage['coverage']*100:.0f}%)")
            print(f"  Critical: {coverage['critical_found']}/{coverage['critical_total']} ({coverage['critical_coverage']*100:.0f}%)")
            print(f"  Structure: {sum(structure.values()):.1f}/4 points")
            print(f"  Time: {baseline['duration']:.1f}s")
            
            if coverage['found']:
                print(f"  Found: {', '.join(coverage['found'][:3])}...")
        else:
            print(f"✗ Failed: {baseline.get('error', 'Unknown error')}")
            coverage = {"coverage": 0, "critical_coverage": 0}
            structure = {"has_structure": 0, "is_parseable": 0, "is_complete": 0, "is_organized": 0}
        
        time.sleep(2)
        
        # SGR test
        print("\n🟢 WITH SGR:")
        sgr = call_sgr(model, EVALUATION_TASK["task"], schema)
        
        if sgr["success"]:
            sgr_coverage = evaluate_coverage(json.dumps(sgr["parsed"]), criteria)
            sgr_structure = evaluate_structure(sgr["content"], sgr["parsed"])
            
            print(f"✓ Valid JSON received")
            print(f"  Coverage: {sgr_coverage['total_found']}/{sgr_coverage['total_issues']} issues ({sgr_coverage['coverage']*100:.0f}%)")
            print(f"  Critical: {sgr_coverage['critical_found']}/{sgr_coverage['critical_total']} ({sgr_coverage['critical_coverage']*100:.0f}%)")
            print(f"  Structure: {sum(sgr_structure.values()):.1f}/4 points")
            print(f"  Time: {sgr['duration']:.1f}s")
            
            # Show structured data
            if sgr["parsed"] and "security_analysis" in sgr["parsed"]:
                vulns = sgr["parsed"]["security_analysis"].get("vulnerabilities", [])
                print(f"  Vulnerabilities: {len(vulns)} found")
                if vulns:
                    print(f"  Types: {', '.join(v.get('type', 'unknown') for v in vulns[:3])}...")
        else:
            print(f"✗ Failed to parse JSON")
            sgr_coverage = {"coverage": 0, "critical_coverage": 0}
            sgr_structure = {"has_structure": 0, "is_parseable": 0, "is_complete": 0, "is_organized": 0}
        
        # Calculate fair comparison
        baseline_score = (
            coverage["coverage"] * 40 +           # 40% weight on coverage
            coverage["critical_coverage"] * 40 +  # 40% weight on critical issues  
            sum(structure.values()) / 4 * 20     # 20% weight on structure
        )
        
        sgr_score = (
            sgr_coverage["coverage"] * 40 +
            sgr_coverage["critical_coverage"] * 40 +
            sum(sgr_structure.values()) / 4 * 20
        )
        
        improvement = ((sgr_score - baseline_score) / max(baseline_score, 1)) * 100
        
        print(f"\n📊 COMPARISON:")
        print(f"  Baseline Score: {baseline_score:.1f}/100")
        print(f"  SGR Score: {sgr_score:.1f}/100")
        print(f"  Improvement: {improvement:+.1f}%")
        
        results.append({
            "model": model["name"],
            "baseline_score": baseline_score,
            "sgr_score": sgr_score,
            "improvement": improvement,
            "baseline_coverage": coverage["coverage"],
            "sgr_coverage": sgr_coverage["coverage"],
            "baseline_critical": coverage["critical_coverage"],
            "sgr_critical": sgr_coverage["critical_coverage"]
        })
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    print("\n🏆 Results by Model:")
    print("-" * 80)
    print(f"{'Model':<20} {'Baseline':<12} {'SGR':<12} {'Improvement':<15} {'Winner'}")
    print("-" * 80)
    
    for r in results:
        winner = "SGR" if r["improvement"] > 0 else "Baseline"
        print(f"{r['model']:<20} {r['baseline_score']:<12.1f} {r['sgr_score']:<12.1f} {r['improvement']:<15.1f}% {winner}")
    
    # Overall verdict
    avg_improvement = sum(r["improvement"] for r in results) / len(results)
    
    print(f"\n📈 Average Improvement: {avg_improvement:+.1f}%")
    
    print("\n🔍 Key Insights:")
    print("-" * 50)
    
    # Coverage comparison
    avg_baseline_coverage = sum(r["baseline_coverage"] for r in results) / len(results)
    avg_sgr_coverage = sum(r["sgr_coverage"] for r in results) / len(results)
    
    print(f"1. Issue Coverage:")
    print(f"   - Baseline: {avg_baseline_coverage*100:.0f}% of issues found")
    print(f"   - SGR: {avg_sgr_coverage*100:.0f}% of issues found")
    
    # Critical issue comparison
    avg_baseline_critical = sum(r["baseline_critical"] for r in results) / len(results)
    avg_sgr_critical = sum(r["sgr_critical"] for r in results) / len(results)
    
    print(f"\n2. Critical Issues:")
    print(f"   - Baseline: {avg_baseline_critical*100:.0f}% found")
    print(f"   - SGR: {avg_sgr_critical*100:.0f}% found")
    
    print(f"\n3. Structure Benefits:")
    print(f"   - SGR provides consistent, machine-readable output")
    print(f"   - Easier to integrate into automated workflows")
    print(f"   - Guaranteed format for downstream processing")
    
    if avg_improvement > 10:
        print(f"\n✅ VERDICT: SGR provides meaningful improvement (+{avg_improvement:.1f}%)")
        print("   Especially valuable for systematic analysis and automation")
    elif avg_improvement > 0:
        print(f"\n📊 VERDICT: SGR provides modest improvement (+{avg_improvement:.1f}%)")
        print("   Consider for tasks requiring structured output")
    else:
        print(f"\n⚠️  VERDICT: SGR shows minimal benefit ({avg_improvement:+.1f}%)")
        print("   Modern models already provide good unstructured analysis")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()