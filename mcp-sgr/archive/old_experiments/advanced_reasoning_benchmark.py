#!/usr/bin/env python3
"""
Advanced Reasoning Benchmark - Testing SGR on truly complex tasks

These tasks require:
- Multi-step reasoning
- Handling ambiguity
- Complex analysis
- Creative problem solving
"""

import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, List, Tuple, Optional

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Really challenging tasks that require deep reasoning
HARD_TASKS = [
    {
        "id": "distributed_systems_debug",
        "name": "Complex Distributed Systems Debugging",
        "task": """Our distributed system has a subtle bug:
        
- 5 microservices: Auth, Users, Orders, Inventory, Notifications
- Bug: Sometimes orders are created but inventory isn't updated
- Happens only during high load (>1000 req/s)
- Logs show all services return 200 OK
- Message queue (RabbitMQ) shows no lost messages
- Database shows orders exist but inventory count unchanged
- Bug appears random - same request sometimes works, sometimes doesn't
- No correlation with specific users or products
- Started after deploying new "performance optimizations" last week
- Rollback didn't fix it (!!)

Analyze this systematically:
1. What are ALL possible root causes? Think beyond the obvious
2. Why didn't rollback fix it?
3. Design a debugging strategy to isolate the issue
4. Propose both emergency fix and proper solution
5. How to prevent similar issues in the future?""",
        "complexity": "very_high"
    },
    {
        "id": "algorithm_design",
        "name": "Novel Algorithm Design",
        "task": """Design an algorithm for this unique problem:

You have a social network where users can "trust" each other with weights 0-1.
Trust is NOT symmetric: A trusts B with 0.8, but B might trust A with 0.3.
Trust is NOT transitive directly, but has influence.

Problem: Calculate a "trustworthiness score" for each user that:
1. Reflects how much they are trusted by others
2. Weights trust from highly trusted users more
3. Prevents gaming (users creating fake accounts to boost score)
4. Updates efficiently as new trust relationships form
5. Handles cycles in trust graph (A trusts B trusts C trusts A)

Requirements:
- Must converge to stable scores
- Must be computationally feasible for 100M users
- Must be explainable to users
- Must handle malicious actors trying to game the system

Design the algorithm with:
- Mathematical foundation
- Pseudocode implementation
- Complexity analysis
- Anti-gaming measures
- Comparison to PageRank and why it's different""",
        "complexity": "very_high"
    },
    {
        "id": "architecture_tradeoffs",
        "name": "Complex Architecture Decision",
        "task": """You're the CTO of a fintech startup. Current situation:

- Monolithic Django app serving 50k daily users
- Team: 8 engineers (2 senior, 4 mid, 2 junior)
- Growth: 30% month-over-month for 6 months
- Expecting 1M daily users in 12 months
- Current issues: deployments take 2 hours, one bug affects everything
- Regulatory requirement: full audit trail, 99.99% uptime SLA
- Budget: $200k/year for infrastructure
- Competitor just raised $50M and is aggressive

The board wants you to "modernize" to microservices. However:
- Team has no microservices experience
- Current system works, just slow deployments
- You need to keep shipping features (can't pause for 6 months)
- Wrong decision could kill the company

Provide:
1. Deep analysis of the REAL problem (not what board thinks)
2. All options with honest pros/cons (not just micro vs mono)
3. Phased migration strategy that doesn't stop feature development
4. Risk mitigation for each phase
5. How to handle the team's lack of experience
6. What would you ACTUALLY do and why?""",
        "complexity": "very_high"
    },
    {
        "id": "ml_system_design",
        "name": "ML System Design Challenge",
        "task": """Design a real-time fraud detection system with these constraints:

- 100k transactions per second
- Must decide in <100ms per transaction
- False positive rate must be <0.1% (merchants hate false declines)
- Must catch >95% of actual fraud
- Fraud patterns change daily (adversarial)
- Must work globally (different fraud patterns by country)
- Must be explainable (regulations require explaining why transaction was declined)
- Training data is heavily imbalanced (0.01% fraud)
- Must handle concept drift
- System must be online 24/7 (no downtime for updates)

Additional complexity:
- Fraudsters test patterns with small transactions first
- Some legitimate users have weird but valid patterns
- Must handle coordinated attacks (thousands of cards at once)
- Privacy laws prevent storing some features

Design:
1. Complete ML pipeline architecture
2. Feature engineering strategy
3. Model selection and why (consider latency!)
4. How to handle the imbalance
5. Online learning approach for concept drift
6. A/B testing strategy that doesn't let fraud through
7. Explainability solution that satisfies regulations
8. Fallback for when ML system is uncertain""",
        "complexity": "extreme"
    }
]

# Enhanced SGR schemas for complex reasoning
ADVANCED_SGR_SCHEMAS = {
    "analysis": {
        "type": "object",
        "properties": {
            "problem_decomposition": {
                "type": "object",
                "properties": {
                    "core_challenge": {"type": "string"},
                    "subproblems": {"type": "array", "items": {"type": "string"}},
                    "hidden_complexities": {"type": "array", "items": {"type": "string"}},
                    "assumptions_to_validate": {"type": "array", "items": {"type": "string"}}
                }
            },
            "reasoning_chain": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "confidence": {"type": "number"},
                        "alternatives_considered": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "key_insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "insight": {"type": "string"},
                        "importance": {"type": "string", "enum": ["critical", "major", "minor"]},
                        "evidence": {"type": "string"}
                    }
                }
            },
            "uncertainty_analysis": {
                "type": "object",
                "properties": {
                    "known_unknowns": {"type": "array", "items": {"type": "string"}},
                    "assumptions_made": {"type": "array", "items": {"type": "string"}},
                    "confidence_level": {"type": "number"}
                }
            }
        }
    },
    "solution": {
        "type": "object",
        "properties": {
            "approach": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string"},
                    "rationale": {"type": "string"},
                    "tradeoffs": {"type": "array", "items": {"type": "string"}}
                }
            },
            "implementation": {
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "technical_details": {"type": "string"},
                    "complexity_analysis": {"type": "string"}
                }
            },
            "risk_assessment": {
                "type": "object",
                "properties": {
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "mitigation_strategies": {"type": "array", "items": {"type": "string"}}
                }
            },
            "success_criteria": {"type": "array", "items": {"type": "string"}}
        }
    }
}

# Improved SGR prompts for complex reasoning
ENHANCED_SGR_PROMPTS = {
    "deep_analysis": """Analyze this complex problem using systematic reasoning:

{task}

Think deeply about:
1. What makes this problem genuinely difficult?
2. What are the non-obvious aspects?
3. What expertise is required?
4. What are the hidden dependencies?

Provide your analysis in this structure:
{schema}

Be thorough - this is a complex problem requiring deep thought.""",
    
    "solution_synthesis": """Based on your analysis:
{analysis}

Now synthesize a comprehensive solution to:
{task}

Requirements for your solution:
1. Address all aspects identified in analysis
2. Consider real-world constraints
3. Provide concrete, actionable steps
4. Acknowledge uncertainties honestly
5. Think like an expert who has solved similar problems

Structure your solution as:
{schema}"""
}


def call_model(model_id: str, messages: List[Dict], temperature: float = 0.1) -> Tuple[Optional[str], float, int]:
    """Call model with configurable temperature."""
    
    start_time = time.time()
    
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4000  # More tokens for complex responses
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
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            return content, time.time() - start_time, tokens
    except Exception as e:
        print(f"\nError: {e}")
        return None, time.time() - start_time, 0


def run_enhanced_sgr(model_id: str, task: str) -> Tuple[Optional[str], float, int]:
    """Run enhanced SGR with better schemas and prompts."""
    
    total_time = 0
    total_tokens = 0
    
    # Phase 1: Deep analysis with structured thinking
    messages = [
        {
            "role": "system", 
            "content": "You are a senior expert with deep experience in distributed systems, algorithms, and system design. Think step by step."
        },
        {
            "role": "user",
            "content": ENHANCED_SGR_PROMPTS["deep_analysis"].format(
                task=task,
                schema=json.dumps(ADVANCED_SGR_SCHEMAS["analysis"], indent=2)
            )
        }
    ]
    
    analysis_response, time1, tokens1 = call_model(model_id, messages, temperature=0.2)
    total_time += time1
    total_tokens += tokens1
    
    if not analysis_response:
        return None, total_time, total_tokens
    
    # Try to extract structured analysis
    analysis = analysis_response
    try:
        if "```json" in analysis_response:
            json_str = analysis_response.split("```json")[1].split("```")[0]
            analysis = json.loads(json_str)
        elif "{" in analysis_response:
            start = analysis_response.find("{")
            end = analysis_response.rfind("}") + 1
            if start >= 0 and end > start:
                analysis = json.loads(analysis_response[start:end])
    except:
        # If JSON parsing fails, use raw response
        pass
    
    # Phase 2: Solution synthesis based on deep analysis
    messages = [
        {
            "role": "system",
            "content": "You are synthesizing a solution based on deep analysis. Be concrete, practical, and thorough."
        },
        {
            "role": "user",
            "content": ENHANCED_SGR_PROMPTS["solution_synthesis"].format(
                analysis=json.dumps(analysis, indent=2) if isinstance(analysis, dict) else str(analysis),
                task=task,
                schema=json.dumps(ADVANCED_SGR_SCHEMAS["solution"], indent=2)
            )
        }
    ]
    
    solution_response, time2, tokens2 = call_model(model_id, messages, temperature=0.1)
    total_time += time2
    total_tokens += tokens2
    
    return solution_response, total_time, total_tokens


def run_baseline(model_id: str, task: str) -> Tuple[Optional[str], float, int]:
    """Run without SGR - direct prompting."""
    
    messages = [
        {
            "role": "system",
            "content": "You are a senior expert. Provide thorough, well-reasoned analysis and solutions."
        },
        {
            "role": "user",
            "content": task
        }
    ]
    
    return call_model(model_id, messages, temperature=0.1)


def evaluate_complex_response(response: str, task: Dict) -> Dict[str, float]:
    """Evaluate response on complex reasoning tasks."""
    
    if not response:
        return {"total": 0}
    
    scores = {}
    response_lower = response.lower()
    
    # Depth of analysis (0-10)
    depth_indicators = [
        "however", "alternatively", "on the other hand", "consider",
        "tradeoff", "complexity", "nuance", "subtle", "non-obvious",
        "edge case", "assumption", "dependency", "constraint"
    ]
    depth_score = sum(2 for indicator in depth_indicators if indicator in response_lower)
    scores["depth"] = min(10, depth_score)
    
    # Problem decomposition (0-10)
    decomposition_indicators = [
        "first", "second", "third", "component", "aspect", "dimension",
        "break down", "decompose", "subproblem", "layer", "level"
    ]
    decomp_score = sum(1.5 for indicator in decomposition_indicators if indicator in response_lower)
    scores["decomposition"] = min(10, decomp_score)
    
    # Solution completeness (0-10)
    solution_indicators = [
        "implement", "approach", "strategy", "solution", "fix", "resolve",
        "mitigate", "prevent", "handle", "address", "phase", "step"
    ]
    solution_score = sum(1.2 for indicator in solution_indicators if indicator in response_lower)
    scores["solution_quality"] = min(10, solution_score)
    
    # Handling uncertainty (0-10)
    uncertainty_indicators = [
        "might", "could", "possibly", "likely", "assume", "depend",
        "uncertain", "risk", "unknown", "caveat", "limitation"
    ]
    uncertainty_score = sum(2 for indicator in uncertainty_indicators if indicator in response_lower)
    scores["uncertainty_handling"] = min(10, uncertainty_score)
    
    # Technical accuracy for specific tasks
    if task["id"] == "distributed_systems_debug":
        technical_terms = ["race condition", "eventual consistency", "idempotent", "saga", "two-phase commit",
                          "distributed transaction", "cache", "replication lag", "message ordering"]
        technical_score = sum(2 for term in technical_terms if term in response_lower)
        scores["technical_accuracy"] = min(10, technical_score)
    elif task["id"] == "algorithm_design":
        technical_terms = ["eigenvector", "convergence", "iteration", "graph", "centrality", "damping",
                          "sparse matrix", "pagerank", "sybil attack", "reputation"]
        technical_score = sum(2 for term in technical_terms if term in response_lower)
        scores["technical_accuracy"] = min(10, technical_score)
    else:
        scores["technical_accuracy"] = 5  # Default middle score
    
    # Length bonus for thoroughness
    length_score = min(10, len(response) / 300)  # Expect ~3000 chars for complex problems
    scores["thoroughness"] = length_score
    
    # Calculate weighted total
    weights = {
        "depth": 0.25,
        "decomposition": 0.20,
        "solution_quality": 0.20,
        "uncertainty_handling": 0.15,
        "technical_accuracy": 0.15,
        "thoroughness": 0.05
    }
    
    scores["total"] = sum(scores[metric] * weights[metric] for metric in weights)
    
    return scores


def main():
    """Run advanced reasoning benchmark."""
    
    print("\n🧠 Advanced Reasoning Benchmark")
    print("=" * 80)
    print("Testing SGR on genuinely complex problems requiring deep reasoning")
    print("=" * 80)
    
    # Test models
    models = [
        {"name": "Mistral-7B", "id": "mistralai/mistral-7b-instruct"},
        {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct"},
        {"name": "Claude-3-Haiku", "id": "anthropic/claude-3-haiku"}
    ]
    
    results = []
    
    # Test on 2 hardest tasks to save time/cost
    for task in HARD_TASKS[:2]:
        print(f"\n\n{'='*80}")
        print(f"🎯 Task: {task['name']}")
        print(f"Complexity: {task['complexity'].upper()}")
        print("="*80)
        print(f"\nTask preview: {task['task'][:200]}...")
        
        task_results = {"task": task["name"], "models": {}}
        
        for model in models[:2]:  # Test Mistral and Qwen
            print(f"\n\n🤖 Testing {model['name']}")
            print("-" * 60)
            
            # Test WITHOUT SGR
            print(f"\n🔴 WITHOUT SGR (Baseline):", end="", flush=True)
            baseline_response, baseline_time, baseline_tokens = run_baseline(model["id"], task["task"])
            
            if baseline_response:
                baseline_scores = evaluate_complex_response(baseline_response, task)
                print(f" ✓ Score: {baseline_scores['total']:.1f}/10, Time: {baseline_time:.1f}s")
                print("  Breakdown:", end="")
                for metric, score in baseline_scores.items():
                    if metric != "total":
                        print(f" {metric}={score:.1f}", end="")
                print(f"\n  Response length: {len(baseline_response)} chars")
                print(f"  First 200 chars: {baseline_response[:200]}...")
            else:
                baseline_scores = {"total": 0}
                print(" ✗ Failed")
            
            time.sleep(3)
            
            # Test WITH Enhanced SGR
            print(f"\n🟢 WITH Enhanced SGR:", end="", flush=True)
            sgr_response, sgr_time, sgr_tokens = run_enhanced_sgr(model["id"], task["task"])
            
            if sgr_response:
                sgr_scores = evaluate_complex_response(sgr_response, task)
                print(f" ✓ Score: {sgr_scores['total']:.1f}/10, Time: {sgr_time:.1f}s")
                print("  Breakdown:", end="")
                for metric, score in sgr_scores.items():
                    if metric != "total":
                        print(f" {metric}={score:.1f}", end="")
                print(f"\n  Response length: {len(sgr_response)} chars")
                print(f"  First 200 chars: {sgr_response[:200]}...")
            else:
                sgr_scores = {"total": 0}
                print(" ✗ Failed")
            
            # Calculate improvement
            if baseline_scores["total"] > 0:
                improvement = ((sgr_scores["total"] - baseline_scores["total"]) / baseline_scores["total"]) * 100
                print(f"\n📊 Improvement: {improvement:+.1f}% ({baseline_scores['total']:.1f} → {sgr_scores['total']:.1f})")
                
                # Cost analysis
                baseline_cost = (baseline_tokens / 1000) * 0.0003  # Rough estimate
                sgr_cost = (sgr_tokens / 1000) * 0.0003
                print(f"💰 Cost: ${baseline_cost:.4f} → ${sgr_cost:.4f} ({(sgr_cost/baseline_cost - 1)*100:+.1f}%)")
            
            task_results["models"][model["name"]] = {
                "baseline": baseline_scores,
                "sgr": sgr_scores,
                "improvement": improvement if baseline_scores["total"] > 0 else 0
            }
            
            time.sleep(3)
        
        results.append(task_results)
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 ADVANCED REASONING SUMMARY")
    print("="*80)
    
    # Average improvements by model
    model_improvements = {}
    
    for task_result in results:
        for model_name, scores in task_result["models"].items():
            if model_name not in model_improvements:
                model_improvements[model_name] = []
            model_improvements[model_name].append(scores["improvement"])
    
    print("\n🎯 Average Improvement with Enhanced SGR:")
    print("-" * 50)
    
    overall_improvement = []
    for model_name, improvements in model_improvements.items():
        avg = sum(improvements) / len(improvements)
        overall_improvement.extend(improvements)
        print(f"  {model_name}: {avg:+.1f}%")
    
    if overall_improvement:
        total_avg = sum(overall_improvement) / len(overall_improvement)
        print(f"\n  OVERALL: {total_avg:+.1f}%")
    
    # Detailed breakdown
    print("\n📈 Detailed Results by Task:")
    print("-" * 50)
    
    for task_result in results:
        print(f"\n{task_result['task']}:")
        for model_name, scores in task_result["models"].items():
            b_total = scores['baseline']['total']
            s_total = scores['sgr']['total']
            imp = scores['improvement']
            print(f"  {model_name}: {b_total:.1f} → {s_total:.1f} ({imp:+.1f}%)")
    
    # Final verdict
    print("\n" + "="*80)
    print("🏁 CONCLUSION")
    print("="*80)
    
    if overall_improvement and total_avg > 15:
        print(f"\n✅ Enhanced SGR shows SIGNIFICANT improvement on complex tasks: {total_avg:+.1f}%")
        print("   Complex problems benefit from structured reasoning approach")
        print("   The improvement justifies the additional API costs")
    elif overall_improvement and total_avg > 5:
        print(f"\n📊 Enhanced SGR shows moderate improvement: {total_avg:+.1f}%")
        print("   Benefits are task-dependent")
        print("   Consider using SGR selectively for complex reasoning tasks")
    else:
        print(f"\n⚠️  Enhanced SGR shows minimal improvement: {total_avg:+.1f}%")
        print("   Current implementation may need further refinement")
        print("   Models may already be near their reasoning capacity")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()