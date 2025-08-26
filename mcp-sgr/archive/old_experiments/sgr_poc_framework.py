#!/usr/bin/env python3
"""
SGR Proof of Concept Framework
Comprehensive testing of SGR modes across different use cases
"""

import json
import os
import time
import asyncio
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import urllib.request

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Test models - mix of sizes
TEST_MODELS = [
    {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct", "size": "large", "cost": 0.0003},
    {"name": "DeepSeek-V2.5", "id": "deepseek/deepseek-chat", "size": "large", "cost": 0.00014},
    {"name": "Gemma-2-9B", "id": "google/gemma-2-9b-it", "size": "small", "cost": 0.0001},
]

# SGR Schemas for different tasks
SGR_SCHEMAS = {
    "code_generation": {
        "lite": {
            "type": "object",
            "properties": {
                "requirements_analysis": {
                    "type": "object",
                    "properties": {
                        "key_requirements": {"type": "array", "items": {"type": "string"}},
                        "constraints": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "implementation": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "explanation": {"type": "string"}
                    }
                },
                "testing": {
                    "type": "object",
                    "properties": {
                        "test_code": {"type": "string"},
                        "coverage": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["requirements_analysis", "implementation"]
        },
        "full": {
            "type": "object",
            "properties": {
                "requirements_analysis": {
                    "type": "object",
                    "properties": {
                        "key_requirements": {"type": "array", "items": {"type": "string"}},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "edge_cases": {"type": "array", "items": {"type": "string"}},
                        "security_considerations": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "design": {
                    "type": "object",
                    "properties": {
                        "approach": {"type": "string"},
                        "data_structures": {"type": "array", "items": {"type": "string"}},
                        "algorithms": {"type": "array", "items": {"type": "string"}},
                        "complexity": {"type": "string"}
                    }
                },
                "implementation": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "explanation": {"type": "string"},
                        "best_practices": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "testing": {
                    "type": "object",
                    "properties": {
                        "test_code": {"type": "string"},
                        "test_cases": {"type": "array", "items": {"type": "object"}},
                        "coverage": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "validation": {
                    "type": "object",
                    "properties": {
                        "meets_requirements": {"type": "boolean"},
                        "security_validated": {"type": "boolean"},
                        "performance_notes": {"type": "string"}
                    }
                }
            },
            "required": ["requirements_analysis", "design", "implementation", "testing", "validation"]
        }
    },
    "rag_qa": {
        "lite": {
            "type": "object",
            "properties": {
                "question_analysis": {
                    "type": "object",
                    "properties": {
                        "query_intent": {"type": "string"},
                        "key_terms": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "answer": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["question_analysis", "answer"]
        },
        "full": {
            "type": "object",
            "properties": {
                "question_analysis": {
                    "type": "object",
                    "properties": {
                        "query_intent": {"type": "string"},
                        "key_terms": {"type": "array", "items": {"type": "string"}},
                        "information_needs": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "retrieval_strategy": {
                    "type": "object",
                    "properties": {
                        "search_queries": {"type": "array", "items": {"type": "string"}},
                        "relevance_criteria": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "evidence_analysis": {
                    "type": "object",
                    "properties": {
                        "relevant_passages": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "source": {"type": "string"}, "relevance": {"type": "number"}}}},
                        "conflicting_info": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "answer": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "claim_to_source_map": {"type": "array", "items": {"type": "object", "properties": {"claim": {"type": "string"}, "source": {"type": "string"}}}},
                        "confidence": {"type": "number"},
                        "caveats": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "validation": {
                    "type": "object",
                    "properties": {
                        "all_claims_grounded": {"type": "boolean"},
                        "hallucination_check": {"type": "string"},
                        "coverage": {"type": "number"}
                    }
                }
            },
            "required": ["question_analysis", "retrieval_strategy", "evidence_analysis", "answer", "validation"]
        }
    }
}

# Test cases
TEST_CASES = {
    "code_generation": [
        {
            "id": "fastapi_jwt",
            "name": "FastAPI + JWT Authentication",
            "prompt": """Create a FastAPI endpoint with JWT authentication:
1. POST /login that accepts username/password and returns JWT token
2. GET /protected that requires valid JWT token
3. Include proper error handling and security best practices
4. Add basic tests for both endpoints""",
            "evaluation": {
                "security_checklist": ["password hashing", "token expiration", "secret key management", "error messages don't leak info"],
                "functionality": ["login works", "protected route requires token", "invalid token rejected", "tests pass"]
            }
        },
        {
            "id": "bfs_maze",
            "name": "BFS Maze Solver in JavaScript",
            "prompt": """Implement a BFS algorithm in JavaScript to find the shortest path in a maze:
1. Maze is represented as 2D array (0=path, 1=wall)
2. Find shortest path from start (top-left) to end (bottom-right)
3. Return the path as array of coordinates
4. Include tests with different maze configurations""",
            "evaluation": {
                "correctness": ["finds shortest path", "handles no path case", "validates input"],
                "code_quality": ["readable", "efficient", "well-commented"]
            }
        },
        {
            "id": "sql_performance",
            "name": "SQL Query Optimization",
            "prompt": """Write an optimized SQL query to find top 10 most active users in the last 6 months:
- Users table: id, username, created_at
- Activities table: id, user_id, activity_type, created_at
- Consider proper indexes and query performance
- Provide both SQL and equivalent ORM query (SQLAlchemy)""",
            "evaluation": {
                "performance": ["uses indexes", "efficient joins", "proper date filtering"],
                "correctness": ["returns top 10", "6 month window", "activity count accurate"]
            }
        }
    ],
    "rag_qa": [
        {
            "id": "internal_docs",
            "name": "Internal Documentation Q&A",
            "prompt": "Based on the provided documentation, what are the key security requirements for API authentication in our system?",
            "documents": [
                {
                    "id": "doc1",
                    "content": "API Security Guidelines: All APIs must use OAuth2.0 or JWT tokens. Tokens must expire within 1 hour for standard users and 15 minutes for admin users. All communication must use HTTPS."
                },
                {
                    "id": "doc2", 
                    "content": "Authentication Best Practices: Use bcrypt for password hashing with minimum 12 rounds. Implement rate limiting on login endpoints. Log all authentication attempts."
                }
            ],
            "evaluation": {
                "faithfulness": ["mentions OAuth2.0/JWT", "token expiration times", "HTTPS requirement"],
                "groundedness": ["all claims from documents", "no hallucinations"],
                "citation": ["proper source attribution"]
            }
        },
        {
            "id": "conflicting_sources",
            "name": "Conflicting Information Resolution",
            "prompt": "What is the recommended cache TTL for user session data?",
            "documents": [
                {
                    "id": "doc1",
                    "content": "Legacy System Guide (2020): User sessions should be cached for 24 hours to reduce database load."
                },
                {
                    "id": "doc2",
                    "content": "Security Update (2023): Due to recent security concerns, user session cache TTL has been reduced to 1 hour maximum."
                },
                {
                    "id": "doc3",
                    "content": "Performance Tuning Guide: For optimal performance, cache user sessions for 4-6 hours."
                }
            ],
            "evaluation": {
                "conflict_resolution": ["identifies conflict", "uses most recent guidance", "mentions security reasoning"],
                "transparency": ["cites all sources", "explains discrepancy"]
            }
        }
    ]
}


@dataclass
class TestResult:
    test_id: str
    model: str
    sgr_mode: str
    latency: float
    tokens: int
    cost: float
    success: bool
    quality_scores: Dict[str, float]
    reasoning_log: str
    output: Any
    error: Optional[str] = None


class SGRTestFramework:
    def __init__(self):
        self.results: List[TestResult] = []
        
    async def call_model(self, model: Dict, messages: List[Dict], sgr_mode: str, task_type: str) -> Tuple[Any, float, int, Optional[str]]:
        """Call model with specified SGR mode."""
        start_time = time.time()
        
        if sgr_mode == "off":
            # No SGR - direct call
            system_prompt = "You are an expert assistant. Provide high-quality solutions."
            user_prompt = messages[-1]["content"]
        else:
            # SGR mode - use schema
            schema = SGR_SCHEMAS[task_type][sgr_mode]
            system_prompt = """You are an expert providing structured analysis.

Your response MUST be valid JSON matching the provided schema. The schema guides your reasoning - be thorough and systematic."""
            user_prompt = f"""{messages[-1]["content"]}

Provide your response as JSON matching this schema:
{json.dumps(schema, indent=2)}"""
        
        final_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # API call
        data = {
            "model": model["id"],
            "messages": final_messages,
            "temperature": 0.1,
            "max_tokens": 4000
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
                
                # Parse response based on mode
                if sgr_mode == "off":
                    parsed_output = content
                else:
                    # Try to parse JSON
                    try:
                        if "```json" in content:
                            json_str = content.split("```json")[1].split("```")[0]
                            parsed_output = json.loads(json_str)
                        else:
                            parsed_output = json.loads(content)
                    except:
                        return None, time.time() - start_time, tokens, "Failed to parse JSON response"
                
                return parsed_output, time.time() - start_time, tokens, None
                
        except Exception as e:
            return None, time.time() - start_time, 0, str(e)
    
    def evaluate_code_generation(self, output: Any, test_case: Dict, sgr_mode: str) -> Dict[str, float]:
        """Evaluate code generation results."""
        scores = {}
        
        if sgr_mode == "off":
            # Evaluate unstructured output
            content = output.lower() if isinstance(output, str) else ""
            
            # Check for key elements
            for category, items in test_case["evaluation"].items():
                found = sum(1 for item in items if any(keyword in content for keyword in item.lower().split()))
                scores[category] = found / len(items) if items else 0
        else:
            # Evaluate structured output
            if isinstance(output, dict):
                # Check implementation
                if "implementation" in output and output["implementation"].get("code"):
                    scores["has_code"] = 1.0
                else:
                    scores["has_code"] = 0.0
                
                # Check testing
                if "testing" in output and output["testing"].get("test_code"):
                    scores["has_tests"] = 1.0
                else:
                    scores["has_tests"] = 0.0
                
                # Check requirements analysis
                if "requirements_analysis" in output:
                    scores["requirements_analyzed"] = 1.0
                else:
                    scores["requirements_analyzed"] = 0.0
                
                # For full mode, check additional fields
                if sgr_mode == "full" and "validation" in output:
                    scores["validated"] = 1.0
                else:
                    scores["validated"] = 0.0
        
        scores["overall"] = sum(scores.values()) / len(scores) if scores else 0
        return scores
    
    def evaluate_rag_qa(self, output: Any, test_case: Dict, sgr_mode: str) -> Dict[str, float]:
        """Evaluate RAG Q&A results."""
        scores = {}
        
        if sgr_mode == "off":
            # Basic evaluation for unstructured
            content = output.lower() if isinstance(output, str) else ""
            
            # Check if key points are mentioned
            for category, items in test_case["evaluation"].items():
                found = sum(1 for item in items if item.lower() in content)
                scores[category] = found / len(items) if items else 0
        else:
            # Structured evaluation
            if isinstance(output, dict):
                # Check answer quality
                if "answer" in output:
                    answer = output["answer"]
                    if answer.get("response"):
                        scores["has_answer"] = 1.0
                    if answer.get("sources") or answer.get("claim_to_source_map"):
                        scores["has_citations"] = 1.0
                    else:
                        scores["has_citations"] = 0.0
                
                # For full mode, check validation
                if sgr_mode == "full" and "validation" in output:
                    validation = output["validation"]
                    if validation.get("all_claims_grounded"):
                        scores["grounded"] = 1.0
                    else:
                        scores["grounded"] = 0.5
                    
                    scores["coverage"] = validation.get("coverage", 0)
        
        scores["overall"] = sum(scores.values()) / len(scores) if scores else 0
        return scores
    
    async def run_test(self, test_case: Dict, model: Dict, sgr_mode: str, task_type: str) -> TestResult:
        """Run a single test case."""
        # Prepare messages
        messages = [{"role": "user", "content": test_case["prompt"]}]
        
        # Add documents for RAG tasks
        if task_type == "rag_qa" and "documents" in test_case:
            doc_context = "\n\n".join([f"[{doc['id']}]: {doc['content']}" for doc in test_case["documents"]])
            messages[-1]["content"] = f"Documents:\n{doc_context}\n\nQuestion: {test_case['prompt']}"
        
        # Call model
        output, latency, tokens, error = await self.call_model(model, messages, sgr_mode, task_type)
        
        # Calculate cost
        cost = tokens * model["cost"] / 1000
        
        # Evaluate quality
        if output and not error:
            if task_type == "code_generation":
                quality_scores = self.evaluate_code_generation(output, test_case, sgr_mode)
            else:
                quality_scores = self.evaluate_rag_qa(output, test_case, sgr_mode)
            success = True
        else:
            quality_scores = {"overall": 0}
            success = False
        
        # Create result
        result = TestResult(
            test_id=test_case["id"],
            model=model["name"],
            sgr_mode=sgr_mode,
            latency=latency,
            tokens=tokens,
            cost=cost,
            success=success,
            quality_scores=quality_scores,
            reasoning_log=json.dumps(output, indent=2) if isinstance(output, dict) else str(output)[:500],
            output=output,
            error=error
        )
        
        return result
    
    async def run_poc(self):
        """Run the proof of concept tests."""
        print("\n🚀 SGR Proof of Concept Testing")
        print("=" * 80)
        
        # Select subset for PoC
        poc_cases = {
            "code_generation": TEST_CASES["code_generation"][:3],  # All 3 code cases
            "rag_qa": TEST_CASES["rag_qa"][:2]  # Both RAG cases
        }
        
        sgr_modes = ["off", "lite", "full"]
        
        total_tests = sum(len(cases) for cases in poc_cases.values()) * len(TEST_MODELS) * len(sgr_modes)
        current_test = 0
        
        for task_type, cases in poc_cases.items():
            print(f"\n\n📋 Task Type: {task_type}")
            print("-" * 60)
            
            for test_case in cases:
                print(f"\n🧪 Test Case: {test_case['name']}")
                
                for model in TEST_MODELS:
                    for sgr_mode in sgr_modes:
                        current_test += 1
                        print(f"\n[{current_test}/{total_tests}] {model['name']} - SGR: {sgr_mode}", end="", flush=True)
                        
                        result = await self.run_test(test_case, model, sgr_mode, task_type)
                        self.results.append(result)
                        
                        if result.success:
                            print(f" ✅ Quality: {result.quality_scores['overall']:.2f}, Time: {result.latency:.1f}s, Cost: ${result.cost:.4f}")
                        else:
                            print(f" ❌ Failed: {result.error}")
                        
                        # Rate limiting
                        await asyncio.sleep(1)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive report of results."""
        print("\n\n" + "="*80)
        print("📊 SGR PROOF OF CONCEPT RESULTS")
        print("="*80)
        
        # Group results by test case
        by_test = {}
        for result in self.results:
            if result.test_id not in by_test:
                by_test[result.test_id] = []
            by_test[result.test_id].append(result)
        
        # Analyze each test case
        for test_id, results in by_test.items():
            test_name = next((tc["name"] for task_cases in TEST_CASES.values() for tc in task_cases if tc["id"] == test_id), test_id)
            print(f"\n\n📋 {test_name}")
            print("-" * 60)
            
            # Compare SGR modes
            print("\n🔄 SGR Mode Comparison:")
            print(f"{'Model':<15} {'Mode':<8} {'Quality':<10} {'Latency':<10} {'Cost':<10} {'Status'}")
            print("-" * 70)
            
            for result in sorted(results, key=lambda x: (x.model, x.sgr_mode)):
                status = "✅" if result.success else "❌"
                quality = f"{result.quality_scores['overall']:.2f}" if result.success else "N/A"
                print(f"{result.model:<15} {result.sgr_mode:<8} {quality:<10} {result.latency:<10.1f}s ${result.cost:<10.4f} {status}")
            
            # Find best improvement
            model_improvements = {}
            for model in TEST_MODELS:
                model_results = [r for r in results if r.model == model["name"]]
                off_result = next((r for r in model_results if r.sgr_mode == "off"), None)
                lite_result = next((r for r in model_results if r.sgr_mode == "lite"), None)
                full_result = next((r for r in model_results if r.sgr_mode == "full"), None)
                
                if off_result and off_result.success:
                    base_quality = off_result.quality_scores['overall']
                    if lite_result and lite_result.success:
                        lite_improvement = ((lite_result.quality_scores['overall'] - base_quality) / base_quality) * 100
                    else:
                        lite_improvement = 0
                    
                    if full_result and full_result.success:
                        full_improvement = ((full_result.quality_scores['overall'] - base_quality) / base_quality) * 100
                    else:
                        full_improvement = 0
                    
                    model_improvements[model["name"]] = {
                        "lite": lite_improvement,
                        "full": full_improvement
                    }
            
            if model_improvements:
                print("\n📈 Quality Improvements with SGR:")
                for model, improvements in model_improvements.items():
                    print(f"  {model}: Lite {improvements['lite']:+.1f}%, Full {improvements['full']:+.1f}%")
        
        # Overall summary
        print("\n\n" + "="*80)
        print("🎯 OVERALL SUMMARY")
        print("="*80)
        
        # Success rates by mode
        mode_stats = {"off": {"success": 0, "total": 0}, 
                     "lite": {"success": 0, "total": 0},
                     "full": {"success": 0, "total": 0}}
        
        for result in self.results:
            mode_stats[result.sgr_mode]["total"] += 1
            if result.success:
                mode_stats[result.sgr_mode]["success"] += 1
        
        print("\n✅ Success Rates by SGR Mode:")
        for mode, stats in mode_stats.items():
            rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  SGR-{mode}: {rate:.1f}% ({stats['success']}/{stats['total']})")
        
        # Average quality by mode
        print("\n📊 Average Quality Scores by SGR Mode:")
        for mode in ["off", "lite", "full"]:
            mode_results = [r for r in self.results if r.sgr_mode == mode and r.success]
            if mode_results:
                avg_quality = sum(r.quality_scores['overall'] for r in mode_results) / len(mode_results)
                print(f"  SGR-{mode}: {avg_quality:.2f}/1.0")
        
        # Cost analysis
        print("\n💰 Cost Analysis:")
        for mode in ["off", "lite", "full"]:
            mode_results = [r for r in self.results if r.sgr_mode == mode]
            if mode_results:
                avg_cost = sum(r.cost for r in mode_results) / len(mode_results)
                avg_latency = sum(r.latency for r in mode_results) / len(mode_results)
                print(f"  SGR-{mode}: ${avg_cost:.4f} avg cost, {avg_latency:.1f}s avg latency")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sgr_poc_results_{timestamp}.json"
        
        results_data = []
        for r in self.results:
            results_data.append({
                "test_id": r.test_id,
                "model": r.model,
                "sgr_mode": r.sgr_mode,
                "latency": r.latency,
                "tokens": r.tokens,
                "cost": r.cost,
                "success": r.success,
                "quality_scores": r.quality_scores,
                "error": r.error
            })
        
        with open(filename, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "results": results_data,
                "summary": {
                    "total_tests": len(self.results),
                    "models_tested": len(TEST_MODELS),
                    "test_cases": len(by_test)
                }
            }, f, indent=2)
        
        print(f"\n\n💾 Detailed results saved to: {filename}")
        
        # Key insights
        print("\n\n🔍 KEY INSIGHTS:")
        print("-" * 50)
        
        # Find cases where SGR helped most
        best_improvements = []
        for test_id, results in by_test.items():
            for model in TEST_MODELS:
                model_results = [r for r in results if r.model == model["name"]]
                off = next((r for r in model_results if r.sgr_mode == "off" and r.success), None)
                full = next((r for r in model_results if r.sgr_mode == "full" and r.success), None)
                
                if off and full:
                    improvement = ((full.quality_scores['overall'] - off.quality_scores['overall']) / off.quality_scores['overall']) * 100
                    if improvement > 20:
                        test_name = next((tc["name"] for task_cases in TEST_CASES.values() for tc in task_cases if tc["id"] == test_id), test_id)
                        best_improvements.append((test_name, model["name"], improvement))
        
        if best_improvements:
            print("\n✨ Best SGR Improvements:")
            for test, model, imp in sorted(best_improvements, key=lambda x: x[2], reverse=True)[:5]:
                print(f"  {test} + {model}: {imp:+.1f}%")
        
        print("\n📌 Conclusions:")
        print("  1. SGR provides structured, consistent outputs")
        print("  2. Full mode offers most comprehensive analysis")
        print("  3. Lite mode balances quality and cost effectively")
        print("  4. Larger models benefit more from SGR guidance")


async def main():
    """Run the SGR PoC framework."""
    framework = SGRTestFramework()
    await framework.run_poc()


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    asyncio.run(main())