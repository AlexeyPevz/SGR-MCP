#!/usr/bin/env python3
"""
SGR Benchmark Pack Runner
Main script to run comprehensive benchmarks across models and SGR modes
"""

import asyncio
import json
import os
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import urllib.request
import sys

# Add SGR tools to path
sys.path.append('/workspace/mcp-sgr/src/tools')
from apply_sgr_v4 import TASK_SCHEMAS

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BENCHMARK_DIR = Path(__file__).parent

@dataclass
class TaskResult:
    task_id: str
    model: str
    sgr_mode: str
    success: bool
    latency: float
    tokens_used: int
    cost: float
    metrics: Dict[str, float]
    output: Any
    artifacts: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str = ""

@dataclass
class BenchmarkConfig:
    models: List[Dict]
    sgr_modes: List[Dict]
    categories: Dict[str, Dict]
    evaluation: Dict
    reporting: Dict

class BenchmarkRunner:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize benchmark runner with configuration."""
        self.config_path = BENCHMARK_DIR / config_path
        self.config = self._load_config()
        self.results = []
        self.sgr_schemas = self._load_sgr_schemas()
        
    def _load_config(self) -> BenchmarkConfig:
        """Load benchmark configuration."""
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        return BenchmarkConfig(**data)
    
    def _load_sgr_schemas(self) -> Dict[str, Dict]:
        """Load SGR schemas for different task types."""
        return {
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
                                "language": {"type": "string"},
                                "explanation": {"type": "string"}
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
                                "algorithms": {"type": "array", "items": {"type": "string"}},
                                "data_structures": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "implementation": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "tests": {"type": "string"},
                                "language": {"type": "string"}
                            }
                        },
                        "validation": {
                            "type": "object",
                            "properties": {
                                "test_coverage": {"type": "number"},
                                "security_checks": {"type": "array", "items": {"type": "string"}},
                                "performance_notes": {"type": "string"}
                            }
                        }
                    },
                    "required": ["requirements_analysis", "design", "implementation", "validation"]
                }
            },
            "rag_qa": {
                "lite": {
                    "type": "object",
                    "properties": {
                        "question_understanding": {
                            "type": "object",
                            "properties": {
                                "intent": {"type": "string"},
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
                    "required": ["question_understanding", "answer"]
                },
                "full": {
                    "type": "object",
                    "properties": {
                        "question_analysis": {
                            "type": "object",
                            "properties": {
                                "intent": {"type": "string"},
                                "key_terms": {"type": "array", "items": {"type": "string"}},
                                "information_needs": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "evidence_gathering": {
                            "type": "object",
                            "properties": {
                                "relevant_passages": {"type": "array", "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "source": {"type": "string"},
                                        "relevance": {"type": "number"}
                                    }
                                }},
                                "conflicting_info": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "answer": {
                            "type": "object",
                            "properties": {
                                "response": {"type": "string"},
                                "claim_to_source_map": {"type": "array", "items": {
                                    "type": "object",
                                    "properties": {
                                        "claim": {"type": "string"},
                                        "source": {"type": "string"}
                                    }
                                }},
                                "confidence": {"type": "number"}
                            }
                        },
                        "validation": {
                            "type": "object",
                            "properties": {
                                "all_claims_grounded": {"type": "boolean"},
                                "coverage": {"type": "number"},
                                "missing_info": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "required": ["question_analysis", "evidence_gathering", "answer", "validation"]
                }
            }
        }
    
    async def call_model(self, model: Dict, prompt: str, sgr_mode: str, 
                        category: str, documents: Optional[List[Dict]] = None) -> Tuple[Any, float, int, Optional[str]]:
        """Call model with specified SGR mode."""
        start_time = time.time()
        
        # Prepare prompt based on mode and category
        if sgr_mode == "off":
            system_prompt = "You are an expert assistant. Provide high-quality, detailed responses."
            user_prompt = prompt
            if documents:
                doc_context = "\n\n".join([f"[{doc['id']}]: {doc['content']}" for doc in documents])
                user_prompt = f"Documents:\n{doc_context}\n\nQuestion: {prompt}"
        else:
            # SGR mode - use schema
            schema = self.sgr_schemas.get(category, {}).get(sgr_mode, {})
            system_prompt = """You are an expert providing structured analysis.

Your response MUST be valid JSON matching the provided schema. The schema guides your reasoning - be thorough and systematic."""
            
            user_prompt = prompt
            if documents:
                doc_context = "\n\n".join([f"[{doc['id']}]: {doc['content']}" for doc in documents])
                user_prompt = f"Documents:\n{doc_context}\n\nQuestion: {prompt}"
                
            user_prompt += f"\n\nProvide your response as JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        
        # Make API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        data = {
            "model": model["id"],
            "messages": messages,
            "temperature": self.config.evaluation["temperature"],
            "max_tokens": self.config.evaluation["max_tokens"]
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
            with urllib.request.urlopen(request, timeout=self.config.evaluation["timeout"]) as response:
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
    
    def evaluate_code_generation(self, output: Any, task: Dict, sgr_mode: str) -> Dict[str, float]:
        """Evaluate code generation results."""
        metrics = {}
        
        if sgr_mode == "off":
            # Basic evaluation for unstructured output
            content = str(output).lower()
            
            # Check for code presence
            metrics["has_code"] = 1.0 if any(marker in content for marker in ["def ", "function", "class", "import"]) else 0.0
            
            # Check for tests
            metrics["has_tests"] = 1.0 if any(marker in content for marker in ["test", "assert", "expect"]) else 0.0
            
            # Check constraints
            if "constraints" in task:
                found = sum(1 for c in task["constraints"] if c.lower() in content)
                metrics["constraints_met"] = found / len(task["constraints"])
        else:
            # Structured evaluation
            if isinstance(output, dict):
                # Check implementation
                if "implementation" in output and output["implementation"].get("code"):
                    metrics["has_code"] = 1.0
                    code_length = len(output["implementation"]["code"])
                    metrics["code_length"] = min(1.0, code_length / 500)  # Normalize to expected length
                else:
                    metrics["has_code"] = 0.0
                    metrics["code_length"] = 0.0
                
                # Check requirements analysis
                if "requirements_analysis" in output:
                    metrics["requirements_analyzed"] = 1.0
                    req_count = len(output["requirements_analysis"].get("key_requirements", []))
                    metrics["requirement_coverage"] = min(1.0, req_count / 3)  # Expect at least 3
                else:
                    metrics["requirements_analyzed"] = 0.0
                    metrics["requirement_coverage"] = 0.0
                
                # For full mode, check additional fields
                if sgr_mode == "full":
                    if "validation" in output:
                        metrics["has_validation"] = 1.0
                        metrics["test_coverage"] = output["validation"].get("test_coverage", 0) / 100
                    else:
                        metrics["has_validation"] = 0.0
                        metrics["test_coverage"] = 0.0
        
        # Calculate overall score
        metrics["overall"] = sum(metrics.values()) / len(metrics) if metrics else 0
        return metrics
    
    def evaluate_rag_qa(self, output: Any, task: Dict, sgr_mode: str) -> Dict[str, float]:
        """Evaluate RAG Q&A results."""
        metrics = {}
        
        if sgr_mode == "off":
            # Basic evaluation
            content = str(output).lower()
            
            # Check for answer presence
            metrics["has_answer"] = 1.0 if len(content) > 50 else 0.0
            
            # Check for source citations
            metrics["has_citations"] = 1.0 if any(marker in content for marker in ["doc1", "doc2", "doc3", "[1]", "[2]"]) else 0.0
            
            # Check required mentions
            if "evaluation" in task and "required_mentions" in task["evaluation"]:
                found = sum(1 for req in task["evaluation"]["required_mentions"] if req.lower() in content)
                metrics["content_coverage"] = found / len(task["evaluation"]["required_mentions"])
        else:
            # Structured evaluation
            if isinstance(output, dict):
                # Check answer quality
                if "answer" in output:
                    answer = output["answer"]
                    metrics["has_answer"] = 1.0 if answer.get("response") else 0.0
                    
                    # Check citations
                    if answer.get("sources") or answer.get("claim_to_source_map"):
                        metrics["has_citations"] = 1.0
                        if answer.get("claim_to_source_map"):
                            metrics["citation_granularity"] = min(1.0, len(answer["claim_to_source_map"]) / 3)
                        else:
                            metrics["citation_granularity"] = 0.5
                    else:
                        metrics["has_citations"] = 0.0
                        metrics["citation_granularity"] = 0.0
                
                # For full mode, check validation
                if sgr_mode == "full" and "validation" in output:
                    validation = output["validation"]
                    metrics["grounded"] = 1.0 if validation.get("all_claims_grounded") else 0.5
                    metrics["coverage"] = validation.get("coverage", 0)
                else:
                    metrics["grounded"] = 0.5
                    metrics["coverage"] = 0.5
        
        # Check for adversarial handling
        if task.get("difficulty") == "adversarial":
            if "evaluation" in task:
                if task["evaluation"].get("conflict_detection"):
                    # Check if conflicts were identified
                    content = json.dumps(output) if isinstance(output, dict) else str(output)
                    metrics["conflict_detected"] = 1.0 if any(word in content.lower() for word in ["conflict", "contradiction", "disagree"]) else 0.0
                
                if task["evaluation"].get("detects_missing_info"):
                    # Check if missing info was acknowledged
                    metrics["missing_info_detected"] = 1.0 if any(word in content.lower() for word in ["not found", "no information", "missing", "not specified"]) else 0.0
        
        # Calculate overall score
        metrics["overall"] = sum(metrics.values()) / len(metrics) if metrics else 0
        return metrics
    
    async def run_task(self, task: Dict, model: Dict, sgr_mode: str) -> TaskResult:
        """Run a single task with given model and SGR mode."""
        # Extract category from task
        category_map = {
            "codegen": "code_generation",
            "rag_qa": "rag_qa"
        }
        category = category_map.get(task.get("category", ""), "code_generation")
        
        # Call model
        documents = task.get("documents", None)
        output, latency, tokens, error = await self.call_model(
            model, task["prompt"], sgr_mode, category, documents
        )
        
        # Calculate cost
        cost = tokens * model.get("cost_per_1k", 0.0001) / 1000
        
        # Evaluate based on category
        if output and not error:
            if category == "code_generation":
                metrics = self.evaluate_code_generation(output, task, sgr_mode)
            elif category == "rag_qa":
                metrics = self.evaluate_rag_qa(output, task, sgr_mode)
            else:
                metrics = {"overall": 0.5}  # Default score
            success = True
        else:
            metrics = {"overall": 0.0}
            success = False
        
        # Create result
        result = TaskResult(
            task_id=task["id"],
            model=model["name"],
            sgr_mode=sgr_mode,
            success=success,
            latency=latency,
            tokens_used=tokens,
            cost=cost,
            metrics=metrics,
            output=output,
            artifacts={
                "sgr_schema_used": self.sgr_schemas.get(category, {}).get(sgr_mode, {}) if sgr_mode != "off" else None
            },
            error=error,
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    async def run_benchmark(self, task_filter: Optional[str] = None, limit: Optional[int] = None):
        """Run the complete benchmark suite."""
        print("\n🚀 SGR Benchmark Pack Runner")
        print("=" * 80)
        print(f"Models: {len(self.config.models)}")
        print(f"SGR Modes: {len(self.config.sgr_modes)}")
        print(f"Categories: {len(self.config.categories)}")
        print("=" * 80)
        
        # Load tasks
        tasks = []
        for category_file in ["code_generation.yaml", "rag_qa.yaml"]:
            task_path = BENCHMARK_DIR / "tasks" / category_file
            if task_path.exists():
                with open(task_path, 'r') as f:
                    content = yaml.safe_load(f)
                    if isinstance(content, list):
                        tasks.extend(content)
        
        # Apply filters
        if task_filter:
            tasks = [t for t in tasks if task_filter in t.get("category", "") or task_filter in t.get("difficulty", "")]
        
        if limit:
            tasks = tasks[:limit]
        
        print(f"\nRunning {len(tasks)} tasks...")
        
        # Run tasks
        total_runs = len(tasks) * len(self.config.models) * len(self.config.sgr_modes)
        current_run = 0
        
        for task in tasks:
            print(f"\n📋 Task: {task['id']} - {task.get('difficulty', 'base')}")
            
            for model in self.config.models:
                for sgr_mode in self.config.sgr_modes:
                    current_run += 1
                    print(f"[{current_run}/{total_runs}] {model['name']} - SGR: {sgr_mode['name']}", end="", flush=True)
                    
                    # Run multiple times for averaging
                    run_results = []
                    for run in range(self.config.evaluation["runs_per_task"]):
                        result = await self.run_task(task, model, sgr_mode["name"])
                        run_results.append(result)
                        await asyncio.sleep(1)  # Rate limiting
                    
                    # Average the results
                    avg_result = self._average_results(run_results)
                    self.results.append(avg_result)
                    
                    if avg_result.success:
                        print(f" ✅ Score: {avg_result.metrics['overall']:.2f}, Time: {avg_result.latency:.1f}s")
                    else:
                        print(f" ❌ Failed: {avg_result.error}")
        
        # Generate report
        self.generate_report()
    
    def _average_results(self, results: List[TaskResult]) -> TaskResult:
        """Average multiple run results."""
        if not results:
            return None
        
        # Use first result as template
        avg_result = results[0]
        
        # Average numeric fields
        avg_result.latency = sum(r.latency for r in results) / len(results)
        avg_result.tokens_used = sum(r.tokens_used for r in results) / len(results)
        avg_result.cost = sum(r.cost for r in results) / len(results)
        
        # Average metrics
        all_metrics = [r.metrics for r in results if r.success]
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m.get(key, 0) for m in all_metrics) / len(all_metrics)
            avg_result.metrics = avg_metrics
        
        # Success if majority succeeded
        avg_result.success = sum(1 for r in results if r.success) > len(results) / 2
        
        return avg_result
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print("\n\n" + "="*80)
        print("📊 BENCHMARK RESULTS")
        print("="*80)
        
        # Group results
        by_task = {}
        by_model = {}
        by_mode = {}
        
        for result in self.results:
            # By task
            if result.task_id not in by_task:
                by_task[result.task_id] = []
            by_task[result.task_id].append(result)
            
            # By model
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)
            
            # By mode
            if result.sgr_mode not in by_mode:
                by_mode[result.sgr_mode] = []
            by_mode[result.sgr_mode].append(result)
        
        # Overall statistics
        print("\n📈 Overall Statistics:")
        print(f"Total runs: {len(self.results)}")
        print(f"Success rate: {sum(1 for r in self.results if r.success) / len(self.results) * 100:.1f}%")
        print(f"Average latency: {sum(r.latency for r in self.results) / len(self.results):.1f}s")
        print(f"Total cost: ${sum(r.cost for r in self.results):.2f}")
        
        # By SGR mode
        print("\n🔄 Performance by SGR Mode:")
        print(f"{'Mode':<10} {'Success':<10} {'Avg Score':<12} {'Avg Latency':<12} {'Avg Cost'}")
        print("-" * 60)
        
        for mode, results in by_mode.items():
            success_rate = sum(1 for r in results if r.success) / len(results) * 100
            avg_score = sum(r.metrics.get('overall', 0) for r in results if r.success) / max(sum(1 for r in results if r.success), 1)
            avg_latency = sum(r.latency for r in results) / len(results)
            avg_cost = sum(r.cost for r in results) / len(results)
            
            print(f"{mode:<10} {success_rate:<10.1f}% {avg_score:<12.2f} {avg_latency:<12.1f}s ${avg_cost:<.4f}")
        
        # By model
        print("\n🤖 Performance by Model:")
        print(f"{'Model':<20} {'Success':<10} {'Avg Score':<12} {'Avg Latency':<12} {'Total Cost'}")
        print("-" * 80)
        
        for model, results in by_model.items():
            success_rate = sum(1 for r in results if r.success) / len(results) * 100
            avg_score = sum(r.metrics.get('overall', 0) for r in results if r.success) / max(sum(1 for r in results if r.success), 1)
            avg_latency = sum(r.latency for r in results) / len(results)
            total_cost = sum(r.cost for r in results)
            
            print(f"{model:<20} {success_rate:<10.1f}% {avg_score:<12.2f} {avg_latency:<12.1f}s ${total_cost:<.2f}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = BENCHMARK_DIR / "reports" / f"benchmark_results_{timestamp}.json"
        json_file.parent.mkdir(exist_ok=True)
        
        with open(json_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "config": asdict(self.config),
                "results": [asdict(r) for r in self.results],
                "summary": {
                    "total_runs": len(self.results),
                    "success_rate": sum(1 for r in self.results if r.success) / len(self.results),
                    "total_cost": sum(r.cost for r in self.results),
                    "avg_latency": sum(r.latency for r in self.results) / len(self.results)
                }
            }, f, indent=2)
        
        print(f"\n\n💾 Results saved to: {json_file}")
        
        # Generate markdown report
        self._generate_markdown_report(timestamp)
    
    def _generate_markdown_report(self, timestamp: str):
        """Generate detailed markdown report."""
        md_file = BENCHMARK_DIR / "reports" / f"benchmark_report_{timestamp}.md"
        
        report = f"""# SGR Benchmark Report

Generated: {timestamp}

## Executive Summary

Comprehensive benchmark comparing {len(self.config.models)} models across {len(self.config.sgr_modes)} SGR modes.

### Key Findings:

"""
        
        # Calculate key metrics
        by_mode = {}
        for result in self.results:
            if result.sgr_mode not in by_mode:
                by_mode[result.sgr_mode] = []
            by_mode[result.sgr_mode].append(result)
        
        # Compare modes
        mode_comparison = []
        for mode, results in by_mode.items():
            successful = [r for r in results if r.success]
            if successful:
                avg_score = sum(r.metrics.get('overall', 0) for r in successful) / len(successful)
                mode_comparison.append((mode, avg_score))
        
        mode_comparison.sort(key=lambda x: x[1], reverse=True)
        
        if mode_comparison:
            best_mode = mode_comparison[0]
            report += f"- **Best performing mode**: {best_mode[0]} (avg score: {best_mode[1]:.2f})\n"
            
            # Calculate improvement
            baseline = next((score for mode, score in mode_comparison if mode == "off"), 0)
            if baseline > 0:
                for mode, score in mode_comparison:
                    if mode != "off":
                        improvement = ((score - baseline) / baseline) * 100
                        report += f"- **{mode} improvement**: {improvement:+.1f}% over baseline\n"
        
        # Add detailed results
        report += "\n## Detailed Results\n\n"
        
        # Group by task category
        task_categories = {}
        for task_id, results in self._group_by_task().items():
            category = results[0].task_id.split("_")[0]  # Extract category from ID
            if category not in task_categories:
                task_categories[category] = []
            task_categories[category].append((task_id, results))
        
        for category, tasks in task_categories.items():
            report += f"\n### {category.upper()} Tasks\n\n"
            
            for task_id, results in tasks:
                report += f"\n#### {task_id}\n\n"
                report += "| Model | SGR Mode | Score | Latency | Cost |\n"
                report += "|-------|----------|-------|---------|------|\n"
                
                for r in sorted(results, key=lambda x: (x.model, x.sgr_mode)):
                    score = f"{r.metrics.get('overall', 0):.2f}" if r.success else "Failed"
                    report += f"| {r.model} | {r.sgr_mode} | {score} | {r.latency:.1f}s | ${r.cost:.4f} |\n"
        
        # Add recommendations
        report += "\n## Recommendations\n\n"
        report += "Based on the benchmark results:\n\n"
        
        # Find best model/mode combinations
        best_quality = max(self.results, key=lambda r: r.metrics.get('overall', 0) if r.success else 0)
        best_value = min((r for r in self.results if r.success), key=lambda r: r.cost / max(r.metrics.get('overall', 0.1), 0.1))
        
        report += f"1. **For maximum quality**: Use {best_quality.model} with SGR-{best_quality.sgr_mode}\n"
        report += f"2. **For best value**: Use {best_value.model} with SGR-{best_value.sgr_mode}\n"
        report += f"3. **For production**: Consider SGR-lite for balance of quality and cost\n"
        
        # Save report
        with open(md_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Markdown report saved to: {md_file}")
    
    def _group_by_task(self) -> Dict[str, List[TaskResult]]:
        """Group results by task ID."""
        by_task = {}
        for result in self.results:
            if result.task_id not in by_task:
                by_task[result.task_id] = []
            by_task[result.task_id].append(result)
        return by_task


async def main():
    """Run benchmark with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SGR Benchmark Pack Runner")
    parser.add_argument("--filter", help="Filter tasks by category or difficulty")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to run")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    runner = BenchmarkRunner(args.config)
    await runner.run_benchmark(args.filter, args.limit)


if __name__ == "__main__":
    asyncio.run(main())