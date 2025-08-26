"""Extended benchmarks for SGR performance comparison across models and configurations.

Features:
- Multiple model providers and tiers
- SGR on/off comparison
- lite/full budget modes
- Detailed metrics: latency, cost estimation, quality scores
- Statistical analysis
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

from src.schemas import SCHEMA_REGISTRY
from src.utils.llm_client import LLMClient
from src.tools.apply_sgr import apply_sgr_tool
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager


# Extended model configuration with pricing
MODELS_EXTENDED: Dict[str, List[Dict[str, Any]]] = {
    "ultra_cheap": [
        {"name": "groq/llama-3.1-8b-instant", "cost_per_1k": 0.0001},
        {"name": "google/gemini-2.0-flash-exp:free", "cost_per_1k": 0.0},
    ],
    "cheap": [
        {"name": "qwen/qwen-2.5-7b-instruct", "cost_per_1k": 0.00035},
        {"name": "mistralai/mistral-7b-instruct", "cost_per_1k": 0.00025},
        {"name": "meta-llama/llama-3.1-8b-instruct", "cost_per_1k": 0.00018},
        {"name": "google/gemini-2.0-flash-lite", "cost_per_1k": 0.0003},
    ],
    "medium": [
        {"name": "anthropic/claude-3.5-haiku", "cost_per_1k": 0.001},
        {"name": "openai/gpt-4o-mini", "cost_per_1k": 0.00015},
        {"name": "qwen/qwen-2.5-72b-instruct", "cost_per_1k": 0.0035},
    ],
    "strong": [
        {"name": "anthropic/claude-3.5-sonnet", "cost_per_1k": 0.003},
        {"name": "openai/gpt-4o", "cost_per_1k": 0.0025},
        {"name": "google/gemini-2.5-pro", "cost_per_1k": 0.00125},
    ],
    "frontier": [
        {"name": "anthropic/claude-3.5-sonnet-20241022", "cost_per_1k": 0.003},
        {"name": "openai/o1-mini", "cost_per_1k": 0.003},
        {"name": "deepseek/deepseek-chat", "cost_per_1k": 0.00014},
    ]
}

# Extended task set with complexity levels
TASKS_EXTENDED: Dict[str, List[Dict[str, str]]] = {
    "analysis": [
        {
            "name": "simple_analysis",
            "task": "Analyze a simple Python function for performance issues",
            "complexity": "low"
        },
        {
            "name": "medium_analysis", 
            "task": "Analyze a REST API with 10 endpoints for security vulnerabilities and performance bottlenecks",
            "complexity": "medium"
        },
        {
            "name": "complex_analysis",
            "task": "Analyze a distributed microservices architecture with 20+ services for scalability issues, data consistency problems, and optimization opportunities",
            "complexity": "high"
        }
    ],
    "planning": [
        {
            "name": "simple_planning",
            "task": "Plan the development of a todo list web app",
            "complexity": "low"
        },
        {
            "name": "medium_planning",
            "task": "Plan migration from monolith to microservices for an e-commerce platform",
            "complexity": "medium"
        },
        {
            "name": "complex_planning",
            "task": "Plan a multi-region disaster recovery strategy for a financial services platform with zero data loss requirement",
            "complexity": "high"
        }
    ],
    "code_generation": [
        {
            "name": "simple_code",
            "task": "Write a Python function to validate email addresses",
            "complexity": "low"
        },
        {
            "name": "medium_code",
            "task": "Implement a rate limiter using Redis with sliding window algorithm",
            "complexity": "medium"
        },
        {
            "name": "complex_code",
            "task": "Implement a distributed lock manager with automatic failover and consensus algorithm",
            "complexity": "high"
        }
    ]
}


@dataclass
class ExtendedRunResult:
    """Extended result with more metrics"""
    valid: bool
    confidence: float
    latency_ms: int
    tokens_used: int = 0
    estimated_cost: float = 0.0
    error: Optional[str] = None
    quality_score: float = 0.0  # Combined metric
    reasoning_depth: int = 0  # How many reasoning steps
    actions_count: int = 0  # Number of suggested actions
    
    
@dataclass 
class BenchmarkRun:
    """Single benchmark run configuration"""
    model: str
    model_info: Dict[str, Any]
    task_category: str
    task_info: Dict[str, str]
    sgr_enabled: bool
    sgr_budget: Optional[str] = None
    
    
async def run_baseline_extended(
    task_category: str,
    task: str,
    model: str,
    model_info: Dict[str, Any]
) -> ExtendedRunResult:
    """Run baseline without SGR"""
    client = LLMClient()
    os.environ["OPENROUTER_DEFAULT_MODEL"] = model
    
    start_time = time.perf_counter()
    tokens_used = 0
    
    try:
        schema = SCHEMA_REGISTRY[task_category]()
        system = (
            "You are a JSON-only assistant. Return strictly valid JSON matching the given schema. "
            "Provide detailed, high-quality responses. No markdown, no extra text."
        )
        prompt = (
            f"Task: {task}\n\n"
            f"Schema: {json.dumps(schema.to_json_schema(), indent=2)}\n\n"
            "Return only valid JSON matching the schema."
        )
        
        # Estimate tokens (rough approximation)
        tokens_used = len(prompt.split()) * 2  # Rough estimate
        
        text = await client.generate(
            prompt,
            backend="openrouter",
            temperature=0.1,
            max_tokens=4000,
            system_prompt=system,
        )
        
        # Parse response
        raw = text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
            
        data = json.loads(raw)
        validation_result = schema.validate(data)
        
        # Calculate quality score
        quality_score = validation_result.confidence
        if validation_result.valid:
            quality_score += 0.2
            
        # Count actions/depth
        actions_count = len(data.get("suggested_actions", []))
        reasoning_depth = 1  # Baseline has single-step reasoning
        
        latency = int((time.perf_counter() - start_time) * 1000)
        estimated_cost = (tokens_used / 1000) * model_info["cost_per_1k"]
        
        return ExtendedRunResult(
            valid=validation_result.valid,
            confidence=validation_result.confidence,
            latency_ms=latency,
            tokens_used=tokens_used,
            estimated_cost=estimated_cost,
            quality_score=quality_score,
            reasoning_depth=reasoning_depth,
            actions_count=actions_count
        )
        
    except Exception as e:
        latency = int((time.perf_counter() - start_time) * 1000)
        return ExtendedRunResult(
            valid=False,
            confidence=0.0,
            latency_ms=latency,
            tokens_used=tokens_used,
            estimated_cost=(tokens_used / 1000) * model_info["cost_per_1k"],
            error=str(e)
        )
    finally:
        await client.close()


async def run_sgr_extended(
    task_category: str,
    task: str,
    model: str,
    model_info: Dict[str, Any],
    budget: str = "lite"
) -> ExtendedRunResult:
    """Run with SGR enabled"""
    os.environ["OPENROUTER_DEFAULT_MODEL"] = model
    
    client = LLMClient()
    cache = CacheManager()
    telemetry = TelemetryManager()
    
    # Disable cache for benchmarks
    cache.enabled = False
    
    await cache.initialize()
    await telemetry.initialize()
    
    start_time = time.perf_counter()
    
    try:
        result = await apply_sgr_tool(
            arguments={
                "task": task,
                "schema_type": task_category,
                "budget": budget,
                "backend": "openrouter",
            },
            llm_client=client,
            cache_manager=cache,
            telemetry=telemetry,
        )
        
        valid = result.get("metadata", {}).get("validation", {}).get("valid", False)
        confidence = float(result.get("confidence", 0.0))
        
        # Extract additional metrics
        reasoning = result.get("reasoning", {})
        actions = result.get("suggested_actions", [])
        
        # Calculate reasoning depth (number of keys in reasoning dict)
        reasoning_depth = len(reasoning) if isinstance(reasoning, dict) else 1
        
        # Quality score combines validation, confidence, and reasoning depth
        quality_score = confidence
        if valid:
            quality_score += 0.2
        quality_score += min(reasoning_depth * 0.05, 0.3)  # Bonus for depth, capped
        
        latency = int((time.perf_counter() - start_time) * 1000)
        
        # Rough token estimation based on budget
        tokens_used = 500 if budget == "lite" else 1500
        estimated_cost = (tokens_used / 1000) * model_info["cost_per_1k"]
        
        return ExtendedRunResult(
            valid=valid,
            confidence=confidence,
            latency_ms=latency,
            tokens_used=tokens_used,
            estimated_cost=estimated_cost,
            quality_score=min(quality_score, 1.0),
            reasoning_depth=reasoning_depth,
            actions_count=len(actions)
        )
        
    except Exception as e:
        latency = int((time.perf_counter() - start_time) * 1000)
        return ExtendedRunResult(
            valid=False,
            confidence=0.0,
            latency_ms=latency,
            error=str(e)
        )
    finally:
        await client.close()
        await cache.close()
        await telemetry.close()


async def run_benchmark_suite(
    model_tiers: List[str],
    task_categories: List[str],
    sgr_modes: List[str],
    max_concurrent: int = 3
) -> Dict[str, Any]:
    """Run complete benchmark suite"""
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "model_tiers": model_tiers,
            "task_categories": task_categories,
            "sgr_modes": sgr_modes
        },
        "models": {},
        "tasks": {},
        "runs": [],
        "summary": {}
    }
    
    # Collect all models and tasks
    for tier in model_tiers:
        results["models"][tier] = MODELS_EXTENDED.get(tier, [])
        
    for category in task_categories:
        results["tasks"][category] = TASKS_EXTENDED.get(category, [])
    
    # Create benchmark runs
    benchmarks = []
    
    for tier in model_tiers:
        for model_info in MODELS_EXTENDED.get(tier, []):
            for category in task_categories:
                for task_info in TASKS_EXTENDED.get(category, []):
                    # Baseline run
                    if "baseline" in sgr_modes:
                        benchmarks.append(BenchmarkRun(
                            model=model_info["name"],
                            model_info=model_info,
                            task_category=category,
                            task_info=task_info,
                            sgr_enabled=False
                        ))
                    
                    # SGR runs
                    if "sgr_lite" in sgr_modes:
                        benchmarks.append(BenchmarkRun(
                            model=model_info["name"],
                            model_info=model_info,
                            task_category=category,
                            task_info=task_info,
                            sgr_enabled=True,
                            sgr_budget="lite"
                        ))
                        
                    if "sgr_full" in sgr_modes:
                        benchmarks.append(BenchmarkRun(
                            model=model_info["name"],
                            model_info=model_info,
                            task_category=category,
                            task_info=task_info,
                            sgr_enabled=True,
                            sgr_budget="full"
                        ))
    
    # Run benchmarks with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_single_benchmark(bench: BenchmarkRun) -> Dict[str, Any]:
        async with semaphore:
            print(f"Running: {bench.model} | {bench.task_info['name']} | SGR: {bench.sgr_enabled} ({bench.sgr_budget or 'N/A'})")
            
            if bench.sgr_enabled:
                result = await run_sgr_extended(
                    bench.task_category,
                    bench.task_info["task"],
                    bench.model,
                    bench.model_info,
                    bench.sgr_budget
                )
            else:
                result = await run_baseline_extended(
                    bench.task_category,
                    bench.task_info["task"],
                    bench.model,
                    bench.model_info
                )
            
            return {
                "model": bench.model,
                "model_tier": next(t for t, models in MODELS_EXTENDED.items() 
                                  if any(m["name"] == bench.model for m in models)),
                "task_category": bench.task_category,
                "task_name": bench.task_info["name"],
                "task_complexity": bench.task_info["complexity"],
                "sgr_enabled": bench.sgr_enabled,
                "sgr_budget": bench.sgr_budget,
                "result": asdict(result)
            }
    
    # Execute all benchmarks
    run_results = await asyncio.gather(*[
        run_single_benchmark(bench) for bench in benchmarks
    ])
    
    results["runs"] = run_results
    
    # Calculate summary statistics
    results["summary"] = calculate_summary_stats(run_results)
    
    return results


def calculate_summary_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from benchmark runs"""
    summary = {
        "by_model_tier": {},
        "by_task_complexity": {},
        "sgr_comparison": {},
        "overall": {}
    }
    
    # Group runs for analysis
    from collections import defaultdict
    
    tier_groups = defaultdict(list)
    complexity_groups = defaultdict(list)
    sgr_groups = defaultdict(list)
    
    for run in runs:
        result = run["result"]
        tier_groups[run["model_tier"]].append(result)
        complexity_groups[run["task_complexity"]].append(result)
        
        sgr_key = "baseline"
        if run["sgr_enabled"]:
            sgr_key = f"sgr_{run['sgr_budget']}"
        sgr_groups[sgr_key].append(result)
    
    # Calculate tier statistics
    for tier, results in tier_groups.items():
        valid_results = [r for r in results if not r.get("error")]
        if valid_results:
            summary["by_model_tier"][tier] = {
                "avg_latency_ms": statistics.mean([r["latency_ms"] for r in valid_results]),
                "avg_quality_score": statistics.mean([r["quality_score"] for r in valid_results]),
                "avg_cost": statistics.mean([r["estimated_cost"] for r in valid_results]),
                "success_rate": len([r for r in valid_results if r["valid"]]) / len(valid_results),
                "total_runs": len(results)
            }
    
    # Calculate complexity statistics
    for complexity, results in complexity_groups.items():
        valid_results = [r for r in results if not r.get("error")]
        if valid_results:
            summary["by_task_complexity"][complexity] = {
                "avg_latency_ms": statistics.mean([r["latency_ms"] for r in valid_results]),
                "avg_quality_score": statistics.mean([r["quality_score"] for r in valid_results]),
                "success_rate": len([r for r in valid_results if r["valid"]]) / len(valid_results),
                "avg_reasoning_depth": statistics.mean([r["reasoning_depth"] for r in valid_results])
            }
    
    # SGR comparison
    for mode, results in sgr_groups.items():
        valid_results = [r for r in results if not r.get("error")]
        if valid_results:
            summary["sgr_comparison"][mode] = {
                "avg_latency_ms": statistics.mean([r["latency_ms"] for r in valid_results]),
                "avg_quality_score": statistics.mean([r["quality_score"] for r in valid_results]),
                "avg_cost": statistics.mean([r["estimated_cost"] for r in valid_results]),
                "success_rate": len([r for r in valid_results if r["valid"]]) / len(valid_results),
                "avg_confidence": statistics.mean([r["confidence"] for r in valid_results]),
                "total_runs": len(results)
            }
    
    # Overall statistics
    all_valid = [r["result"] for r in runs if not r["result"].get("error")]
    if all_valid:
        summary["overall"] = {
            "total_runs": len(runs),
            "total_errors": len([r for r in runs if r["result"].get("error")]),
            "avg_latency_ms": statistics.mean([r["latency_ms"] for r in all_valid]),
            "total_estimated_cost": sum([r["estimated_cost"] for r in all_valid]),
            "overall_success_rate": len([r for r in all_valid if r["valid"]]) / len(all_valid)
        }
    
    return summary


def write_extended_report(results: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path]:
    """Write detailed benchmark report"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Write JSON
    json_path = output_dir / f"extended_benchmark_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2))
    
    # Write detailed markdown report
    md_path = output_dir / f"extended_benchmark_{timestamp}.md"
    
    lines = [
        f"# Extended SGR Benchmark Report",
        f"*Generated: {results['timestamp']}*",
        "",
        "## Configuration",
        f"- Model Tiers: {', '.join(results['configuration']['model_tiers'])}",
        f"- Task Categories: {', '.join(results['configuration']['task_categories'])}",
        f"- SGR Modes: {', '.join(results['configuration']['sgr_modes'])}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Overall stats
    overall = results["summary"]["overall"]
    lines.extend([
        f"- **Total Runs**: {overall['total_runs']}",
        f"- **Success Rate**: {overall['overall_success_rate']:.1%}",
        f"- **Average Latency**: {overall['avg_latency_ms']:.0f}ms",
        f"- **Total Estimated Cost**: ${overall['total_estimated_cost']:.4f}",
        "",
        "## SGR Performance Comparison",
        "",
        "| Mode | Success Rate | Avg Quality | Avg Latency | Avg Cost | Avg Confidence |",
        "|------|-------------|------------|-------------|----------|----------------|"
    ])
    
    # SGR comparison table
    for mode in ["baseline", "sgr_lite", "sgr_full"]:
        if mode in results["summary"]["sgr_comparison"]:
            stats = results["summary"]["sgr_comparison"][mode]
            lines.append(
                f"| {mode} | {stats['success_rate']:.1%} | {stats['avg_quality_score']:.2f} | "
                f"{stats['avg_latency_ms']:.0f}ms | ${stats['avg_cost']:.4f} | {stats['avg_confidence']:.2f} |"
            )
    
    lines.extend([
        "",
        "## Performance by Model Tier",
        "",
        "| Tier | Success Rate | Avg Quality | Avg Latency | Avg Cost |",
        "|------|-------------|------------|-------------|----------|"
    ])
    
    # Model tier comparison
    for tier in results["configuration"]["model_tiers"]:
        if tier in results["summary"]["by_model_tier"]:
            stats = results["summary"]["by_model_tier"][tier]
            lines.append(
                f"| {tier} | {stats['success_rate']:.1%} | {stats['avg_quality_score']:.2f} | "
                f"{stats['avg_latency_ms']:.0f}ms | ${stats['avg_cost']:.4f} |"
            )
    
    lines.extend([
        "",
        "## Performance by Task Complexity",
        "",
        "| Complexity | Success Rate | Avg Quality | Avg Latency | Avg Reasoning Depth |",
        "|------------|-------------|------------|-------------|-------------------|"
    ])
    
    # Task complexity comparison
    for complexity in ["low", "medium", "high"]:
        if complexity in results["summary"]["by_task_complexity"]:
            stats = results["summary"]["by_task_complexity"][complexity]
            lines.append(
                f"| {complexity} | {stats['success_rate']:.1%} | {stats['avg_quality_score']:.2f} | "
                f"{stats['avg_latency_ms']:.0f}ms | {stats['avg_reasoning_depth']:.1f} |"
            )
    
    # Detailed results by model
    lines.extend([
        "",
        "## Detailed Results by Model",
        ""
    ])
    
    # Group results by model
    model_results = {}
    for run in results["runs"]:
        model = run["model"]
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(run)
    
    for model, runs in model_results.items():
        tier = runs[0]["model_tier"]
        lines.extend([
            f"### {model} ({tier})",
            "",
            "| Task | Complexity | Mode | Valid | Quality | Confidence | Latency | Cost |",
            "|------|------------|------|-------|---------|------------|---------|------|"
        ])
        
        for run in runs:
            result = run["result"]
            mode = "baseline"
            if run["sgr_enabled"]:
                mode = f"sgr_{run['sgr_budget']}"
            
            valid = "✓" if result["valid"] else "✗"
            if result.get("error"):
                valid = "ERROR"
                
            lines.append(
                f"| {run['task_name']} | {run['task_complexity']} | {mode} | {valid} | "
                f"{result['quality_score']:.2f} | {result['confidence']:.2f} | "
                f"{result['latency_ms']}ms | ${result['estimated_cost']:.4f} |"
            )
        
        lines.append("")
    
    # Key findings
    lines.extend([
        "## Key Findings",
        "",
        "### SGR Impact",
        ""
    ])
    
    # Calculate SGR improvements
    sgr_comp = results["summary"]["sgr_comparison"]
    if "baseline" in sgr_comp and "sgr_lite" in sgr_comp:
        quality_improvement = (sgr_comp["sgr_lite"]["avg_quality_score"] - 
                             sgr_comp["baseline"]["avg_quality_score"]) / sgr_comp["baseline"]["avg_quality_score"]
        success_improvement = sgr_comp["sgr_lite"]["success_rate"] - sgr_comp["baseline"]["success_rate"]
        
        lines.extend([
            f"- **Quality Improvement (lite)**: {quality_improvement:.1%}",
            f"- **Success Rate Delta (lite)**: {success_improvement:.1%}",
            f"- **Latency Overhead (lite)**: {sgr_comp['sgr_lite']['avg_latency_ms'] - sgr_comp['baseline']['avg_latency_ms']:.0f}ms",
            ""
        ])
    
    if "baseline" in sgr_comp and "sgr_full" in sgr_comp:
        quality_improvement = (sgr_comp["sgr_full"]["avg_quality_score"] - 
                             sgr_comp["baseline"]["avg_quality_score"]) / sgr_comp["baseline"]["avg_quality_score"]
        success_improvement = sgr_comp["sgr_full"]["success_rate"] - sgr_comp["baseline"]["success_rate"]
        
        lines.extend([
            f"- **Quality Improvement (full)**: {quality_improvement:.1%}",
            f"- **Success Rate Delta (full)**: {success_improvement:.1%}",
            f"- **Latency Overhead (full)**: {sgr_comp['sgr_full']['avg_latency_ms'] - sgr_comp['baseline']['avg_latency_ms']:.0f}ms",
            ""
        ])
    
    md_path.write_text("\n".join(lines))
    
    return json_path, md_path


async def main():
    parser = argparse.ArgumentParser(description="Extended SGR benchmarking")
    parser.add_argument(
        "--model-tiers",
        type=str,
        default="cheap,medium",
        help="Comma-separated model tiers: ultra_cheap,cheap,medium,strong,frontier"
    )
    parser.add_argument(
        "--task-categories",
        type=str,
        default="analysis,planning,code_generation",
        help="Comma-separated task categories"
    )
    parser.add_argument(
        "--sgr-modes",
        type=str,
        default="baseline,sgr_lite,sgr_full",
        help="Comma-separated SGR modes to test"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent API calls"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    model_tiers = [t.strip() for t in args.model_tiers.split(",")]
    task_categories = [t.strip() for t in args.task_categories.split(",")]
    sgr_modes = [m.strip() for m in args.sgr_modes.split(",")]
    
    # Ensure output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Disable cache for benchmarks
    os.environ["CACHE_ENABLED"] = "false"
    
    print(f"Starting extended benchmarks...")
    print(f"Model tiers: {model_tiers}")
    print(f"Task categories: {task_categories}")
    print(f"SGR modes: {sgr_modes}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()
    
    # Run benchmarks
    results = await run_benchmark_suite(
        model_tiers,
        task_categories,
        sgr_modes,
        args.max_concurrent
    )
    
    # Write reports
    json_path, md_path = write_extended_report(results, output_dir)
    
    print(f"\nBenchmark complete!")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    
    # Print summary
    sgr_comp = results["summary"]["sgr_comparison"]
    if "baseline" in sgr_comp and "sgr_lite" in sgr_comp:
        print(f"\nSGR Lite vs Baseline:")
        print(f"  Quality: {sgr_comp['sgr_lite']['avg_quality_score']:.2f} vs {sgr_comp['baseline']['avg_quality_score']:.2f}")
        print(f"  Success: {sgr_comp['sgr_lite']['success_rate']:.1%} vs {sgr_comp['baseline']['success_rate']:.1%}")
        print(f"  Latency: {sgr_comp['sgr_lite']['avg_latency_ms']:.0f}ms vs {sgr_comp['baseline']['avg_latency_ms']:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())