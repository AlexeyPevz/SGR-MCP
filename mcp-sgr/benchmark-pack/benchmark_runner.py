#!/usr/bin/env python3
"""
SGR Benchmark Runner - Main execution script
Supports multiple configurations and cost tracking
"""

import json
import os
import sys
import time
import yaml
import argparse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import traceback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from eval.metrics import (
        RAGASMetrics, 
        CodeQualityMetrics,
        SummarizationMetrics,
        WorkflowMetrics,
        calculate_composite_score
    )
except ImportError:
    print("Warning: Advanced metrics not available, using basic scoring")
    RAGASMetrics = None

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    models: List[Dict[str, Any]]
    sgr_modes: List[Dict[str, Any]]
    test_categories: Dict[str, Dict[str, Any]]
    evaluation: Dict[str, Any]
    reporting: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

@dataclass
class TaskResult:
    """Result from a single task execution."""
    task_id: str
    model: str
    sgr_mode: str
    success: bool
    response: str
    metrics: Dict[str, float] = field(default_factory=dict)
    latency: float = 0.0
    cost: float = 0.0
    error: Optional[str] = None
    tokens_used: int = 0

def load_tasks(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """Load tasks based on configuration."""
    tasks = []
    
    for category, cat_config in config.test_categories.items():
        # Map category names to file names
        category_files = {
            'code_generation': 'code_generation.yaml',
            'rag_qa': 'rag_qa.yaml',
            'summarization': 'summarization.yaml',
            'planning_decision': 'planning_decision.yaml',
            'data_etl': 'data_etl.yaml',
            'agent_workflow': 'agent_workflow.yaml'
        }
        
        task_file = Path('tasks') / category_files.get(category, f'{category}.yaml')
        
        if task_file.exists():
            with open(task_file, 'r') as f:
                all_tasks = yaml.safe_load(f)
            
            # Filter tasks based on config
            if 'tasks' in cat_config:
                # Specific tasks listed
                selected_tasks = [t for t in all_tasks if t['id'] in cat_config['tasks']]
            else:
                # Take first N tasks
                selected_tasks = all_tasks[:cat_config.get('count', len(all_tasks))]
            
            tasks.extend(selected_tasks)
    
    return tasks

def call_model(task: Dict[str, Any], model_config: Dict[str, Any], 
               sgr_mode: str, sgr_config: Dict[str, Any],
               api_key: str) -> TaskResult:
    """Call model with task and return result."""
    
    start_time = time.time()
    
    try:
        # Prepare prompt based on SGR mode
        if sgr_mode == "off":
            prompt = task['prompt']
        else:
            # Add SGR instructions
            schema_fields = sgr_config.get('schema_fields', [])
            schema_prompt = f"""
Please structure your response with the following sections:
{chr(10).join(f'- {field}' for field in schema_fields)}

Provide your response in JSON format.
"""
            prompt = f"{task['prompt']}\n\n{schema_prompt}"
        
        # Prepare request data
        data = json.dumps({
            "model": model_config['id'],
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }).encode('utf-8')
        
        # Call OpenRouter API using urllib
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            return TaskResult(
                task_id=task['id'],
                model=model_config['name'],
                sgr_mode=sgr_mode,
                success=False,
                response="",
                error=f"API error {e.code}: {error_body}",
                latency=time.time() - start_time
            )
        
        content = result['choices'][0]['message']['content']
        tokens = result.get('usage', {}).get('total_tokens', 0)
        
        # Calculate cost
        cost = (tokens / 1000) * model_config['cost_per_1k']
        
        # Evaluate response
        metrics = evaluate_response(task, content, sgr_mode)
        
        return TaskResult(
            task_id=task['id'],
            model=model_config['name'],
            sgr_mode=sgr_mode,
            success=True,
            response=content,
            metrics=metrics,
            latency=time.time() - start_time,
            cost=cost,
            tokens_used=tokens
        )
        
    except Exception as e:
        return TaskResult(
            task_id=task['id'],
            model=model_config['name'],
            sgr_mode=sgr_mode,
            success=False,
            response="",
            error=str(e),
            latency=time.time() - start_time
        )

def evaluate_response(task: Dict[str, Any], response: str, sgr_mode: str) -> Dict[str, float]:
    """Evaluate response based on task type."""
    
    metrics = {}
    category = task['category']
    
    # Basic scoring
    if response:
        metrics['response_length'] = len(response)
        metrics['has_content'] = 1.0
    else:
        metrics['response_length'] = 0
        metrics['has_content'] = 0.0
        metrics['overall'] = 0.0
        return metrics
    
    # Category-specific evaluation
    if category == 'code_generation' or category == 'codegen':
        # Check for code presence
        has_code = '```' in response or 'def ' in response or 'function ' in response
        metrics['has_code'] = 1.0 if has_code else 0.0
        
        # Check for test mention
        has_tests = 'test' in response.lower() or 'assert' in response
        metrics['has_tests'] = 1.0 if has_tests else 0.5
        
        metrics['overall'] = (metrics['has_code'] + metrics['has_tests']) / 2
        
    elif category == 'rag_qa':
        # Check for answer and citations
        metrics['has_answer'] = 1.0 if len(response) > 50 else 0.5
        metrics['has_citations'] = 1.0 if any(x in response for x in ['[1]', 'source:', 'according to']) else 0.0
        
        metrics['overall'] = (metrics['has_answer'] + metrics['has_citations']) / 2
        
    elif category == 'summarization':
        # Check compression and key points
        original_length = len(task.get('document', {}).get('content', ''))
        if original_length > 0:
            metrics['compression_ratio'] = len(response) / original_length
            metrics['good_compression'] = 1.0 if 0.1 < metrics['compression_ratio'] < 0.5 else 0.5
        else:
            metrics['good_compression'] = 0.5
            
        metrics['overall'] = metrics['good_compression']
        
    else:
        # Generic evaluation
        metrics['overall'] = 0.7 if len(response) > 100 else 0.3
    
    # Boost for SGR structure
    if sgr_mode != "off":
        try:
            # Check if response is valid JSON
            json.loads(response)
            metrics['structured'] = 1.0
            metrics['overall'] = metrics['overall'] * 1.2  # 20% boost
        except:
            metrics['structured'] = 0.0
    
    # Ensure overall is between 0 and 1
    metrics['overall'] = min(1.0, max(0.0, metrics['overall']))
    
    return metrics

def run_benchmark(config: BenchmarkConfig, api_key: str, limit: Optional[int] = None) -> List[TaskResult]:
    """Run the complete benchmark."""
    
    results = []
    tasks = load_tasks(config)
    
    if limit:
        tasks = tasks[:limit]
    
    total_runs = len(tasks) * len(config.models) * len(config.sgr_modes)
    current_run = 0
    
    print(f"\n🚀 Starting benchmark with {len(tasks)} tasks, {len(config.models)} models, {len(config.sgr_modes)} SGR modes")
    print(f"Total runs: {total_runs}")
    
    # Create reports directory
    Path('reports').mkdir(exist_ok=True)
    
    # Log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path('reports') / f'benchmark_{timestamp}.log'
    
    for task in tasks:
        for model in config.models:
            for sgr_mode_config in config.sgr_modes:
                current_run += 1
                sgr_mode = sgr_mode_config['name']
                
                print(f"\n[{current_run}/{total_runs}] Running {task['id']} with {model['name']} in {sgr_mode} mode...")
                
                # Log progress
                with open(log_file, 'a') as f:
                    f.write(f"[{datetime.now()}] Task: {task['id']}, Model: {model['name']}, Mode: {sgr_mode}\n")
                
                # Run task
                result = call_model(task, model, sgr_mode, sgr_mode_config, api_key)
                results.append(result)
                
                # Log result
                with open(log_file, 'a') as f:
                    f.write(f"  Result: {'✓' if result.success else '✗'}, "
                           f"Score: {result.metrics.get('overall', 0):.2f}, "
                           f"Latency: {result.latency:.1f}s, "
                           f"Cost: ${result.cost:.4f}\n")
                
                if result.success:
                    print(f"  ✓ Success! Score: {result.metrics.get('overall', 0):.2f}, "
                          f"Latency: {result.latency:.1f}s, Cost: ${result.cost:.4f}")
                else:
                    print(f"  ✗ Failed: {result.error}")
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
    
    return results

def generate_report(results: List[TaskResult], config: BenchmarkConfig, output_dir: str = "reports"):
    """Generate comprehensive report from results."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate summary statistics
    total_cost = sum(r.cost for r in results)
    total_tokens = sum(r.tokens_used for r in results)
    success_rate = sum(1 for r in results if r.success) / len(results) * 100 if results else 0
    
    # Group results
    by_mode = {}
    by_model = {}
    
    for r in results:
        # By mode
        if r.sgr_mode not in by_mode:
            by_mode[r.sgr_mode] = []
        by_mode[r.sgr_mode].append(r)
        
        # By model
        if r.model not in by_model:
            by_model[r.model] = []
        by_model[r.model].append(r)
    
    # Generate JSON report
    json_report = {
        "timestamp": timestamp,
        "summary": {
            "total_runs": len(results),
            "success_rate": success_rate,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "average_latency": sum(r.latency for r in results) / len(results) if results else 0
        },
        "by_sgr_mode": {},
        "by_model": {},
        "results": [vars(r) for r in results]
    }
    
    # Calculate mode statistics
    for mode, mode_results in by_mode.items():
        successful = [r for r in mode_results if r.success]
        json_report["by_sgr_mode"][mode] = {
            "success_rate": len(successful) / len(mode_results) * 100 if mode_results else 0,
            "average_score": sum(r.metrics.get('overall', 0) for r in successful) / len(successful) if successful else 0,
            "average_latency": sum(r.latency for r in mode_results) / len(mode_results) if mode_results else 0,
            "total_cost": sum(r.cost for r in mode_results)
        }
    
    # Calculate model statistics
    for model, model_results in by_model.items():
        successful = [r for r in model_results if r.success]
        json_report["by_model"][model] = {
            "success_rate": len(successful) / len(model_results) * 100 if model_results else 0,
            "average_score": sum(r.metrics.get('overall', 0) for r in successful) / len(successful) if successful else 0,
            "average_latency": sum(r.latency for r in model_results) / len(model_results) if model_results else 0,
            "total_cost": sum(r.cost for r in model_results)
        }
    
    # Save JSON report
    json_path = Path(output_dir) / f"benchmark_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    # Generate Markdown report
    md_report = f"""# SGR Benchmark Report
Generated: {timestamp}

## 📊 Summary

- **Total Runs**: {len(results)}
- **Success Rate**: {success_rate:.1f}%
- **Total Cost**: ${total_cost:.4f}
- **Total Tokens**: {total_tokens:,}
- **Average Latency**: {json_report['summary']['average_latency']:.1f}s

## 🔄 Performance by SGR Mode

| Mode | Success Rate | Avg Score | Avg Latency | Total Cost |
|------|-------------|-----------|-------------|------------|
"""
    
    for mode, stats in json_report["by_sgr_mode"].items():
        md_report += f"| {mode} | {stats['success_rate']:.1f}% | {stats['average_score']:.2f} | {stats['average_latency']:.1f}s | ${stats['total_cost']:.4f} |\n"
    
    md_report += "\n## 🤖 Performance by Model\n\n"
    md_report += "| Model | Success Rate | Avg Score | Avg Latency | Total Cost |\n"
    md_report += "|-------|-------------|-----------|-------------|------------|\n"
    
    for model, stats in json_report["by_model"].items():
        md_report += f"| {model} | {stats['success_rate']:.1f}% | {stats['average_score']:.2f} | {stats['average_latency']:.1f}s | ${stats['total_cost']:.4f} |\n"
    
    # Save Markdown report
    md_path = Path(output_dir) / f"benchmark_report_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    print(f"\n✅ Reports saved:")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")
    
    return json_path, md_path

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Run SGR Benchmark")
    parser.add_argument('--config', default='config_free.yaml', help='Configuration file')
    parser.add_argument('--limit', type=int, help='Limit number of tasks')
    parser.add_argument('--api-key', help='OpenRouter API key (or use env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: No API key provided. Set OPENROUTER_API_KEY or use --api-key")
        sys.exit(1)
    
    # Load configuration
    config = BenchmarkConfig.from_yaml(args.config)
    
    print(f"📋 Loaded configuration from {args.config}")
    
    # Run benchmark
    results = run_benchmark(config, api_key, args.limit)
    
    # Generate reports
    if results:
        generate_report(results, config)
        
        # Show summary
        print("\n" + "="*60)
        print("📊 BENCHMARK COMPLETE!")
        print("="*60)
        
        total_cost = sum(r.cost for r in results)
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        
        print(f"\nTotal cost: ${total_cost:.4f}")
        print(f"Success rate: {success_rate:.1f}%")
        
        # Show best model (only if we have successful results)
        successful_results = [r for r in results if r.success]
        if successful_results:
            by_model = {}
            for r in successful_results:
                if r.model not in by_model:
                    by_model[r.model] = []
                by_model[r.model].append(r.metrics.get('overall', 0))
            
            if by_model:
                best_model = max(by_model.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
                print(f"Best model: {best_model[0]} (avg score: {sum(best_model[1])/len(best_model[1]):.2f})")

if __name__ == "__main__":
    main()