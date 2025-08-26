#!/usr/bin/env python3
"""
Visualize benchmark results with charts and detailed analysis
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

# ASCII chart generation (no external dependencies)
def create_bar_chart(data: Dict[str, float], title: str, width: int = 50) -> str:
    """Create ASCII bar chart."""
    chart = f"\n{title}\n" + "=" * (width + 20) + "\n"
    
    if not data:
        return chart + "No data available\n"
    
    max_value = max(data.values()) if data.values() else 1
    max_label_len = max(len(str(k)) for k in data.keys())
    
    for label, value in data.items():
        bar_length = int((value / max_value) * width) if max_value > 0 else 0
        bar = "█" * bar_length
        chart += f"{label:<{max_label_len}} | {bar} {value:.2f}\n"
    
    return chart

def create_comparison_table(data: List[Dict[str, Any]], columns: List[str]) -> str:
    """Create formatted comparison table."""
    if not data or not columns:
        return "No data available\n"
    
    # Calculate column widths
    col_widths = {}
    for col in columns:
        max_width = len(col)
        for row in data:
            val_len = len(str(row.get(col, "")))
            max_width = max(max_width, val_len)
        col_widths[col] = min(max_width + 2, 20)  # Cap at 20 chars
    
    # Create header
    header = "|"
    for col in columns:
        header += f" {col:<{col_widths[col]-2}} |"
    
    separator = "+" + "+".join(["-" * col_widths[col] for col in columns]) + "+"
    
    # Create rows
    table = separator + "\n" + header + "\n" + separator + "\n"
    
    for row in data:
        row_str = "|"
        for col in columns:
            val = str(row.get(col, ""))
            if len(val) > col_widths[col] - 2:
                val = val[:col_widths[col]-5] + "..."
            row_str += f" {val:<{col_widths[col]-2}} |"
        table += row_str + "\n"
    
    table += separator
    return table

def analyze_results(results_file: str):
    """Analyze and visualize benchmark results."""
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", [])
    if not results:
        print("No results found in file")
        return
    
    print("\n" + "="*80)
    print("📊 SGR BENCHMARK ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing: {results_file}")
    print(f"Generated: {data.get('timestamp', 'Unknown')}")
    print(f"Total runs: {len(results)}")
    
    # Group results by various dimensions
    by_model = {}
    by_mode = {}
    by_category = {}
    by_task = {}
    
    for r in results:
        # By model
        model = r["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)
        
        # By SGR mode
        mode = r["sgr_mode"]
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(r)
        
        # By category (extract from task_id)
        category = r["task_id"].split("_")[0]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(r)
        
        # By specific task
        task = r["task_id"]
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(r)
    
    # 1. Success Rate Analysis
    print("\n\n📈 SUCCESS RATE ANALYSIS")
    print("-" * 50)
    
    success_by_mode = {}
    for mode, results in by_mode.items():
        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        success_by_mode[mode] = success_rate
    
    print(create_bar_chart(success_by_mode, "Success Rate by SGR Mode (%)"))
    
    # 2. Quality Score Analysis
    print("\n\n🏆 QUALITY SCORE ANALYSIS")
    print("-" * 50)
    
    # Average quality by mode
    quality_by_mode = {}
    for mode, results in by_mode.items():
        successful = [r for r in results if r["success"]]
        if successful:
            avg_quality = statistics.mean(r["metrics"].get("overall", 0) for r in successful)
            quality_by_mode[mode] = avg_quality
    
    print(create_bar_chart(quality_by_mode, "Average Quality Score by SGR Mode"))
    
    # Quality improvement analysis
    print("\n📊 Quality Improvement with SGR")
    improvements = []
    
    for task, task_results in by_task.items():
        # Find baseline (off mode)
        baseline = next((r for r in task_results if r["sgr_mode"] == "off" and r["success"]), None)
        if not baseline:
            continue
            
        base_score = baseline["metrics"].get("overall", 0)
        
        # Compare with SGR modes
        for r in task_results:
            if r["sgr_mode"] != "off" and r["success"]:
                improvement = ((r["metrics"].get("overall", 0) - base_score) / base_score * 100) if base_score > 0 else 0
                improvements.append({
                    "task": task,
                    "model": r["model"],
                    "mode": r["sgr_mode"],
                    "improvement": improvement
                })
    
    # Top improvements
    top_improvements = sorted(improvements, key=lambda x: x["improvement"], reverse=True)[:10]
    
    print("\nTop 10 SGR Improvements:")
    improvement_table = create_comparison_table(
        top_improvements,
        ["task", "model", "mode", "improvement"]
    )
    print(improvement_table)
    
    # 3. Performance Analysis
    print("\n\n⏱️  PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    # Latency by mode
    latency_by_mode = {}
    for mode, results in by_mode.items():
        latencies = [r["latency"] for r in results]
        latency_by_mode[mode] = {
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "avg": statistics.mean(latencies)
        }
    
    print("\nLatency Statistics (seconds):")
    latency_data = []
    for mode, stats in latency_by_mode.items():
        latency_data.append({
            "mode": mode,
            "avg": f"{stats['avg']:.1f}",
            "p50": f"{stats['p50']:.1f}",
            "p95": f"{stats['p95']:.1f}"
        })
    
    print(create_comparison_table(latency_data, ["mode", "avg", "p50", "p95"]))
    
    # 4. Cost Analysis
    print("\n\n💰 COST ANALYSIS")
    print("-" * 50)
    
    # Cost by mode
    cost_by_mode = {}
    for mode, results in by_mode.items():
        total_cost = sum(r["cost"] for r in results)
        avg_cost = total_cost / len(results)
        cost_by_mode[mode] = avg_cost
    
    print(create_bar_chart(cost_by_mode, "Average Cost per Request by SGR Mode ($)"))
    
    # Cost efficiency (quality per dollar)
    print("\n💎 Value Analysis (Quality per Dollar)")
    value_analysis = {}
    
    for mode in by_mode:
        if mode in quality_by_mode and mode in cost_by_mode and cost_by_mode[mode] > 0:
            value = quality_by_mode[mode] / cost_by_mode[mode]
            value_analysis[mode] = value
    
    print(create_bar_chart(value_analysis, "Quality per Dollar by SGR Mode"))
    
    # 5. Model Comparison
    print("\n\n🤖 MODEL COMPARISON")
    print("-" * 50)
    
    model_stats = []
    for model, results in by_model.items():
        successful = [r for r in results if r["success"]]
        
        stats = {
            "model": model,
            "success_rate": f"{len(successful)/len(results)*100:.1f}%",
            "avg_quality": f"{statistics.mean(r['metrics'].get('overall', 0) for r in successful):.2f}" if successful else "N/A",
            "avg_latency": f"{statistics.mean(r['latency'] for r in results):.1f}s",
            "total_cost": f"${sum(r['cost'] for r in results):.2f}"
        }
        model_stats.append(stats)
    
    print(create_comparison_table(model_stats, ["model", "success_rate", "avg_quality", "avg_latency", "total_cost"]))
    
    # 6. Category Analysis
    print("\n\n📂 CATEGORY ANALYSIS")
    print("-" * 50)
    
    for category, results in by_category.items():
        successful = [r for r in results if r["success"]]
        if not successful:
            continue
            
        print(f"\n{category.upper()}:")
        
        # SGR effectiveness by category
        mode_scores = {}
        for mode in ["off", "lite", "full"]:
            mode_results = [r for r in successful if r["sgr_mode"] == mode]
            if mode_results:
                avg_score = statistics.mean(r["metrics"].get("overall", 0) for r in mode_results)
                mode_scores[mode] = avg_score
        
        if mode_scores:
            # Calculate improvements
            if "off" in mode_scores:
                base = mode_scores["off"]
                for mode in ["lite", "full"]:
                    if mode in mode_scores and base > 0:
                        improvement = ((mode_scores[mode] - base) / base) * 100
                        print(f"  SGR-{mode} improvement: {improvement:+.1f}%")
    
    # 7. Key Insights
    print("\n\n🔍 KEY INSIGHTS")
    print("-" * 50)
    
    # Best performing model
    best_model_quality = max(by_model.items(), 
                            key=lambda x: statistics.mean(r["metrics"].get("overall", 0) 
                                                        for r in x[1] if r["success"]))
    print(f"\n1. Best Model for Quality: {best_model_quality[0]}")
    
    # Best value model
    model_values = {}
    for model, results in by_model.items():
        successful = [r for r in results if r["success"]]
        if successful:
            avg_quality = statistics.mean(r["metrics"].get("overall", 0) for r in successful)
            avg_cost = statistics.mean(r["cost"] for r in results)
            if avg_cost > 0:
                model_values[model] = avg_quality / avg_cost
    
    if model_values:
        best_value_model = max(model_values.items(), key=lambda x: x[1])
        print(f"2. Best Value Model: {best_value_model[0]} ({best_value_model[1]:.0f} quality/$)")
    
    # SGR effectiveness
    if "off" in quality_by_mode:
        base_quality = quality_by_mode["off"]
        sgr_improvements = []
        for mode in ["lite", "full"]:
            if mode in quality_by_mode and base_quality > 0:
                imp = ((quality_by_mode[mode] - base_quality) / base_quality) * 100
                sgr_improvements.append((mode, imp))
        
        if sgr_improvements:
            best_sgr = max(sgr_improvements, key=lambda x: x[1])
            print(f"3. Most Effective SGR Mode: {best_sgr[0]} (+{best_sgr[1]:.1f}% quality)")
    
    # Task complexity
    task_difficulties = {}
    for task, results in by_task.items():
        successful = [r for r in results if r["success"]]
        if successful:
            # Tasks where SGR helps most
            baseline = next((r["metrics"].get("overall", 0) for r in successful if r["sgr_mode"] == "off"), 0)
            sgr_scores = [r["metrics"].get("overall", 0) for r in successful if r["sgr_mode"] != "off"]
            if sgr_scores and baseline > 0:
                avg_improvement = statistics.mean((s - baseline) / baseline * 100 for s in sgr_scores)
                task_difficulties[task] = avg_improvement
    
    if task_difficulties:
        best_sgr_task = max(task_difficulties.items(), key=lambda x: x[1])
        print(f"4. Task Most Improved by SGR: {best_sgr_task[0]} (+{best_sgr_task[1]:.1f}%)")
    
    # Save analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"analysis_report_{timestamp}.txt"
    
    print(f"\n\n💾 Analysis saved to: {report_file}")

def main():
    """Run visualization on latest results."""
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Find latest results file
        results_files = list(Path(".").glob("reports/benchmark_results_*.json"))
        if not results_files:
            print("❌ No results files found. Run benchmark first.")
            return
        
        results_file = max(results_files, key=lambda f: f.stat().st_mtime)
        print(f"Using latest results: {results_file}")
    
    analyze_results(str(results_file))

if __name__ == "__main__":
    main()