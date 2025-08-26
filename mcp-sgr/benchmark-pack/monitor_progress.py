#!/usr/bin/env python3
"""Monitor benchmark progress in real-time"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def find_latest_log():
    """Find the latest benchmark log file"""
    logs = list(Path("reports").glob("benchmark_*.log"))
    if not logs:
        return None
    return max(logs, key=lambda f: f.stat().st_mtime)

def parse_log_progress(log_file):
    """Parse progress from log file"""
    stats = {
        "total": 0,
        "completed": 0,
        "success": 0,
        "failed": 0,
        "models": {},
        "categories": {},
        "sgr_modes": {"off": 0, "lite": 0, "full": 0},
        "avg_latency": 0,
        "total_cost": 0
    }
    
    latencies = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if "Task:" in line and "Model:" in line and "Mode:" in line:
            stats["total"] += 1
            
            # Extract model
            model = line.split("Model: ")[1].split(",")[0]
            if model not in stats["models"]:
                stats["models"][model] = {"total": 0, "success": 0}
            stats["models"][model]["total"] += 1
            
            # Extract category
            task = line.split("Task: ")[1].split(",")[0]
            category = task.split("_")[0]
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1
            
            # Extract mode
            mode = line.split("Mode: ")[1].strip()
            if mode in stats["sgr_modes"]:
                stats["sgr_modes"][mode] += 1
        
        elif "Result:" in line:
            stats["completed"] += 1
            
            if "✓" in line:
                stats["success"] += 1
                # Update model success
                for model in stats["models"]:
                    if model in lines[stats["completed"]-1]:
                        stats["models"][model]["success"] += 1
            else:
                stats["failed"] += 1
            
            # Extract metrics
            if "Latency:" in line:
                try:
                    latency = float(line.split("Latency: ")[1].split("s")[0])
                    latencies.append(latency)
                except:
                    pass
            
            if "Cost:" in line:
                try:
                    cost = float(line.split("Cost: $")[1].strip())
                    stats["total_cost"] += cost
                except:
                    pass
    
    if latencies:
        stats["avg_latency"] = sum(latencies) / len(latencies)
    
    return stats

def display_progress(stats):
    """Display progress in a nice format"""
    os.system('clear')
    
    print("="*80)
    print("📊 BENCHMARK PROGRESS MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Overall progress
    if stats["total"] > 0:
        progress = stats["completed"] / stats["total"] * 100
        success_rate = stats["success"] / stats["completed"] * 100 if stats["completed"] > 0 else 0
        
        print(f"Progress: {stats['completed']}/{stats['total']} ({progress:.1f}%)")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Cost: ${stats['total_cost']:.4f}")
        print(f"Avg Latency: {stats['avg_latency']:.1f}s")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\n[{bar}] {progress:.0f}%")
        
        # Model performance
        print("\n📈 Model Performance:")
        for model, data in sorted(stats["models"].items()):
            if data["total"] > 0:
                model_success = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
                print(f"  {model:<20} {data['success']}/{data['total']} ({model_success:.0f}%)")
        
        # Category coverage
        print("\n📁 Category Coverage:")
        for category, count in sorted(stats["categories"].items()):
            print(f"  {category:<20} {count} tasks")
        
        # SGR Mode distribution
        print("\n🔄 SGR Modes:")
        for mode, count in stats["sgr_modes"].items():
            print(f"  {mode:<10} {count} runs")
        
        # Estimated time remaining
        if stats["completed"] > 0 and stats["avg_latency"] > 0:
            remaining = stats["total"] - stats["completed"]
            eta_seconds = remaining * (stats["avg_latency"] + 0.5)  # +0.5s for overhead
            eta_minutes = eta_seconds / 60
            print(f"\n⏱️  Estimated time remaining: {eta_minutes:.1f} minutes")

def main():
    """Main monitoring loop"""
    print("Looking for benchmark log...")
    
    while True:
        log_file = find_latest_log()
        if log_file:
            stats = parse_log_progress(log_file)
            display_progress(stats)
            
            if stats["completed"] >= stats["total"] and stats["total"] > 0:
                print("\n✅ Benchmark Complete!")
                break
        else:
            print("Waiting for benchmark to start...")
        
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    main()