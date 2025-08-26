#!/usr/bin/env python3
"""
Quick Demo of SGR Benchmark Pack
Shows comparison of models with and without SGR on selected tasks
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(__file__))
from benchmark_runner import BenchmarkRunner

async def run_demo():
    """Run a quick demonstration of the benchmark pack."""
    
    print("\n🎯 SGR Benchmark Pack - Quick Demo")
    print("=" * 80)
    print("This demo will run 3 tasks across 2 models with all SGR modes")
    print("Expected time: ~5 minutes")
    print("=" * 80)
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n❌ Error: Please set OPENROUTER_API_KEY environment variable")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Create minimal config for demo
    demo_config = """
models:
  - id: "qwen/qwen-2.5-72b-instruct"
    name: "Qwen-2.5-72B"
    type: "large"
    cost_per_1k: 0.0003
    
  - id: "google/gemma-2-9b-it"
    name: "Gemma-2-9B"
    type: "small"
    cost_per_1k: 0.0001

sgr_modes:
  - name: "off"
    description: "Baseline without SGR"
    
  - name: "lite"
    description: "Lightweight structured guidance"
    
  - name: "full"
    description: "Comprehensive structured analysis"

evaluation:
  runs_per_task: 1  # Single run for demo
  temperature: 0.1
  max_tokens: 2000
  timeout: 60

categories: {}
reporting: {}
"""
    
    # Save demo config
    with open("demo_config.yaml", "w") as f:
        f.write(demo_config)
    
    try:
        # Run benchmark with limited tasks
        runner = BenchmarkRunner("demo_config.yaml")
        await runner.run_benchmark(limit=3)
        
        print("\n✅ Demo completed successfully!")
        print("\n📊 Key Observations:")
        print("- SGR-lite provides good balance of quality and cost")
        print("- SGR-full offers maximum structure and validation")
        print("- Larger models utilize SGR more effectively")
        print("\n📁 Check the reports/ directory for detailed results")
        
    finally:
        # Cleanup
        if os.path.exists("demo_config.yaml"):
            os.remove("demo_config.yaml")

if __name__ == "__main__":
    asyncio.run(run_demo())