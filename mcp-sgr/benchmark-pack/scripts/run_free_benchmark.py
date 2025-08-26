#!/usr/bin/env python3
"""
Run SGR benchmark with FREE and ultra-cheap models
Total cost: < $0.20 for full test, $0.00 for free models only
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path

def calculate_estimated_cost(config):
    """Calculate estimated cost for the benchmark run."""
    
    models = config['models']
    tasks_count = sum(cat['count'] for cat in config['test_categories'].values())
    modes_count = len(config['sgr_modes'])
    runs_per_task = config['evaluation']['runs_per_task']
    
    print("\n💰 COST ESTIMATION")
    print("="*50)
    print(f"Tasks: {tasks_count}")
    print(f"Models: {len(models)}")
    print(f"SGR Modes: {modes_count}")
    print(f"Runs per task: {runs_per_task}")
    print(f"Total API calls: {tasks_count * len(models) * modes_count * runs_per_task}")
    
    # Estimate tokens
    avg_tokens_per_call = 2000  # Conservative estimate
    total_tokens = tasks_count * len(models) * modes_count * runs_per_task * avg_tokens_per_call
    
    print(f"\nEstimated total tokens: {total_tokens:,}")
    
    # Calculate costs
    total_cost = 0
    free_models = []
    paid_models = []
    
    for model in models:
        model_tokens = tasks_count * modes_count * runs_per_task * avg_tokens_per_call
        model_cost = (model_tokens / 1000) * model['cost_per_1k']
        
        if model['cost_per_1k'] == 0:
            free_models.append(model['name'])
        else:
            paid_models.append({
                'name': model['name'],
                'cost': model_cost,
                'cost_per_1k': model['cost_per_1k']
            })
            total_cost += model_cost
    
    print("\n🆓 FREE Models:")
    for model in free_models:
        print(f"  - {model}: $0.00")
    
    print("\n💵 Paid Models:")
    for model in paid_models:
        print(f"  - {model['name']}: ${model['cost']:.4f} (${model['cost_per_1k']}/1k tokens)")
    
    print(f"\n📊 TOTAL ESTIMATED COST: ${total_cost:.2f}")
    
    return total_cost

def create_budget_runner_script(config_file):
    """Create a modified runner that uses the budget config."""
    
    runner_script = f"""#!/usr/bin/env python3
# Auto-generated budget runner
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main runner
from benchmark_runner import main

# Override config file
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0], '--config', '{config_file}']

# Add any additional arguments
if len(original_argv) > 1:
    sys.argv.extend(original_argv[1:])

# Run with budget config
if __name__ == "__main__":
    main()
"""
    
    runner_path = "run_budget_benchmark.py"
    with open(runner_path, 'w') as f:
        f.write(runner_script)
    
    os.chmod(runner_path, 0o755)
    return runner_path

def main():
    """Set up and explain how to run free/budget benchmark."""
    
    print("\n🚀 SGR BENCHMARK - FREE & BUDGET EDITION")
    print("="*60)
    
    # Load config
    config_file = "config_free.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate costs
    total_cost = calculate_estimated_cost(config)
    
    # Show options
    print("\n📋 BENCHMARK OPTIONS:")
    print("\n1. 🆓 FREE ONLY (Mistral, Gemma, Llama)")
    print("   python run_free_benchmark.py --free-only")
    print("   Cost: $0.00")
    print("   Time: ~15 minutes")
    
    print("\n2. 💰 ULTRA BUDGET (Free + DeepSeek + Qwen)")
    print("   python run_free_benchmark.py --ultra-budget")
    print("   Cost: < $0.05")
    print("   Time: ~20 minutes")
    
    print("\n3. 📊 FULL BUDGET TEST (All 6 models)")
    print("   python run_free_benchmark.py")
    print(f"   Cost: ~${total_cost:.2f}")
    print("   Time: ~30 minutes")
    
    print("\n4. ⚡ QUICK TEST (2 tasks, 2 models)")
    print("   python run_free_benchmark.py --quick")
    print("   Cost: < $0.01")
    print("   Time: ~2 minutes")
    
    # Create runner variants
    print("\n✨ Creating benchmark scripts...")
    
    # Free only config
    free_config = config.copy()
    free_config['models'] = [m for m in config['models'] if m['cost_per_1k'] == 0]
    
    with open('config_free_only.yaml', 'w') as f:
        yaml.dump(free_config, f)
    
    # Ultra budget config
    budget_config = config.copy()
    budget_config['models'] = [m for m in config['models'] 
                              if m['cost_per_1k'] == 0 or m['name'] in ['DeepSeek-Chat', 'Qwen-2.5-7B']]
    
    with open('config_ultra_budget.yaml', 'w') as f:
        yaml.dump(budget_config, f)
    
    # Quick test config
    quick_config = config.copy()
    quick_config['test_categories'] = {
        'code_generation': {'count': 1, 'tasks': ['code_simple_001']},
        'rag_qa': {'count': 1, 'tasks': ['rag_simple_001']}
    }
    quick_config['models'] = quick_config['models'][:2]  # Just first 2 models
    quick_config['sgr_modes'] = quick_config['sgr_modes'][:2]  # Just off and lite
    
    with open('config_quick_test.yaml', 'w') as f:
        yaml.dump(quick_config, f)
    
    print("\n✅ Setup complete! Choose your benchmark option above.")
    print("\n💡 TIP: Start with option 4 (Quick Test) to verify everything works!")
    print("\n📝 Make sure you have OPENROUTER_API_KEY environment variable set")
    
    # Check for API key
    if not os.environ.get('OPENROUTER_API_KEY'):
        print("\n⚠️  WARNING: OPENROUTER_API_KEY not found in environment!")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'")
    else:
        print("\n✅ API key found. Ready to run!")

if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--free-only":
            print("\nRunning FREE ONLY benchmark...")
            os.system("python benchmark_runner.py --config config_free_only.yaml")
        elif sys.argv[1] == "--ultra-budget":
            print("\nRunning ULTRA BUDGET benchmark...")
            os.system("python benchmark_runner.py --config config_ultra_budget.yaml")
        elif sys.argv[1] == "--quick":
            print("\nRunning QUICK TEST...")
            os.system("python benchmark_runner.py --config config_quick_test.yaml")
        else:
            print("\nRunning FULL BUDGET benchmark...")
            os.system("python benchmark_runner.py --config config_free.yaml")
    else:
        main()