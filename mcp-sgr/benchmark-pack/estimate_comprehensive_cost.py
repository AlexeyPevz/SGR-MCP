#!/usr/bin/env python3
"""Estimate cost for comprehensive benchmark"""

# Model costs (per 1k tokens)
models = {
    "GPT-4o": {"input": 2.5, "output": 10.0, "name": "GPT-4o (Top)"},
    "Claude-3.5-Sonnet": {"input": 3.0, "output": 15.0, "name": "Claude-3.5-Sonnet (Top)"},
    "GPT-4-Turbo": {"input": 10.0, "output": 30.0, "name": "GPT-4-Turbo (Top)"},
    "GPT-3.5-Turbo": {"input": 0.5, "output": 1.5, "name": "GPT-3.5-Turbo (Mid)"},
    "Claude-3-Haiku": {"input": 0.25, "output": 1.25, "name": "Claude-3-Haiku (Mid)"},
    "Mistral-7B-Free": {"input": 0.0, "output": 0.0, "name": "Mistral-7B-Free (Free)"},
    "Ministral-8B": {"input": 0.02, "output": 0.02, "name": "Ministral-8B (Budget)"},
    "DeepSeek-Chat": {"input": 0.14, "output": 0.28, "name": "DeepSeek-Chat (Budget)"},
    "Qwen-2.5-7B": {"input": 0.15, "output": 0.15, "name": "Qwen-2.5-7B (Budget)"},
    "Qwen-2.5-72B": {"input": 0.9, "output": 0.9, "name": "Qwen-2.5-72B (Budget)"}
}

# Test parameters
num_tasks = 40
sgr_modes = 3  # off, lite, full
runs_per_task = 2
avg_input_tokens = 300
avg_output_tokens = 400

print("💰 Cost Estimation for Comprehensive SGR Benchmark")
print("="*60)
print(f"\nTest Configuration:")
print(f"- Tasks: {num_tasks}")
print(f"- SGR Modes: {sgr_modes} (off, lite, full)")
print(f"- Runs per task: {runs_per_task}")
print(f"- Total requests per model: {num_tasks * sgr_modes * runs_per_task}")
print(f"\nToken estimates:")
print(f"- Input: ~{avg_input_tokens} tokens")
print(f"- Output: ~{avg_output_tokens} tokens")

print("\n💵 Cost Breakdown by Model:")
print("-"*60)

total_cost = 0
top_models_cost = 0
budget_models_cost = 0

for model_id, costs in models.items():
    requests = num_tasks * sgr_modes * runs_per_task
    input_cost = (avg_input_tokens / 1000) * costs["input"] * requests
    output_cost = (avg_output_tokens / 1000) * costs["output"] * requests
    model_total = input_cost + output_cost
    total_cost += model_total
    
    if "Top" in costs["name"]:
        top_models_cost += model_total
    elif "Budget" in costs["name"] or "Free" in costs["name"]:
        budget_models_cost += model_total
    
    print(f"{costs['name']:.<40} ${model_total:>8.2f}")

print("-"*60)
print(f"{'TOTAL COST:':.<40} ${total_cost:>8.2f}")
print(f"\n📊 Cost Analysis:")
print(f"- Top models (3): ${top_models_cost:.2f} ({top_models_cost/total_cost*100:.0f}% of total)")
print(f"- Budget/Free models (5): ${budget_models_cost:.2f} ({budget_models_cost/total_cost*100:.0f}% of total)")
print(f"- Mid-tier models (2): ${total_cost - top_models_cost - budget_models_cost:.2f}")

print("\n🎯 Recommendations:")
print("1. Run budget models first (very cheap)")
print("2. Run mid-tier models (reasonable cost)")
print("3. Run top models selectively (expensive)")
print("\n💡 Or create a 'quick' config with fewer tasks for top models!")

# Quick config suggestion
quick_tasks = 10
quick_cost_top = 0
for model_id, costs in models.items():
    if "Top" in costs["name"]:
        requests = quick_tasks * sgr_modes * 1  # 1 run only
        model_cost = ((avg_input_tokens + avg_output_tokens) / 1000) * \
                    ((costs["input"] + costs["output"]) / 2) * requests
        quick_cost_top += model_cost

print(f"\n⚡ Quick Test Option:")
print(f"- Test top models on only {quick_tasks} tasks: ~${quick_cost_top:.2f}")
print(f"- Full test on budget models: ~${budget_models_cost:.2f}")
print(f"- Total for mixed approach: ~${quick_cost_top + budget_models_cost:.2f}")