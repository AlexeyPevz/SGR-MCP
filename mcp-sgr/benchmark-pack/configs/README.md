# 📋 Benchmark Configuration Guide

## Standard Configuration

Use `standard_config.yaml` as a template for all benchmark configurations.

## Configuration Structure

```yaml
models:           # List of models to test
sgr_modes:        # SGR modes (off/lite/full)
categories:       # Task categories to test
test_settings:    # Test parameters
evaluation:       # Evaluation metrics
reporting:        # Output format settings
```

## Quick Configs

### 1. Quick Test (5 minutes)
```yaml
# Use: config_quick_test.yaml
- 1 free model (Mistral-7B)
- 3 tasks
- All SGR modes
- Cost: $0
```

### 2. Standard Test (30 minutes)
```yaml
# Use: standard_config.yaml
- 5 models (free + budget)
- 15 tasks
- All SGR modes
- Cost: ~$0.10
```

### 3. Comprehensive Test (2 hours)
```yaml
# Use: config_comprehensive.yaml
- 10 models
- 40 tasks
- All SGR modes
- Cost: ~$2.00
```

## Model Recommendations

### Free Models (Best Value)
- `mistralai/mistral-7b-instruct:free` ⭐ Best free model
- `meta-llama/llama-3.2-3b-instruct:free`

### Budget Models
- `ministral/ministral-8b-2410` - $0.02/1k tokens
- `deepseek/deepseek-chat` - $0.14/1k tokens
- `qwen/qwen-2.5-7b-instruct` - $0.15/1k tokens

### Premium Models (Test Sparingly)
- `openai/gpt-3.5-turbo` - $0.50/1k tokens
- `openai/gpt-4o` - $2.50/1k tokens

## Usage

```bash
# Run with standard config
python scripts/benchmark_runner.py --config configs/standard_config.yaml

# Quick test
python scripts/benchmark_runner.py --config configs/config_quick_test.yaml --limit 5

# Custom config
cp configs/standard_config.yaml configs/my_config.yaml
# Edit my_config.yaml
python scripts/benchmark_runner.py --config configs/my_config.yaml
```