# MCP-SGR Benchmarking Guide

## Overview

This directory contains comprehensive benchmarking tools to compare SGR (Schema-Guided Reasoning) performance across different models and configurations.

## Quick Start

```bash
# From the mcp-sgr directory, run:
./run_benchmarks.sh
```

This will prompt for your OpenRouter API key and run benchmarks with default settings.

## Benchmark Scripts

### 1. `extended_benchmarks.py` - Comprehensive Benchmarking

The main benchmarking script that tests:
- Multiple model tiers (ultra_cheap, cheap, medium, strong, frontier)
- Different task complexities (low, medium, high)
- SGR modes: baseline (no SGR), sgr_lite, sgr_full
- Detailed metrics: quality scores, latency, cost estimation

### 2. `run_benchmarks.py` - Original Simple Benchmarks

Basic benchmarking script for quick tests with fewer models and tasks.

## Configuration Options

### Model Tiers

- **ultra_cheap**: Free or nearly free models (Groq, Gemini Flash free tier)
- **cheap**: Budget models ($0.00018-$0.00035 per 1k tokens)
- **medium**: Mid-tier models ($0.00015-$0.0035 per 1k tokens)  
- **strong**: Premium models ($0.00125-$0.003 per 1k tokens)
- **frontier**: Cutting-edge models (Claude 3.5, GPT-4o, o1-mini)

### Task Categories

- **analysis**: Code/system analysis tasks
- **planning**: Project planning and architecture tasks
- **code_generation**: Code writing tasks
- **decision**: Decision-making tasks
- **summarization**: Text summarization tasks

### SGR Modes

- **baseline**: Direct model call without SGR
- **sgr_lite**: SGR with lite budget (faster, less detailed)
- **sgr_full**: SGR with full budget (slower, more comprehensive)

## Running Custom Benchmarks

### Using the Shell Script

```bash
# Custom configuration
./run_benchmarks.sh "cheap,medium" "analysis,planning" "baseline,sgr_lite" "3"

# Parameters:
# 1. Model tiers (comma-separated)
# 2. Task categories (comma-separated)
# 3. SGR modes (comma-separated)
# 4. Max concurrent API calls
```

### Direct Python Execution

```bash
# Set up environment
export OPENROUTER_API_KEY="your-key-here"
export CACHE_ENABLED=false

# Run extended benchmarks
python benchmarks/extended_benchmarks.py \
    --model-tiers ultra_cheap,cheap,medium \
    --task-categories analysis,planning,code_generation \
    --sgr-modes baseline,sgr_lite,sgr_full \
    --max-concurrent 3 \
    --output-dir reports
```

## Understanding Results

### Report Files

Benchmarks generate two types of reports in the `reports/` directory:

1. **JSON Report** (`extended_benchmark_TIMESTAMP.json`)
   - Complete raw data
   - Detailed run information
   - For programmatic analysis

2. **Markdown Report** (`extended_benchmark_TIMESTAMP.md`)
   - Human-readable summary
   - Tables and statistics
   - Key findings and comparisons

### Key Metrics

- **Success Rate**: Percentage of valid responses according to schema
- **Quality Score**: Combined metric (0-1) based on validation, confidence, and reasoning depth
- **Confidence**: Model's self-reported confidence (0-1)
- **Latency**: Response time in milliseconds
- **Cost**: Estimated cost based on token usage
- **Reasoning Depth**: Number of reasoning steps (SGR only)

### Interpreting SGR Impact

The reports show comparisons between:
- Baseline (no SGR) vs SGR modes
- Quality improvements with SGR
- Latency/cost trade-offs
- Performance across different model tiers

## Example Results Interpretation

```markdown
## SGR Performance Comparison

| Mode | Success Rate | Avg Quality | Avg Latency | Avg Cost |
|------|-------------|------------|-------------|----------|
| baseline | 75.0% | 0.65 | 1200ms | $0.0012 |
| sgr_lite | 85.0% | 0.78 | 1800ms | $0.0018 |
| sgr_full | 92.0% | 0.85 | 3500ms | $0.0035 |
```

This shows:
- SGR improves success rate by 10-17%
- Quality scores increase by 20-30%
- Latency increases by 50-190%
- Cost increases proportionally

## Best Practices

1. **Start Small**: Test with a few models first
2. **Monitor Costs**: Use cost estimates to budget
3. **Concurrent Limits**: Keep max-concurrent low (2-3) to avoid rate limits
4. **Task Selection**: Choose tasks relevant to your use case
5. **Multiple Runs**: Run benchmarks multiple times for consistency

## Troubleshooting

### Common Issues

1. **Rate Limits**: Reduce `--max-concurrent` parameter
2. **API Errors**: Check your API key and model availability
3. **Memory Issues**: Run fewer models/tasks at once
4. **Missing Dependencies**: Run `pip install -e .` in the mcp-sgr directory

### Debug Mode

For detailed logging:
```bash
export LOG_LEVEL=DEBUG
python benchmarks/extended_benchmarks.py ...
```

## Cost Estimation

Approximate costs for full benchmark suite:

| Configuration | Estimated Cost |
|---------------|----------------|
| Minimal (cheap models, 2 tasks) | $0.05-0.10 |
| Standard (cheap+medium, 3 tasks) | $0.20-0.50 |
| Comprehensive (all tiers, all tasks) | $2.00-5.00 |

## Contributing

To add new models or tasks:

1. Edit `MODELS_EXTENDED` in `extended_benchmarks.py`
2. Add new task categories to `TASKS_EXTENDED`
3. Update cost estimates based on provider pricing

## Next Steps

After running benchmarks:

1. Analyze which models benefit most from SGR
2. Identify optimal budget modes for your use case
3. Consider cost/quality trade-offs
4. Use results to configure production deployments