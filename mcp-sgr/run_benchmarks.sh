#!/bin/bash

# MCP-SGR Benchmark Runner Script
# This script sets up the environment and runs benchmarks

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}MCP-SGR Benchmark Runner${NC}"
echo "========================="
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Not in mcp-sgr directory${NC}"
    echo "Please run this script from the mcp-sgr directory"
    exit 1
fi

# Setup Python environment
echo -e "${YELLOW}Setting up Python environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv || {
        echo -e "${RED}Failed to create virtual environment${NC}"
        echo "Please install python3-venv package"
        exit 1
    }
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -e . || {
    echo -e "${RED}Failed to install dependencies${NC}"
    exit 1
}

# Create reports directory
mkdir -p reports

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${YELLOW}OpenRouter API key not found in environment${NC}"
    echo "Please enter your OpenRouter API key (it will not be saved):"
    read -s OPENROUTER_API_KEY
    export OPENROUTER_API_KEY
    echo
fi

# Default configuration
DEFAULT_MODEL_TIERS="cheap,medium"
DEFAULT_TASK_CATEGORIES="analysis,planning"
DEFAULT_SGR_MODES="baseline,sgr_lite,sgr_full"
DEFAULT_MAX_CONCURRENT="2"

# Parse command line arguments
MODEL_TIERS="${1:-$DEFAULT_MODEL_TIERS}"
TASK_CATEGORIES="${2:-$DEFAULT_TASK_CATEGORIES}"
SGR_MODES="${3:-$DEFAULT_SGR_MODES}"
MAX_CONCURRENT="${4:-$DEFAULT_MAX_CONCURRENT}"

# Show configuration
echo -e "${GREEN}Benchmark Configuration:${NC}"
echo "- Model Tiers: $MODEL_TIERS"
echo "- Task Categories: $TASK_CATEGORIES"
echo "- SGR Modes: $SGR_MODES"
echo "- Max Concurrent: $MAX_CONCURRENT"
echo

# Set environment variables
export CACHE_ENABLED=false
export LLM_BACKENDS=openrouter
export ROUTER_DEFAULT_BACKEND=openrouter

# Run benchmarks
echo -e "${YELLOW}Starting benchmarks...${NC}"
echo "This may take several minutes depending on the configuration."
echo

# Run the extended benchmarks
python benchmarks/extended_benchmarks.py \
    --model-tiers "$MODEL_TIERS" \
    --task-categories "$TASK_CATEGORIES" \
    --sgr-modes "$SGR_MODES" \
    --max-concurrent "$MAX_CONCURRENT" \
    --output-dir reports

# Check if benchmark was successful
if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}Benchmark completed successfully!${NC}"
    echo "Check the reports directory for results."
    
    # Show latest report files
    echo
    echo "Latest reports:"
    ls -lt reports/extended_benchmark_*.md | head -1
    ls -lt reports/extended_benchmark_*.json | head -1
else
    echo -e "${RED}Benchmark failed!${NC}"
    exit 1
fi

# Deactivate virtual environment
deactivate