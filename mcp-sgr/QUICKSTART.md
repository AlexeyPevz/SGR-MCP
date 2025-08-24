# MCP-SGR Quick Start Guide

Get up and running with MCP-SGR in 5 minutes!

## Prerequisites

- Python 3.11+
- Ollama (for local LLM) or OpenRouter API key

## Installation

### Option 1: Quick Setup Script

```bash
# Clone the repository
git clone https://github.com/your-org/mcp-sgr
cd mcp-sgr

# Run setup script
./setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Copy environment file
cp .env.example .env
```

## Configuration

Edit `.env` file:

```env
# For local Ollama
LLM_BACKENDS=ollama
OLLAMA_HOST=http://localhost:11434

# OR for OpenRouter
LLM_BACKENDS=openrouter
OPENROUTER_API_KEY=your-key-here
```

## Start Ollama (if using local LLM)

```bash
# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama3.1:8b
```

## Test Installation

```bash
# Run basic example
python examples/basic_usage.py
```

## Usage Examples

### 1. Command Line Interface

```bash
# Analyze a task
python -m src.cli analyze "Design a caching system for a web API" --schema analysis

# Enhance a prompt
python -m src.cli enhance "Write a Python function" --level comprehensive

# View cache statistics
python -m src.cli cache-stats
```

### 2. Python API

```python
import asyncio
from src.utils.llm_client import LLMClient
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager
from src.tools import apply_sgr_tool

async def analyze_task():
    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()
    
    await cache_manager.initialize()
    await telemetry.initialize()
    
    # Apply SGR analysis
    result = await apply_sgr_tool(
        arguments={
            "task": "Build a recommendation system",
            "schema_type": "analysis",
            "budget": "full"
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry
    )
    
    print(f"Confidence: {result['confidence']}")
    print(f"Actions: {result['suggested_actions']}")
    
    # Cleanup
    await llm_client.close()
    await cache_manager.close()

# Run the analysis
asyncio.run(analyze_task())
```

### 3. MCP Server Mode

```bash
# Start as MCP server (for Cursor/other MCP clients)
python -m src.server

# The server will be available via stdio transport
```

### 4. Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f mcp-sgr

# Stop services
docker-compose down
```

## Common Tasks

### Analyze Code

```python
result = await apply_sgr_tool(
    arguments={
        "task": "Review this Python code for security issues: ...",
        "schema_type": "code_generation",
        "budget": "full"
    },
    # ... clients
)
```

### Make Decisions

```python
result = await apply_sgr_tool(
    arguments={
        "task": "Choose between MongoDB and PostgreSQL for our app",
        "schema_type": "decision",
        "context": {
            "app_type": "e-commerce",
            "scale": "medium"
        }
    },
    # ... clients
)
```

### Wrap Existing Agents

```python
from src.tools import wrap_agent_call_tool

result = await wrap_agent_call_tool(
    arguments={
        "agent_endpoint": "http://localhost:8000/generate",
        "agent_request": {"prompt": "Create a REST API"},
        "sgr_config": {
            "schema_type": "code_generation",
            "pre_analysis": True,
            "post_analysis": True
        }
    },
    # ... clients
)
```

## Troubleshooting

### Ollama Connection Error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### No Models Available

```bash
# Pull required models
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b  # For code generation
```

### Cache/Database Errors

```bash
# Create data directory
mkdir -p data

# Clear cache if corrupted
rm -f data/*.db
```

## Next Steps

1. Read the [full documentation](README.md)
2. Explore more [examples](examples/)
3. Create [custom schemas](examples/custom_schema.py)
4. Integrate with your [existing agents](examples/agent_wrapper.py)
5. Set up [n8n integration](integrations/n8n-node/)

## Getting Help

- Check the [README](README.md) for detailed documentation
- Open an [issue](https://github.com/your-org/mcp-sgr/issues) for bugs
- Join our [discussions](https://github.com/your-org/mcp-sgr/discussions) for questions

Happy reasoning! 🚀