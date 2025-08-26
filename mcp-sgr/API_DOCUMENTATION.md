# 🔌 MCP-SGR API Documentation

## Overview

MCP-SGR provides multiple ways to integrate Schema-Guided Reasoning into your applications:

1. **Python API** - Direct integration
2. **MCP Server** - Model Context Protocol server
3. **HTTP API** - REST endpoints
4. **CLI** - Command-line interface

## 🐍 Python API

### Basic Usage

```python
from sgr import apply_sgr

# Simple SGR call
result = await apply_sgr(
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}],
    mode="lite"  # or "full" or "off"
)

print(result["content"])
```

### Advanced Usage

```python
from sgr import apply_sgr, SGRConfig

# Configure SGR
config = SGRConfig(
    mode="full",
    model="mistralai/mistral-7b-instruct:free",
    temperature=0.7,
    max_tokens=1500
)

# Custom schema
schema = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "code": {"type": "string"},
        "tests": {"type": "array", "items": {"type": "string"}}
    }
}

result = await apply_sgr(
    messages=messages,
    config=config,
    custom_schema=schema
)
```

### Available Functions

#### `apply_sgr(messages, mode="lite", **kwargs)`
Main function to apply SGR to messages.

**Parameters:**
- `messages` (List[Dict]): Chat messages
- `mode` (str): SGR mode - "off", "lite", or "full"
- `model` (str, optional): Override model selection
- `temperature` (float, optional): Temperature (0-1)
- `max_tokens` (int, optional): Max response tokens
- `custom_schema` (dict, optional): Custom JSON schema

**Returns:**
- Dict with `content`, `usage`, `model`, `latency`

#### `detect_task_type(messages)`
Detect the type of task from messages.

**Returns:**
- TaskType enum: ANALYSIS, CODE_GENERATION, SUMMARIZATION, etc.

#### `select_model(task_type, sgr_mode)`
Select best model for task and SGR mode.

**Returns:**
- Model ID string

## 🖥️ MCP Server

### Starting the Server

```bash
# Default (stdio)
python -m sgr.server

# With specific transport
python -m sgr.server --transport stdio
```

### MCP Tools Available

#### `apply_sgr`
```json
{
  "name": "apply_sgr",
  "description": "Apply Schema-Guided Reasoning",
  "inputSchema": {
    "type": "object",
    "properties": {
      "messages": {"type": "array"},
      "mode": {"type": "string", "enum": ["off", "lite", "full"]}
    }
  }
}
```

#### `enhance_prompt`
```json
{
  "name": "enhance_prompt",
  "description": "Enhance prompt with SGR",
  "inputSchema": {
    "type": "object",
    "properties": {
      "prompt": {"type": "string"},
      "task_type": {"type": "string"}
    }
  }
}
```

## 🌐 HTTP API

### Starting HTTP Server

```bash
# Start on default port 8000
python -m sgr.http_server

# Custom port
python -m sgr.http_server --port 8080
```

### Endpoints

#### POST `/apply_sgr`

Apply SGR to messages.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Explain quantum computing"}
  ],
  "mode": "lite"
}
```

**Response:**
```json
{
  "content": "Structured response...",
  "usage": {
    "input_tokens": 50,
    "output_tokens": 200
  },
  "model": "mistralai/mistral-7b-instruct:free",
  "latency": 1.5
}
```

#### GET `/health`

Check server health.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## 💻 CLI Usage

### Basic Commands

```bash
# Apply SGR to a prompt
sgr apply "Write a sorting function" --mode lite

# Enhance a prompt
sgr enhance "Explain AI" --task-type analysis

# Test SGR
sgr test --model mistral-7b-free
```

### CLI Options

```
Options:
  --mode {off,lite,full}  SGR mode (default: lite)
  --model MODEL          Model to use
  --temperature FLOAT    Temperature (0-1)
  --max-tokens INT       Max tokens
  --output {json,text}   Output format
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
OPENROUTER_API_KEY=your-key
OPENAI_API_KEY=your-key

# Default settings
SGR_DEFAULT_MODE=lite
SGR_DEFAULT_MODEL=mistralai/mistral-7b-instruct:free
SGR_CACHE_ENABLED=true
```

### Config File

Create `~/.sgr/config.yaml`:

```yaml
defaults:
  mode: lite
  model: mistral-7b-free
  temperature: 0.7

models:
  preferred:
    - mistralai/mistral-7b-instruct:free
    - ministral/ministral-8b-2410
  
routing:
  code_generation: qwen-2.5-72b
  analysis: claude-3-haiku
```

## 📦 Integration Examples

### LangChain Integration

```python
from langchain.llms import BaseLLM
from sgr.integrations.langchain import SGRLangChain

# Create SGR-enhanced LLM
llm = SGRLangChain(
    base_llm=your_llm,
    sgr_mode="lite"
)

# Use in chains
response = llm("Analyze this data...")
```

### FastAPI Integration

```python
from fastapi import FastAPI
from sgr import apply_sgr

app = FastAPI()

@app.post("/chat")
async def chat(messages: List[Dict]):
    result = await apply_sgr(messages, mode="lite")
    return result
```

### Gradio Integration

```python
import gradio as gr
from sgr import apply_sgr

async def process(text, mode):
    messages = [{"role": "user", "content": text}]
    result = await apply_sgr(messages, mode=mode)
    return result["content"]

interface = gr.Interface(
    fn=process,
    inputs=[
        gr.Textbox(label="Input"),
        gr.Radio(["off", "lite", "full"], label="SGR Mode")
    ],
    outputs="text"
)

interface.launch()
```

## 📊 Response Format

### Standard Response

```json
{
  "content": "The response content",
  "structured_output": {
    "task_understanding": "...",
    "solution": "..."
  },
  "usage": {
    "input_tokens": 100,
    "output_tokens": 500,
    "total_tokens": 600
  },
  "model": "mistralai/mistral-7b-instruct:free",
  "sgr_mode": "lite",
  "latency": 2.5,
  "cached": false
}
```

### Error Response

```json
{
  "error": {
    "type": "ModelError",
    "message": "Model API error",
    "details": "Rate limit exceeded"
  },
  "fallback": true,
  "fallback_model": "ministral-8b"
}
```

## 🔒 Best Practices

1. **Start with SGR-Lite** - It provides 80% of benefits with minimal overhead
2. **Use free models** - Mistral-7B-Free works great with SGR
3. **Cache responses** - Enable caching for repeated queries
4. **Handle errors** - Always have fallback models configured
5. **Monitor usage** - Track tokens and costs

## 📚 More Examples

See the `/examples` directory for:
- `basic_usage.py` - Simple SGR usage
- `custom_schema.py` - Custom schemas
- `agent_wrapper.py` - Agent integration
- `n8n-integration-guide.md` - n8n workflow