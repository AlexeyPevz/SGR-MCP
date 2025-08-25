# MCP-SGR Integrations

This directory contains integrations with popular AI agent frameworks.

## Available Integrations

### ✅ LangChain
- Location: `langchain/`
- Features:
  - SGRRunnable for direct SGR analysis
  - SGRChainWrapper to enhance existing chains
  - Specialized chains (Analysis, Planning, Decision)
- Example: `langchain/example_langchain.py`

### ✅ AutoGen
- Location: `autogen/`
- Features:
  - SGRAgent with built-in reasoning
  - SGRGroupChatManager for conversation analysis
  - Wrapper for existing AutoGen agents
- Example: `autogen/example_autogen.py`

### ✅ CrewAI
- Location: `crewai/`
- Features:
  - SGRAgent with role-specific schemas
  - SGRCrew with orchestration analysis
  - Task enhancement utilities
- Example: `crewai/example_crewai.py`

### ✅ n8n
- Location: `n8n/`
- Features:
  - HTTP API integration guide
  - Ready-to-import workflow examples
  - No custom node required
- Documentation: `n8n/README.md`

## Quick Start

### LangChain
```python
from integrations.langchain import create_sgr_chain

chain = create_sgr_chain(schema_type="analysis")
result = chain.invoke("Analyze the security of this code...")
```

### AutoGen
```python
from integrations.autogen import create_sgr_assistant

assistant = create_sgr_assistant(
    name="Analyst",
    system_message="You analyze problems",
    schema_type="analysis"
)
```

### CrewAI
```python
from integrations.crewai import create_sgr_agent

agent = create_sgr_agent(
    role="Researcher",
    goal="Conduct analysis",
    backstory="Expert researcher",
    schema_type="analysis"
)
```

### n8n
Use HTTP Request node with:
- URL: `http://your-server:8080/v1/apply-sgr`
- Method: POST
- Body: `{"task": "...", "schema_type": "auto"}`

## Installation

Each integration may require its framework:
```bash
# LangChain
pip install langchain langchain-community

# AutoGen
pip install pyautogen

# CrewAI
pip install crewai

# n8n - no installation needed, uses HTTP API
```

## Contributing

To add a new integration:
1. Create a directory with framework name
2. Add main integration module
3. Include usage examples
4. Update this README

PR welcome!
