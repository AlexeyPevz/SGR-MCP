#!/bin/bash
# Quick setup script for using MCP-SGR with Gemini

echo "=== MCP-SGR + Gemini Quick Setup ==="
echo

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY environment variable is not set"
    echo "Please set it: export GEMINI_API_KEY=your-key-here"
    exit 1
fi

echo "✓ Gemini API key found"

# Create .env file for MCP-SGR
cat > .env << EOF
# Gemini Configuration via Custom Backend
LLM_BACKENDS=custom
ROUTER_DEFAULT_BACKEND=custom
CUSTOM_LLM_URL=http://localhost:8001/v1/chat/completions

# SGR Configuration
SGR_BUDGET_DEPTH=lite
SGR_PRE_ANALYSIS=auto
SGR_POST_ANALYSIS=lite

# Cache Configuration
CACHE_ENABLED=true
CACHE_STORE=sqlite:///./data/cache.db

# HTTP Server
HTTP_ENABLED=true
HTTP_PORT=8080
HTTP_HOST=0.0.0.0
HTTP_REQUIRE_AUTH=false

# Logging
LOG_LEVEL=INFO
EOF

echo "✓ Created .env configuration"

# Create directories
mkdir -p data logs
echo "✓ Created data directories"

# Install dependencies if not in Docker
if [ ! -f "/.dockerenv" ]; then
    echo "Installing Python dependencies..."
    pip install google-generativeai fastapi uvicorn
fi

echo
echo "=== Setup Complete! ==="
echo
echo "To start MCP-SGR with Gemini:"
echo "1. In terminal 1: python examples/gemini_proxy.py"
echo "2. In terminal 2: python -m src.http_server"
echo
echo "Then test with:"
echo "curl -X POST http://localhost:8080/v1/apply-sgr \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"task\": \"Analyze this Python code for security issues\", \"schema_type\": \"analysis\"}'"
echo