#!/bin/bash
# Setup script for MCP-SGR

echo "Setting up MCP-SGR..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "Error: Python 3.11+ is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing MCP-SGR..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs

# Copy environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if models are available
    echo "Checking Ollama models..."
    if ollama list | grep -q "llama3.1:8b"; then
        echo "✓ llama3.1:8b model is available"
    else
        echo "⚠️  llama3.1:8b model not found. Run: ollama pull llama3.1:8b"
    fi
else
    echo "⚠️  Ollama not found. Install from https://ollama.ai"
fi

# Run basic tests
echo "Running basic tests..."
if python -m pytest tests/test_schemas.py -v; then
    echo "✓ Tests passed"
else
    echo "⚠️  Some tests failed"
fi

echo ""
echo "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Start Ollama: ollama serve"
echo "3. Pull required models: ollama pull llama3.1:8b"
echo "4. Run examples: python examples/basic_usage.py"
echo "5. Start MCP server: python -m src.server"
echo ""
echo "For more information, see README.md"