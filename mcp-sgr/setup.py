#!/usr/bin/env python3
"""
Simple setup script for MCP-SGR
"""

import os
import sys
import subprocess

def setup():
    """Setup MCP-SGR environment"""
    
    print("🚀 Setting up MCP-SGR...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    
    print("✅ Python version OK:", sys.version)
    
    # Create necessary directories
    dirs = [
        "reports",
        "cache",
        "logs"
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n⚠️  No API key found!")
        print("To use MCP-SGR, you need an OpenRouter API key:")
        print("1. Get a free key at https://openrouter.ai")
        print("2. Set it: export OPENROUTER_API_KEY='your-key'")
    else:
        print("✅ API key found")
    
    # Create a test script
    test_script = """#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.simple_sgr_demo import apply_sgr

# Quick test
result = apply_sgr("Hello, SGR!", mode="lite")
if result["success"]:
    print("✅ SGR is working!")
    print("Response:", result["content"][:100] + "...")
else:
    print("❌ Error:", result["content"])
"""
    
    with open("test_sgr.py", "w") as f:
        f.write(test_script)
    os.chmod("test_sgr.py", 0o755)
    print("✅ Created test_sgr.py")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Set your API key: export OPENROUTER_API_KEY='your-key'")
    print("2. Run the demo: python examples/simple_sgr_demo.py")
    print("3. Test SGR: python test_sgr.py")
    print("\nSee QUICKSTART_GUIDE.md for more!")
    
    return True

if __name__ == "__main__":
    setup()