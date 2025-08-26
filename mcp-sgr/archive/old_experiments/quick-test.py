#!/usr/bin/env python3
"""Quick test to check if core components can be imported."""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing MCP-SGR imports...")

try:
    # Test core imports
    print("✓ Testing schemas...")
    from src.schemas import SCHEMA_REGISTRY
    from src.schemas.base import BaseSchema
    print(f"  Found {len(SCHEMA_REGISTRY)} schemas: {list(SCHEMA_REGISTRY.keys())}")
    
    print("\n✓ Testing tools...")
    from src.tools import apply_sgr_tool, enhance_prompt_tool
    
    print("\n✓ Testing utils...")
    from src.utils.llm_client import LLMClient
    from src.utils.cache import CacheManager
    
    print("\n✓ Testing server...")
    from src.server import SGRServer
    
    print("\n✓ Testing HTTP server...")
    from src.http_server import app
    
    print("\n✅ All core imports successful!")
    print("\nProject structure looks good.")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nMissing dependencies. Run: pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)