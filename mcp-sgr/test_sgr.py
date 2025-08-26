#!/usr/bin/env python3
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
