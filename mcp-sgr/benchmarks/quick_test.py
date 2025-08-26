"""Quick test script to verify benchmark setup before running full suite."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.llm_client import LLMClient
from src.tools.apply_sgr import apply_sgr_tool
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager


async def test_baseline():
    """Test basic LLM call without SGR."""
    print("Testing baseline LLM call...")
    
    client = LLMClient()
    try:
        response = await client.generate(
            "Say 'Hello, SGR benchmarks are working!' in JSON format with a 'message' field.",
            backend="openrouter",
            temperature=0.1,
            max_tokens=100
        )
        print(f"✓ Baseline response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Baseline failed: {e}")
        return False
    finally:
        await client.close()


async def test_sgr_lite():
    """Test SGR with lite budget."""
    print("\nTesting SGR lite mode...")
    
    client = LLMClient()
    cache = CacheManager()
    telemetry = TelemetryManager()
    
    # Disable cache for testing
    cache.enabled = False
    
    await cache.initialize()
    await telemetry.initialize()
    
    try:
        result = await apply_sgr_tool(
            arguments={
                "task": "List 3 benefits of using structured reasoning",
                "schema_type": "analysis",
                "budget": "lite",
                "backend": "openrouter",
            },
            llm_client=client,
            cache_manager=cache,
            telemetry=telemetry,
        )
        
        print(f"✓ SGR lite confidence: {result.get('confidence', 0):.2f}")
        print(f"✓ Valid response: {result.get('metadata', {}).get('validation', {}).get('valid', False)}")
        return True
        
    except Exception as e:
        print(f"✗ SGR lite failed: {e}")
        return False
    finally:
        await client.close()
        await cache.close()
        await telemetry.close()


async def main():
    """Run quick tests."""
    print("MCP-SGR Quick Test")
    print("==================")
    print()
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set!")
        print("Please set: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print(f"✓ OpenRouter API key found")
    
    # Set default model for testing
    os.environ["OPENROUTER_DEFAULT_MODEL"] = "google/gemini-2.0-flash-exp:free"
    print(f"✓ Using model: {os.environ['OPENROUTER_DEFAULT_MODEL']}")
    
    # Disable cache
    os.environ["CACHE_ENABLED"] = "false"
    
    # Run tests
    baseline_ok = await test_baseline()
    sgr_ok = await test_sgr_lite()
    
    print("\n" + "="*50)
    if baseline_ok and sgr_ok:
        print("✅ All tests passed! Ready to run benchmarks.")
        print("\nNext steps:")
        print("1. Run minimal benchmark: ./run_benchmarks.sh ultra_cheap analysis baseline,sgr_lite 1")
        print("2. Run full benchmark: ./run_benchmarks.sh")
    else:
        print("❌ Some tests failed. Please check:")
        print("- API key is valid")
        print("- Network connection is working")
        print("- OpenRouter has available credits")


if __name__ == "__main__":
    asyncio.run(main())