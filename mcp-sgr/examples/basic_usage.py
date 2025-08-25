"""Basic usage examples for MCP-SGR."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import apply_sgr_tool, enhance_prompt_tool
from src.utils.cache import CacheManager
from src.utils.llm_client import LLMClient
from src.utils.telemetry import TelemetryManager


async def example_apply_sgr():
    """Example of applying SGR to analyze a task."""
    print("\n=== Example: Apply SGR Analysis ===\n")

    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()

    await cache_manager.initialize()
    await telemetry.initialize()

    # Example task
    task = """
    Design a scalable API rate limiting system that can handle 100K requests per second,
    support multiple rate limiting strategies (fixed window, sliding window, token bucket),
    work across distributed servers, and provide real-time analytics.
    """

    # Apply SGR analysis
    result = await apply_sgr_tool(
        arguments={
            "task": task,
            "context": {"environment": "production", "scale": "high", "criticality": "high"},
            "schema_type": "analysis",
            "budget": "full",
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry,
    )

    print("Task:", task.strip())
    print("\nAnalysis Results:")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nKey Understanding:")
    if "reasoning" in result and "understanding" in result["reasoning"]:
        understanding = result["reasoning"]["understanding"]
        print(f"- Summary: {understanding.get('task_summary', 'N/A')}")
        print(f"- Key Aspects: {', '.join(understanding.get('key_aspects', []))}")

    print("\nIdentified Risks:")
    if "reasoning" in result and "risks" in result["reasoning"]:
        for risk in result["reasoning"]["risks"][:3]:
            if isinstance(risk, dict):
                print(f"- {risk.get('risk', 'Unknown')}: {risk.get('mitigation', 'N/A')}")

    print("\nSuggested Actions:")
    for action in result.get("suggested_actions", [])[:5]:
        print(f"- {action}")

    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def example_enhance_prompt():
    """Example of enhancing a simple prompt."""
    print("\n=== Example: Enhance Prompt ===\n")

    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()

    await cache_manager.initialize()

    # Simple prompt
    original_prompt = "Write a Python function to validate email addresses"

    print(f"Original prompt: {original_prompt}")

    # Enhance with different levels
    for level in ["minimal", "standard", "comprehensive"]:
        print(f"\n--- Enhancement Level: {level} ---")

        result = await enhance_prompt_tool(
            arguments={
                "original_prompt": original_prompt,
                "enhancement_level": level,
                "target_model": "llama3.1:8b",
            },
            llm_client=llm_client,
            cache_manager=cache_manager,
        )

        print(f"\nEnhanced prompt ({level}):")
        print(
            result["enhanced_prompt"][:500] + "..."
            if len(result["enhanced_prompt"]) > 500
            else result["enhanced_prompt"]
        )

        print(f"\nDetected intent: {result['metadata']['detected_intent']}")
        print(f"Suggested schema: {result['metadata']['suggested_schema']}")

    # Cleanup
    await llm_client.close()
    await cache_manager.close()


async def example_code_generation():
    """Example of using SGR for code generation."""
    print("\n=== Example: Code Generation with SGR ===\n")

    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()

    await cache_manager.initialize()
    await telemetry.initialize()

    # Code generation task
    task = """
    Create a Python decorator that implements retry logic with exponential backoff.
    It should support configurable max attempts, base delay, and exception types to retry on.
    Include proper error handling and logging.
    """

    # Apply code generation schema
    result = await apply_sgr_tool(
        arguments={
            "task": task,
            "context": {"language": "python", "style": "clean and documented"},
            "schema_type": "code_generation",
            "budget": "full",
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry,
    )

    print("Task:", task.strip())
    print(f"\nConfidence: {result['confidence']:.2f}")

    if "reasoning" in result:
        reasoning = result["reasoning"]

        # Show design approach
        if "design" in reasoning and "approach" in reasoning["design"]:
            print(f"\nDesign Approach: {reasoning['design']['approach']}")

        # Show implementation details
        if "implementation" in reasoning:
            impl = reasoning["implementation"]
            print(f"\nImplementation Language: {impl.get('language', 'N/A')}")
            if "dependencies" in impl:
                print(f"Dependencies: {', '.join(impl['dependencies'])}")

        # Show validation plan
        if "validation" in reasoning and "test_cases" in reasoning["validation"]:
            print("\nTest Cases:")
            for test in reasoning["validation"]["test_cases"][:3]:
                print(f"- {test}")

    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def example_decision_making():
    """Example of using SGR for decision making."""
    print("\n=== Example: Decision Making with SGR ===\n")

    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()

    await cache_manager.initialize()
    await telemetry.initialize()

    # Decision task
    task = """
    Choose the best message queue technology for our microservices architecture.
    Requirements: high throughput (50K msg/sec), exactly-once delivery, 
    easy scaling, good Python support, and operational simplicity.
    Options: RabbitMQ, Apache Kafka, Redis Streams, AWS SQS.
    """

    # Apply decision schema
    result = await apply_sgr_tool(
        arguments={
            "task": task,
            "context": {"team_size": "small", "experience": "intermediate", "cloud": "AWS"},
            "schema_type": "decision",
            "budget": "full",
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry,
    )

    print("Task:", task.strip())
    print(f"\nConfidence: {result['confidence']:.2f}")

    if "reasoning" in result:
        reasoning = result["reasoning"]

        # Show decision
        if "decision" in reasoning:
            decision = reasoning["decision"]
            print(f"\nSelected Option: {decision.get('selected_option', 'N/A')}")

            if "rationale" in decision:
                print("\nRationale:")
                for reason in decision["rationale"][:3]:
                    print(f"- {reason}")

        # Show analysis scores
        if "analysis" in reasoning and "scoring" in reasoning["analysis"]:
            print("\nOption Scores:")
            for option in reasoning["analysis"]["scoring"]:
                print(f"- {option.get('option', 'Unknown')}: {option.get('weighted_total', 0):.1f}")

    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def main():
    """Run all examples."""
    print("MCP-SGR Basic Usage Examples")
    print("=" * 50)

    # Run examples
    await example_apply_sgr()
    await example_enhance_prompt()
    await example_code_generation()
    await example_decision_making()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Note: These examples assume you have a local LLM running (e.g., Ollama)
    # or have configured OpenRouter API key in .env

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with LLM configuration")
        print("2. Started Ollama or configured OpenRouter")
        print("3. Installed all dependencies: pip install -e .")
