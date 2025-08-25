"""Example of creating and using custom SGR schemas."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas.custom import SchemaBuilder, create_bug_report_schema
from src.tools import apply_sgr_tool, learn_schema_tool
from src.utils.cache import CacheManager
from src.utils.llm_client import LLMClient
from src.utils.telemetry import TelemetryManager


async def example_custom_schema_builder():
    """Example of building a custom schema programmatically."""
    print("\n=== Example: Custom Schema Builder ===\n")

    # Build a custom schema for API endpoint documentation
    builder = SchemaBuilder(
        schema_id="api_endpoint_doc", description="Schema for documenting REST API endpoints"
    )

    # Add fields
    builder.add_field(
        name="endpoint",
        field_type="string",
        description="The API endpoint path",
        required=True,
        pattern=r"^/[a-zA-Z0-9/_-]+$",
    )

    builder.add_field(
        name="method",
        field_type="string",
        description="HTTP method",
        required=True,
        enum=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )

    builder.add_field(
        name="description",
        field_type="string",
        description="What this endpoint does",
        required=True,
        min_length=20,
    )

    builder.add_object_field(
        name="request",
        properties={
            "headers": {"type": "object", "description": "Required headers"},
            "query_params": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "required": {"type": "boolean"},
                        "description": {"type": "string"},
                    },
                },
            },
            "body_schema": {"type": "object", "description": "Request body JSON schema"},
        },
        required=True,
    )

    builder.add_object_field(
        name="response",
        properties={
            "success_codes": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Successful HTTP status codes",
            },
            "error_codes": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Possible error codes",
            },
            "schema": {"type": "object", "description": "Response body schema"},
        },
        required=True,
    )

    builder.add_array_field(
        name="examples",
        item_type="object",
        description="Request/response examples",
        required=False,
        min_items=1,
    )

    # Build the schema
    api_schema = builder.build()

    print("Created Custom Schema:")
    print(json.dumps(api_schema.to_json_schema(), indent=2))

    # Use the custom schema
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()

    await cache_manager.initialize()
    await telemetry.initialize()

    task = """
    Document the user profile update endpoint:
    - Updates user profile information
    - Requires authentication
    - Validates email format
    - Returns updated profile
    """

    result = await apply_sgr_tool(
        arguments={
            "task": task,
            "schema_type": "custom",
            "custom_schema": api_schema.to_json_schema(),
            "budget": "full",
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry,
    )

    print("\n--- Applied Custom Schema ---")
    print(f"Confidence: {result['confidence']:.2f}")

    if "reasoning" in result:
        reasoning = result["reasoning"]
        print(f"\nEndpoint: {reasoning.get('endpoint', 'N/A')}")
        print(f"Method: {reasoning.get('method', 'N/A')}")
        print(f"Description: {reasoning.get('description', 'N/A')}")

    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def example_bug_report_schema():
    """Example using the pre-built bug report schema."""
    print("\n=== Example: Bug Report Schema ===\n")

    # Create bug report schema
    bug_schema = create_bug_report_schema()

    print("Bug Report Schema Fields:")
    for field in bug_schema.get_fields()[:5]:
        print(f"- {field.name}: {field.type} {'(required)' if field.required else '(optional)'}")

    # Initialize components
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry = TelemetryManager()

    await cache_manager.initialize()
    await telemetry.initialize()

    # Bug report to analyze
    bug_report = """
    Users are reporting that the shopping cart is not updating when they add items.
    This happens intermittently, about 30% of the time. When it fails, the UI shows
    a loading spinner that never disappears. The browser console shows a 504 Gateway
    Timeout error. This started happening after the last deployment on Friday.
    Mobile app is not affected, only web.
    """

    result = await apply_sgr_tool(
        arguments={
            "task": f"Analyze this bug report: {bug_report}",
            "schema_type": "custom",
            "custom_schema": bug_schema.to_json_schema(),
            "budget": "full",
        },
        llm_client=llm_client,
        cache_manager=cache_manager,
        telemetry=telemetry,
    )

    print("\n--- Bug Analysis Results ---")
    print(f"Confidence: {result['confidence']:.2f}")

    if "reasoning" in result:
        reasoning = result["reasoning"]

        # Issue identification
        if "issue_identification" in reasoning:
            issue = reasoning["issue_identification"]
            print(f"\nIssue: {issue.get('title', 'N/A')}")
            print(f"Severity: {issue.get('severity', 'N/A')}")
            print(f"Category: {issue.get('category', 'N/A')}")

        # Reproduction info
        if "reproduction" in reasoning:
            repro = reasoning["reproduction"]
            print(f"\nFrequency: {repro.get('frequency', 'N/A')}")
            if "steps" in repro:
                print("Steps to reproduce:")
                for step in repro["steps"][:3]:
                    print(f"  - {step}")

        # Recommendation
        if "recommendation" in reasoning:
            rec = reasoning["recommendation"]
            print(f"\nPriority: {rec.get('priority', 'N/A')}")
            print(f"Suggested fix: {rec.get('fix_approach', 'N/A')}")

    # Cleanup
    await llm_client.close()
    await cache_manager.close()
    await telemetry.close()


async def example_learn_schema():
    """Example of learning a schema from examples."""
    print("\n=== Example: Learn Schema from Examples ===\n")

    # Initialize components
    llm_client = LLMClient()

    # Prepare examples for a "security review" schema
    examples = [
        {
            "input": {"code": "password = request.GET['password']"},
            "expected_reasoning": {
                "vulnerabilities": [
                    {
                        "type": "security",
                        "severity": "high",
                        "description": "Password in GET request",
                        "fix": "Use POST request for sensitive data",
                    }
                ],
                "risk_score": 8,
                "recommendations": ["Use POST", "Add HTTPS enforcement"],
            },
        },
        {
            "input": {"code": 'query = f"SELECT * FROM users WHERE id = {user_id}"'},
            "expected_reasoning": {
                "vulnerabilities": [
                    {
                        "type": "sql_injection",
                        "severity": "critical",
                        "description": "Direct string interpolation in SQL",
                        "fix": "Use parameterized queries",
                    }
                ],
                "risk_score": 10,
                "recommendations": ["Use prepared statements", "Input validation"],
            },
        },
        {
            "input": {"code": "eval(user_input)"},
            "expected_reasoning": {
                "vulnerabilities": [
                    {
                        "type": "code_injection",
                        "severity": "critical",
                        "description": "Direct eval of user input",
                        "fix": "Never use eval with user input",
                    }
                ],
                "risk_score": 10,
                "recommendations": ["Remove eval", "Use safe alternatives"],
            },
        },
    ]

    print("Learning from examples...")
    print(f"Number of examples: {len(examples)}")

    # Learn the schema
    result = await learn_schema_tool(
        arguments={
            "examples": examples,
            "task_type": "security_review",
            "description": "Schema for reviewing code security vulnerabilities",
        },
        llm_client=llm_client,
    )

    if result["status"] == "success":
        print("\n--- Learned Schema ---")
        print(f"Task type: {result['metadata']['task_type']}")
        print(f"Field count: {result['metadata']['field_count']}")
        print(f"Validation rate: {result['metadata']['validation_rate']:.2f}")

        print("\nCommon patterns found:")
        for pattern in result["metadata"]["common_patterns"][:3]:
            print(f"- {pattern}")

        print("\nSchema structure:")
        schema = result["schema"]
        if "properties" in schema:
            for prop_name in list(schema["properties"].keys())[:5]:
                print(f"- {prop_name}")

        print("\n--- Usage Guide Preview ---")
        print(result["usage_guide"][:500] + "...")
    else:
        print(f"Learning failed: {result.get('error', 'Unknown error')}")

    # Cleanup
    await llm_client.close()


async def example_schema_validation():
    """Example of schema validation and error handling."""
    print("\n=== Example: Schema Validation ===\n")

    # Create a strict schema
    builder = SchemaBuilder(schema_id="strict_config", description="Strict configuration schema")

    builder.add_field(
        name="version", field_type="string", pattern=r"^\d+\.\d+\.\d+$", required=True
    )

    builder.add_field(name="port", field_type="integer", required=True)

    builder.add_field(
        name="environment",
        field_type="string",
        enum=["development", "staging", "production"],
        required=True,
    )

    strict_schema = builder.build()

    # Test data
    test_cases = [
        {
            "name": "Valid config",
            "data": {"version": "1.0.0", "port": 8080, "environment": "production"},
        },
        {
            "name": "Invalid version format",
            "data": {
                "version": "1.0",  # Missing patch version
                "port": 8080,
                "environment": "production",
            },
        },
        {
            "name": "Invalid environment",
            "data": {"version": "1.0.0", "port": 8080, "environment": "testing"},  # Not in enum
        },
        {
            "name": "Missing required field",
            "data": {
                "version": "1.0.0",
                "environment": "production",
                # Missing port
            },
        },
    ]

    print("Testing schema validation...\n")

    for test in test_cases:
        print(f"Test: {test['name']}")
        result = strict_schema.validate(test["data"])

        if result.valid:
            print(f"✓ Valid (confidence: {result.confidence:.2f})")
        else:
            print(f"✗ Invalid")
            for error in result.errors:
                print(f"  - {error}")

        if result.warnings:
            print("  Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        print()


async def main():
    """Run all custom schema examples."""
    print("MCP-SGR Custom Schema Examples")
    print("=" * 50)

    await example_custom_schema_builder()
    await example_bug_report_schema()
    await example_learn_schema()
    await example_schema_validation()

    print("\n" + "=" * 50)
    print("Custom schema examples completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
