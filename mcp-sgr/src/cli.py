"""CLI interface for MCP-SGR."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from .server import SGRServer
from .tools import apply_sgr_tool, enhance_prompt_tool
from .utils.cache import CacheManager
from .utils.llm_client import LLMClient
from .utils.telemetry import TelemetryManager

# Load environment variables
load_dotenv()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """MCP-SGR: Structured Guided Reasoning for LLM agents."""
    pass


@cli.command()
@click.option("--stdio", is_flag=True, help="Use stdio transport (default)")
@click.option("--http", is_flag=True, help="Enable HTTP facade")
@click.option("--port", default=8080, help="HTTP port (default: 8080)")
def server(stdio: bool, http: bool, port: int):
    """Start the MCP-SGR server."""
    # Default to stdio if nothing specified
    if not http:
        stdio = True

    if stdio:
        click.echo("Starting MCP-SGR server with stdio transport...")
        asyncio.run(SGRServer().run())
    elif http:
        click.echo(f"Starting MCP-SGR HTTP server on port {port}...")
        from .http_server import run_http_server

        run_http_server(port=port)


@cli.command()
@click.argument("task")
@click.option(
    "--schema",
    "-s",
    default="auto",
    type=click.Choice(
        ["auto", "analysis", "planning", "decision", "code_generation", "summarization"]
    ),
    help="Schema type to use",
)
@click.option(
    "--budget",
    "-b",
    default="lite",
    type=click.Choice(["none", "lite", "full"]),
    help="Reasoning budget depth",
)
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON")
def analyze(task: str, schema: str, budget: str, output: Optional[str], output_json: bool):
    """Analyze a task using SGR."""

    async def run_analysis():
        # Initialize components
        llm_client = LLMClient()
        cache_manager = CacheManager()
        telemetry = TelemetryManager()

        await cache_manager.initialize()
        await telemetry.initialize()

        try:
            # Apply SGR
            result = await apply_sgr_tool(
                arguments={"task": task, "schema_type": schema, "budget": budget},
                llm_client=llm_client,
                cache_manager=cache_manager,
                telemetry=telemetry,
            )

            # Format output
            if output_json:
                output_text = json.dumps(result, indent=2)
            else:
                output_text = format_analysis_result(result)

            # Write output
            if output:
                Path(output).write_text(output_text)
                click.echo(f"Analysis saved to {output}")
            else:
                click.echo(output_text)

        finally:
            await llm_client.close()
            await cache_manager.close()
            await telemetry.close()

    asyncio.run(run_analysis())


@cli.command()
@click.argument("prompt")
@click.option(
    "--level",
    "-l",
    default="standard",
    type=click.Choice(["minimal", "standard", "comprehensive"]),
    help="Enhancement level",
)
@click.option("--target", "-t", help="Target model")
def enhance(prompt: str, level: str, target: Optional[str]):
    """Enhance a prompt with SGR structure."""

    async def run_enhancement():
        llm_client = LLMClient()
        cache_manager = CacheManager()

        await cache_manager.initialize()

        try:
            result = await enhance_prompt_tool(
                arguments={
                    "original_prompt": prompt,
                    "enhancement_level": level,
                    "target_model": target,
                },
                llm_client=llm_client,
                cache_manager=cache_manager,
            )

            click.echo("Original prompt:")
            click.echo(f"  {prompt}\n")

            click.echo("Enhanced prompt:")
            click.echo(result["enhanced_prompt"])

            click.echo(f"\nDetected intent: {result['metadata']['detected_intent']}")
            click.echo(f"Suggested schema: {result['metadata']['suggested_schema']}")

        finally:
            await llm_client.close()
            await cache_manager.close()

    asyncio.run(run_enhancement())


@cli.command()
def cache_stats():
    """Show cache statistics."""

    async def show_stats():
        cache_manager = CacheManager()
        await cache_manager.initialize()

        try:
            stats = await cache_manager.get_cache_stats()

            click.echo("Cache Statistics:")
            click.echo(f"  Enabled: {stats.get('enabled', False)}")

            if stats.get("enabled"):
                click.echo(f"  Total entries: {stats.get('total_entries', 0)}")
                click.echo(f"  Active entries: {stats.get('active_entries', 0)}")
                click.echo(f"  Total hits: {stats.get('total_hits', 0)}")
                click.echo(f"  Hit rate: {stats.get('hit_rate', 0):.2%}")

        finally:
            await cache_manager.close()

    asyncio.run(show_stats())


@cli.command()
@click.option("--limit", "-n", default=10, help="Number of traces to show")
@click.option("--tool", "-t", help="Filter by tool name")
def traces(limit: int, tool: Optional[str]):
    """Show recent reasoning traces."""

    async def show_traces():
        cache_manager = CacheManager()
        await cache_manager.initialize()

        try:
            traces = await cache_manager.get_recent_traces(limit=limit, tool_name=tool)

            if not traces:
                click.echo("No traces found.")
                return

            click.echo(f"Recent Traces (showing {len(traces)} of {limit} requested):\n")

            for trace in traces:
                click.echo(f"ID: {trace['id']}")
                click.echo(f"Tool: {trace['tool_name']}")
                click.echo(f"Time: {trace['created_at']}")
                click.echo(f"Duration: {trace['duration_ms']}ms")

                # Show abbreviated arguments
                args = trace.get("arguments", {})
                if "task" in args:
                    task_preview = (
                        args["task"][:100] + "..." if len(args["task"]) > 100 else args["task"]
                    )
                    click.echo(f"Task: {task_preview}")

                click.echo("-" * 60)

        finally:
            await cache_manager.close()

    asyncio.run(show_traces())


@cli.command()
@click.option("--clear-cache", is_flag=True, help="Clear all cache entries")
@click.option("--clear-traces", is_flag=True, help="Clear all traces")
def cleanup(clear_cache: bool, clear_traces: bool):
    """Clean up cache and traces."""
    if not clear_cache and not clear_traces:
        click.echo("Nothing to clean. Use --clear-cache or --clear-traces")
        return

    async def run_cleanup():
        cache_manager = CacheManager()
        await cache_manager.initialize()

        try:
            if clear_cache:
                count = await cache_manager.clear_expired()
                click.echo(f"Cleared {count} expired cache entries")

            if clear_traces:
                # This would need to be implemented
                click.echo("Trace cleanup not yet implemented")

        finally:
            await cache_manager.close()

    asyncio.run(run_cleanup())


def format_analysis_result(result: dict) -> str:
    """Format analysis result for human readability."""
    output = []

    output.append(f"Confidence: {result.get('confidence', 0):.2%}")
    output.append("")

    # Show reasoning summary
    if "reasoning" in result:
        reasoning = result["reasoning"]

        if "understanding" in reasoning:
            output.append("Understanding:")
            understanding = reasoning["understanding"]
            if "task_summary" in understanding:
                output.append(f"  {understanding['task_summary']}")
            output.append("")

        if "goals" in reasoning:
            output.append("Goals:")
            goals = reasoning["goals"]
            if "primary" in goals:
                output.append(f"  Primary: {goals['primary']}")
            if "success_criteria" in goals:
                output.append("  Success Criteria:")
                for criterion in goals["success_criteria"][:3]:
                    output.append(f"    - {criterion}")
            output.append("")

        if "risks" in reasoning:
            output.append("Risks:")
            for risk in reasoning["risks"][:3]:
                if isinstance(risk, dict):
                    output.append(f"  - {risk.get('risk', 'Unknown')}")
                    output.append(f"    Mitigation: {risk.get('mitigation', 'N/A')}")
            output.append("")

    # Show suggested actions
    if "suggested_actions" in result and result["suggested_actions"]:
        output.append("Suggested Actions:")
        for action in result["suggested_actions"][:5]:
            output.append(f"  - {action}")
        output.append("")

    # Show metadata
    if "metadata" in result:
        metadata = result["metadata"]
        output.append("Metadata:")
        output.append(f"  Schema: {metadata.get('schema_type', 'N/A')}")
        output.append(f"  Budget: {metadata.get('budget', 'N/A')}")
        output.append(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")

    return "\n".join(output)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
