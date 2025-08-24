"""MCP-SGR Server implementation."""

import os
import sys
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, Resource, TextContent, ImageContent, EmbeddedResource,
    ListResourcesResult, ReadResourceResult, ListToolsResult, CallToolResult
)
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .schemas import SCHEMA_REGISTRY
from .schemas.custom import CustomSchema
from .tools.apply_sgr import apply_sgr_tool
from .tools.wrap_agent import wrap_agent_call_tool
from .tools.enhance_prompt import enhance_prompt_tool
from .tools.learn_schema import learn_schema_tool
from .utils.llm_client import LLMClient
from .utils.cache import CacheManager
from .utils.telemetry import TelemetryManager

# Load environment variables
load_dotenv()

# Configure logging with rotation
from .utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


class SGRServer:
    """MCP server for Schema-Guided Reasoning."""
    
    def __init__(self):
        self.server = Server("sgr-reasoning")
        self.llm_client = LLMClient()
        self.cache_manager = CacheManager()
        self.telemetry = TelemetryManager()
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all MCP handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available SGR tools."""
            tools = [
                Tool(
                    name="apply_sgr",
                    description="Apply a structured reasoning schema to analyze and structure a task",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task or problem to analyze"
                            },
                            "context": {
                                "type": "object",
                                "description": "Additional context information"
                            },
                            "schema_type": {
                                "type": "string",
                                "enum": ["auto", "analysis", "planning", "decision", "search", "code_generation", "summarization", "custom"],
                                "default": "auto"
                            },
                            "custom_schema": {
                                "type": "object",
                                "description": "Custom schema definition if schema_type is 'custom'"
                            },
                            "budget": {
                                "type": "string",
                                "enum": ["none", "lite", "full"],
                                "default": "lite"
                            }
                        },
                        "required": ["task"]
                    }
                ),
                Tool(
                    name="wrap_agent_call",
                    description="Wrap any agent call with pre/post SGR analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_endpoint": {
                                "type": "string",
                                "description": "The endpoint or identifier of the agent"
                            },
                            "agent_request": {
                                "type": "object",
                                "description": "The request payload for the agent"
                            },
                            "sgr_config": {
                                "type": "object",
                                "description": "SGR configuration",
                                "properties": {
                                    "schema_type": {"type": "string", "default": "auto"},
                                    "budget": {"type": "string", "default": "lite"},
                                    "pre_analysis": {"type": "boolean", "default": True},
                                    "post_analysis": {"type": "boolean", "default": True}
                                }
                            }
                        },
                        "required": ["agent_endpoint", "agent_request"]
                    }
                ),
                Tool(
                    name="enhance_prompt_with_sgr",
                    description="Transform a simple prompt into a structured prompt",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "original_prompt": {
                                "type": "string",
                                "description": "The original simple prompt"
                            },
                            "target_model": {
                                "type": "string",
                                "description": "Target model identifier"
                            },
                            "enhancement_level": {
                                "type": "string",
                                "enum": ["minimal", "standard", "comprehensive"],
                                "default": "standard"
                            }
                        },
                        "required": ["original_prompt"]
                    }
                ),
                Tool(
                    name="learn_schema_from_examples",
                    description="Learn a new SGR schema from examples (roadmap feature)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "examples": {
                                "type": "array",
                                "description": "Example inputs and reasoning outputs",
                                "minItems": 3
                            },
                            "task_type": {
                                "type": "string",
                                "description": "Name for the new schema"
                            }
                        },
                        "required": ["examples", "task_type"]
                    }
                )
            ]
            return ListToolsResult(tools=tools)
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            logger.info(f"Tool called: {name} with arguments: {arguments}")
            
            try:
                if name == "apply_sgr":
                    result = await apply_sgr_tool(
                        arguments, 
                        self.llm_client, 
                        self.cache_manager,
                        self.telemetry
                    )
                elif name == "wrap_agent_call":
                    result = await wrap_agent_call_tool(
                        arguments,
                        self.llm_client,
                        self.cache_manager,
                        self.telemetry
                    )
                elif name == "enhance_prompt_with_sgr":
                    result = await enhance_prompt_tool(
                        arguments,
                        self.llm_client,
                        self.cache_manager
                    )
                elif name == "learn_schema_from_examples":
                    result = await learn_schema_tool(
                        arguments,
                        self.llm_client
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return CallToolResult(
                    content=[TextContent(text=json.dumps(result, indent=2))]
                )
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
                return CallToolResult(
                    content=[TextContent(text=json.dumps({
                        "error": str(e),
                        "type": type(e).__name__
                    }))]
                )
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources."""
            resources = [
                Resource(
                    uri="sgr://schemas",
                    name="schema_library",
                    description="Available SGR schemas",
                    mimeType="application/json"
                ),
                Resource(
                    uri="sgr://policy",
                    name="policy",
                    description="Current routing and budget policy",
                    mimeType="application/yaml"
                ),
                Resource(
                    uri="sgr://traces",
                    name="traces",
                    description="Recent reasoning traces",
                    mimeType="application/json"
                )
            ]
            return ListResourcesResult(resources=resources)
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read a resource."""
            logger.info(f"Reading resource: {uri}")
            
            if uri == "sgr://schemas":
                # Return schema library
                schemas = {}
                for name, schema_class in SCHEMA_REGISTRY.items():
                    schema = schema_class()
                    schemas[name] = {
                        "description": schema.get_description(),
                        "schema": schema.to_json_schema(),
                        "examples": schema.get_examples()
                    }
                
                content = json.dumps(schemas, indent=2)
                return ReadResourceResult(
                    content=[TextContent(text=content)]
                )
            
            elif uri == "sgr://policy":
                # Return current policy
                policy_file = Path("router_policy.yaml")
                if policy_file.exists():
                    content = policy_file.read_text()
                else:
                    content = """# Default router policy
router:
  rules:
    - when: task_type == "code_generation"
      use: default
    - when: task_type in ["analysis", "summarization"]
      use: default
  retry:
    max_attempts: 2
    backoff: 0.8
"""
                return ReadResourceResult(
                    content=[TextContent(text=content)]
                )
            
            elif uri == "sgr://traces":
                # Return recent traces
                traces = await self.cache_manager.get_recent_traces(limit=10)
                content = json.dumps(traces, indent=2)
                return ReadResourceResult(
                    content=[TextContent(text=content)]
                )
            
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def run(self):
        """Run the MCP server."""
        logger.info("Starting SGR MCP Server...")
        
        # Initialize components
        await self.cache_manager.initialize()
        await self.telemetry.initialize()
        
        try:
            # Run the server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(read_stream, write_stream)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down gracefully...")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("Shutting down SGR server...")
        
        try:
            # Close LLM client
            if hasattr(self.llm_client, 'close'):
                await self.llm_client.close()
            
            # Close cache manager
            if self.cache_manager:
                await self.cache_manager.close()
            
            # Close telemetry
            if self.telemetry:
                await self.telemetry.close()
            
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


async def main():
    """Main entry point."""
    server = SGRServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())