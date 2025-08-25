"""HTTP facade for MCP-SGR server."""

import os
import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import time
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .utils.llm_client import LLMClient
from .utils.cache import CacheManager
from .utils.telemetry import TelemetryManager
from .tools import (
    apply_sgr_tool,
    wrap_agent_call_tool,
    enhance_prompt_tool,
    learn_schema_tool
)
from .utils.logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)


class SimpleRateLimiter:
    """Naive in-memory rate limiter (per key per minute)."""
    def __init__(self, enabled: bool = False, max_rpm: int = 120):
        self.enabled = enabled
        self.max_rpm = max_rpm
        self._buckets: dict[str, dict[str, float | int]] = {}

    def _now_minute(self) -> int:
        return int(time.time() // 60)

    def allow(self, key: str) -> bool:
        if not self.enabled:
            return True
        minute = self._now_minute()
        bucket = self._buckets.get(key)
        if not bucket or bucket.get("minute") != minute:
            self._buckets[key] = {"minute": minute, "count": 1}
            return True
        count = int(bucket.get("count", 0)) + 1
        bucket["count"] = count
        return count <= self.max_rpm


# Pydantic models for requests/responses
class ApplySGRRequest(BaseModel):
    task: str = Field(..., description="The task or problem to analyze")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    schema_type: str = Field(default="auto", description="Schema type to use")
    custom_schema: Optional[Dict[str, Any]] = Field(default=None, description="Custom schema definition")
    budget: str = Field(default="lite", description="Reasoning budget depth")


class WrapAgentRequest(BaseModel):
    agent_endpoint: str = Field(..., description="Agent endpoint URL or identifier")
    agent_request: Dict[str, Any] = Field(..., description="Request payload for the agent")
    sgr_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="SGR configuration")


class EnhancePromptRequest(BaseModel):
    original_prompt: str = Field(..., description="The original prompt to enhance")
    target_model: Optional[str] = Field(default=None, description="Target model identifier")
    enhancement_level: str = Field(default="standard", description="Enhancement level")


class LearnSchemaRequest(BaseModel):
    examples: List[Dict[str, Any]] = Field(..., description="Example inputs and expected reasoning", min_length=3)
    task_type: str = Field(..., description="Name for the new schema/task type")
    description: Optional[str] = Field(default=None, description="Description of the schema")


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_enabled: bool
    trace_enabled: bool


# Global instances
llm_client: Optional[LLMClient] = None
cache_manager: Optional[CacheManager] = None
telemetry_manager: Optional[TelemetryManager] = None
rate_limiter: Optional[SimpleRateLimiter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global llm_client, cache_manager, telemetry_manager, rate_limiter
    
    # Startup
    logger.info("Starting HTTP facade...")
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry_manager = TelemetryManager()
    
    await cache_manager.initialize()
    await telemetry_manager.initialize()

    # Rate limiter
    rate_enabled = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    max_rpm = int(os.getenv("RATE_LIMIT_MAX_RPM", "120"))
    rate_limiter = SimpleRateLimiter(rate_enabled, max_rpm)
    
    yield
    
    # Shutdown
    logger.info("Shutting down HTTP facade...")
    if llm_client:
        await llm_client.close()
    if cache_manager:
        await cache_manager.close()
    if telemetry_manager:
        await telemetry_manager.close()


# Create FastAPI app
app = FastAPI(
    title="MCP-SGR HTTP API",
    description="HTTP facade for Schema-Guided Reasoning middleware",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware (configurable via ENV)
cors_origins_env = os.getenv("HTTP_CORS_ORIGINS", "*")
allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Optional API key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> bool:
    """Verify API key if configured or required."""
    expected_key = os.getenv("HTTP_AUTH_TOKEN")
    require_auth = os.getenv("HTTP_REQUIRE_AUTH", "false").lower() == "true"
    if not require_auth and not expected_key:
        return True  # No auth enforced
    if expected_key and x_api_key == expected_key:
        return True
    return False


async def verify_rate_limit(request: Request) -> bool:
    """Simple per-minute rate limiter by client key (ip or api key)."""
    if not rate_limiter or not rate_limiter.enabled:
        return True
    # Prefer API key for bucketing; fallback to client host
    api_key = request.headers.get("x-api-key") or request.headers.get("x_api_key")
    client = request.client.host if request.client else "unknown"
    key = api_key or client
    allowed = rate_limiter.allow(key)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        cache_enabled=cache_manager.enabled if cache_manager else False,
        trace_enabled=telemetry_manager.enabled if telemetry_manager else False
    )


@app.post("/v1/apply-sgr")
async def apply_sgr(
    request: ApplySGRRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Apply SGR schema to analyze a task."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not llm_client or not cache_manager or not telemetry_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        result = await apply_sgr_tool(
            arguments=request.dict(),
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry_manager
        )
        return result
    except Exception as e:
        logger.error(f"Error in apply_sgr: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/wrap-agent")
async def wrap_agent(
    request: WrapAgentRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Wrap agent call with pre/post analysis."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not llm_client or not cache_manager or not telemetry_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        result = await wrap_agent_call_tool(
            arguments=request.dict(),
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry_manager
        )
        return result
    except Exception as e:
        logger.error(f"Error in wrap_agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/enhance-prompt")
async def enhance_prompt(
    request: EnhancePromptRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Enhance a prompt with SGR structure."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not llm_client or not cache_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        result = await enhance_prompt_tool(
            arguments=request.dict(),
            llm_client=llm_client,
            cache_manager=cache_manager
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhance_prompt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/learn-schema")
async def learn_schema(
    request: LearnSchemaRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Learn new schema from examples."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not llm_client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        result = await learn_schema_tool(
            arguments=request.dict(),
            llm_client=llm_client
        )
        return result
    except Exception as e:
        logger.error(f"Error in learn_schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/schemas")
async def list_schemas(
    authorized: bool = Depends(verify_api_key), _: bool = Depends(verify_rate_limit)
):
    """List available schemas."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    from .schemas import SCHEMA_REGISTRY
    
    schemas: Dict[str, Any] = {}
    for name, schema_class in SCHEMA_REGISTRY.items():
        schema = schema_class()
        schemas[name] = {
            "description": schema.get_description(),
            "fields": [f.name for f in schema.get_fields()],
        }
    
    return schemas


@app.get("/v1/cache-stats")
async def get_cache_stats(
    authorized: bool = Depends(verify_api_key), _: bool = Depends(verify_rate_limit)
):
    """Get cache statistics."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not cache_manager:
        return {"enabled": False}
    
    return await cache_manager.get_cache_stats()


@app.get("/v1/traces")
async def get_traces(
    limit: int = 10,
    tool_name: Optional[str] = None,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Get recent reasoning traces."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not cache_manager:
        return []
    
    return await cache_manager.get_recent_traces(limit=limit, tool_name=tool_name)


def run_http_server(host: str = "127.0.0.1", port: int = 8080):
    """Run the HTTP server."""
    uvicorn.run(
        "src.http_server:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    run_http_server()