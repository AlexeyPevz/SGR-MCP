"""HTTP facade for MCP-SGR server."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml
import re
import jwt

from .tools import apply_sgr_tool, enhance_prompt_tool, learn_schema_tool, wrap_agent_call_tool
from .utils.cache import CacheManager
from .utils.llm_client import LLMClient
from .utils.logging_config import setup_logging
from .utils.telemetry import TelemetryManager

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


# Enhanced input validation patterns
import re
from pydantic import validator

# Security patterns for input validation
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # XSS attempts
    r'javascript:',  # JavaScript execution
    r'eval\s*\(',  # Eval function
    r'exec\s*\(',  # Exec function
    r'import\s+os',  # OS imports
    r'subprocess',  # Subprocess calls
    r'__import__',  # Dynamic imports
]

def validate_safe_input(value: str) -> str:
    """Validate input for security threats."""
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(f"Potentially dangerous input detected")
    
    # Length limit
    if len(value) > 50000:  # 50KB limit
        raise ValueError("Input too long (max 50KB)")
    
    return value

# Pydantic models for requests/responses
class ApplySGRRequest(BaseModel):
    task: str = Field(..., description="The task or problem to analyze", max_length=10000)
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    schema_type: str = Field(default="auto", description="Schema type to use", regex="^[a-zA-Z0-9_-]+$")
    custom_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom schema definition"
    )
    budget: str = Field(default="lite", description="Reasoning budget depth", regex="^(lite|standard|full)$")
    
    @validator('task')
    def validate_task_safety(cls, v):
        return validate_safe_input(v)


class WrapAgentRequest(BaseModel):
    agent_endpoint: str = Field(..., description="Agent endpoint URL or identifier", max_length=1000)
    agent_request: Dict[str, Any] = Field(..., description="Request payload for the agent")
    sgr_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="SGR configuration"
    )
    
    @validator('agent_endpoint')
    def validate_endpoint_safety(cls, v):
        if not v.startswith(('http://', 'https://')):
            # Allow named endpoints for internal routing
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("Invalid endpoint format")
        return v


class EnhancePromptRequest(BaseModel):
    original_prompt: str = Field(..., description="The original prompt to enhance", max_length=10000)
    target_model: Optional[str] = Field(default=None, description="Target model identifier", max_length=100)
    enhancement_level: str = Field(default="standard", description="Enhancement level", regex="^(minimal|standard|aggressive)$")
    
    @validator('original_prompt')
    def validate_prompt_safety(cls, v):
        return validate_safe_input(v)
    
    @validator('target_model')
    def validate_model_name(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9/_.-]+$', v):
            raise ValueError("Invalid model name format")
        return v


class LearnSchemaRequest(BaseModel):
    examples: List[Dict[str, Any]] = Field(
        ..., description="Example inputs and expected reasoning", min_length=3, max_length=20
    )
    task_type: str = Field(..., description="Name for the new schema/task type", max_length=50, regex="^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(default=None, description="Description of the schema", max_length=1000)
    
    @validator('description')
    def validate_description_safety(cls, v):
        if v:
            return validate_safe_input(v)
        return v


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
_redis_client = None  # type: ignore[var-annotated]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global llm_client, cache_manager, telemetry_manager, rate_limiter, _redis_client

    # Startup
    logger.info("Starting HTTP facade...")
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry_manager = TelemetryManager()

    await cache_manager.initialize()
    await telemetry_manager.initialize()

    # Rate limiter (memory or redis)
    rate_enabled = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    max_rpm = int(os.getenv("RATE_LIMIT_MAX_RPM", "120"))
    backend = os.getenv("RATE_LIMIT_BACKEND", "memory").lower()
    if backend == "redis" and rate_enabled:
        try:
            import redis.asyncio as aioredis  # type: ignore

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            # Global assignments
            global _redis_client
            _redis_client = aioredis.from_url(redis_url, decode_responses=True)

            class RedisRateLimiter:
                def __init__(self, client, max_rpm: int):
                    self.client = client
                    self.max_rpm = max_rpm

                async def allow(self, key: str) -> bool:
                    # Key per minute per key
                    minute_key = f"rate:{key}:{int(time.time() // 60)}"
                    count = await self.client.incr(minute_key)
                    if count == 1:
                        await self.client.expire(minute_key, 65)
                    return int(count) <= self.max_rpm

            rate_limiter = RedisRateLimiter(_redis_client, max_rpm)  # type: ignore[assignment]
        except Exception:
            rate_limiter = SimpleRateLimiter(rate_enabled, max_rpm)
    else:
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
    if _redis_client is not None:
        try:
            await _redis_client.close()
        except Exception:
            pass


# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="MCP-SGR HTTP API",
    description="""
    ## Schema-Guided Reasoning API
    
    A powerful middleware for transparent and managed LLM agent reasoning via MCP and SGR.
    
    ### Features
    - 🧠 **Schema-Guided Reasoning**: Structured thinking for LLM agents
    - 🔄 **Multi-LLM Support**: Works with Ollama, OpenRouter, vLLM and more
    - 🚀 **Budget Optimization**: Free models perform like premium ones
    - 🔒 **Enterprise Security**: JWT auth, rate limiting, input validation
    - 📊 **Observability**: OpenTelemetry integration, caching, tracing
    
    ### Quick Start
    1. Get API key from admin
    2. Set `X-API-Key` header in requests
    3. Use `/v1/apply-sgr` for structured reasoning
    4. Monitor usage via `/v1/cache-stats` and `/v1/traces`
    
    ### Rate Limits
    - Default: 120 requests per minute per API key
    - Contact admin for higher limits
    """,
    version="0.1.0",
    contact={
        "name": "MCP-SGR Team", 
        "email": "team@mcp-sgr.dev",
        "url": "https://github.com/mcp-sgr/mcp-sgr"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",  # Standard Swagger UI
    redoc_url="/redoc",  # Standard ReDoc
    openapi_tags=[
        {"name": "reasoning", "description": "Core SGR operations"},
        {"name": "agents", "description": "Agent wrapper functionality"},
        {"name": "prompts", "description": "Prompt enhancement tools"},
        {"name": "schemas", "description": "Schema management"},
        {"name": "monitoring", "description": "System monitoring and stats"},
        {"name": "health", "description": "Health checks and status"}
    ],
    lifespan=lifespan,
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'none'; object-src 'none'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response

# Rate Limiting Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting based on IP address or API key."""
    if rate_limiter and rate_limiter.enabled:
        # Use API key if present, otherwise IP address
        api_key = request.headers.get("x-api-key")
        client_id = api_key if api_key else request.client.host if request.client else "unknown"
        
        if not rate_limiter.allow(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
    
    return await call_next(request)

# Add CORS middleware (configurable via ENV)
cors_origins_env = os.getenv("HTTP_CORS_ORIGINS", "*")
allow_origins = (
    [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # More restrictive
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],  # More restrictive
)


# Optional API key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)) -> bool:
    """Verify auth based on configured mode: token (default) or jwt.

    - token mode: uses x-api-key header and HTTP_AUTH_TOKEN
    - jwt mode: checks Authorization: Bearer <token> against HTTP_JWT_SECRET (HS256)
    Both modes may co-exist; passing either valid credential grants access.
    """
    require_auth = os.getenv("HTTP_REQUIRE_AUTH", "false").lower() == "true"
    if not require_auth:
        return True

    # Token auth
    expected_key = os.getenv("HTTP_AUTH_TOKEN")
    if expected_key and x_api_key == expected_key:
        return True

    # JWT auth (optional)
    mode = os.getenv("HTTP_AUTH_MODE", "token").lower()
    if mode == "jwt" and authorization:
        match = re.match(r"^Bearer\s+(.+)$", authorization.strip(), re.IGNORECASE)
        if match:
            token = match.group(1)
            try:
                secret = os.getenv("HTTP_JWT_SECRET")
                if not secret:
                    return False
                options = {"verify_aud": False}
                issuer = os.getenv("HTTP_JWT_ISSUER")
                audience = os.getenv("HTTP_JWT_AUDIENCE")
                kwargs = {"algorithms": ["HS256"], "options": options}
                if issuer:
                    kwargs["issuer"] = issuer
                if audience:
                    kwargs["audience"] = audience
                jwt.decode(token, secret, **kwargs)  # type: ignore[arg-type]
                return True
            except Exception:
                return False

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


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        cache_enabled=cache_manager.enabled if cache_manager else False,
        trace_enabled=telemetry_manager.enabled if telemetry_manager else False,
    )


@app.post("/v1/apply-sgr", tags=["reasoning"])
async def apply_sgr(
    request: ApplySGRRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
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
            telemetry=telemetry_manager,
        )
        return result
    except Exception as e:
        logger.error(f"Error in apply_sgr: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/wrap-agent", tags=["agents"])
async def wrap_agent(
    request: WrapAgentRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
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
            telemetry=telemetry_manager,
        )
        return result
    except Exception as e:
        logger.error(f"Error in wrap_agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/enhance-prompt", tags=["prompts"])
async def enhance_prompt(
    request: EnhancePromptRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    """Enhance a prompt with SGR structure."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not llm_client or not cache_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        result = await enhance_prompt_tool(
            arguments=request.dict(), llm_client=llm_client, cache_manager=cache_manager
        )
        return result
    except Exception as e:
        logger.error(f"Error in enhance_prompt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/learn-schema", tags=["schemas"])
async def learn_schema(
    request: LearnSchemaRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    """Learn new schema from examples."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not llm_client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        result = await learn_schema_tool(arguments=request.dict(), llm_client=llm_client)
        return result
    except Exception as e:
        logger.error(f"Error in learn_schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/schemas", tags=["schemas"])
async def list_schemas(
    authorized: bool = Depends(verify_api_key), _: bool = Depends(verify_rate_limit)
):
    """List available schemas."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    from .schemas import SCHEMA_REGISTRY

    schemas: Dict[str, Any] = {}
    for name, schema_factory in SCHEMA_REGISTRY.items():
        schema = schema_factory()
        schemas[name] = {
            "description": schema.get_description(),
            "fields": [f.name for f in schema.get_fields()],
        }

    return schemas


@app.get("/v1/cache-stats", tags=["monitoring"])
async def get_cache_stats(
    authorized: bool = Depends(verify_api_key), _: bool = Depends(verify_rate_limit)
):
    """Get cache statistics."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not cache_manager:
        return {"enabled": False}

    return await cache_manager.get_cache_stats()


@app.get("/v1/traces", tags=["monitoring"])
async def get_traces(
    limit: int = 10,
    tool_name: Optional[str] = None,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    """Get recent reasoning traces."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not cache_manager:
        return []

    return await cache_manager.get_recent_traces(limit=limit, tool_name=tool_name)


@app.get("/v1/performance-stats", tags=["monitoring"])
async def get_performance_stats(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Get LLM client performance statistics."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    return llm_client.get_performance_stats()


@app.post("/v1/performance-stats/reset", tags=["monitoring"])
async def reset_performance_stats(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Reset LLM client performance statistics."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    llm_client.reset_performance_stats()
    return {"message": "Performance statistics reset successfully"}


@app.get("/v1/health-check", tags=["monitoring"])
async def detailed_health_check(
    timeout: float = 10.0,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Perform detailed health check on all backends."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    try:
        health_status = await llm_client.health_check(timeout)
        
        # Add overall status
        all_healthy = all(
            status.get("status") == "healthy" 
            for status in health_status.values()
        )
        
        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "backends": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/v1/batch-apply-sgr", tags=["reasoning"])
async def batch_apply_sgr(
    requests: List[ApplySGRRequest],
    max_concurrent: int = 5,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Apply SGR to multiple tasks concurrently."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not llm_client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if len(requests) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 20)")
    
    if max_concurrent > 10:  # Limit concurrency
        max_concurrent = 10

    try:
        # Process requests concurrently
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request: ApplySGRRequest):
            async with semaphore:
                return await apply_sgr_tool(
                    arguments=request.dict(),
                    llm_client=llm_client,
                    cache_manager=cache_manager,
                    telemetry_manager=telemetry_manager
                )
        
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                processed_results.append({
                    "error": str(result),
                    "request_index": i
                })
            else:
                processed_results.append(result)
        
        return {
            "results": processed_results,
            "total_requests": len(requests),
            "successful_requests": sum(1 for r in results if not isinstance(r, Exception)),
            "failed_requests": sum(1 for r in results if isinstance(r, Exception))
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/openapi.yaml")
async def openapi_yaml():
    """Serve OpenAPI schema in YAML."""
    try:
        schema = app.openapi()
        content = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)
        return Response(content=content, media_type="application/yaml")
    except Exception as e:
        logger.error(f"Failed to render OpenAPI YAML: {e}")
        raise HTTPException(status_code=500, detail="Failed to render OpenAPI")


@app.get("/docs/swagger")
async def swagger_ui():
    """Custom Swagger UI with enhanced documentation."""
    from fastapi.openapi.docs import get_swagger_ui_html
    
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="MCP-SGR API Documentation",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
    )


@app.get("/docs/redoc")  
async def redoc():
    """ReDoc documentation interface."""
    from fastapi.openapi.docs import get_redoc_html
    
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="MCP-SGR API Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js",
    )


def run_http_server(host: str = "127.0.0.1", port: int = 8080):
    """Run the HTTP server."""
    uvicorn.run(
        "src.http_server:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    run_http_server()
