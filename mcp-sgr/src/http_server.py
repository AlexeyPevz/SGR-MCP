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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
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
    schema_type: str = Field(default="auto", description="Schema type to use", pattern="^[a-zA-Z0-9_-]+$")
    custom_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom schema definition"
    )
    budget: str = Field(default="lite", description="Reasoning budget depth", pattern="^(lite|standard|full)$")
    
    @field_validator('task')
    @classmethod
    def validate_task_safety(cls, v: str) -> str:
        return validate_safe_input(v)


class WrapAgentRequest(BaseModel):
    agent_endpoint: str = Field(..., description="Agent endpoint URL or identifier", max_length=1000)
    agent_request: Dict[str, Any] = Field(..., description="Request payload for the agent")
    sgr_config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="SGR configuration"
    )
    
    @field_validator('agent_endpoint')
    @classmethod
    def validate_endpoint_safety(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            # Allow named endpoints for internal routing
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError("Invalid endpoint format")
        return v
# Utility: sanitize arbitrary data to be JSON-safe (avoid Mock/AsyncMock recursion)
def _sanitize_for_json(obj: Any) -> Any:
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_sanitize_for_json(v) for v in obj)
        # Pydantic models
        try:
            from pydantic import BaseModel as _BM
            if isinstance(obj, _BM):
                return _sanitize_for_json(obj.model_dump())
        except Exception:
            pass
        # Fallback to string
        return str(obj)
    except Exception:
        return str(obj)



class EnhancePromptRequest(BaseModel):
    original_prompt: str = Field(..., description="The original prompt to enhance", max_length=10000)
    target_model: Optional[str] = Field(default=None, description="Target model identifier", max_length=100)
    enhancement_level: str = Field(default="standard", description="Enhancement level", pattern="^(minimal|standard|aggressive)$")
    
    @field_validator('original_prompt')
    @classmethod
    def validate_prompt_safety(cls, v: str) -> str:
        return validate_safe_input(v)
    
    @field_validator('target_model')
    @classmethod
    def validate_model_name(cls, v: str | None) -> str | None:
        if v and not re.match(r'^[a-zA-Z0-9/_.-]+$', v):
            raise ValueError("Invalid model name format")
        return v


class LearnSchemaRequest(BaseModel):
    examples: List[Dict[str, Any]] = Field(
        ..., description="Example inputs and expected reasoning", min_length=3, max_length=20
    )
    task_type: str = Field(..., description="Name for the new schema/task type", max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(default=None, description="Description of the schema", max_length=1000)
    
    @field_validator('description')
    @classmethod
    def validate_description_safety(cls, v: str | None) -> str | None:
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
rbac_manager: Optional['RBACManager'] = None
audit_logger: Optional['AuditLogger'] = None
adaptive_router: Optional['AdaptiveRouter'] = None
smart_cache: Optional['SmartCacheManager'] = None
cost_optimizer: Optional['CostOptimizer'] = None
_redis_client = None  # type: ignore[var-annotated]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global llm_client, cache_manager, telemetry_manager, rate_limiter, rbac_manager, audit_logger, _redis_client

    # Startup
    logger.info("Starting HTTP facade...")
    llm_client = LLMClient()
    cache_manager = CacheManager()
    telemetry_manager = TelemetryManager()
    
    # Initialize auth system
    from .auth.rbac import RBACManager
    from .auth.audit import AuditLogger
    rbac_manager = RBACManager(cache_manager)
    audit_logger = AuditLogger(cache_manager)
    
    # Initialize AI optimization (if enabled)
    global adaptive_router, smart_cache, cost_optimizer
    ai_optimization_enabled = os.getenv("AI_OPTIMIZATION_ENABLED", "true").lower() == "true"

    if ai_optimization_enabled:
        try:
            from .ai_optimization.adaptive_router import AdaptiveRouter
            from .ai_optimization.smart_cache import SmartCacheManager
            from .ai_optimization.cost_optimizer import CostOptimizer
        except Exception:
            logger.warning("AI optimization modules unavailable; disabling optimization features")
            ai_optimization_enabled = False

    if ai_optimization_enabled:
        adaptive_router = AdaptiveRouter(cache_manager)
        smart_cache = SmartCacheManager(cache_manager)
        cost_optimizer = CostOptimizer(cache_manager)
        
        await adaptive_router.initialize()
        await smart_cache.initialize()
        await cost_optimizer.initialize()
        
        logger.info("AI optimization modules initialized")

    await cache_manager.initialize()
    await telemetry_manager.initialize()
    await audit_logger.initialize()

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
    if audit_logger:
        await audit_logger.close()
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
    # Re-evaluate env flag per-request to avoid cross-test leakage
    env_enabled = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    if rate_limiter and rate_limiter.enabled and env_enabled:
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


# Enhanced authentication and authorization
async def get_current_user(
    x_api_key: Optional[str] = Header(default=None), 
    authorization: Optional[str] = Header(default=None),
    request: Request = None
) -> tuple[Optional['User'], Optional['APIKey']]:
    """Get current authenticated user and API key."""
    if not rbac_manager:
        return None, None
    
    require_auth = os.getenv("HTTP_REQUIRE_AUTH", "false").lower() == "true"
    if not require_auth:
        return None, None
    
    # Try API key authentication first
    if x_api_key:
        api_key = await rbac_manager.validate_api_key(x_api_key)
        if api_key:
            user = await rbac_manager.get_user(api_key.user_id)
            if user and user.is_active:
                # Log authentication event
                if audit_logger:
                    await audit_logger.log_authentication(
                        action="api_key_used",
                        organization_id=user.organization_id,
                        user_id=user.id,
                        api_key_id=api_key.id,
                        ip_address=request.client.host if request and request.client else None,
                        user_agent=request.headers.get("user-agent") if request else None
                    )
                return user, api_key
        else:
            # Log failed authentication
            if audit_logger:
                await audit_logger.log_authentication(
                    action="api_key_invalid",
                    organization_id="unknown",
                    ip_address=request.client.host if request and request.client else None,
                    error_message="Invalid API key"
                )
    
    # Legacy token authentication (for backwards compatibility)
    expected_key = os.getenv("HTTP_AUTH_TOKEN")
    if expected_key and x_api_key == expected_key:
        # Create a pseudo-user for legacy auth
        return None, None  # Indicates legacy auth success
    
    return None, None


async def verify_api_key(
    x_api_key: Optional[str] = Header(default=None), 
    authorization: Optional[str] = Header(default=None),
    request=None
) -> bool:
    """Legacy API key verification for backwards compatibility."""
    user, api_key = await get_current_user(x_api_key, authorization, request)
    
    # If we have a valid user/API key, allow access
    if user and api_key:
        return True
    
    # Check legacy auth
    require_auth = os.getenv("HTTP_REQUIRE_AUTH", "false").lower() == "true"
    if not require_auth:
        return True
    
    expected_key = os.getenv("HTTP_AUTH_TOKEN")
    if expected_key and x_api_key == expected_key:
        return True
    
    return False


def require_permission(permission: 'Permission'):
    """Dependency factory returning an async dependency callable.

    FastAPI expects a callable (not a coroutine object) in Depends().
    """
    async def check_permission(
        user_and_key: tuple = Depends(get_current_user),
        request=None,
    ):
        user, api_key = user_and_key
        
        # Legacy auth bypass
        if not user and not api_key:
            legacy_auth = await verify_api_key(request=request)
            if legacy_auth:
                return True
        
        if not user or not rbac_manager:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Check permission
        has_permission = await rbac_manager.check_permission(
            user.id, 
            permission, 
            api_key.id if api_key else None
        )
        
        if not has_permission:
            # Log permission denied
            if audit_logger:
                await audit_logger.log_action(
                    action="permission_denied",
                    resource_type="permission",
                    organization_id=user.organization_id,
                    user_id=user.id,
                    api_key_id=api_key.id if api_key else None,
                    ip_address=request.client.host if request and request.client else None,
                    metadata={"required_permission": permission.value}
                )
            
            raise HTTPException(
                status_code=403, 
                detail=f"Permission {permission.value} required"
            )
        
        return user

    return check_permission


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
            arguments=request.model_dump(),
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry_manager,
        )
        return JSONResponse(content=_sanitize_for_json(result))
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
            arguments=request.model_dump(),
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry_manager,
        )
        return JSONResponse(content=_sanitize_for_json(result))
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
            arguments=request.model_dump(), llm_client=llm_client, cache_manager=cache_manager
        )
        return JSONResponse(content=_sanitize_for_json(result))
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
        result = await learn_schema_tool(arguments=request.model_dump(), llm_client=llm_client)
        return JSONResponse(content=_sanitize_for_json(result))
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
                    arguments=request.model_dump(),
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


# Admin endpoints for user and organization management
@app.post("/v1/admin/organizations", tags=["admin"])
async def create_organization(
    org_data: dict,
    user: 'User' = Depends(require_permission('Permission.ADMIN_ORGS'))
):
    """Create a new organization."""
    if not rbac_manager:
        raise HTTPException(status_code=503, detail="RBAC not initialized")
    
    try:
        org = await rbac_manager.create_organization(
            name=org_data["name"],
            domain=org_data.get("domain"),
            plan=org_data.get("plan", "free")
        )
        
        if audit_logger:
            await audit_logger.log_admin_action(
                action="create_organization",
                target_resource_type="organization",
                target_resource_id=org.id,
                organization_id=user.organization_id,
                admin_user_id=user.id,
                changes={"name": org.name, "plan": org.plan}
            )
        
        return org.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/admin/users", tags=["admin"])
async def create_user(
    user_data: dict,
    current_user: 'User' = Depends(require_permission('Permission.ADMIN_USERS'))
):
    """Create a new user."""
    if not rbac_manager:
        raise HTTPException(status_code=503, detail="RBAC not initialized")
    
    try:
        from .auth.models import UserRole
        
        user = await rbac_manager.create_user(
            email=user_data["email"],
            name=user_data["name"],
            organization_id=user_data["organization_id"],
            role=UserRole(user_data.get("role", "viewer"))
        )
        
        if audit_logger:
            await audit_logger.log_admin_action(
                action="create_user",
                target_resource_type="user",
                target_resource_id=user.id,
                organization_id=current_user.organization_id,
                admin_user_id=current_user.id,
                changes={"email": user.email, "role": user.role.value}
            )
        
        return user.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/admin/api-keys", tags=["admin"])
async def create_api_key(
    key_data: dict,
    current_user: 'User' = Depends(require_permission('Permission.ADMIN_USERS'))
):
    """Create API key for a user."""
    if not rbac_manager:
        raise HTTPException(status_code=503, detail="RBAC not initialized")
    
    try:
        from .auth.models import Permission
        
        permissions = []
        if key_data.get("permissions"):
            permissions = [Permission(p) for p in key_data["permissions"]]
        
        api_key, raw_key = await rbac_manager.create_api_key(
            user_id=key_data["user_id"],
            name=key_data["name"],
            permissions=permissions,
            expires_in_days=key_data.get("expires_in_days")
        )
        
        if audit_logger:
            await audit_logger.log_admin_action(
                action="create_api_key",
                target_resource_type="api_key",
                target_resource_id=api_key.id,
                organization_id=current_user.organization_id,
                admin_user_id=current_user.id,
                changes={"user_id": api_key.user_id, "name": api_key.name}
            )
        
        return {
            "api_key_id": api_key.id,
            "api_key": raw_key,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "permissions": [p.value for p in api_key.permissions]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/admin/organizations/{org_id}/stats", tags=["admin"])
async def get_organization_stats(
    org_id: str,
    user: 'User' = Depends(require_permission('Permission.ADMIN_ORGS'))
):
    """Get organization statistics."""
    if not rbac_manager:
        raise HTTPException(status_code=503, detail="RBAC not initialized")
    
    # Check if user can access this org (admin can see all, others only their own)
    if user.role != 'UserRole.SUPER_ADMIN' and user.organization_id != org_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    stats = await rbac_manager.get_organization_stats(org_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    return stats


@app.get("/v1/admin/audit-logs", tags=["admin"])
async def get_audit_logs(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    action: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100,
    current_user: 'User' = Depends(require_permission('Permission.ADMIN_SYSTEM'))
):
    """Get audit logs for organization."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logging not initialized")
    
    try:
        # Parse time filters
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        
        logs = await audit_logger.search_logs(
            organization_id=current_user.organization_id,
            start_time=start_dt,
            end_time=end_dt,
            action=action,
            user_id=user_id,
            limit=limit
        )
        
        return [log.model_dump() for log in logs]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/admin/security-events", tags=["admin"])
async def get_security_events(
    hours: int = 24,
    current_user: 'User' = Depends(require_permission('Permission.ADMIN_SYSTEM'))
):
    """Get security events for the last N hours."""
    if not audit_logger:
        raise HTTPException(status_code=503, detail="Audit logging not initialized")
    
    events = await audit_logger.get_security_events(
        organization_id=current_user.organization_id,
        hours=hours
    )
    
    return [event.model_dump() for event in events]


@app.get("/v1/profile", tags=["users"])
async def get_user_profile(
    user_and_key: tuple = Depends(get_current_user)
):
    """Get current user profile."""
    user, api_key = user_and_key
    
    if not user:
        # Legacy auth fallback
        return {"message": "Legacy authentication - no profile available"}
    
    if not rbac_manager:
        raise HTTPException(status_code=503, detail="RBAC not initialized")
    
    stats = await rbac_manager.get_user_stats(user.id)
    permissions = await rbac_manager.get_user_permissions(user.id)
    
    return {
        "user": user.model_dump(),
        "api_key": api_key.model_dump() if api_key else None,
        "stats": stats,
        "permissions": [p.value for p in permissions]
    }


# AI Optimization endpoints
@app.get("/v1/ai/routing-stats", tags=["ai-optimization"])
async def get_routing_stats(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Get adaptive routing statistics."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not adaptive_router:
        return {"message": "Adaptive routing not enabled"}
    
    return await adaptive_router.get_routing_stats()


@app.get("/v1/ai/cache-stats", tags=["ai-optimization"])
async def get_smart_cache_stats(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Get smart cache statistics."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not smart_cache:
        return {"message": "Smart cache not enabled"}
    
    return await smart_cache.get_cache_stats()


@app.post("/v1/ai/cache/preload", tags=["ai-optimization"])
async def preload_cache(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit)
):
    """Trigger predictive cache preloading."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not smart_cache:
        raise HTTPException(status_code=503, detail="Smart cache not enabled")
    
    preloaded_count = await smart_cache.preload_predicted_keys()
    return {"preloaded_keys": preloaded_count}


@app.get("/v1/ai/cost-insights", tags=["ai-optimization"])
async def get_cost_insights(
    user_and_key: tuple = Depends(get_current_user)
):
    """Get cost insights for organization."""
    user, api_key = user_and_key
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not cost_optimizer:
        return {"message": "Cost optimization not enabled"}
    
    return await cost_optimizer.get_cost_insights(user.organization_id)


@app.post("/v1/ai/cost/budget", tags=["ai-optimization"])
async def set_cost_budget(
    budget_data: dict,
    user: 'User' = Depends(require_permission('Permission.ADMIN_BILLING'))
):
    """Set monthly cost budget for organization."""
    if not cost_optimizer:
        raise HTTPException(status_code=503, detail="Cost optimization not enabled")
    
    monthly_budget = budget_data.get("monthly_budget")
    if not monthly_budget or monthly_budget <= 0:
        raise HTTPException(status_code=400, detail="Valid monthly_budget required")
    
    await cost_optimizer.set_budget(user.organization_id, monthly_budget)
    
    return {"message": f"Budget set to ${monthly_budget:.2f}/month"}


@app.post("/v1/ai/model-recommendation", tags=["ai-optimization"])
async def get_model_recommendation(
    request_data: dict,
    user_and_key: tuple = Depends(get_current_user)
):
    """Get AI-powered model recommendation."""
    user, api_key = user_and_key
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not cost_optimizer or not adaptive_router:
        raise HTTPException(status_code=503, detail="AI optimization not enabled")
    
    try:
        from .ai_optimization.cost_optimizer import OptimizationStrategy
        
        # Extract parameters
        task_complexity = request_data.get("task_complexity", "medium")
        quality_requirement = request_data.get("quality_requirement", 0.7)
        latency_requirement = request_data.get("latency_requirement", 5000)
        strategy = OptimizationStrategy(request_data.get("strategy", "balanced"))
        available_models = request_data.get("available_models", [
            ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
            ("openrouter", "qwen/qwen-2.5-72b-instruct"),
            ("ollama", "llama3.1:8b")
        ])
        
        backend, model, details = await cost_optimizer.recommend_model(
            task_complexity=task_complexity,
            quality_requirement=quality_requirement,
            latency_requirement=latency_requirement,
            organization_id=user.organization_id,
            available_models=available_models,
            strategy=strategy
        )
        
        return {
            "recommended_backend": backend,
            "recommended_model": model,
            "recommendation_details": details
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/ai/optimization-stats", tags=["ai-optimization"])
async def get_ai_optimization_stats(
    user: 'User' = Depends(require_permission('Permission.METRICS_READ'))
):
    """Get comprehensive AI optimization statistics."""
    stats = {
        "adaptive_routing": {},
        "smart_cache": {},
        "cost_optimization": {},
        "enabled_features": []
    }
    
    if adaptive_router:
        stats["adaptive_routing"] = await adaptive_router.get_routing_stats()
        stats["enabled_features"].append("adaptive_routing")
    
    if smart_cache:
        stats["smart_cache"] = await smart_cache.get_cache_stats()
        stats["enabled_features"].append("smart_cache")
    
    if cost_optimizer:
        stats["cost_optimization"] = await cost_optimizer.get_optimizer_stats()
        stats["enabled_features"].append("cost_optimization")
    
    return stats


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
