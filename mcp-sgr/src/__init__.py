"""MCP-SGR: Structured Guided Reasoning middleware for LLM agents."""

__version__ = "0.1.0"
__author__ = "MCP-SGR Team"

# Avoid importing server-level objects at module import time to prevent circular imports.
# Export utilities that are safe to import from submodules and tests.
__all__ = ["include_routers"]

# Re-export router factory for app
from fastapi import FastAPI


def include_routers(app: FastAPI) -> None:  # pragma: no cover
    """Include all APIRouter modules into given FastAPI app."""
    from .api.routers import reasoning, agents, prompts, schemas, monitoring  # noqa: F401

    for router in [
        reasoning.router,
        agents.router,
        prompts.router,
        schemas.router,
        monitoring.router,
    ]:
        app.include_router(router)
