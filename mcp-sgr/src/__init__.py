"""MCP-SGR: Structured Guided Reasoning middleware for LLM agents."""

__version__ = "0.1.0"
__author__ = "MCP-SGR Team"

from .server import SGRServer

__all__ = ["SGRServer"]

# Re-export router factory for app
from fastapi import FastAPI


def include_routers(app: FastAPI) -> None:  # pragma: no cover
    """Include all APIRouter modules into given FastAPI app."""
    from .api.routers import reasoning  # noqa: F401

    for router in [reasoning.router]:
        app.include_router(router)
