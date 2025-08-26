"""Collection of APIRouter modules for MCP-SGR."""

from .reasoning import router as reasoning_router
from .agents import router as agents_router

__all__ = ["reasoning_router", "agents_router"]