"""Collection of APIRouter modules for MCP-SGR."""

from .reasoning import router as reasoning_router
from .agents import router as agents_router
from .prompts import router as prompts_router

__all__ = ["reasoning_router", "agents_router", "prompts_router"]