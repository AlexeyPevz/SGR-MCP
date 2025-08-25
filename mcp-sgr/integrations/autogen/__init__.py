"""AutoGen integration for MCP-SGR."""

from .sgr_autogen import (
    SGRAgent,
    SGRGroupChatManager,
    create_sgr_assistant,
    wrap_autogen_agent
)

__all__ = [
    "SGRAgent",
    "SGRGroupChatManager",
    "create_sgr_assistant",
    "wrap_autogen_agent"
]