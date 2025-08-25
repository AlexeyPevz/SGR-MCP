"""CrewAI integration for MCP-SGR."""

from .sgr_crewai import (
    SGRAgent,
    SGRCrew,
    create_sgr_agent,
    enhance_crew_with_sgr,
    enhance_task_description
)

__all__ = [
    "SGRAgent",
    "SGRCrew",
    "create_sgr_agent",
    "enhance_crew_with_sgr",
    "enhance_task_description"
]