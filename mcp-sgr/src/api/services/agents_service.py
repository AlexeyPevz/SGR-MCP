"""Service layer for agent wrapper operations."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from ...http_server import (
    wrap_agent_call_tool,  # type: ignore
    llm_client,
    cache_manager,
    telemetry_manager,
    logger,
    WrapAgentRequest,
)


async def wrap_agent_service(request: WrapAgentRequest) -> Dict[str, Any]:
    """Wrap an agent call with pre/post SGR analysis."""
    if not llm_client or not cache_manager or not telemetry_manager:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return await wrap_agent_call_tool(  # type: ignore[arg-type]
            arguments=request.dict(),
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry_manager,
        )
    except Exception as exc:
        logger.error("Error in wrap_agent_service: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc