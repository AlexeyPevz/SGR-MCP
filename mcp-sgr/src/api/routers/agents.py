"""Agent wrapper API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...http_server import (
    verify_api_key,
    verify_rate_limit,
    logger,
    WrapAgentRequest,
)
from ..services import agents_service

router = APIRouter(prefix="/v1", tags=["agents"])


@router.post("/wrap-agent")
async def wrap_agent(
    request: WrapAgentRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    """Wrap an agent call with SGR pre/post analysis."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        return await agents_service.wrap_agent_service(request)
    except Exception as exc:  # pragma: no cover
        logger.error("Error in wrap_agent: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))