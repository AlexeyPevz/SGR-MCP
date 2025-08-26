"""Prompt enhancement API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...http_server import (
    verify_api_key,
    verify_rate_limit,
    logger,
    EnhancePromptRequest,
)
from ..services import prompts_service

router = APIRouter(prefix="/v1", tags=["prompts"])


@router.post("/enhance-prompt")
async def enhance_prompt(
    request: EnhancePromptRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    """Enhance a prompt with SGR technique."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        return await prompts_service.enhance_prompt_service(request)
    except Exception as exc:  # pragma: no cover
        logger.error("Error in enhance_prompt endpoint: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))