"""Service layer for prompt enhancement operations."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from ...http_server import (
    enhance_prompt_tool,  # type: ignore
    llm_client,
    cache_manager,
    logger,
    EnhancePromptRequest,
)


async def enhance_prompt_service(request: EnhancePromptRequest) -> Dict[str, Any]:
    """Enhance a prompt using SGR-backed logic."""
    if not llm_client or not cache_manager:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return await enhance_prompt_tool(  # type: ignore[arg-type]
            arguments=request.dict(),
            llm_client=llm_client,
            cache_manager=cache_manager,
        )
    except Exception as exc:
        logger.error("Error in enhance_prompt_service: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc