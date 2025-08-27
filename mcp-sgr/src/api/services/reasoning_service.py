"""Service layer for reasoning operations (SGR apply, batch, etc.)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from fastapi import HTTPException

from ...http_server import (
    apply_sgr_tool,  # type: ignore
    llm_client,
    cache_manager,
    telemetry_manager,
    logger,
)
from ...http_server import ApplySGRRequest  # re-export original model


async def apply_sgr_service(request: ApplySGRRequest) -> Dict[str, Any]:
    """Apply SGR to a single task and return structured result."""

    if not llm_client or not cache_manager or not telemetry_manager:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return await apply_sgr_tool(  # type: ignore[arg-type]
            arguments=request.model_dump(),
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry_manager,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Error in apply_sgr_service: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def batch_apply_sgr_service(
    requests: List[ApplySGRRequest],
    max_concurrent: int = 5,
) -> Dict[str, Any]:
    """Apply SGR concurrently to a batch of requests."""
    if not llm_client:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Service not initialized")

    if len(requests) > 20:
        raise HTTPException(status_code=400, detail="Batch size too large (max 20)")

    if max_concurrent > 10:
        max_concurrent = 10

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process(req: ApplySGRRequest):
        async with semaphore:
            return await apply_sgr_service(req)

    tasks = [_process(r) for r in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed: List[Dict[str, Any]] = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error("Batch request %d failed: %s", idx, res)
            processed.append({"error": str(res), "request_index": idx})
        else:
            processed.append(res)  # type: ignore[arg-type]

    return {
        "results": processed,
        "total_requests": len(requests),
        "successful_requests": sum(1 for r in results if not isinstance(r, Exception)),
        "failed_requests": sum(1 for r in results if isinstance(r, Exception)),
    }