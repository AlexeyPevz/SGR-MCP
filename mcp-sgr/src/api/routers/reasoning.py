from __future__ import annotations

"""Reasoning-related API routes (SGR core operations)."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request

from ...http_server import (
    verify_api_key,
    verify_rate_limit,
    logger,
)
from ...http_server import ApplySGRRequest  # keep model import temporarily
from ..services import reasoning_service

router = APIRouter(prefix="/v1", tags=["reasoning"])


@router.post("/apply-sgr")
async def apply_sgr(
    request: ApplySGRRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    """Apply SGR schema to analyze a task."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        result = await reasoning_service.apply_sgr_service(request)
        return result
    except Exception as e:  # pragma: no cover
        logger.error(f"Error in apply_sgr: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-apply-sgr")
async def batch_apply_sgr(
    requests: List[ApplySGRRequest],
    max_concurrent: int = 5,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    """Apply SGR to multiple tasks concurrently."""
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if len(requests) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 20)")

    if max_concurrent > 10:
        max_concurrent = 10

    import asyncio

    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_request(req: ApplySGRRequest):
            async with semaphore:
                return await reasoning_service.apply_sgr_service(req)

        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                processed_results.append({"error": str(result), "request_index": i})
            else:
                processed_results.append(result)  # type: ignore[arg-type]

        return {
            "results": processed_results,
            "total_requests": len(requests),
            "successful_requests": sum(1 for r in results if not isinstance(r, Exception)),
            "failed_requests": sum(1 for r in results if isinstance(r, Exception)),
        }
    except Exception as e:  # pragma: no cover
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))