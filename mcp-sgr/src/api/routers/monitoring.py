"""Monitoring and stats API routes."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ...http_server import verify_api_key, verify_rate_limit, logger
from ..services import monitoring_service

router = APIRouter(prefix="/v1", tags=["monitoring"])


@router.get("/cache-stats")
async def cache_stats(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return monitoring_service.cache_stats_service()


@router.get("/traces")
async def traces(
    limit: int = 10,
    tool_name: Optional[str] = None,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return monitoring_service.traces_service(limit=limit, tool_name=tool_name)


@router.get("/performance-stats")
async def performance_stats(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return monitoring_service.performance_stats_service()


@router.post("/performance-stats/reset")
async def reset_performance_stats(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return monitoring_service.reset_performance_stats_service()


@router.get("/health-check")
async def detailed_health_check(
    timeout: float = 10.0,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return await monitoring_service.detailed_health_check_service(timeout)