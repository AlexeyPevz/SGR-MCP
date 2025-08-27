"""Service layer for monitoring and stats endpoints."""
from __future__ import annotations

from typing import Any, Dict
from datetime import datetime

from fastapi import HTTPException

from ...http_server import cache_manager, llm_client, telemetry_manager, logger


def cache_stats_service() -> Dict[str, Any]:
    if not cache_manager:
        return {"enabled": False}
    return asyncio.run(cache_manager.get_cache_stats())  # sync wrapper


def traces_service(limit: int = 10, tool_name: str | None = None):
    if not cache_manager:
        return []
    return asyncio.run(cache_manager.get_recent_traces(limit=limit, tool_name=tool_name))


import asyncio


def performance_stats_service() -> Dict[str, Any]:
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    return llm_client.get_performance_stats()


def reset_performance_stats_service():
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    llm_client.reset_performance_stats()
    return {"message": "Performance statistics reset successfully"}


async def detailed_health_check_service(timeout: float = 10.0):
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    health_status = await llm_client.health_check(timeout)
    all_healthy = all(status.get("status") == "healthy" for status in health_status.values())
    return {
        "overall_status": "healthy" if all_healthy else "degraded",
        "backends": health_status,
        "timestamp": datetime.utcnow().isoformat(),
    }