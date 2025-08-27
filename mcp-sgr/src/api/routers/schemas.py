"""Schema management API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...http_server import (
    verify_api_key,
    verify_rate_limit,
    logger,
    LearnSchemaRequest,
)
from ..services import schemas_service

router = APIRouter(prefix="/v1", tags=["schemas"])


@router.get("/schemas")
async def list_schemas(
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return schemas_service.list_schemas_service()


@router.post("/learn-schema")
async def learn_schema(
    request: LearnSchemaRequest,
    authorized: bool = Depends(verify_api_key),
    _: bool = Depends(verify_rate_limit),
):
    if not authorized:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return await schemas_service.learn_schema_service(request)