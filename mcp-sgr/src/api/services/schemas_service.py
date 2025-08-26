"""Service layer for schema management operations."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException

from ...http_server import (
    learn_schema_tool,  # type: ignore
    llm_client,
    cache_manager,
    logger,
    LearnSchemaRequest,
)

from ...schemas import SCHEMA_REGISTRY  # local registry


def list_schemas_service() -> Dict[str, Any]:
    """Return metadata about available schemas."""
    schemas: Dict[str, Any] = {}
    for name, schema_factory in SCHEMA_REGISTRY.items():
        schema = schema_factory()
        schemas[name] = {
            "description": schema.get_description(),
            "fields": [f.name for f in schema.get_fields()],
        }
    return schemas


async def learn_schema_service(request: LearnSchemaRequest) -> Dict[str, Any]:
    """Learn new schema from provided examples via tool call."""
    if not llm_client:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return await learn_schema_tool(arguments=request.dict(), llm_client=llm_client)  # type: ignore[arg-type]
    except Exception as exc:
        logger.error("Error in learn_schema_service: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc