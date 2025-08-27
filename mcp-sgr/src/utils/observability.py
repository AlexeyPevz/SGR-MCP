"""Unified observability utilities – logging & OpenTelemetry setup."""
from __future__ import annotations

from .logging_config import setup_logging as _setup_logging
from .telemetry import TelemetryManager

__all__ = ["setup_observability", "get_telemetry_manager"]


_telemetry_manager: TelemetryManager | None = None


def setup_observability() -> None:  # pragma: no cover
    """Configure logging stack and initialize global TelemetryManager."""
    global _telemetry_manager
    _setup_logging()
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
        import asyncio

        # Initialize in background event loop if already running, else new loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.create_task(_telemetry_manager.initialize())


def get_telemetry_manager() -> TelemetryManager | None:
    return _telemetry_manager