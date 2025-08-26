"""Telemetry manager for observability."""

import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,  # type: ignore[import-not-found]
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Manages telemetry and observability."""

    def __init__(self):
        """Initialize telemetry manager."""
        self.enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "mcp-sgr")
        self.endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        self.tracer = None
        self._spans = {}  # Track active spans
        self._initialized = False

    async def initialize(self):
        """Initialize telemetry provider."""
        if self._initialized:
            return

        if self.enabled and OTEL_AVAILABLE:
            try:
                # Create resource
                resource = Resource.create(
                    {
                        ResourceAttributes.SERVICE_NAME: self.service_name,
                        ResourceAttributes.SERVICE_VERSION: "0.1.0",
                    }
                )

                # Setup tracer provider
                provider = TracerProvider(resource=resource)

                # Setup exporter
                exporter = OTLPSpanExporter(endpoint=self.endpoint, insecure=True)

                # Add span processor
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)

                # Set global provider
                trace.set_tracer_provider(provider)

                # Get tracer
                self.tracer = trace.get_tracer(__name__)

                self._initialized = True
                logger.info(f"Telemetry initialized with endpoint: {self.endpoint}")

            except Exception as e:
                logger.error(f"Failed to initialize telemetry: {e}")
                self.enabled = False
        else:
            if self.enabled and not OTEL_AVAILABLE:
                logger.warning("OpenTelemetry not available, disabling telemetry")
                self.enabled = False

    async def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> str:
        """Start a new span.

        Args:
            name: Span name
            attributes: Initial attributes

        Returns:
            Span ID for tracking
        """
        span_id = str(uuid.uuid4())

        if self.enabled and self.tracer:
            try:
                span = self.tracer.start_span(name)

                # Set attributes
                if attributes:
                    for key, value in attributes.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(key, value)
                        else:
                            span.set_attribute(key, str(value))

                self._spans[span_id] = {"span": span, "start_time": time.time(), "name": name}

            except Exception as e:
                logger.error(f"Failed to start span: {e}")
        else:
            # Track locally even if telemetry is disabled
            self._spans[span_id] = {
                "span": None,
                "start_time": time.time(),
                "name": name,
                "attributes": attributes or {},
            }

        return span_id

    async def end_span(self, span_id: str, attributes: Optional[Dict[str, Any]] = None):
        """End a span.

        Args:
            span_id: Span ID to end
            attributes: Additional attributes to set
        """
        if span_id not in self._spans:
            logger.warning(f"Unknown span ID: {span_id}")
            return

        span_info = self._spans.pop(span_id)
        duration_ms = int((time.time() - span_info["start_time"]) * 1000)

        if self.enabled and span_info.get("span"):
            try:
                span = span_info["span"]

                # Set final attributes
                if attributes:
                    for key, value in attributes.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(key, value)
                        else:
                            span.set_attribute(key, str(value))

                span.set_attribute("duration.ms", duration_ms)

                # End the span
                span.end()

            except Exception as e:
                logger.error(f"Failed to end span: {e}")

        # Log locally
        logger.debug(
            f"Span ended: {span_info['name']} "
            f"(duration: {duration_ms}ms, attributes: {attributes})"
        )

    async def record_error(self, span_id: str, error: Exception):
        """Record an error in a span.

        Args:
            span_id: Span ID
            error: Exception to record
        """
        if span_id not in self._spans:
            return

        if self.enabled and self._spans[span_id].get("span"):
            try:
                span = self._spans[span_id]["span"]
                span.record_exception(error)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
            except Exception as e:
                logger.error(f"Failed to record error: {e}")

    async def add_event(self, span_id: str, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to a span.

        Args:
            span_id: Span ID
            name: Event name
            attributes: Event attributes
        """
        if span_id not in self._spans:
            return

        if self.enabled and self._spans[span_id].get("span"):
            try:
                span = self._spans[span_id]["span"]
                span.add_event(name, attributes=attributes or {})
            except Exception as e:
                logger.error(f"Failed to add event: {e}")

    def get_trace_id(self, span_id: str) -> Optional[str]:
        """Get trace ID for a span.

        Args:
            span_id: Span ID

        Returns:
            Trace ID if available
        """
        if span_id not in self._spans:
            return None

        if self.enabled and self._spans[span_id].get("span"):
            try:
                span = self._spans[span_id]["span"]
                context = span.get_span_context()
                return format(context.trace_id, "032x")
            except Exception as e:
                logger.error(f"Failed to get trace id: {e}")

        return None

    async def create_metric(
        self, name: str, value: float, unit: str = "", labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric (placeholder for future metrics support).

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            labels: Metric labels
        """
        # Record metric using OpenTelemetry
        if not self.enabled:
            return

        try:
            from opentelemetry import metrics

            if not hasattr(self, "_meter"):
                self._meter = metrics.get_meter("mcp-sgr", "1.0.0")
                self._counters = {}
                self._histograms = {}
                self._gauges = {}
                self._last_gauge_values = {}

            # Determine metric type and record
            if name.endswith("_total") or name.endswith("_count"):
                # Counter
                if name not in self._counters:
                    self._counters[name] = self._meter.create_counter(
                        name=name, description=f"Counter for {name}", unit=unit
                    )
                self._counters[name].add(value, labels)

            elif name.endswith("_duration") or name.endswith("_latency"):
                # Histogram
                if name not in self._histograms:
                    self._histograms[name] = self._meter.create_histogram(
                        name=name, description=f"Histogram for {name}", unit=unit
                    )
                self._histograms[name].record(value, labels)

            else:
                # Gauge (up-down counter)
                if name not in self._gauges:
                    self._gauges[name] = self._meter.create_up_down_counter(
                        name=name, description=f"Gauge for {name}", unit=unit
                    )
                previous = self._last_gauge_values.get(name, 0)
                delta = value - previous
                if delta != 0:
                    self._gauges[name].add(delta, labels)
                self._last_gauge_values[name] = value

        except ImportError:
            logger.debug(f"OpenTelemetry metrics not available. Metric: {name}={value}{unit}")
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")

    async def close(self):
        """Close telemetry and flush remaining data."""
        # End any remaining spans
        for span_id in list(self._spans.keys()):
            await self.end_span(span_id, {"closed": "forced"})

        if self.enabled and OTEL_AVAILABLE:
            try:
                # Force flush
                provider = trace.get_tracer_provider()
                if hasattr(provider, "force_flush"):
                    provider.force_flush()
            except Exception as e:
                logger.error(f"Error during telemetry close: {e}")
