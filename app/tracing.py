from __future__ import annotations

import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def configure_tracing(app: FastAPI, *, endpoint: str | None, service_name: str) -> None:
    """Enable OTLP tracing when configured, while keeping local demo startup dependency-light."""
    if not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OTLP endpoint configured but OpenTelemetry packages are unavailable")
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
