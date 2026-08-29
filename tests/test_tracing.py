import builtins
import logging

from fastapi import FastAPI

from app.tracing import configure_tracing


def test_tracing_degrades_safely_when_optional_packages_are_missing(monkeypatch, caplog):
    original_import = builtins.__import__

    def import_without_otlp(name, *args, **kwargs):
        if name == "opentelemetry.exporter.otlp.proto.http.trace_exporter":
            raise ImportError("optional tracing dependency is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_otlp)

    with caplog.at_level(logging.WARNING):
        configure_tracing(
            FastAPI(),
            endpoint="http://collector.example:4318/v1/traces",
            service_name="test-service",
        )

    assert "OpenTelemetry packages are unavailable" in caplog.text
