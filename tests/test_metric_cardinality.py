from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import SecretStr

import app.main as main
from app.rate_limit import SlidingWindowRateLimiter


def test_unknown_and_rejected_paths_have_bounded_private_metrics(monkeypatch, caplog):
    registry = CollectorRegistry()
    counter = Counter("bounded_requests", "Requests", ("method", "route", "status"), registry=registry)
    histogram = Histogram("bounded_latency", "Latency", ("method", "route"), registry=registry)
    monkeypatch.setattr(main, "REQUEST_COUNT", counter)
    monkeypatch.setattr(main, "REQUEST_DURATION", histogram)
    monkeypatch.setattr(main.settings, "api_access_key", SecretStr("test-key"))
    monkeypatch.setattr(main, "rate_limiter", SlidingWindowRateLimiter(1))
    with TestClient(main.app) as client:
        for index in range(20):
            assert client.get(f"/private-marker-{index}").status_code == 404
            assert client.get(f"/api/private-marker-{index}").status_code in {401, 429}
            assert client.request(f"CUSTOM{index}", "/private-marker").status_code == 404
    samples = [sample for metric in counter.collect() for sample in metric.samples]
    assert {sample.labels["route"] for sample in samples} == {"unmatched"}
    assert {sample.labels["method"] for sample in samples} == {"GET", "OTHER"}
    assert len([s for s in samples if s.name.endswith("_total")]) == 4
    assert b"private-marker" not in generate_latest(registry)
    application_logs = [record for record in caplog.records if record.name == "app.main"]
    assert all(getattr(record, "path", "unmatched") == "unmatched" for record in application_logs)
