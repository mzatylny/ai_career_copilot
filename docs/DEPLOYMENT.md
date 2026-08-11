# Deployment and operations

## Container release

CI runs linting, tests with coverage, Bandit, dependency auditing, and deterministic RAG evaluation. A tag such as `v3.0.0` triggers the container workflow, which publishes an image to GitHub Container Registry with provenance and an SBOM.

## Kubernetes demo deployment

1. Copy `deploy/kubernetes/secret.example.yaml` outside the repository and replace its example values.
2. Apply the secret, then `deploy/kubernetes/app.yaml`.
3. Terminate TLS and authenticate traffic at an ingress or API gateway.
4. Scrape `/metrics` from the cluster network and apply `prometheus-rules.yaml` when the Prometheus Operator is installed.
5. Install the `observability` extra and set `OTEL_EXPORTER_OTLP_ENDPOINT` to export distributed traces through OTLP/HTTP.

The manifest intentionally uses one replica because the included persistence and limiter are local. Before horizontal scaling, replace SQLite, local objects, Chroma persistence, in-process jobs, and the limiter with managed shared services.

The liveness probe uses `/api/health`; readiness uses `/api/ready` and removes the pod from service when local persistence is unavailable. Production startup also rejects missing or shorter-than-32-character API keys.

## Suggested service objectives

- Availability: 99.5% for the API excluding the external AI provider.
- Latency: p95 under 500 ms for health/session/document metadata and under 3 seconds for non-ingestion mock AI requests.
- Reliability: 99% of accepted ingestion jobs reach a terminal state within five minutes for documents inside configured bounds.
- Quality: retrieval recall@2 at or above the CI threshold on the versioned evaluation set.

## Load test

Install `.[loadtest]`, start the API in mock mode, then run:

```bash
locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
```

Use the results to record throughput, p50/p95/p99 latency, error rate, CPU, memory, and queue depth. Avoid using real CV data or a paid AI key during load tests.
