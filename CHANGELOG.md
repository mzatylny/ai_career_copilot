# Changelog

## 3.0.0

- Added tenant API-key identities, server-generated sessions, and enforced session ownership.
- Added bounded request limiting, OpenAI time/output controls, hashed embedding caching, structured logs, Prometheus metrics, optional OTLP tracing, and alert rules.
- Added asynchronous PDF ingestion with SQLite job metadata and a replaceable local object-store adapter.
- Added a responsive portfolio UI, deterministic RAG evaluation, load testing, a threat model, architecture decisions, deployment guidance, Kubernetes manifests, and a release container workflow.
- Preserved the free local mock demo and synchronous upload API for backwards compatibility.
- Added explicit generation/degradation and RAG grounding status to prevent fallback or unsupported output from appearing authoritative.
- Added extracted-text and chunk-count limits, strong production-key validation, readiness checks, and serialized ingestion/deletion.
- Hardened Compose with a read-only root filesystem, dropped capabilities, and no-new-privileges.

## 2.2.0

- Added optional API-key protection for data endpoints.
- Validated resource-limit configuration at startup.
- Added Python 3.11/3.12 CI, coverage enforcement, static security checks, dependency auditing, and Dependabot.
- Added container build exclusions and security/contribution guidance.
- Switched the example environment to safe mock defaults.
- Improved project documentation and removed duplicate package metadata.

## 2.1.0

- Added production-minded FastAPI validation, session-scoped document RAG, deterministic demo modes, tests, and container hardening.
