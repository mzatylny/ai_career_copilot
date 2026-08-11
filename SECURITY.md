# Security policy

## Supported version

Security fixes are applied to the latest version on `main`.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature in the repository Security tab. Do not open a public issue containing secrets, personal CV data, exploit details, or uploaded documents.

## Deployment boundary

`AI_COPILOT_TENANT_KEYS` maps unique API keys to tenant identities, and production rejects credentials shorter than 32 characters. Server-generated session ownership is enforced in SQLite before Chroma access. Ingestion and deletion share a per-session lock so deleted documents cannot be recreated by a queued local job. The included limiter, background runner, metadata database, and object store are intentionally single-instance implementations. Public horizontal deployments still require an identity-aware gateway, distributed limits, managed persistence, a durable queue, TLS termination, and secret management.

Review the complete [`threat model`](docs/THREAT_MODEL.md) before deployment.
