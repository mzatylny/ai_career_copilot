# Security policy

## Supported version

Security fixes are applied to the latest version on `main`.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature in the repository Security tab. Do not open a public issue containing secrets, personal CV data, exploit details, or uploaded documents.

## Deployment boundary

`AI_COPILOT_TENANT_KEYS` maps unique API keys to tenant identities, and production rejects credentials shorter than 32 characters. Server-generated session ownership is enforced in SQLite before Chroma access. Ingestion and deletion share a per-session lock so deleted documents cannot be recreated by a queued local job. The included limiter, background runner, metadata database, and object store are intentionally single-instance implementations. Public horizontal deployments still require an identity-aware gateway, distributed limits, managed persistence, a durable queue, TLS termination, and secret management.

Review the complete [`threat model`](docs/THREAT_MODEL.md) before deployment.

## Unfixed upstream ChromaDB advisories

As of 2026-08-29, the latest compatible ChromaDB release (`1.5.9`) is reported by
`pip-audit` under PYSEC-2026-311 and CVE-2026-45830, CVE-2026-45831, and
CVE-2026-45833. No fixed release is listed. The advisories concern Chroma's HTTP server,
remote model loading, and server-side authorization providers. This application uses an
in-process `PersistentClient` and does not expose Chroma's HTTP API, which keeps those
server entry points outside the included deployment boundary.

The dependency audit remains fail-closed and these findings are not ignored. Do not expose
a separate Chroma service from this deployment. Upgrade ChromaDB as soon as a fixed release
is available and rerun the full tenant-isolation and RAG test suites before deployment.
