# Move existing Chroma data to Qdrant (3.1)

Version 3.1 replaces ChromaDB with `qdrant-client` in persistent local mode.
New installations use `QDRANT_PATH=./qdrant_db` and
`QDRANT_COLLECTION=career_documents`. The health response now reports `qdrant`.
The dependency audit remains mandatory, with no ignored advisories.

## Existing installations

Stop the old application and workers for the entire migration. Back up the
Chroma directory, SQLite session database, object storage, and configuration
together. Export from a **copy** of the Chroma directory: opening an old store
with a Chroma client can apply its own storage migrations.

Keep the old Python environment for export. Create a **fresh** environment for
3.1 so the unused vulnerable Chroma package is not left installed. The exporter
is a standalone script and requires only the old environment's existing Chroma
installation; do not add Chroma to the new app or its CI dependencies.

From the new source checkout, using absolute paths to your old Python and backup:

```bash
/path/to/old-venv/bin/python scripts/migrate_vectors.py export \
  --chroma-path /path/to/backup-copy/chroma_db \
  --collection career_documents --dimensions 1536 \
  --output /path/to/private-backup/vectors.jsonl

python -m venv .venv-qdrant
.venv-qdrant/bin/python -m pip install -e '.[dev]'
.venv-qdrant/bin/python -m scripts.migrate_vectors import \
  --input /path/to/private-backup/vectors.jsonl \
  --qdrant-path /path/to/NEW/qdrant_db --collection career_documents
```

Use the original collection name and embedding dimensions if yours differ.
The exporter reads existing vectors; it does not call OpenAI or re-embed text.
The importer preserves external chunk IDs, text, session IDs, source names,
pages, snippets and vectors. Qdrant normalizes vectors for cosine search.
Internal point UUIDs are deterministic mappings of the original chunk IDs.

The import checks record count, checksum, unique IDs, session metadata and
embedding dimensions. It builds a temporary store and publishes the target
directory only after validation succeeds. Existing targets and export files
are never overwritten. If an import fails, fix the export and retry with a new
target; an interrupted process may leave a hidden `.NAME-migration-*` directory
which can be removed after confirming the process has stopped.

Replace `CHROMA_PATH`/`CHROMA_COLLECTION` in `.env` with the Qdrant variables and
the new target path. Old variable names deliberately cause a configuration
error so an upgrade cannot silently ignore a configured legacy database.
Keep `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, mock-embedding mode,
`SESSION_DATABASE_PATH`, tenant identities/keys and object-storage configuration
unchanged. The SQLite ownership database is **not** part of the vector export;
retain it to preserve access to existing sessions.

Update the Compose volume mount to the migrated Qdrant directory before starting
the new app. The Kubernetes example now uses `/app/data/qdrant` on the existing
data volume. Confirm document counts, a representative search in each tenant,
and session-specific deletion on a test copy before reopening access.
The JSONL export contains document text; protect it like the source backups.

## Rollback and limits

For rollback, stop 3.1 and restore the old application environment, configuration
and matching pre-migration backups. Writes made after migration do not replicate
back to Chroma. No production data is migrated automatically by an app upgrade.

Qdrant local mode is intended for this single-process portfolio deployment. It
locks its storage directory; run one app worker per directory. The application
shares one client and serializes its operations across worker threads. For large
collections or multiple replicas, adopt a managed/shared vector service and
benchmark retrieval before deployment. See the [Qdrant client local-mode guide](https://github.com/qdrant/qdrant-client#local-mode).
