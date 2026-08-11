from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

JOB_TRANSITIONS = {
    "queued": {"processing", "failed"},
    "processing": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    session_id: str
    tenant_id: str
    filename: str
    status: str
    chunks_indexed: int
    error: str | None
    created_at: str
    updated_at: str


class SessionStore:
    """SQLite ownership and job metadata with one short-lived connection per operation."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_tenant
                    ON sessions (tenant_id);

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'processing', 'completed', 'failed')
                    ),
                    chunks_indexed INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_tenant
                    ON ingestion_jobs (tenant_id, created_at);
                """
            )

    def create_session(self, tenant_id: str) -> str:
        session_id = secrets.token_urlsafe(18)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (session_id, tenant_id, created_at) VALUES (?, ?, ?)",
                (session_id, tenant_id, _utc_now()),
            )
        return session_id

    def ensure_owner(self, session_id: str, tenant_id: str, *, create: bool = False) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT tenant_id FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None and create:
                connection.execute(
                    "INSERT INTO sessions (session_id, tenant_id, created_at) VALUES (?, ?, ?)",
                    (session_id, tenant_id, _utc_now()),
                )
                return True
            return row is not None and secrets.compare_digest(str(row["tenant_id"]), tenant_id)

    def create_job(self, session_id: str, tenant_id: str, filename: str) -> JobRecord:
        job_id = secrets.token_urlsafe(18)
        timestamp = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, session_id, tenant_id, filename, status,
                    chunks_indexed, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', 0, NULL, ?, ?)
                """,
                (job_id, session_id, tenant_id, filename, timestamp, timestamp),
            )
        job = self.get_job(job_id, tenant_id)
        if job is None:
            raise RuntimeError("Failed to persist ingestion job")
        return job

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        chunks_indexed: int = 0,
        error: str | None = None,
    ) -> bool:
        if status not in JOB_TRANSITIONS:
            raise ValueError(f"Unknown ingestion job status: {status}")
        with self._connect() as connection:
            current = connection.execute(
                "SELECT status FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if current is None:
                return False
            current_status = str(current["status"])
            if status not in JOB_TRANSITIONS[current_status]:
                raise ValueError(f"Invalid ingestion job transition: {current_status} -> {status}")
            cursor = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, chunks_indexed = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, chunks_indexed, error, _utc_now(), job_id),
            )
        return cursor.rowcount == 1

    def get_job(self, job_id: str, tenant_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ? AND tenant_id = ?",
                (job_id, tenant_id),
            ).fetchone()
        return JobRecord(**dict(row)) if row else None

    def get_job_for_worker(self, job_id: str) -> JobRecord | None:
        """Internal lookup used only after a queued task has acquired its session lock."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return JobRecord(**dict(row)) if row else None

    def delete_session(self, session_id: str, tenant_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_id = ? AND tenant_id = ?",
                (session_id, tenant_id),
            )
        return cursor.rowcount > 0

    def ping(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT 1").fetchone()
        return row is not None and row[0] == 1
