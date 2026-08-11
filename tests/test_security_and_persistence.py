import threading
from pathlib import Path

from app.auth import APIKeyRegistry
from app.config import Settings
from app.object_store import LocalObjectStore
from app.rate_limit import SlidingWindowRateLimiter
from app.session_locks import SessionMutationCoordinator
from app.session_store import SessionStore


def test_production_key_strength_is_measurable():
    weak = APIKeyRegistry(Settings(_env_file=None, AI_COPILOT_TENANT_KEYS="alpha:short-key"))
    strong = APIKeyRegistry(Settings(_env_file=None, AI_COPILOT_TENANT_KEYS=f"alpha:{'x' * 32}"))

    assert weak.meets_minimum_key_length() is False
    assert strong.meets_minimum_key_length() is True


def test_legacy_key_cannot_duplicate_a_tenant_key():
    settings = Settings(
        _env_file=None,
        AI_COPILOT_API_KEY="shared-key",
        AI_COPILOT_TENANT_KEYS="alpha:shared-key",
    )

    try:
        APIKeyRegistry(settings)
    except ValueError as exc:
        assert "must not duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate legacy and tenant credentials were accepted")


def test_rate_limiter_enforces_sliding_window():
    limiter = SlidingWindowRateLimiter(2)

    assert limiter.check("tenant", now=0) == (True, 0)
    assert limiter.check("tenant", now=1) == (True, 0)
    allowed, retry_after = limiter.check("tenant", now=2)
    assert allowed is False
    assert retry_after > 0
    assert limiter.check("tenant", now=61) == (True, 0)


def test_session_store_enforces_owner_and_job_visibility(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create_session("alpha")
    job = store.create_job(session_id, "alpha", "resume.pdf")

    assert store.ensure_owner(session_id, "alpha") is True
    assert store.ensure_owner(session_id, "beta") is False
    assert store.get_job(job.job_id, "alpha") is not None
    assert store.get_job(job.job_id, "beta") is None

    store.update_job(job.job_id, status="processing")
    store.update_job(job.job_id, status="completed", chunks_indexed=7)
    completed = store.get_job(job.job_id, "alpha")
    assert completed is not None
    assert completed.status == "completed"
    assert completed.chunks_indexed == 7
    assert store.ping() is True


def test_session_deletion_cancels_queued_job_metadata(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create_session("alpha")
    job = store.create_job(session_id, "alpha", "resume.pdf")

    assert store.delete_session(session_id, "alpha") is True
    assert store.get_job_for_worker(job.job_id) is None


def test_job_state_machine_rejects_terminal_rewrites(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    session_id = store.create_session("alpha")
    job = store.create_job(session_id, "alpha", "resume.pdf")
    store.update_job(job.job_id, status="failed", error="invalid PDF")

    try:
        store.update_job(job.job_id, status="processing")
    except ValueError as exc:
        assert "Invalid ingestion job transition" in str(exc)
    else:
        raise AssertionError("Terminal ingestion job was rewritten")


def test_session_mutations_are_serialized_and_lock_entries_are_released():
    coordinator = SessionMutationCoordinator()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_worker():
        with coordinator.hold("session-a"):
            first_entered.set()
            release_first.wait(timeout=2)

    def second_worker():
        first_entered.wait(timeout=2)
        with coordinator.hold("session-a"):
            second_entered.set()

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    second.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert second_entered.is_set()
    assert coordinator.active_sessions() == 0


def test_local_object_store_rejects_unsafe_identifiers(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4")
    store = LocalObjectStore(tmp_path / "objects")

    stored = store.put("safe-object_1", source)
    assert stored.read_bytes() == b"%PDF-1.4"

    try:
        store.put("../unsafe", source)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe object identifier was accepted")

    store.delete("safe-object_1")
    assert not Path(stored).exists()
