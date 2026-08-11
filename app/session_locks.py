from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class _Entry:
    lock: threading.Lock
    users: int = 0


class SessionMutationCoordinator:
    """Serialize ingestion and deletion for one session inside a service instance."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    @contextmanager
    def hold(self, session_id: str) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = _Entry(lock=threading.Lock())
                self._entries[session_id] = entry
            entry.users += 1

        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(session_id, None)

    def active_sessions(self) -> int:
        with self._guard:
            return len(self._entries)
