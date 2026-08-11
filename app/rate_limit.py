from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    """Small single-process limiter; replace with Redis for multi-instance deployments."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str, now: float | None = None) -> tuple[bool, int]:
        timestamp = now if now is not None else time.monotonic()
        cutoff = timestamp - 60.0

        with self._lock:
            events = self._events[identity]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= self.requests_per_minute:
                retry_after = max(1, int(60 - (timestamp - events[0])))
                return False, retry_after

            events.append(timestamp)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
