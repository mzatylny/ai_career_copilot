from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from app.config import Settings


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    authenticated: bool


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class APIKeyRegistry:
    """Resolve API keys to tenants without retaining plaintext keys in memory."""

    def __init__(self, settings: Settings):
        entries = settings.tenant_api_keys
        self._keys = {_fingerprint(key): tenant for tenant, key in entries.items()}
        key_lengths = [len(key) for key in entries.values()]

        legacy = settings.api_access_key
        if legacy is not None:
            legacy_value = legacy.get_secret_value()
            legacy_fingerprint = _fingerprint(legacy_value)
            if legacy_fingerprint in self._keys:
                raise ValueError("AI_COPILOT_API_KEY must not duplicate a tenant API key")
            self._keys[legacy_fingerprint] = "default"
            key_lengths.append(len(legacy_value))
        self._minimum_key_length = min(key_lengths, default=0)

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def meets_minimum_key_length(self, minimum: int = 32) -> bool:
        return self.enabled and self._minimum_key_length >= minimum

    def resolve(self, supplied_key: str | None) -> Principal | None:
        if not self.enabled:
            return Principal(tenant_id="local-demo", authenticated=False)
        if not supplied_key:
            return None

        supplied_fingerprint = _fingerprint(supplied_key)
        for expected_fingerprint, tenant_id in self._keys.items():
            if secrets.compare_digest(supplied_fingerprint, expected_fingerprint):
                return Principal(tenant_id=tenant_id, authenticated=True)
        return None


def principal_from_request(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return principal
