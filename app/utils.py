from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Iterable


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    return name[:120] or f"upload_{secrets.token_hex(4)}.pdf"


def short_snippet(text: str, limit: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            output.append(item.strip())
    return output
