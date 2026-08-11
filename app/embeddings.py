from __future__ import annotations

import hashlib
import math
import re


def hash_embedding(text: str, dimensions: int) -> list[float]:
    """Deterministic dependency-free embedding for demos, tests and retrieval evaluation."""
    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
