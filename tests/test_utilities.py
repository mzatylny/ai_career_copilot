import math

from app.embeddings import hash_embedding
from app.utils import short_snippet, unique_preserve_order


def test_hash_embedding_is_deterministic_normalized_and_input_sensitive():
    first = hash_embedding("Python FastAPI", 32)
    repeated = hash_embedding("Python FastAPI", 32)
    different = hash_embedding("Kubernetes observability", 32)

    assert first == repeated
    assert first != different
    assert len(first) == 32
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_short_snippet_compacts_whitespace_and_enforces_its_limit():
    assert short_snippet("  Python\n\nFastAPI  ") == "Python FastAPI"
    assert short_snippet("abcdefghij", limit=6) == "abcde…"


def test_unique_preserve_order_normalizes_for_deduplication():
    assert unique_preserve_order([" Python ", "python", "", "FastAPI", " fastapi "]) == [
        "Python",
        "FastAPI",
    ]
