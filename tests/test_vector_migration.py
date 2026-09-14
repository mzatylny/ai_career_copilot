import hashlib
import json

import pytest

from app.vector_store import VectorCollection
from scripts.migrate_vectors import export_collection, import_collection


class LegacyCollection:
    def count(self):
        return 3

    def get(self, *, offset, limit, include):
        indices = list(range(3))[offset:offset + limit]
        return {
            "ids": [f"legacy-{i}" for i in indices],
            "documents": [f"Życiorys {i}" for i in indices],
            "metadatas": [{"session_id": "alpha" if i < 2 else "beta",
                           "source": "CV.pdf", "page": i + 1} for i in indices],
            "embeddings": [[1.0, 0.0, 0.0] for _ in indices],
        }


def test_migration_preserves_records_and_leaves_export_unchanged(tmp_path):
    export = tmp_path / "vectors.jsonl"
    target = tmp_path / "qdrant"
    assert export_collection(LegacyCollection(), export, 3, batch_size=2) == 3
    before = export.read_bytes()
    assert import_collection(export, target, "documents", batch_size=2) == 3
    assert export.read_bytes() == before
    store = VectorCollection(target, "documents", 3)
    try:
        data = store.get(where={"session_id": "alpha"}, include=["documents", "metadatas", "embeddings"])
        records = dict(zip(data["ids"], zip(data["documents"], data["metadatas"], data["embeddings"], strict=True), strict=True))
        assert set(records) == {"legacy-0", "legacy-1"}
        assert records["legacy-1"][0] == "Życiorys 1"
        assert records["legacy-1"][1] == {"session_id": "alpha", "source": "CV.pdf", "page": 2}
        assert records["legacy-1"][2] == [1.0, 0.0, 0.0]
        store.delete(ids=["legacy-0", "legacy-1"])
        assert store.get(where={"session_id": "beta"}, include=[])["ids"] == ["legacy-2"]
    finally:
        store.close()
    with pytest.raises(ValueError, match="Target already exists"):
        import_collection(export, target, "documents")


@pytest.mark.parametrize("failure", ["truncated", "checksum", "duplicate", "dimension", "scope"])
def test_failed_import_never_publishes_partial_database(tmp_path, failure):
    export = tmp_path / "vectors.jsonl"
    export_collection(LegacyCollection(), export, 3)
    lines = export.read_text().splitlines(keepends=True)
    if failure == "truncated":
        lines = lines[:-1]
    elif failure == "checksum":
        lines[1] = lines[1].replace("Życiorys", "Changed")
    else:
        row = json.loads(lines[2])
        if failure == "duplicate":
            row["id"] = "legacy-0"
        elif failure == "dimension":
            row["embedding"] = [1.0]
        else:
            row["metadata"] = {}
        lines[2] = json.dumps(row) + "\n"
        lines[-1] = json.dumps({"count": 3, "sha256": hashlib.sha256("".join(lines[1:-1]).encode()).hexdigest()}) + "\n"
    export.write_text("".join(lines))
    target = tmp_path / "qdrant"
    with pytest.raises(ValueError):
        import_collection(export, target, "documents", batch_size=1)
    assert not target.exists()
    assert not list(tmp_path.glob(".qdrant-migration-*"))


def test_export_refuses_to_overwrite_existing_file(tmp_path):
    export = tmp_path / "vectors.jsonl"
    export.write_text("Keep me")
    with pytest.raises(FileExistsError):
        export_collection(LegacyCollection(), export, 3)
    assert export.read_text() == "Keep me"
