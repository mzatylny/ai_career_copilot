"""Offline Chroma export / Qdrant import. Export runs in the OLD environment."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

FORMAT = "ai-career-copilot-vectors-v1"


def export_collection(collection, output: Path, dimensions: int, batch_size=256):
    """Read an offline legacy collection without modifying its records."""
    total = collection.count()
    digest = hashlib.sha256()
    count = 0
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"format": FORMAT, "dimensions": dimensions}) + "\n")
        for offset in range(0, total, batch_size):
            batch = collection.get(
                limit=batch_size, offset=offset, include=["documents", "metadatas", "embeddings"]
            )
            for chunk_id, text, metadata, vector in zip(
                batch["ids"], batch["documents"], batch["metadatas"], batch["embeddings"],
                strict=True,
            ):
                if len(vector) != dimensions:
                    raise ValueError("Export dimensions do not match the stored embeddings")
                row = json.dumps({
                    "id": chunk_id, "document": text, "metadata": metadata,
                    "embedding": [float(value) for value in vector],
                }, ensure_ascii=False, allow_nan=False) + "\n"
                digest.update(row.encode("utf-8"))
                handle.write(row)
                count += 1
        if count != total or collection.count() != total:
            raise ValueError("Source collection changed; stop the old app and export again")
        handle.write(json.dumps({"count": count, "sha256": digest.hexdigest()}) + "\n")
    return count


def import_collection(source: Path, target: Path, name: str, batch_size=256):
    """Publish a new store only after a complete, validated import succeeds."""
    from app.vector_store import VectorCollection

    target = target.resolve()
    if target.exists():
        raise ValueError("Target already exists; choose a NEW Qdrant directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-migration-", dir=target.parent))
    collection = None
    try:
        with source.open(encoding="utf-8") as handle:
            header = json.loads(next(handle))
            dimensions = header.get("dimensions")
            if header.get("format") != FORMAT or type(dimensions) is not int or not 1 <= dimensions <= 4096:
                raise ValueError("Invalid vector export header")
            collection = VectorCollection(staging, name, dimensions)
            digest = hashlib.sha256()
            seen = set()
            batch = []
            complete = False
            for line in handle:
                row = json.loads(line)
                if "sha256" in row:
                    if row.get("count") != len(seen) or row["sha256"] != digest.hexdigest():
                        raise ValueError("Export count/checksum does not match")
                    if handle.read():
                        raise ValueError("Unexpected data after the export footer")
                    complete = True
                    break
                chunk_id = row["id"]
                if chunk_id in seen:
                    raise ValueError("Duplicate chunk ID in export")
                seen.add(chunk_id)
                digest.update(line.encode("utf-8"))
                batch.append(row)
                if len(batch) >= batch_size:
                    _write_batch(collection, batch)
                    batch.clear()
            if not complete:
                raise ValueError("Export is incomplete: missing checksum footer")
            _write_batch(collection, batch)
            if collection.count() != len(seen):
                raise ValueError("Imported point count does not match")
        collection.close()
        collection = None
        if target.exists():
            raise ValueError("Target appeared during import; refusing to replace it")
        staging.rename(target)
        return len(seen)
    finally:
        if collection is not None:
            collection.close()
        if staging.exists():
            shutil.rmtree(staging)


def _write_batch(collection, rows):
    collection.upsert(
        ids=[r["id"] for r in rows], documents=[r["document"] for r in rows],
        metadatas=[r["metadata"] for r in rows], embeddings=[r["embedding"] for r in rows],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--chroma-path", type=Path, required=True)
    export.add_argument("--collection", default="career_documents")
    export.add_argument("--dimensions", type=int, default=1536)
    export.add_argument("--output", type=Path, required=True)
    load = commands.add_parser("import")
    load.add_argument("--input", type=Path, required=True)
    load.add_argument("--qdrant-path", type=Path, required=True)
    load.add_argument("--collection", default="career_documents")
    args = parser.parse_args()
    try:
        if args.command == "export":
            if not (args.chroma_path / "chroma.sqlite3").is_file():
                raise ValueError("Source is not an existing Chroma database")
            # Only the offline export process loads Chroma, from the old environment.
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(args.chroma_path), settings=Settings(anonymized_telemetry=False)
            )
            legacy = client.get_collection(args.collection, embedding_function=None)
            count = export_collection(legacy, args.output, args.dimensions)
            print(f"Exported {count} chunks to {args.output}")
        else:
            count = import_collection(args.input, args.qdrant_path, args.collection)
            print(f"Imported {count} chunks to {args.qdrant_path}")
    except (ValueError, OSError, KeyError, ImportError, StopIteration) as exc:
        parser.exit(2, f"Migration failed: {exc}\n")


if __name__ == "__main__":
    main()
