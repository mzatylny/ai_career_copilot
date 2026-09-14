"""Persistent, process-local Qdrant storage for explicitly embedded document chunks."""

import math
import threading
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"ai-career-copilot:chunk:{chunk_id}"))


def _session_filter(where: dict) -> models.Filter:
    clauses = where.get("$and", [where])
    conditions = []
    session_found = False
    for clause in clauses:
        for key, value in clause.items():
            if isinstance(value, dict) and set(value) == {"$eq"}:
                value = value["$eq"]
            if key not in {"session_id", "source"} or not isinstance(value, str) or not value:
                raise ValueError("Only non-empty session/source equality filters are supported")
            session_found |= key == "session_id"
            conditions.append(models.FieldCondition(
                key=f"metadata.{key}", match=models.MatchValue(value=value)
            ))
    if not session_found:
        raise ValueError("Vector reads require an explicit session_id filter")
    return models.Filter(must=conditions)


class VectorCollection:
    """Small adapter preserving the RAG layer's chunk/result format.

    Qdrant local uses SQLite internally. One client owns a path, and every client
    operation is serialized so FastAPI worker threads can safely share it.
    """

    def __init__(self, path: str | Path, name: str, dimensions: int):
        if (Path(path) / "chroma.sqlite3").exists():
            raise ValueError("This is a legacy Chroma directory; migrate it to a new Qdrant path")
        self.name, self.dimensions = name, dimensions
        self._lock = threading.RLock()
        self._client = QdrantClient(path=str(path), force_disable_check_same_thread=True)
        try:
            if self._client.collection_exists(name):
                config = self._client.get_collection(name).config.params.vectors
                if not isinstance(config, models.VectorParams) or (
                    config.size != dimensions or config.distance != models.Distance.COSINE
                ):
                    raise ValueError("Vector collection dimensions/distance differ; migrate or re-index")
            else:
                self._client.create_collection(
                    name, vectors_config=models.VectorParams(
                        size=dimensions, distance=models.Distance.COSINE
                    )
                )
        except Exception:
            self._client.close()
            raise

    def close(self):
        with self._lock:
            self._client.close()

    def count(self):
        with self._lock:
            return self._client.count(self.name, exact=True).count

    def upsert(self, *, ids, documents, metadatas, embeddings):
        points = []
        for chunk_id, document, metadata, vector in zip(
            ids, documents, metadatas, embeddings, strict=True
        ):
            if not isinstance(chunk_id, str) or not chunk_id or not isinstance(document, str):
                raise ValueError("Chunks require a non-empty string ID and document text")
            _session_filter({"session_id": metadata.get("session_id")})
            vector = [float(value) for value in vector]
            if len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector):
                raise ValueError("Embedding dimensions differ or contain non-finite values")
            points.append(models.PointStruct(
                id=_point_id(chunk_id), vector=vector,
                payload={"chunk_id": chunk_id, "document": document, "metadata": metadata},
            ))
        if len({p.id for p in points}) != len(points):
            raise ValueError("Duplicate chunk IDs in a batch")
        if points:
            with self._lock:
                self._client.upsert(self.name, points=points, wait=True)

    def get(self, *, where, include):
        query_filter = _session_filter(where)
        result = {"ids": [], **{key: [] for key in include}}
        offset = None
        with self._lock:
            while True:
                records, offset = self._client.scroll(
                    self.name, scroll_filter=query_filter, limit=256, offset=offset,
                    with_payload=True, with_vectors="embeddings" in include,
                )
                for record in records:
                    payload = record.payload
                    result["ids"].append(payload["chunk_id"])
                    for key, value in {
                        "documents": payload["document"], "metadatas": payload["metadata"],
                        "embeddings": record.vector,
                    }.items():
                        if key in include:
                            result[key].append(value)
                if offset is None:
                    return result

    def query(self, *, query_embeddings, n_results, where, include):
        query_filter = _session_filter(where)
        result = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        with self._lock:
            for vector in query_embeddings:
                if len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector):
                    raise ValueError("Invalid query embedding")
                points = self._client.query_points(
                    self.name, query=vector, query_filter=query_filter,
                    limit=n_results, with_payload=True,
                ).points
                result["ids"].append([p.payload["chunk_id"] for p in points])
                result["documents"].append([p.payload["document"] for p in points])
                result["metadatas"].append([p.payload["metadata"] for p in points])
                result["distances"].append([1.0 - p.score for p in points])
        return result

    def delete(self, *, ids):
        if ids:
            with self._lock:
                self._client.delete(
                    self.name, points_selector=[_point_id(value) for value in ids], wait=True
                )


_collections = {}
_collections_lock = threading.RLock()


def get_collection(path: str, name: str, dimensions: int) -> VectorCollection:
    key = str(Path(path).resolve())
    with _collections_lock:
        if key not in _collections:
            _collections[key] = VectorCollection(key, name, dimensions)
        collection = _collections[key]
        if collection.name != name or collection.dimensions != dimensions:
            raise ValueError("One local vector path must use one collection and embedding size")
        return collection


def close_collections():
    with _collections_lock:
        for collection in _collections.values():
            collection.close()
        _collections.clear()
