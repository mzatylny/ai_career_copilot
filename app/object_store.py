from __future__ import annotations

import os
import shutil
from pathlib import Path


class LocalObjectStore:
    """Filesystem-backed object store with a narrow API that can be replaced by S3/GCS."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, object_id: str, source: str | Path) -> Path:
        destination = self._path(object_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    def delete(self, object_id: str) -> None:
        self._path(object_id).unlink(missing_ok=True)

    def is_ready(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK)

    def _path(self, object_id: str) -> Path:
        safe_id = "".join(
            character for character in object_id if character.isalnum() or character in "-_"
        )
        if safe_id != object_id or not safe_id:
            raise ValueError("Invalid object identifier")
        return self.root / safe_id
