"""Content-addressed on-disk cache for LLM completions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class CompletionCache:
    """Store one JSON file per request, keyed by a hash of every request parameter.

    Safe to share across threads: each key is its own file, written atomically.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(**request: Any) -> str:
        payload = json.dumps(request, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return str(json.loads(path.read_text())["response"])

    def put(self, key: str, request: dict[str, Any], response: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"request": request, "response": response}, f, ensure_ascii=False)
        os.replace(tmp, path)
