"""
EmbeddingStore — in-memory face embedding cache backed by a .pkl file.

Used for Phases 1–4 (no database).
In Phase 5, add load_from_db() / save_to_db() alongside pkl methods.
"""
import pickle
import threading
from pathlib import Path

import numpy as np

from app.utils.logger import logger


class EmbeddingStore:
    def __init__(self, pkl_path: str):
        self._path = Path(pkl_path)
        self._store: dict[int, np.ndarray] = {}  # employee_id → embedding
        self._lock = threading.Lock()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "rb") as f:
                    self._store = pickle.load(f)
                logger.info(f"EmbeddingStore loaded {len(self._store)} embeddings from {self._path}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings from {self._path}: {e}. Starting fresh.")
                self._store = {}
        else:
            logger.info(f"No embeddings file at {self._path} — starting fresh")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._store, f)

    # ── Public API ────────────────────────────────────────────────────

    def add(self, employee_id: int, embedding: np.ndarray) -> None:
        """Add or replace an employee's embedding and persist to disk."""
        with self._lock:
            self._store[employee_id] = embedding.copy()
            self._save()
        logger.info(f"EmbeddingStore: saved embedding for employee {employee_id}")

    def remove(self, employee_id: int) -> None:
        """Remove an employee's embedding and persist."""
        with self._lock:
            if employee_id in self._store:
                del self._store[employee_id]
                self._save()
                logger.info(f"EmbeddingStore: removed embedding for employee {employee_id}")

    def get(self, employee_id: int) -> np.ndarray | None:
        return self._store.get(employee_id)

    def get_all(self) -> dict[int, np.ndarray]:
        with self._lock:
            return {k: v.copy() for k, v in self._store.items()}

    def has(self, employee_id: int) -> bool:
        return employee_id in self._store

    def count(self) -> int:
        return len(self._store)
