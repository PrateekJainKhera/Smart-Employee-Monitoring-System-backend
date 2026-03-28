"""
EmbeddingStore — in-memory face embedding cache backed by a .pkl file.

Each employee can have multiple embeddings (one per registered photo/angle).
During recognition, all stored embeddings for a candidate are checked and the
best-matching one is used — this makes recognition robust to angle variation.

Storage format (pkl):
    dict[int, list[np.ndarray]]   employee_id → [emb_angle1, emb_angle2, ...]
"""
import pickle
import threading
from pathlib import Path

import numpy as np

from app.utils.logger import logger


class EmbeddingStore:
    def __init__(self, pkl_path: str):
        self._path = Path(pkl_path)
        # employee_id → list of normalized 512-dim embeddings
        self._store: dict[int, list[np.ndarray]] = {}
        self._lock = threading.Lock()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            logger.info(f"No embeddings file at {self._path} — starting fresh")
            return
        try:
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            # Migrate old format: dict[int, np.ndarray] → dict[int, list[np.ndarray]]
            migrated = {}
            for eid, val in data.items():
                if isinstance(val, np.ndarray):
                    migrated[eid] = [val]   # wrap single embedding in a list
                else:
                    migrated[eid] = val      # already a list
            self._store = migrated
            total_photos = sum(len(v) for v in self._store.values())
            logger.info(
                f"EmbeddingStore: loaded {len(self._store)} employee(s), "
                f"{total_photos} photo(s) from {self._path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load embeddings from {self._path}: {e}. Starting fresh.")
            self._store = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._store, f)

    # ── Public API ────────────────────────────────────────────────────

    def add(self, employee_id: int, embedding: np.ndarray) -> int:
        """
        Append a new embedding for an employee.
        Returns the total number of embeddings stored for this employee.
        """
        with self._lock:
            if employee_id not in self._store:
                self._store[employee_id] = []
            self._store[employee_id].append(embedding.copy())
            self._save()
        count = len(self._store[employee_id])
        logger.info(
            f"EmbeddingStore: saved embedding for employee {employee_id} "
            f"({count} total)"
        )
        return count

    def remove(self, employee_id: int) -> None:
        """Remove all embeddings for an employee."""
        with self._lock:
            if employee_id in self._store:
                del self._store[employee_id]
                self._save()
                logger.info(f"EmbeddingStore: removed all embeddings for employee {employee_id}")

    def remove_one(self, employee_id: int, index: int) -> bool:
        """
        Remove a single embedding by index (0-based).
        Returns True if removed, False if index out of range.
        """
        with self._lock:
            embs = self._store.get(employee_id, [])
            if index < 0 or index >= len(embs):
                return False
            embs.pop(index)
            if not embs:
                del self._store[employee_id]
            self._save()
        return True

    def get_all(self) -> dict[int, list[np.ndarray]]:
        """Returns employee_id → list of embeddings."""
        with self._lock:
            return {k: [e.copy() for e in v] for k, v in self._store.items()}

    def get(self, employee_id: int) -> list[np.ndarray]:
        """Returns all embeddings for an employee, or empty list."""
        return self._store.get(employee_id, [])

    def photo_count(self, employee_id: int) -> int:
        return len(self._store.get(employee_id, []))

    def has(self, employee_id: int) -> bool:
        return employee_id in self._store and len(self._store[employee_id]) > 0

    def count(self) -> int:
        """Total number of employees with at least one embedding."""
        return len(self._store)
