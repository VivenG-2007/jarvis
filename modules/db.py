"""
modules/db.py — MongoDB interface for person registry and recognition logs.

Schema (persons collection):
  {
    "_id": ObjectId,
    "person_id": str,          # e.g. "EMP-001"
    "name": str,
    "department": str,
    "role": str,
    "embedding": list[float],  # 512-dim face vector
    "enrolled_at": datetime,
    "photo_path": str | None   # optional reference image path
  }
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

import config

logger = logging.getLogger("jarvis.db")


class PersonDB:
    """Handles all MongoDB operations for person data and recognition logs."""

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.persons = None
        self.logs = None
        self._connect()

    # ── Connection ────────────────────────────────────────────

    def _connect(self):
        try:
            self.client = MongoClient(
                config.MONGO_URI,
                serverSelectionTimeoutMS=3000
            )
            # Verify connection
            self.client.admin.command("ping")
            self.db       = self.client[config.MONGO_DB_NAME]
            self.persons  = self.db[config.MONGO_COLLECTION_PERSONS]
            self.logs     = self.db[config.MONGO_COLLECTION_LOGS]
            self._ensure_indexes()
            logger.info("[OK]  MongoDB connected → %s / %s", config.MONGO_URI, config.MONGO_DB_NAME)
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error("[FAIL]  MongoDB connection failed: %s", e)
            self.client = None

    def is_connected(self) -> bool:
        return self.client is not None

    def _ensure_indexes(self):
        self.persons.create_index("person_id", unique=True)
        self.persons.create_index("name")
        self.logs.create_index("timestamp")
        self.logs.create_index("person_id")

    # ── Person CRUD ───────────────────────────────────────────

    def enroll_person(
        self,
        person_id: str,
        name: str,
        department: str,
        role: str,
        embedding: np.ndarray,
        photo_path: Optional[str] = None
    ) -> bool:
        """Add or update a person record with their face embedding."""
        if not self.is_connected():
            logger.warning("DB not connected — enrollment skipped.")
            return False
        doc = {
            "person_id":   person_id,
            "name":        name,
            "department":  department,
            "role":        role,
            "embedding":   embedding.tolist(),
            "enrolled_at": datetime.now(timezone.utc),
            "photo_path":  photo_path,
        }
        result = self.persons.update_one(
            {"person_id": person_id},
            {"$set": doc},
            upsert=True
        )
        logger.info("Enrolled person '%s' (id=%s)", name, person_id)
        return result.acknowledged

    def get_all_persons(self) -> list[dict]:
        """Return all person records with embeddings as np.ndarray."""
        if not self.is_connected():
            return []
        docs = list(self.persons.find({}, {"_id": 0}))
        for d in docs:
            d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        return docs

    def get_person_by_id(self, person_id: str) -> Optional[dict]:
        if not self.is_connected():
            return None
        doc = self.persons.find_one({"person_id": person_id}, {"_id": 0})
        if doc:
            doc["embedding"] = np.array(doc["embedding"], dtype=np.float32)
        return doc

    def delete_person(self, person_id: str) -> bool:
        if not self.is_connected():
            return False
        result = self.persons.delete_one({"person_id": person_id})
        return result.deleted_count > 0

    def list_persons(self) -> list[dict]:
        """Return summary list (no embeddings) for display."""
        if not self.is_connected():
            return []
        return list(self.persons.find({}, {"_id": 0, "embedding": 0}))

    # ── Recognition Logging ───────────────────────────────────

    def log_recognition(self, person_id: str, confidence: float, bbox: list):
        """Log a recognition event for audit trail."""
        if not self.is_connected():
            return
        self.logs.insert_one({
            "person_id":  person_id,
            "confidence": round(confidence, 4),
            "bbox":       bbox,
            "timestamp":  datetime.now(timezone.utc),
        })

    def get_recent_logs(self, limit: int = 20) -> list[dict]:
        if not self.is_connected():
            return []
        return list(
            self.logs.find({}, {"_id": 0})
                     .sort("timestamp", -1)
                     .limit(limit)
        )

    # ── Utility ───────────────────────────────────────────────

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")
