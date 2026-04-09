from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config

logger = logging.getLogger("jarvis.db")

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.errors import PyMongoError

    MONGO_AVAILABLE = True
except ImportError:
    MongoClient = None
    Collection = Any
    PyMongoError = Exception
    MONGO_AVAILABLE = False


@dataclass
class PersonRecord:
    person_id: str
    name: str
    department: str
    role: str
    embedding: list[float]
    photo_path: Optional[str] = None
    notes: str = ""
    enrolled_at: Optional[str] = None


class PersonDB:
    def __init__(self, registry_path: Optional[Path] = None, events_path: Optional[Path] = None):
        self.registry_path = Path(registry_path or config.APP.memory_registry_path)
        self.events_path = Path(events_path or config.APP.memory_events_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []

        self.client = None
        self.database = None
        self.persons_collection: Optional[Collection] = None
        self.logs_collection: Optional[Collection] = None
        self.objects_collection: Optional[Collection] = None
        self._mongo_connected = False

        self._connect()
        self._load()

    def _connect(self) -> None:
        if not MONGO_AVAILABLE:
            logger.warning("pymongo is not installed. Falling back to local JSON memory store.")
            return
        try:
            assert MongoClient is not None
            self.client = MongoClient(config.APP.mongo_uri, serverSelectionTimeoutMS=1500)
            self.client.admin.command("ping")
            self.database = self.client[config.APP.mongo_db_name]
            self.persons_collection = self.database[config.APP.mongo_collection_persons]
            self.logs_collection = self.database[config.APP.mongo_collection_logs]
            self.objects_collection = self.database[config.APP.mongo_collection_objects]
            self.persons_collection.create_index("person_id", unique=True)
            self.logs_collection.create_index([("timestamp", -1)])
            self.objects_collection.create_index([("timestamp", -1)])
            self._repair_object_indexes()
            self._mongo_connected = True
            logger.info("[OK] MongoDB connected -> %s / %s", config.APP.mongo_uri, config.APP.mongo_db_name)
        except Exception as exc:
            self._mongo_connected = False
            self.client = None
            self.database = None
            self.persons_collection = None
            self.logs_collection = None
            self.objects_collection = None
            logger.warning("MongoDB unavailable, using local JSON fallback: %s", exc)

    def _load(self) -> None:
        if self._mongo_connected:
            self._load_from_mongo()
            return
        self._load_from_file()

    def _repair_object_indexes(self) -> None:
        if self.objects_collection is None:
            return
        try:
            indexes = self.objects_collection.index_information()
            for name, info in indexes.items():
                if name == "_id_":
                    continue
                keys = info.get("key", [])
                is_label_only = len(keys) == 1 and keys[0][0] == "label"
                if is_label_only and info.get("unique"):
                    self.objects_collection.drop_index(name)
                    logger.info("Dropped legacy unique object index: %s", name)
        except PyMongoError as exc:
            logger.warning("Failed to repair object indexes: %s", exc)

    def _load_from_file(self) -> None:
        if self.registry_path.exists():
            try:
                self._records = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Registry file is invalid JSON. Starting with an empty registry.")
                self._records = []
        else:
            self._persist_file()

    def _load_from_mongo(self) -> None:
        assert self.persons_collection is not None
        try:
            records = list(self.persons_collection.find({}, {"_id": 0}).sort("person_id", 1))
            if records:
                self._records = records
                self._persist_file()
                return
        except PyMongoError as exc:
            logger.warning("Failed to load persons from MongoDB: %s", exc)

        self._load_from_file()
        if self._records:
            self._sync_file_records_to_mongo()

    def _persist_file(self) -> None:
        serialized = [self._serialize_record(item) for item in self._records]
        self.registry_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def _sync_file_records_to_mongo(self) -> None:
        if not self._mongo_connected or not self.persons_collection:
            return
        for record in self._records:
            try:
                self.persons_collection.replace_one({"person_id": record["person_id"]}, record, upsert=True)
            except PyMongoError as exc:
                logger.warning("Failed to sync person %s to MongoDB: %s", record.get("person_id"), exc)

    def _normalize_person(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        normalized["embedding"] = np.asarray(item["embedding"], dtype=np.float32)
        if isinstance(normalized.get("enrolled_at"), datetime):
            normalized["enrolled_at"] = normalized["enrolled_at"].isoformat()
        return normalized

    def _serialize_record(self, item: dict[str, Any]) -> dict[str, Any]:
        serialized = dict(item)
        if isinstance(serialized.get("enrolled_at"), datetime):
            serialized["enrolled_at"] = serialized["enrolled_at"].isoformat()
        return serialized

    def _serialize_event(self, item: dict[str, Any]) -> dict[str, Any]:
        serialized = dict(item)
        if isinstance(serialized.get("timestamp"), datetime):
            serialized["timestamp"] = serialized["timestamp"].isoformat()
        return serialized

    def is_connected(self) -> bool:
        return self._mongo_connected

    def enroll_person(
        self,
        person_id: str,
        name: str,
        department: str,
        role: str,
        embedding: np.ndarray,
        photo_path: Optional[str] = None,
        notes: str = "",
    ) -> bool:
        record = PersonRecord(
            person_id=person_id,
            name=name,
            department=department,
            role=role,
            embedding=embedding.astype(np.float32).tolist(),
            photo_path=photo_path,
            notes=notes,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
        )
        payload = asdict(record)

        self._records = [existing for existing in self._records if existing["person_id"] != person_id]
        self._records.append(payload)
        self._records.sort(key=lambda item: item["person_id"])
        self._persist_file()

        if self._mongo_connected and self.persons_collection is not None:
            try:
                self.persons_collection.replace_one({"person_id": person_id}, payload, upsert=True)
            except PyMongoError as exc:
                logger.warning("Failed to upsert person %s into MongoDB: %s", person_id, exc)
                return False
        return True

    def get_all_persons(self) -> list[dict[str, Any]]:
        if self._mongo_connected and self.persons_collection is not None:
            try:
                self._records = list(self.persons_collection.find({}, {"_id": 0}).sort("person_id", 1))
                self._persist_file()
            except PyMongoError as exc:
                logger.warning("Failed to refresh persons from MongoDB: %s", exc)
        return [self._normalize_person(item) for item in self._records]

    def get_person_by_id(self, person_id: str) -> Optional[dict[str, Any]]:
        if self._mongo_connected and self.persons_collection is not None:
            try:
                item = self.persons_collection.find_one({"person_id": person_id}, {"_id": 0})
                if item is not None:
                    return self._normalize_person(item)
            except PyMongoError as exc:
                logger.warning("Failed to fetch person %s from MongoDB: %s", person_id, exc)
        for item in self._records:
            if item["person_id"] == person_id:
                return self._normalize_person(item)
        return None

    def find_person_by_name(self, query: str) -> Optional[dict[str, Any]]:
        needle = query.strip().lower()
        if not needle:
            return None

        people = self.get_all_persons()
        exact = next((person for person in people if str(person.get("name", "")).strip().lower() == needle), None)
        if exact is not None:
            return exact

        return next((person for person in people if needle in str(person.get("name", "")).strip().lower()), None)

    def list_persons(self) -> list[dict[str, Any]]:
        if self._mongo_connected and self.persons_collection is not None:
            try:
                self._records = list(self.persons_collection.find({}, {"_id": 0}).sort("person_id", 1))
                self._persist_file()
            except PyMongoError as exc:
                logger.warning("Failed to list persons from MongoDB: %s", exc)
        return [{k: v for k, v in item.items() if k != "embedding"} for item in self._records]

    def delete_person(self, person_id: str) -> bool:
        deleted = False
        before = len(self._records)
        self._records = [item for item in self._records if item["person_id"] != person_id]
        if len(self._records) != before:
            deleted = True
            self._persist_file()

        if self._mongo_connected and self.persons_collection is not None:
            try:
                result = self.persons_collection.delete_one({"person_id": person_id})
                deleted = deleted or result.deleted_count > 0
            except PyMongoError as exc:
                logger.warning("Failed to delete person %s from MongoDB: %s", person_id, exc)
        return deleted

    def log_recognition(self, person_id: str, confidence: float, bbox: list[int]) -> None:
        self._append_event(
            {
                "type": "recognition",
                "person_id": person_id,
                "confidence": round(confidence, 3),
                "bbox": bbox,
            }
        )

    def log_object(self, label: str, confidence: float) -> None:
        self._append_event(
            {
                "type": "object",
                "label": label,
                "confidence": round(confidence, 3),
            }
        )

    def get_recent_logs(self, limit: int = 20) -> list[dict[str, Any]]:
        mongo_events: list[dict[str, Any]] = []
        if self._mongo_connected and self.logs_collection is not None:
            try:
                mongo_events.extend(list(self.logs_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)))
            except PyMongoError as exc:
                logger.warning("Failed to read recognition logs from MongoDB: %s", exc)
        if self._mongo_connected and self.objects_collection is not None:
            try:
                mongo_events.extend(list(self.objects_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)))
            except PyMongoError as exc:
                logger.warning("Failed to read object logs from MongoDB: %s", exc)
        if mongo_events:
            serialized_events = [self._serialize_event(item) for item in mongo_events]
            serialized_events.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
            return serialized_events[:limit]

        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events[-limit:]

    def get_person_context(self, person_id: str, limit: int = 8) -> Optional[dict[str, Any]]:
        person = self.get_person_by_id(person_id)
        if person is None:
            return None

        recognition_events = self._get_person_recognition_events(person_id, limit=limit)
        confidences = [float(event.get("confidence", 0.0)) for event in recognition_events if event.get("confidence") is not None]
        last_seen = recognition_events[0].get("timestamp") if recognition_events else None

        return {
            "person_id": person["person_id"],
            "name": person.get("name", ""),
            "department": person.get("department", ""),
            "role": person.get("role", ""),
            "notes": person.get("notes", ""),
            "enrolled_at": person.get("enrolled_at"),
            "recent_sightings": len(recognition_events),
            "last_seen": last_seen,
            "average_confidence": round(sum(confidences) / len(confidences), 1) if confidences else None,
        }

    def get_object_memory(self, limit: int = 20) -> list[dict[str, Any]]:
        if self._mongo_connected and self.objects_collection is not None:
            try:
                records = list(self.objects_collection.find({}, {"_id": 0}).sort("last_seen", -1).limit(limit))
                return [self._serialize_event(item) for item in records]
            except PyMongoError as exc:
                logger.warning("Failed to read object memory from MongoDB: %s", exc)

        if not self.events_path.exists():
            return []

        summary: dict[str, dict[str, Any]] = {}
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") != "object":
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            entry = summary.setdefault(
                label,
                {
                    "type": "object",
                    "label": label,
                    "count": 0,
                    "confidence": 0.0,
                    "timestamp": item.get("timestamp"),
                    "last_seen": item.get("timestamp"),
                },
            )
            entry["count"] += 1
            entry["confidence"] = max(float(entry.get("confidence", 0.0)), float(item.get("confidence", 0.0)))
            timestamp = item.get("timestamp")
            if timestamp and (entry.get("last_seen") is None or timestamp > entry["last_seen"]):
                entry["timestamp"] = timestamp
                entry["last_seen"] = timestamp
        objects = sorted(summary.values(), key=lambda item: item.get("last_seen") or "", reverse=True)
        return objects[:limit]

    def _get_person_recognition_events(self, person_id: str, limit: int = 8) -> list[dict[str, Any]]:
        if self._mongo_connected and self.logs_collection is not None:
            try:
                events = list(
                    self.logs_collection.find({"person_id": person_id}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                )
                return [self._serialize_event(item) for item in events]
            except PyMongoError as exc:
                logger.warning("Failed to read recognition history for %s: %s", person_id, exc)

        if not self.events_path.exists():
            return []

        matches: list[dict[str, Any]] = []
        for line in reversed(self.events_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") == "recognition" and item.get("person_id") == person_id:
                matches.append(item)
            if len(matches) >= limit:
                break
        return matches

    def _append_event(self, payload: dict[str, Any]) -> None:
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

        if not self._mongo_connected:
            return

        collection = None
        if payload.get("type") == "recognition":
            collection = self.logs_collection
        elif payload.get("type") == "object":
            collection = self.objects_collection

        if collection is None:
            return
        try:
            if payload.get("type") == "object":
                collection.update_one(
                    {"label": payload["label"]},
                    {
                        "$set": {
                            "type": "object",
                            "label": payload["label"],
                            "confidence": payload["confidence"],
                            "timestamp": payload["timestamp"],
                            "last_seen": payload["timestamp"],
                        },
                        "$inc": {"count": 1},
                    },
                    upsert=True,
                )
            else:
                collection.insert_one(dict(payload))
        except PyMongoError as exc:
            logger.warning("Failed to write event to MongoDB: %s", exc)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            logger.info("MongoDB connection closed.")
