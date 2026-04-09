from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import config
from modules.db import PersonDB


class SemanticContextBuilder:
    def __init__(self, db: PersonDB | None = None) -> None:
        self.db = db
        self._object_memory_cache: list[dict[str, Any]] = []
        self._object_memory_ts = 0.0
        self.activity_rules = {
            "cell phone": "someone may be using a phone",
            "laptop": "someone may be working on a laptop",
            "keyboard": "someone may be typing",
            "book": "someone may be reading",
            "tv": "attention may be focused on a display",
            "cup": "someone may be drinking",
            "bottle": "someone may be drinking",
            "backpack": "a bag is present",
        }

    def build_context(
        self,
        faces: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        voice_query: str = "",
        source: str = "camera",
        requested_target: str = "",
    ) -> dict[str, Any]:
        object_memory = self._build_object_memory()
        object_memory_by_label = {str(item.get("label", "")).strip().lower(): item for item in object_memory if item.get("label")}
        enriched_objects = [self._merge_object_memory(obj, object_memory_by_label) for obj in objects]
        named_people = [face["name"] for face in faces if face.get("is_known")]
        unknown_count = sum(1 for face in faces if not face.get("is_known"))
        object_names = [obj.get("label", "unknown") for obj in enriched_objects]
        object_counts = dict(Counter(object_names))
        interactions = self._infer_interactions(len(faces), sorted(object_counts))
        visible_records = self._build_visible_records(faces)
        recent_events = self._build_recent_events()

        focus_target = next((face for face in faces if face.get("lock_candidate")), None)
        focus_summary = None
        if focus_target:
            focus_summary = {
                "person_id": focus_target.get("person_id"),
                "name": focus_target.get("name"),
                "department": focus_target.get("department"),
                "confidence": focus_target.get("confidence", 0.0),
                "bbox": focus_target.get("bbox"),
                "record": self.db.get_person_context(focus_target["person_id"]) if self.db and focus_target.get("person_id") else None,
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "people": {
                "known": named_people,
                "unknown_count": unknown_count,
                "tracks": faces,
            },
            "objects": {
                "labels": sorted(object_counts),
                "counts": object_counts,
                "tracks": enriched_objects,
            },
            "scene": {
                "summary": self._scene_summary(named_people, unknown_count, object_counts),
                "interactions": interactions,
            },
            "focus_target": focus_summary,
            "requested_target": requested_target.strip(),
            "memory": {
                "visible_people": visible_records,
                "recent_events": recent_events,
                "object_memory": object_memory,
            },
            "voice_query": voice_query.strip(),
        }

    def build_json(
        self,
        faces: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        voice_query: str = "",
        source: str = "camera",
        requested_target: str = "",
    ) -> str:
        return json.dumps(
            self.build_context(faces, objects, voice_query=voice_query, source=source, requested_target=requested_target),
            indent=2,
        )

    def _build_visible_records(self, faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.db is None:
            return []

        records: list[dict[str, Any]] = []
        seen_person_ids: set[str] = set()
        for face in faces:
            if not face.get("is_known") or not face.get("person_id"):
                continue
            person_id = str(face["person_id"]).strip()
            if not person_id or person_id in seen_person_ids:
                continue
            seen_person_ids.add(person_id)
            record = self.db.get_person_context(person_id)
            if record is not None:
                records.append(record)
        return records

    def _build_recent_events(self, limit: int = 8) -> list[dict[str, Any]]:
        if self.db is None:
            return []

        recent_events = self.db.get_recent_logs(limit=limit)
        compact_events: list[dict[str, Any]] = []
        for event in recent_events:
            compact_events.append(
                {
                    "type": event.get("type"),
                    "person_id": event.get("person_id"),
                    "label": event.get("label"),
                    "confidence": event.get("confidence"),
                    "timestamp": event.get("timestamp"),
                }
            )
        return compact_events

    def _build_object_memory(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        now = datetime.now(timezone.utc).timestamp()
        if self._object_memory_cache and now - self._object_memory_ts < config.APP.object_memory_refresh_sec:
            return self._object_memory_cache[:limit]

        objects = self.db.get_object_memory(limit=limit)
        compact_objects: list[dict[str, Any]] = []
        for item in objects:
            compact_objects.append(
                {
                    "label": item.get("label"),
                    "count": item.get("count", 0),
                    "confidence": item.get("confidence"),
                    "last_seen": item.get("last_seen") or item.get("timestamp"),
                }
            )
        self._object_memory_cache = compact_objects
        self._object_memory_ts = now
        return compact_objects

    @staticmethod
    def _merge_object_memory(obj: dict[str, Any], memory_by_label: dict[str, dict[str, Any]]) -> dict[str, Any]:
        merged = dict(obj)
        label = str(obj.get("label", "")).strip().lower()
        memory = memory_by_label.get(label)
        if memory is not None:
            merged["db_count"] = memory.get("count", 0)
            merged["db_last_seen"] = memory.get("last_seen")
        return merged

    def _infer_interactions(self, people_count: int, object_names: list[str]) -> list[str]:
        interactions: list[str] = []
        if people_count > 1:
            interactions.append("multiple people are present")
        for name in object_names:
            if name in self.activity_rules:
                interactions.append(self.activity_rules[name])
        return interactions

    def _scene_summary(self, people: list[str], unknown_count: int, objects: dict[str, int]) -> str:
        people_fragment = ", ".join(people) if people else "no known people"
        unknown_fragment = f"{unknown_count} unknown" if unknown_count else "no unknowns"
        object_fragment = ", ".join(f"{count} {label}" for label, count in sorted(objects.items())) or "no notable objects"
        return f"Known people: {people_fragment}. Unknowns: {unknown_fragment}. Objects: {object_fragment}."
