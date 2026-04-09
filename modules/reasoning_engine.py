from __future__ import annotations

import base64
import difflib
import json
import logging
import re
from typing import Any

import cv2
import numpy as np
import requests

import config

logger = logging.getLogger("jarvis.reasoning")


class ReasoningEngine:
    def __init__(self, model_name: str | None = None, ollama_host: str | None = None):
        self.model = model_name or config.APP.ollama_model
        host = (ollama_host or config.APP.ollama_host).rstrip("/")
        self.api_url = f"{host}/api/chat"
        self.system_prompt = (
            "You are JARVIS, a local multimodal assistant running through Ollama. "
            "You receive a live scene image and structured JSON context. "
            "Use only the image and the provided JSON as evidence. "
            "Prefer JSON for identities, database records, target locks, and stored object counts. "
            "Use the image for spatial relationships, motion clues, and scene description. "
            "If evidence is missing, say you cannot confirm it. "
            "Keep answers under 3 short sentences and stay operational, direct, and grounded."
        )

    def analyze_scene(self, context: dict[str, Any], user_query: str, frame: np.ndarray | None = None) -> str:
        if not user_query.strip():
            return self._fallback_response(context, "")
        try:
            return self._ask_ollama(context, user_query, frame=frame)
        except Exception as exc:
            logger.warning("Falling back to rules-based reasoning: %s", exc)
            return self._fallback_response(context, user_query)

    def resolve_person_name(self, spoken_name: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        cleaned = spoken_name.strip()
        if not cleaned or not candidates:
            return None
        exact = next((item for item in candidates if str(item.get("name", "")).strip().lower() == cleaned.lower()), None)
        if exact is not None:
            return exact
        try:
            resolved_id = self._ask_ollama_for_name_match(cleaned, candidates)
            if resolved_id:
                matched = next((item for item in candidates if str(item.get("person_id", "")).strip() == resolved_id), None)
                if matched is not None:
                    return matched
        except Exception as exc:
            logger.warning("AI name matching failed, falling back to local similarity: %s", exc)
        return self._fallback_name_match(cleaned, candidates)

    def _ask_ollama(self, context: dict[str, Any], user_query: str, frame: np.ndarray | None = None) -> str:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": (
                f"Live context JSON:\n{json.dumps(context, indent=2)}\n\n"
                f"User question: {user_query}\n\n"
                "Answer as JARVIS using only the current scene evidence."
            ),
        }
        encoded_frame = self._encode_frame(frame)
        if encoded_frame is not None:
            user_message["images"] = [encoded_frame]

        payload = {
            "model": self.model,
            "stream": False,
            "think": config.APP.ollama_think,
            "keep_alive": config.APP.ollama_keep_alive,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                user_message,
            ],
            "options": {
                "temperature": config.APP.ollama_temperature,
                "top_p": config.APP.ollama_top_p,
                "top_k": config.APP.ollama_top_k,
                "num_ctx": config.APP.ollama_num_ctx,
            },
        }
        response = requests.post(self.api_url, json=payload, timeout=config.APP.reasoning_timeout_sec)
        response.raise_for_status()
        body = response.json()
        content = self._clean_response(body.get("message", {}).get("content", ""))
        return content or self._fallback_response(context, user_query)

    def _ask_ollama_for_name_match(self, spoken_name: str, candidates: list[dict[str, Any]]) -> str:
        shortlist = [
            {
                "person_id": item.get("person_id", ""),
                "name": item.get("name", ""),
                "department": item.get("department", ""),
                "role": item.get("role", ""),
            }
            for item in candidates[:50]
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": config.APP.ollama_keep_alive,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You resolve noisy spoken person names against a local employee list. "
                        "Return only one person_id if there is a strong likely match. "
                        "If there is no strong match, return NONE. "
                        "Do not explain."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Spoken name: {spoken_name}\n"
                        f"Candidates JSON: {json.dumps(shortlist, ensure_ascii=True)}\n"
                        "Answer with exactly one person_id or NONE."
                    ),
                },
            ],
            "options": {
                "temperature": 0.0,
                "top_p": 0.2,
                "top_k": 20,
                "num_ctx": min(config.APP.ollama_num_ctx, 2048),
            },
        }
        response = requests.post(self.api_url, json=payload, timeout=config.APP.name_match_timeout_sec)
        response.raise_for_status()
        body = response.json()
        content = self._clean_response(body.get("message", {}).get("content", ""))
        answer = content.strip().splitlines()[0].strip() if content.strip() else ""
        if not answer or answer.upper() == "NONE":
            return ""
        return answer

    def _encode_frame(self, frame: np.ndarray | None) -> str | None:
        if frame is None or not config.APP.ollama_vision_enabled:
            return None

        image = frame
        h, w = image.shape[:2]
        max_width = max(320, config.APP.ollama_vision_max_width)
        if w > max_width:
            scale = max_width / float(w)
            image = cv2.resize(image, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(config.APP.ollama_vision_jpeg_quality)],
        )
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _clean_response(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"<\|channel\>thought\s*", "", cleaned)
        cleaned = re.sub(r"<channel\|>", "", cleaned)
        cleaned = re.sub(r"<\|[^>]+?\|>", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _fallback_name_match(self, spoken_name: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        normalized = spoken_name.lower().strip()
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in candidates:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            base_score = difflib.SequenceMatcher(None, normalized, name.lower()).ratio()
            token_score = max(
                [base_score]
                + [
                    difflib.SequenceMatcher(None, normalized, token).ratio()
                    for token in re.findall(r"[a-z0-9]+", name.lower())
                ]
            )
            if normalized in name.lower():
                token_score = max(token_score, 0.95)
            scored.append((token_score, item))
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1] if scored[0][0] >= 0.62 else None

    def _fallback_response(self, context: dict[str, Any], user_query: str) -> str:
        query = user_query.lower()
        people = context["people"]["tracks"]
        objects = context["objects"]["counts"]
        focus = context.get("focus_target")
        requested_target = context.get("requested_target", "")
        visible_records = context.get("memory", {}).get("visible_people", [])
        object_memory = context.get("memory", {}).get("object_memory", [])

        if any(term in query for term in ["how many", "list", "last seen", "objects database", "object database"]):
            if object_memory:
                match_label = next((item for item in object_memory if item.get("label") and item["label"].lower() in query), None)
                if match_label:
                    last_seen = match_label.get("last_seen") or "unknown"
                    return f"There are {match_label.get('count', 0)} {match_label['label']} records. Last seen: {last_seen}."
                items = ", ".join(
                    f"{item.get('label')} x{item.get('count', 0)} last seen {item.get('last_seen', 'unknown')}"
                    for item in object_memory[:5]
                )
                return f"Object memory shows {items}."
            return "I do not have any object records stored yet."

        if any(term in query for term in ["record", "database", "db", "history", "profile"]):
            record = None
            if focus and focus.get("record"):
                record = focus["record"]
            elif visible_records:
                record = visible_records[0]
            if record:
                last_seen = record.get("last_seen") or "not recently"
                return (
                    f"{record['name']} is {record.get('role', 'unassigned')} in {record.get('department', 'unknown department')}. "
                    f"Recent sightings: {record.get('recent_sightings', 0)}. Last seen: {last_seen}."
                )
            return "I do not have a matching person record in view right now."

        if any(term in query for term in ["spot", "track", "focus", "find"]):
            if focus and focus.get("name"):
                return f"I have {focus['name']} isolated in view and can keep tracking them."
            if requested_target:
                return f"I am looking for {requested_target}, but they are not clearly identified on screen yet."
            return "Tell me which enrolled person to spot, and I will lock onto them."

        if "who is that" in query or "who is this" in query:
            if focus and focus.get("name"):
                return f"That appears to be {focus['name']} from {focus.get('department', 'an unknown department')}."
            if people:
                first_known = next((person for person in people if person.get("is_known")), None)
                if first_known:
                    return f"I can identify {first_known['name']} in view."
            return "I can see a person, but I cannot identify them from the local registry."

        if "what is happening" in query or "what's happening" in query:
            summary = context["scene"]["summary"]
            interactions = context["scene"]["interactions"]
            if interactions:
                return f"{summary} Likely activity: {', '.join(interactions[:2])}."
            return summary

        if "what should i do" in query:
            if context["people"]["unknown_count"] > 0:
                return "An unknown person is visible. Verify identity before taking action."
            if focus and focus.get("name"):
                return f"{focus['name']} is the active focus target. You can continue monitoring them."
            return "The scene looks stable. Keep observing and ask for a specific analysis if needed."

        if "object" in query or "see" in query:
            if objects:
                items = ", ".join(f"{count} {name}" for name, count in list(objects.items())[:4])
                if object_memory:
                    memory_hint = ", ".join(
                        f"{item.get('label')} last seen {item.get('last_seen', 'unknown')}" for item in object_memory[:3]
                    )
                    return f"I currently see {items}. Stored object memory: {memory_hint}."
                return f"I currently see {items}."
            return "I do not see any notable tracked objects right now."

        return context["scene"]["summary"]
