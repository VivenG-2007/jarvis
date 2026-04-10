from __future__ import annotations

import base64
import difflib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import requests

import config

logger = logging.getLogger("jarvis.reasoning")


@dataclass
class VoiceCommandPlan:
    action: str = "answer_query"
    person_id: str = ""
    person_name: str = ""
    object_label: str = ""
    query: str = ""
    response: str = ""
    fields: dict[str, str] | None = None
    confidence: float = 0.0


class ReasoningEngine:
    def __init__(self, model_name: str | None = None, ollama_host: str | None = None):
        self.model = model_name or config.APP.ollama_model
        host = (ollama_host or config.APP.ollama_host).rstrip("/")
        self.api_url = f"{host}/api/chat"
        self.groq_api_url = f"{config.APP.groq_api_base}/chat/completions"
        self.groq_chat_model = config.APP.groq_chat_model
        self.http = requests.Session()
        self.system_prompt = (
            "You are JARVIS, a local multimodal assistant running through Ollama. "
            "You receive a live scene image and structured JSON context. "
            "Use only the image and the provided JSON as evidence. "
            "Prefer JSON for identities, database records, target locks, and stored object counts. "
            "Use the image for spatial relationships, motion clues, and scene description. "
            "If evidence is missing, say you cannot confirm it. "
            "Keep answers under 3 short sentences and stay operational, direct, and grounded."
        )
        self.command_prompt = (
            "You translate noisy voice commands into one JSON action for a local assistant. "
            "You can select only one action from: "
            "focus_person, clear_focus, list_people, get_person, create_person, update_person, delete_person, "
            "list_objects, get_object_records, answer_query. "
            "Use create_person only when the user clearly asks to create/add a database record. "
            "If the user says 'add this person', 'enroll this face', or similar, it means the current visible face "
            "should be enrolled using the provided name and employee ID. "
            "Use update_person only when the user clearly asks to change stored metadata like name, department, role, or notes. "
            "Use delete_person only when the user clearly asks to remove a record. "
            "For 'show viven', 'find viven', 'track viven', or similar requests, choose focus_person. "
            "For object database requests like bottle records, TV records, laptop records, or object memory, "
            "choose list_objects or get_object_records. "
            "Return only valid JSON with keys: action, person_id, person_name, object_label, query, response, fields, confidence."
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
        local_match = self._fallback_name_match(cleaned, candidates)
        if local_match is not None:
            return local_match
        try:
            resolved_id = self._ask_ollama_for_name_match(cleaned, candidates)
            if resolved_id:
                matched = next((item for item in candidates if str(item.get("person_id", "")).strip() == resolved_id), None)
                if matched is not None:
                    return matched
        except Exception as exc:
            logger.warning("AI name matching failed, falling back to local similarity: %s", exc)
        return None

    def plan_voice_command(
        self,
        spoken_command: str,
        candidates: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> VoiceCommandPlan:
        cleaned = spoken_command.strip()
        if not cleaned:
            return VoiceCommandPlan()

        try:
            if config.APP.groq_api_key:
                return self._ask_groq_for_command(cleaned, candidates, context or {})
        except Exception as exc:
            logger.warning("Groq command planning failed, using local fallback: %s", exc)
        return self._fallback_command_plan(cleaned, candidates, context or {})

    def resolve_object_label(self, spoken_label: str, object_memory: list[dict[str, Any]]) -> str:
        cleaned = spoken_label.strip().lower()
        if not cleaned or not object_memory:
            return ""
        labels = [str(item.get("label", "")).strip() for item in object_memory if str(item.get("label", "")).strip()]
        if not labels:
            return ""
        exact = next((label for label in labels if label.lower() == cleaned), None)
        if exact:
            return exact
        if len(cleaned) <= 3:
            direct_alias = next(
                (
                    label
                    for label in labels
                    if cleaned in self._build_object_aliases(label)
                ),
                "",
            )
            return direct_alias

        scored: list[tuple[float, str]] = []
        for label in labels:
            label_lower = label.lower()
            aliases = self._build_object_aliases(label_lower)
            token_score = max(
                difflib.SequenceMatcher(None, self._normalize_spoken_name(cleaned), self._normalize_spoken_name(alias)).ratio()
                for alias in aliases
            )
            skeleton_score = max(
                difflib.SequenceMatcher(None, self._name_skeleton(cleaned), self._name_skeleton(alias)).ratio()
                for alias in aliases
            )
            if any(cleaned in alias for alias in aliases):
                token_score = max(token_score, 0.95)
            scored.append((max(token_score, skeleton_score), label))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1] if scored and scored[0][0] >= 0.56 else ""

    @staticmethod
    def _build_object_aliases(label: str) -> set[str]:
        cleaned = label.lower().replace("-", " ").strip()
        aliases = {cleaned, cleaned.replace(" ", "")}
        synonym_map = {
            "cell phone": {"phone", "mobile", "mobile phone", "cellphone"},
            "tv": {"television", "screen"},
            "bottle": {"water bottle"},
            "wine glass": {"glass"},
            "backpack": {"bag"},
        }
        for canonical, synonyms in synonym_map.items():
            if cleaned == canonical:
                aliases.update(synonyms)
            if cleaned in synonyms:
                aliases.add(canonical)
        return {alias for alias in aliases if alias}

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
        response = self.http.post(self.api_url, json=payload, timeout=config.APP.reasoning_timeout_sec)
        response.raise_for_status()
        body = response.json()
        content = self._clean_response(body.get("message", {}).get("content", ""))
        return content or self._fallback_response(context, user_query)

    def _ask_groq_for_command(
        self,
        spoken_command: str,
        candidates: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> VoiceCommandPlan:
        shortlist = [
            {
                "person_id": item.get("person_id", ""),
                "name": item.get("name", ""),
                "department": item.get("department", ""),
                "role": item.get("role", ""),
                "notes": item.get("notes", ""),
            }
            for item in candidates[:100]
        ]
        visible_names = [item.get("name", "") for item in context.get("memory", {}).get("visible_people", [])[:10]]
        object_memory = [
            {
                "label": item.get("label", ""),
                "count": item.get("count", 0),
                "last_seen": item.get("last_seen", ""),
            }
            for item in context.get("memory", {}).get("object_memory", [])[:50]
        ]
        payload = {
            "model": self.groq_chat_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.command_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "spoken_command": spoken_command,
                            "visible_people": visible_names,
                            "candidate_records": shortlist,
                            "object_memory": object_memory,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        }
        response = self.http.post(
            self.groq_api_url,
            headers={
                "Authorization": f"Bearer {config.APP.groq_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.APP.groq_chat_timeout_sec,
        )
        response.raise_for_status()
        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content)
        return self._normalize_command_plan(parsed, spoken_command, candidates)

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
        response = self.http.post(self.api_url, json=payload, timeout=config.APP.name_match_timeout_sec)
        response.raise_for_status()
        body = response.json()
        content = self._clean_response(body.get("message", {}).get("content", ""))
        answer = content.strip().splitlines()[0].strip() if content.strip() else ""
        if not answer or answer.upper() == "NONE":
            return ""
        return answer

    def _normalize_command_plan(
        self,
        payload: dict[str, Any],
        spoken_command: str,
        candidates: list[dict[str, Any]],
    ) -> VoiceCommandPlan:
        action = str(payload.get("action", "answer_query")).strip() or "answer_query"
        person_id = str(payload.get("person_id", "")).strip()
        person_name = str(payload.get("person_name", "")).strip()
        object_label = str(payload.get("object_label", "")).strip()
        query = str(payload.get("query", "")).strip() or spoken_command
        response = str(payload.get("response", "")).strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        raw_fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        fields = {str(key): str(value) for key, value in raw_fields.items() if value is not None}

        if action != "create_person" and person_name and not person_id:
            matched = self.resolve_person_name(person_name, candidates)
            if matched is not None:
                person_id = str(matched.get("person_id", "")).strip()
                person_name = str(matched.get("name", "")).strip()
        elif action != "create_person" and person_id and not person_name:
            matched = next((item for item in candidates if str(item.get("person_id", "")).strip() == person_id), None)
            if matched is not None:
                person_name = str(matched.get("name", "")).strip()

        return VoiceCommandPlan(
            action=action,
            person_id=person_id,
            person_name=person_name,
            object_label=object_label,
            query=query,
            response=response,
            fields=fields,
            confidence=confidence,
        )

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
            aliases = self._build_name_aliases(name)
            base_score = max(difflib.SequenceMatcher(None, normalized, alias).ratio() for alias in aliases)
            token_score = max(
                [base_score]
                + [
                    difflib.SequenceMatcher(None, normalized, token).ratio()
                    for alias in aliases
                    for token in re.findall(r"[a-z0-9]+", alias)
                ]
            )
            spoken_core = self._normalize_spoken_name(normalized)
            alias_core_scores = [
                difflib.SequenceMatcher(None, spoken_core, self._normalize_spoken_name(alias)).ratio()
                for alias in aliases
            ]
            alias_skeleton_scores = [
                difflib.SequenceMatcher(None, self._name_skeleton(spoken_core), self._name_skeleton(alias)).ratio()
                for alias in aliases
            ]
            token_score = max(token_score, max(alias_core_scores, default=0.0), max(alias_skeleton_scores, default=0.0))
            if any(normalized in alias for alias in aliases):
                token_score = max(token_score, 0.95)
            scored.append((token_score, item))
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1] if scored[0][0] >= 0.56 else None

    @staticmethod
    def _build_name_aliases(name: str) -> list[str]:
        cleaned = name.lower().replace("_", " ").replace("-", " ").strip()
        aliases = {cleaned}
        aliases.add(cleaned.replace(" ", ""))
        aliases.add(cleaned.replace("g", "j"))
        aliases.add(cleaned.replace("v", "w"))
        aliases.add(cleaned.replace("w", "v"))
        aliases.add(cleaned.replace("b", "v"))
        aliases.add(cleaned.replace("ph", "f"))
        return [alias for alias in aliases if alias]

    @staticmethod
    def _normalize_spoken_name(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
        normalized = normalized.replace("vv", "v").replace("ee", "i")
        normalized = normalized.replace("win", "vin")
        normalized = normalized.replace("wi", "vi")
        normalized = normalized.replace("bhe", "ve")
        normalized = normalized.replace("bh", "v")
        normalized = normalized.replace("w", "v")
        normalized = normalized.replace("y", "i")
        return normalized

    @classmethod
    def _name_skeleton(cls, value: str) -> str:
        normalized = cls._normalize_spoken_name(value)
        return re.sub(r"[aeiou]", "", normalized) or normalized

    def _fallback_command_plan(
        self,
        spoken_command: str,
        candidates: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> VoiceCommandPlan:
        lowered = spoken_command.lower().strip()
        object_memory = list(context.get("memory", {}).get("object_memory", []))
        if lowered in {"clear target", "clear focus", "release target", "stop tracking", "track everyone", "show everyone"}:
            return VoiceCommandPlan(action="clear_focus", query=spoken_command, confidence=0.95)

        if any(phrase in lowered for phrase in ["list all people", "show all records", "show database", "list database records"]):
            return VoiceCommandPlan(action="list_people", query=spoken_command, confidence=0.8)

        if any(phrase in lowered for phrase in ["object database", "object memory", "list objects", "show object records"]):
            return VoiceCommandPlan(action="list_objects", query=spoken_command, confidence=0.8)

        object_record_match = re.search(
            r"(?:(?:how many|show|give|list|find|what are)\s+)?(?P<label>[a-z0-9 _-]+?)\s+(?:records|record|objects|object)\b",
            lowered,
        )
        if object_record_match and object_memory:
            spoken_label = object_record_match.group("label").strip(" ,.!?")
            resolved_label = self.resolve_object_label(spoken_label, object_memory)
            return VoiceCommandPlan(
                action="get_object_records",
                object_label=resolved_label or spoken_label,
                query=spoken_command,
                confidence=0.78 if resolved_label else 0.48,
            )

        create_verbs = ("add", "create", "enroll", "register", "save")
        if any(lowered.startswith(prefix) for prefix in create_verbs):
            name_match = re.search(
                r"(?:with\s+name|named|name\s+is)\s+(?P<name>[a-z0-9][a-z0-9 _-]*?)(?=\s+(?:and|with)\s+(?:emp(?:loyee)?\s*)?id\b|$)",
                lowered,
            )
            id_match = re.search(r"(?:emp(?:loyee)?\s*)?id\s+(?P<id>[a-z0-9-]+)\b", lowered)
            if "this person" in lowered or "this face" in lowered or "to database" in lowered or name_match or id_match:
                fields: dict[str, str] = {}
                person_name = name_match.group("name").strip(" ,.!?") if name_match else ""
                person_id = id_match.group("id").strip(" ,.!?") if id_match else ""
                if person_name:
                    fields["name"] = person_name
                if person_id:
                    fields["person_id"] = person_id
                return VoiceCommandPlan(
                    action="create_person",
                    person_id=person_id,
                    person_name=person_name,
                    query=spoken_command,
                    fields=fields,
                    confidence=0.84 if person_name and person_id else 0.62,
                )

        focus_match = re.match(
            r"^(?:show(?: me)?|focus on|lock on|watch|track|spot|find)\s+(?:person\s+|employee\s+|user\s+)?(?P<name>.+)$",
            lowered,
        )
        if focus_match:
            target_name = focus_match.group("name").strip(" ,.!?")
            matched = self.resolve_person_name(target_name, candidates)
            return VoiceCommandPlan(
                action="focus_person",
                person_id=str(matched.get("person_id", "")).strip() if matched else "",
                person_name=str(matched.get("name", "")).strip() if matched else target_name,
                query=spoken_command,
                confidence=0.82 if matched else 0.45,
            )

        for prefix in ["show record for", "show profile for", "show db record for", "who is record for"]:
            if lowered.startswith(prefix):
                target_name = lowered[len(prefix) :].strip()
                matched = self.resolve_person_name(target_name, candidates)
                return VoiceCommandPlan(
                    action="get_person",
                    person_id=str(matched.get("person_id", "")).strip() if matched else "",
                    person_name=str(matched.get("name", "")).strip() if matched else target_name,
                    query=spoken_command,
                    confidence=0.75,
                )

        return VoiceCommandPlan(action="answer_query", query=spoken_command, confidence=0.3)

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

    def close(self) -> None:
        self.http.close()
