from __future__ import annotations

import difflib
import logging
import queue
import re
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Optional

import cv2
import numpy as np

import config
from modules.audio_engine import WhisperEngine
from modules.context_builder import SemanticContextBuilder
from modules.db import PersonDB
from modules.face_engine import FaceEngine, FaceMatch
from modules.hud_overlay import HUDOverlay
from modules.object_engine import ObjectDetection, ObjectEngine
from modules.reasoning_engine import ReasoningEngine, VoiceCommandPlan
from modules.rules_engine import RulesEngine
from modules.screen_capture import ScreenCapture
from modules.tts_engine import TTSEngine

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(config.APP.log_file, encoding="utf-8")],
)
logger = logging.getLogger("jarvis.core")


@dataclass
class SharedState:
    latest_camera_frame: Optional[np.ndarray] = None
    faces: list[FaceMatch] = None
    objects: list[ObjectDetection] = None
    latest_context: dict[str, Any] = None
    latest_response: str = "System online. Say 'Jarvis' and ask a grounded question."
    latest_voice_text: str = ""
    alerts: list[str] = None
    target_person_id: str = ""
    target_person_name: str = ""
    suppressed_person_ids: set[str] = None

    def __post_init__(self) -> None:
        self.faces = []
        self.objects = []
        self.latest_context = {}
        self.alerts = []
        self.suppressed_person_ids = set()


class JarvisSystem:
    def __init__(self):
        cv2.setUseOptimized(True)
        self.db = PersonDB()
        self.face_engine = FaceEngine(self.db)
        self.object_engine = ObjectEngine()
        self.audio_engine = WhisperEngine()
        self.context_builder = SemanticContextBuilder(self.db)
        self.rules_engine = RulesEngine()
        self.reasoning_engine = ReasoningEngine()
        self.hud = HUDOverlay()
        self.screen = ScreenCapture()
        self.tts = TTSEngine(enabled=config.APP.tts_enabled)

        self.cap = cv2.VideoCapture(config.APP.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.APP.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.APP.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, config.APP.camera_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.state = SharedState()
        self.state_lock = threading.Lock()
        self.reasoning_queue: queue.Queue[str] = queue.Queue()
        self.wake_active = False
        self.last_wake_time = 0.0
        self.wake_deadline = 0.0

    def start(self) -> None:
        self.audio_engine.start_background_listening()
        self.tts.start()
        threading.Thread(target=self._capture_worker, daemon=True, name="CaptureWorker").start()
        threading.Thread(target=self._vision_worker, daemon=True, name="VisionWorker").start()
        threading.Thread(target=self._reasoning_worker, daemon=True, name="ReasoningWorker").start()
        self._run_ui_loop()

    def _publish_response(self, text: str, *, speak: bool = False) -> None:
        message = " ".join(text.strip().split())
        if not message:
            return
        with self.state_lock:
            self.state.latest_response = message
        if speak:
            self.tts.speak(message)

    def _capture_worker(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if config.APP.mirror_camera:
                frame = cv2.flip(frame, 1)
            with self.state_lock:
                self.state.latest_camera_frame = frame

    def _vision_worker(self) -> None:
        target_interval = 1.0 / max(config.APP.vision_target_fps, 1.0)
        while self.running:
            cycle_started = time.time()
            with self.state_lock:
                camera_frame = None if self.state.latest_camera_frame is None else self.state.latest_camera_frame.copy()
                target_person_id = self.state.target_person_id
                target_person_name = self.state.target_person_name
                voice_query = self.state.latest_voice_text
                suppressed_person_ids = set(self.state.suppressed_person_ids)
            if camera_frame is None:
                time.sleep(0.01)
                continue
            source_frame, source_name = self._choose_frame(camera_frame, include_screen=config.APP.enable_screen_input)
            faces = self.face_engine.process_frame(source_frame, focus_person_id=target_person_id)
            for face in faces:
                person_id = str(face.person_id or "").strip()
                if person_id and person_id in suppressed_person_ids:
                    face.lock_candidate = False
            objects = self.object_engine.detect(source_frame, source=source_name)
            for obj in objects:
                if obj.confidence >= config.APP.object_db_conf_threshold:
                    self.db.log_object(obj.label, obj.confidence)
            faces_payload = [self._face_to_payload(face, suppressed_person_ids) for face in faces]
            object_payload = [asdict(obj) for obj in objects]
            context = self.context_builder.build_context(
                faces_payload,
                object_payload,
                voice_query=voice_query,
                source=source_name,
                requested_target=target_person_name,
            )
            rule_result = self.rules_engine.process_context(context)
            with self.state_lock:
                self.state.faces = faces
                self.state.objects = objects
                self.state.latest_context = context
                self.state.alerts = rule_result["alerts"]
            remaining = target_interval - (time.time() - cycle_started)
            if remaining > 0:
                time.sleep(min(remaining, 0.02))

    def _choose_frame(self, camera_frame: np.ndarray, include_screen: bool = False) -> tuple[np.ndarray, str]:
        mode = config.APP.fusion_mode.lower()
        screen_frame = self.screen.grab() if include_screen and mode in {"screen", "pip"} else None
        if mode == "screen" and screen_frame is not None:
            return screen_frame, "screen"
        if mode == "pip" and screen_frame is not None:
            screen_small = cv2.resize(screen_frame, (camera_frame.shape[1] // 3, camera_frame.shape[0] // 3))
            camera_frame[20 : 20 + screen_small.shape[0], 20 : 20 + screen_small.shape[1]] = screen_small
            return camera_frame, "camera+screen"
        return camera_frame, "camera"

    def _reasoning_worker(self) -> None:
        while self.running:
            try:
                query = self.reasoning_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.state_lock:
                context = dict(self.state.latest_context)
                frame = None if self.state.latest_camera_frame is None else self.state.latest_camera_frame.copy()
            response = self.reasoning_engine.analyze_scene(context, query, frame=frame)
            self._publish_response(response, speak=True)

    def _run_ui_loop(self) -> None:
        window_name = "JARVIS Local Assistant"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if config.APP.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        last_frame_time = time.time()
        while self.running:
            self._poll_audio()
            with self.state_lock:
                camera_frame = None if self.state.latest_camera_frame is None else self.state.latest_camera_frame.copy()
                raw_faces = list(self.state.faces)
                context = dict(self.state.latest_context)
                objects = list(context.get("objects", {}).get("tracks", [])) or [asdict(obj) for obj in self.state.objects]
                alerts = list(self.state.alerts)
                response = self.state.latest_response
                heard_text = self.state.latest_voice_text
                focus_mode_active = bool(self.state.target_person_id)
                suppressed_person_ids = set(self.state.suppressed_person_ids)
            faces = [self._face_to_payload(face, suppressed_person_ids) for face in raw_faces]
            if camera_frame is None:
                time.sleep(0.01)
                continue
            frame, _ = self._choose_frame(camera_frame, include_screen=config.APP.enable_screen_input)
            overlay = self.hud.render(
                frame,
                faces,
                objects,
                alerts,
                context_summary=response,
                heard_text=heard_text,
                is_listening=self.audio_engine.voice_active,
                wake_active=self.wake_active,
                audio_level=self.audio_engine.audio_level,
                focus_mode_active=focus_mode_active,
            )
            fps = 1.0 / max(time.time() - last_frame_time, 1e-6)
            last_frame_time = time.time()
            cv2.putText(overlay, f"{fps:04.1f} FPS", (overlay.shape[1] - 100, overlay.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.APP.hud_color_primary, 1, cv2.LINE_AA)
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                self.running = False
            elif key == ord("s"):
                self._save_screenshot(overlay)
            elif key == ord("f"):
                current = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                next_state = cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, next_state)
        self.shutdown()

    def _poll_audio(self) -> None:
        transcript = self.audio_engine.get_latest_command()
        if not transcript:
            if self.wake_active and time.time() > self.wake_deadline:
                self.wake_active = False
                self.wake_deadline = 0.0
                self._publish_response("Voice mode closed. Say Jarvis to start again.")
            return
        self.state.latest_voice_text = transcript
        normalized = transcript.lower().strip()
        wake_detected, normalized = self._extract_wake_command(normalized)
        if wake_detected:
            self.wake_active = True
            self.last_wake_time = time.time()
            self.wake_deadline = self.last_wake_time + config.APP.command_timeout_sec
            logger.info("Wake word detected in transcript: %s", transcript)
            self._publish_response("Listening for your command.")
        if self.wake_active and normalized:
            self._publish_response(f"Command received: {normalized}. Processing now.")
            handled = self._handle_voice_command(normalized)
            if not handled:
                logger.info("Queueing grounded voice query: %s", normalized)
                self.reasoning_queue.put(normalized)
            self.wake_active = False
            self.wake_deadline = 0.0
            if handled:
                return

    def _extract_wake_command(self, transcript: str) -> tuple[bool, str]:
        tokens = re.findall(r"[a-z0-9']+", transcript.lower())
        if not tokens:
            return False, ""

        wake_word = config.APP.wake_word.lower().strip()
        wake_aliases = {wake_word}
        if wake_word == "jarvis":
            wake_aliases.update({"javis", "jervis", "jarvish", "jarviss", "jarviz"})

        remaining: list[str] = []
        matched = False
        for token in tokens:
            if not matched and self._is_wake_token(token, wake_word, wake_aliases):
                matched = True
                continue
            remaining.append(token)
        return matched, " ".join(remaining).strip()

    @staticmethod
    def _is_wake_token(token: str, wake_word: str, wake_aliases: set[str]) -> bool:
        if token in wake_aliases:
            return True
        if len(token) < max(3, len(wake_word) - 2):
            return False
        if token[0] != wake_word[0]:
            return False
        ratio = difflib.SequenceMatcher(None, token, wake_word).ratio()
        return ratio >= 0.78

    def _handle_voice_command(self, normalized: str) -> bool:
        command = normalized.strip()
        if not command:
            return False
        if not self._is_probable_command(command):
            self._publish_response("I heard audio after Jarvis, but it did not sound like a command.", speak=True)
            logger.info("Ignored non-command speech after wake word: %s", command)
            return True

        if self._handle_focus_command(command):
            return True

        people = self.db.list_persons()
        with self.state_lock:
            context = dict(self.state.latest_context)
        plan = self.reasoning_engine.plan_voice_command(command, people, context=context)
        return self._execute_voice_plan(plan)

    @staticmethod
    def _is_probable_command(command: str) -> bool:
        cleaned = re.sub(r"\s+", " ", command.lower()).strip(" .,!?:;")
        if not cleaned:
            return False

        filler_phrases = {
            "uh",
            "um",
            "hmm",
            "ah",
            "okay",
            "ok",
            "yes",
            "yeah",
            "hello",
            "hi",
            "huh",
            "hmmm",
        }
        if cleaned in filler_phrases:
            return False

        tokens = re.findall(r"[a-z0-9']+", cleaned)
        if not tokens:
            return False
        if len(tokens) == 1 and tokens[0] in filler_phrases:
            return False

        command_verbs = {
            "show",
            "find",
            "track",
            "focus",
            "lock",
            "watch",
            "spot",
            "clear",
            "release",
            "list",
            "create",
            "add",
            "update",
            "change",
            "delete",
            "remove",
            "who",
            "what",
            "where",
            "when",
            "which",
            "tell",
            "scan",
            "identify",
            "read",
            "open",
            "close",
        }
        return len(tokens) >= 2 or tokens[0] in command_verbs

    def _execute_voice_plan(self, plan: VoiceCommandPlan) -> bool:
        action = plan.action.strip().lower()
        if not action or action == "answer_query":
            return False

        if action == "clear_focus":
            with self.state_lock:
                self.state.target_person_id = ""
                self.state.target_person_name = ""
            self._publish_response("Focus target cleared. Monitoring the full scene again.", speak=True)
            logger.info("Cleared requested focus target.")
            return True

        if action == "focus_person":
            return self._apply_focus_target(plan.person_name or plan.query, explicit_person_id=plan.person_id)

        if action == "list_people":
            records = self.db.list_persons()
            if not records:
                response = "I do not have any person records stored yet."
            else:
                preview = ", ".join(
                    f"{item.get('name', 'Unknown')} ({item.get('person_id', 'N/A')})"
                    for item in records[:8]
                )
                response = f"Database records: {preview}."
            self._publish_response(response, speak=True)
            return True

        if action == "list_objects":
            records = self.db.get_object_memory(limit=12)
            if not records:
                response = "I do not have any object records stored yet."
            else:
                preview = ", ".join(
                    f"{item.get('label', 'unknown')} x{item.get('count', 0)}"
                    for item in records[:8]
                )
                response = f"Object database records: {preview}."
            self._publish_response(response, speak=True)
            return True

        if action == "get_object_records":
            record = self._resolve_object_record(plan.object_label or plan.query)
            response = self._summarize_object_record(record, missing_label=plan.object_label or plan.query)
            self._publish_response(response, speak=True)
            return True

        if action == "get_person":
            person = self._resolve_db_person(plan.person_id, plan.person_name)
            response = self._summarize_person_record(person, missing_name=plan.person_name)
            if person is not None:
                self._set_focus_target(person["person_id"], person["name"])
            self._publish_response(response, speak=True)
            return True

        if action == "create_person":
            fields = dict(plan.fields or {})
            person_id = self._normalize_person_id(plan.person_id or fields.get("person_id", "").strip())
            person_name = plan.person_name or fields.get("name", "").strip()
            if not person_id or not person_name:
                response = "I need both a person ID and a name to create a record."
            else:
                enrollment_face, face_error = self._get_enrollment_face()
                if enrollment_face is None:
                    response = face_error
                    self._publish_response(response, speak=True)
                    return True
                existing = self.db.get_person_by_id(person_id)
                created = self.db.enroll_person(
                    person_id=person_id,
                    name=person_name,
                    department=fields.get("department", ""),
                    role=fields.get("role", ""),
                    embedding=enrollment_face.embedding,
                    notes=fields.get("notes", ""),
                    photo_path=fields.get("photo_path"),
                )
                if created:
                    with self.state_lock:
                        self.state.suppressed_person_ids.discard(person_id)
                        self._set_focus_target(person_id, person_name)
                    action_word = "Updated" if existing is not None else "Created"
                    response = f"{action_word} live face record for {person_name} with ID {person_id}. Tracking started now."
                else:
                    response = f"I could not save the live face record for {person_name}."
            self._publish_response(response, speak=True)
            return True

        if action == "update_person":
            person = self._resolve_db_person(plan.person_id, plan.person_name)
            fields = dict(plan.fields or {})
            if person is None:
                response = f"I could not find {plan.person_name or plan.person_id or 'that person'} in the database."
            elif not fields:
                response = "I need at least one field to update, such as department, role, name, or notes."
            else:
                updated = self.db.update_person_metadata(str(person["person_id"]), fields)
                response = (
                    f"Updated {updated['name']}'s record."
                    if updated is not None
                    else f"I could not update {person['name']}'s record."
                )
            self._publish_response(response, speak=True)
            return True

        if action == "delete_person":
            person = self._resolve_db_person(plan.person_id, plan.person_name)
            if person is None:
                response = f"I could not find {plan.person_name or plan.person_id or 'that person'} in the database."
            else:
                deleted = self.db.delete_person(str(person["person_id"]))
                if deleted:
                    with self.state_lock:
                        if self.state.target_person_id == str(person["person_id"]):
                            self.state.target_person_id = ""
                            self.state.target_person_name = ""
                    response = f"Deleted the record for {person['name']}."
                else:
                    response = f"I could not delete the record for {person['name']}."
            self._publish_response(response, speak=True)
            return True

        return False

    @staticmethod
    def _normalize_person_id(raw_value: str) -> str:
        cleaned = raw_value.strip().upper()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\s+", "", cleaned)
        if cleaned.isdigit():
            return f"EMP-{cleaned}"
        match = re.fullmatch(r"EMP[-_ ]?(\d+)", cleaned)
        if match:
            return f"EMP-{match.group(1)}"
        return cleaned

    def _get_enrollment_face(self) -> tuple[FaceMatch | None, str]:
        with self.state_lock:
            visible_faces = list(self.state.faces)
            focus_person_id = self.state.target_person_id.strip()

        candidates = [face for face in visible_faces if face.embedding is not None]
        if not candidates:
            return None, "I need one clear visible face before I can enroll this person."

        if focus_person_id:
            focused = [face for face in candidates if str(face.person_id or "").strip() == focus_person_id]
            if len(focused) == 1:
                return focused[0], ""

        if len(candidates) == 1:
            return candidates[0], ""

        lock_candidates = [face for face in candidates if face.lock_candidate]
        if len(lock_candidates) == 1:
            return lock_candidates[0], ""

        return None, "I can see multiple faces right now. Bring one person into view or lock onto them first, then ask again."

    def _handle_focus_command(self, command: str) -> bool:
        if command in {"clear target", "clear focus", "release target", "stop tracking", "track everyone", "show everyone"}:
            with self.state_lock:
                self.state.target_person_id = ""
                self.state.target_person_name = ""
                self.state.suppressed_person_ids.clear()
            self._publish_response("Focus target cleared. Monitoring the full scene again.", speak=True)
            logger.info("Cleared requested focus target.")
            return True

        hidden_name = self._extract_hidden_target(command)
        if hidden_name:
            return self._suppress_target(hidden_name)

        target_name = self._extract_focus_target(command)
        if not target_name:
            return False

        return self._apply_focus_target(target_name)

    def _apply_focus_target(self, target_name: str, explicit_person_id: str = "") -> bool:
        target_name = target_name.strip()
        if not target_name and not explicit_person_id:
            return False

        target_name = re.sub(r"\b(?:in|from)\s+the\s+crowd\b.*$", "", target_name).strip(" ,.!?")
        people = self.db.get_all_persons()
        person = None
        if explicit_person_id:
            person = next((item for item in people if str(item.get("person_id", "")).strip() == explicit_person_id.strip()), None)
        if person is None:
            person = self.reasoning_engine.resolve_person_name(target_name, people)
        if person is None:
            response = f"I could not find {target_name} in the enrolled records."
        else:
            with self.state_lock:
                self.state.suppressed_person_ids.discard(str(person["person_id"]))
                self._set_focus_target(person["person_id"], person["name"])
            if person["name"].strip().lower() != target_name.strip().lower():
                response = f"Tracking {person['name']}. I matched that from your spoken request and will isolate them in the crowd."
            else:
                response = f"Tracking {person['name']}. I will isolate them from the crowd when they appear."
        self._publish_response(response, speak=True)
        logger.info("Focus command resolved target=%s matched=%s", target_name, person["person_id"] if person else "none")
        return True

    def _set_focus_target(self, person_id: str, person_name: str) -> None:
        self.state.target_person_id = str(person_id)
        self.state.target_person_name = str(person_name)

    def _extract_focus_target(self, command: str) -> str:
        patterns = [
            r"^(?:focus on|lock on|watch|track|spot|show(?: me)?)\s+(?P<name>.+)$",
            r"^find\s+(?:person\s+|employee\s+|user\s+)?(?P<name>.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, command)
            if match:
                return match.group("name").strip()
        return ""

    def _extract_hidden_target(self, command: str) -> str:
        patterns = [
            r"^(?:remove|hide|ignore|don't show|do not show)\s+(?:person\s+|employee\s+|user\s+)?(?P<name>.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, command)
            if match:
                return match.group("name").strip()
        return ""

    def _suppress_target(self, target_name: str) -> bool:
        people = self.db.get_all_persons()
        person = self.reasoning_engine.resolve_person_name(target_name, people)
        if person is None:
            self._publish_response(f"I could not find {target_name} in the enrolled records.", speak=True)
            return True

        with self.state_lock:
            person_id = str(person["person_id"])
            self.state.suppressed_person_ids.add(person_id)
            if self.state.target_person_id == person_id:
                self.state.target_person_id = ""
                self.state.target_person_name = ""
        self._publish_response(
            f"Okay. I will stop marking {person['name']} until you ask for them again.",
            speak=True,
        )
        logger.info("Suppressed overlay marking for target=%s", person_id)
        return True

    def _resolve_db_person(self, person_id: str, person_name: str) -> dict[str, Any] | None:
        if person_id.strip():
            person = self.db.get_person_by_id(person_id.strip())
            if person is not None:
                return person
        if person_name.strip():
            person = self.db.find_person_by_name(person_name.strip())
            if person is not None:
                return person
            people = self.db.get_all_persons()
            return self.reasoning_engine.resolve_person_name(person_name.strip(), people)
        return None

    def _resolve_object_record(self, label_hint: str) -> dict[str, Any] | None:
        object_memory = self.db.get_object_memory(limit=50)
        if not label_hint.strip():
            return object_memory[0] if object_memory else None
        resolved_label = self.reasoning_engine.resolve_object_label(label_hint, object_memory)
        if not resolved_label:
            return None
        return next(
            (item for item in object_memory if str(item.get("label", "")).strip().lower() == resolved_label.strip().lower()),
            None,
        )

    @staticmethod
    def _summarize_person_record(person: dict[str, Any] | None, missing_name: str = "") -> str:
        if person is None:
            label = missing_name or "that person"
            return f"I could not find {label} in the database."
        return (
            f"{person.get('name', 'Unknown')} has ID {person.get('person_id', 'N/A')}, "
            f"role {person.get('role', 'unknown')}, department {person.get('department', 'unknown')}, "
            f"notes {person.get('notes', 'none')}."
        )

    @staticmethod
    def _summarize_object_record(record: dict[str, Any] | None, missing_label: str = "") -> str:
        if record is None:
            label = missing_label or "that object"
            return f"I could not find records for {label} in the object database."
        last_seen = record.get("last_seen") or record.get("timestamp") or "unknown"
        confidence = record.get("confidence")
        confidence_text = f"{float(confidence) * 100:.0f}% top confidence" if confidence is not None else "confidence unavailable"
        return (
            f"I found {int(record.get('count', 0))} records for {record.get('label', 'that object')}. "
            f"Last seen {last_seen}, {confidence_text}."
        )

    def _face_to_payload(self, face: FaceMatch, suppressed_person_ids: set[str] | None = None) -> dict[str, Any]:
        suppressed = str(face.person_id or "").strip() in (suppressed_person_ids or set())
        return {
            "id": face.track_id,
            "bbox": face.bbox.as_list(),
            "person_id": face.person_id or "UNKNOWN",
            "name": face.name or "Unknown",
            "department": face.department or "N/A",
            "role": face.role or "N/A",
            "confidence": face.confidence,
            "is_known": face.is_known and not suppressed,
            "lock_candidate": face.lock_candidate,
            "suppressed_overlay": suppressed,
        }

    def _save_screenshot(self, frame: np.ndarray) -> None:
        output = config.SCREENSHOTS_DIR / f"jarvis_{int(time.time())}.png"
        cv2.imwrite(str(output), frame)
        logger.info("Saved screenshot to %s", output)

    def shutdown(self) -> None:
        self.audio_engine.stop()
        self.tts.stop()
        self.cap.release()
        self.reasoning_engine.close()
        self.db.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    JarvisSystem().start()
