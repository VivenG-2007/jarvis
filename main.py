from __future__ import annotations

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
from modules.reasoning_engine import ReasoningEngine
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

    def __post_init__(self) -> None:
        self.faces = []
        self.objects = []
        self.latest_context = {}
        self.alerts = []


class JarvisSystem:
    def __init__(self):
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

    def start(self) -> None:
        self.audio_engine.start_background_listening()
        self.tts.start()
        threading.Thread(target=self._capture_worker, daemon=True, name="CaptureWorker").start()
        threading.Thread(target=self._vision_worker, daemon=True, name="VisionWorker").start()
        threading.Thread(target=self._reasoning_worker, daemon=True, name="ReasoningWorker").start()
        self._run_ui_loop()

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
            if camera_frame is None:
                time.sleep(0.01)
                continue
            source_frame, source_name = self._choose_frame(camera_frame, include_screen=config.APP.enable_screen_input)
            faces = self.face_engine.process_frame(source_frame, focus_person_id=target_person_id)
            objects = self.object_engine.detect(source_frame, source=source_name)
            for obj in objects:
                if obj.confidence >= config.APP.object_db_conf_threshold:
                    self.db.log_object(obj.label, obj.confidence)
            faces_payload = [self._face_to_payload(face) for face in faces]
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
            with self.state_lock:
                self.state.latest_response = response
            self.tts.speak(response)

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
                faces = [self._face_to_payload(face) for face in self.state.faces]
                context = dict(self.state.latest_context)
                objects = list(context.get("objects", {}).get("tracks", [])) or [asdict(obj) for obj in self.state.objects]
                alerts = list(self.state.alerts)
                response = self.state.latest_response
                heard_text = self.state.latest_voice_text
                focus_mode_active = bool(self.state.target_person_id)
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
            if self.wake_active and time.time() - self.last_wake_time > config.APP.command_timeout_sec:
                self.wake_active = False
            return
        self.state.latest_voice_text = transcript
        normalized = transcript.lower().strip()
        wake_word = config.APP.wake_word.lower()
        if wake_word in normalized:
            self.wake_active = True
            self.last_wake_time = time.time()
            logger.info("Wake word detected in transcript: %s", transcript)
            normalized = normalized.replace(wake_word, "").strip(" ,.!?")
            with self.state_lock:
                self.state.latest_response = "Listening for a grounded follow-up."
        if self.wake_active and normalized:
            self.last_wake_time = time.time()
            if self._handle_focus_command(normalized):
                return
            logger.info("Queueing grounded voice query: %s", normalized)
            self.reasoning_queue.put(normalized)

    def _handle_focus_command(self, normalized: str) -> bool:
        command = normalized.strip()
        if not command:
            return False

        if command in {"clear target", "clear focus", "release target", "stop tracking", "track everyone"}:
            with self.state_lock:
                self.state.target_person_id = ""
                self.state.target_person_name = ""
                self.state.latest_response = "Focus target cleared. Monitoring the full scene again."
            logger.info("Cleared requested focus target.")
            return True

        target_name = self._extract_focus_target(command)
        if not target_name:
            return False

        target_name = re.sub(r"\b(?:in|from)\s+the\s+crowd\b.*$", "", target_name).strip(" ,.!?")
        people = self.db.get_all_persons()
        person = self.reasoning_engine.resolve_person_name(target_name, people)
        with self.state_lock:
            if person is None:
                self.state.latest_response = f"I could not find {target_name} in the enrolled records."
            else:
                self.state.target_person_id = person["person_id"]
                self.state.target_person_name = person["name"]
                if person["name"].strip().lower() != target_name.strip().lower():
                    self.state.latest_response = f"Tracking {person['name']}. I matched that from your spoken request and will isolate them in the crowd."
                else:
                    self.state.latest_response = f"Tracking {person['name']}. I will isolate them from the crowd when they appear."
        logger.info("Focus command resolved target=%s matched=%s", target_name, person["person_id"] if person else "none")
        return True

    def _extract_focus_target(self, command: str) -> str:
        patterns = [
            r"^(?:focus on|lock on|watch|track|spot)\s+(?P<name>.+)$",
            r"^find\s+(?:person\s+|employee\s+|user\s+)?(?P<name>.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, command)
            if match:
                return match.group("name").strip()
        return ""

    def _face_to_payload(self, face: FaceMatch) -> dict[str, Any]:
        return {
            "id": face.track_id,
            "bbox": face.bbox.as_list(),
            "person_id": face.person_id or "UNKNOWN",
            "name": face.name or "Unknown",
            "department": face.department or "N/A",
            "role": face.role or "N/A",
            "confidence": face.confidence,
            "is_known": face.is_known,
            "lock_candidate": face.lock_candidate,
        }

    def _save_screenshot(self, frame: np.ndarray) -> None:
        output = config.SCREENSHOTS_DIR / f"jarvis_{int(time.time())}.png"
        cv2.imwrite(str(output), frame)
        logger.info("Saved screenshot to %s", output)

    def shutdown(self) -> None:
        self.audio_engine.stop()
        self.tts.stop()
        self.cap.release()
        self.db.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    JarvisSystem().start()
