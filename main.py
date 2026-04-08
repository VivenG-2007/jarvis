"""
main.py — JARVIS Core Execution Pipeline

Synthesizes Vision, Voice, Context, Security Rules, and LLM Intelligence 
into a highly-concurrent > 20 FPS edge-ready pipeline.
"""

import os
import cv2
import time
import logging
import threading
import queue
from typing import List, Dict, Any

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")


os.environ["ORT_LOGGING_LEVEL"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Suppress HuggingFace download noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress httpx INFO logs (used internally by HuggingFace Hub)
import logging as _logging
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("httpcore").setLevel(_logging.WARNING)
_logging.getLogger("huggingface_hub").setLevel(_logging.WARNING)

# ── Import JARVIS Modules ──────────────────────────────────────
from modules.face_engine import FaceEngine
from modules.object_engine import ObjectEngine
from modules.context_builder import SemanticContextBuilder
from modules.rules_engine import RulesEngine
from modules.reasoning_engine import ReasoningEngine
from modules.hud_overlay import HUDOverlay
from modules.audio_engine import WhisperEngine
from modules.db import PersonDB

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("JARVIS.Core")

class JarvisSystem:
    def __init__(self, camera_index: int = 0):
        logger.info("Initializing JARVIS Subsystems...")
        
        # 1. Vision Hardware
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 2. Engines
        self.db = PersonDB()
        self.face_engine = FaceEngine(db=self.db)
        self.object_engine = ObjectEngine(model_path="yolov8s.pt")
        self.context_builder = SemanticContextBuilder()
        self.rules_engine = RulesEngine()
        self.reasoning_engine = ReasoningEngine(model_name="llama3")
        self.hud = HUDOverlay()
        
        self.audio_engine = WhisperEngine(model_size="tiny.en")
        
        # 3. State Memory (For sharing data between threads)
        self.running = True
        self.current_frame = None
        self.latest_objects = []
        self.latest_faces = []
        self.face_matches_cache = []
        
        self.latest_context_json = ""
        self.active_alerts = []
        self.ai_insight = ""
        
        # 4. Wake Word & Session State
        self.wake_active = False
        self.last_voice_time = 0
        
        # 5. Asynchronous Queues
        self.llm_query_queue = queue.Queue()
        
        logger.info("JARVIS Systems Online.")

    # ── Background Thread: YOLO Object Detection ───────────────
    def _object_worker(self):
        """Runs YOLO detection independently to secure high FPS."""
        while self.running:
            if self.current_frame is not None:
                detections = self.object_engine.detect(self.current_frame.copy(), conf=0.50)
                self.latest_objects = detections
                
                # Log each detected object into MongoDB (count + last_seen)
                for obj in detections:
                    if obj.class_id != 0:  # Skip persons (handled by FaceEngine)
                        self.db.log_object(obj.label, obj.confidence)
                        
            time.sleep(0.1)

    # ── Background Thread: Face Recognition ─────────────────────
    def _face_worker(self):
        """Offloads heavy InsightFace math from the main UI thread."""
        while self.running:
            if self.current_frame is not None:
                # InsightFace runs as fast as the CPU allows without blocking the camera
                self.face_matches_cache = self.face_engine.process_frame(self.current_frame.copy())
            time.sleep(0.05)

    # ── Background Thread: AI Semantic Reasoning ───────────────
    def _reasoning_worker(self):
        """Processes heavy LLM JSON-Context inference without freezing the UI."""
        while self.running:
            try:
                # Wait for an audio query or system prompt
                query = self.llm_query_queue.get(timeout=1)
                
                # Snapshot the exact semantic memory of the room
                context_snapshot = self.latest_context_json
                logger.info(f"LLM Thinking about: '{query}'...")
                
                # Generate grounded response
                self.ai_insight = self.reasoning_engine.analyze_scene(context_snapshot, query)
                logger.info(f"JARVIS: {self.ai_insight}")
                
            except queue.Empty:
                pass

    # ── Core Pipeline ──────────────────────────────────────────
    def run(self):
        """Primary System Loop."""
        
        # Boot Daemon Threads
        threading.Thread(target=self._object_worker, daemon=True, name="YOLO").start()
        threading.Thread(target=self._face_worker, daemon=True, name="InsightFace").start()
        threading.Thread(target=self._reasoning_worker, daemon=True, name="LLM").start()
        self.audio_engine.start_background_listening()
        
        logger.info("Booting Camera Feed...")
        frame_time = time.time()
        fps = 0.0

        while self.running:
            start_t = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.current_frame = frame.copy()

            # 1. Map Vision Thread Arrays
            # We no longer block the UI loop for AI processing! We only map the results cache asynchronously.
            faces_payload = []
            faces_names = []
            for m in self.face_matches_cache:
                bbox = m.bbox.as_list()
                faces_payload.append({
                    "id": m.track_id,
                    "bbox": bbox,
                    "name": m.name or "Unknown",
                    "department": m.department or "",
                    "role": m.role or "",
                    "confidence": m.confidence
                })
                faces_names.append(m.name or "Unknown")
                
            objects_payload = []
            objects_labels = []
            for obj in self.latest_objects:
                if obj.class_id == 0: continue # Exclude human bounding boxes from YOLO (InsightFace handles them)
                objects_payload.append({"bbox": obj.bbox, "label": obj.label})
                objects_labels.append(obj.label)

            # 2. Semantic Context Builder
            self.latest_context_json = self.context_builder.build_context(faces_names, objects_labels)

            # 3. Security Rules Engine
            rules_output_json = self.rules_engine.process_context(self.latest_context_json)
            import json
            rules_output = json.loads(rules_output_json)
            self.active_alerts = rules_output.get("alerts", [])

            # 4. Audio Input Intercept
            voice_command = self.audio_engine.get_latest_command()
            if voice_command:
                logger.info(f"AUDIO CAPTURED: '{voice_command}'")
                cmd_lower = voice_command.lower().strip(".,! ")
                
                # Check for Wake Word variations
                if "hey jarvis" in cmd_lower or "jarvis" == cmd_lower:
                    self.wake_active = True
                    self.last_voice_time = time.time()
                    self.active_alerts.append("JARVIS: System Wakened.")
                    # Strip wake word if present in a larger command
                    voice_command = voice_command.replace("Hey JARVIS", "").replace("hey jarvis", "").strip()

                # Process command ONLY if wake word is active
                if self.wake_active:
                    self.last_voice_time = time.time()
                    if voice_command:
                        self.llm_query_queue.put(voice_command)
                        self.active_alerts.append(f"Command: '{voice_command}'")

            # Auto-timeout Session (5 seconds)
            if self.wake_active and (time.time() - self.last_voice_time > 5.0):
                self.wake_active = False
                self.active_alerts.append("JARVIS: Entering Standby.")

            # 5. UI Overlay Render
            display_text = self.ai_insight if self.ai_insight else "JARVIS Standby. Say 'Hey JARVIS' to activate."

            hud_frame = self.hud.render(
                frame=frame,
                faces=faces_payload,
                objects=objects_payload,
                alerts=self.active_alerts,
                context_summary=display_text,
                is_listening=self.wake_active
            )
            
            # Draw FPS
            cv2.putText(hud_frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 6. Output to Screen
            cv2.imshow("JARVIS Main HUD", hud_frame)

            # Calculate precise main-loop FPS
            elapsed = time.time() - start_t
            fps = 1.0 / elapsed if elapsed > 0 else 0
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                self.running = False
                break

        # Shutdown sequence
        logger.info("Deactivating JARVIS Systems...")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    jarvis = JarvisSystem()
    jarvis.run()
