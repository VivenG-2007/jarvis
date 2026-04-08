"""
main.py — JARVIS Face Recognition & HUD Display
Entry point for the real-time recognition pipeline.

Controls:
  Q / ESC   — quit
  F         — toggle fullscreen
  S         — screenshot
  R         — reload known persons from DB
"""

import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config
from modules.db import PersonDB
from modules.face_engine import FaceEngine, BBox, FaceMatch
from modules.object_engine import ObjectEngine, ObjectDetection
from modules.hud import draw_hud

# ── Logging setup ──────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
handlers = [logging.StreamHandler()]
if config.LOG_TO_FILE:
    handlers.append(logging.FileHandler(config.LOG_FILE, encoding="utf-8"))

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("jarvis.main")


# ── Shared state for multi-threaded pipeline ───────────────────

class SharedState:
    def __init__(self):
        self.frame:    np.ndarray | None = None
        self.matches:  list              = []
        self.objects:  list              = []
        self.fps:      float             = 0.0
        self.lock      = threading.Lock()
        self.running   = True


# ── Recognition worker thread ──────────────────────────────────

def recognition_worker(state: SharedState, engine: FaceEngine):
    """Runs in a background thread; processes frames and updates matches."""
    fps_buf = deque(maxlen=30)
    prev_t  = time.time()

    while state.running:
        with state.lock:
            frame = state.frame.copy() if state.frame is not None else None

        if frame is None:
            time.sleep(0.005)
            continue

        t0      = time.time()
        matches = engine.process_frame(frame)
        elapsed = time.time() - t0
        fps_buf.append(1.0 / max(elapsed, 1e-6))

        with state.lock:
            state.matches = matches
            state.fps     = sum(fps_buf) / len(fps_buf)

        # Cap recognition rate to avoid overloading CPU
        sleep_ms = max(0, 0.01 - elapsed)  # Reduced sleep for higher throughput
        time.sleep(sleep_ms)


# ── Object Detection worker thread ─────────────────────────────

def object_worker(state: SharedState, engine: ObjectEngine):
    """Runs in a background thread; processes every other frame for objects."""
    frame_count = 0
    while state.running:
        with state.lock:
            frame = state.frame.copy() if state.frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        # Run YOLO every 2 frames as requested
        if frame_count % 2 == 0:
            t0      = time.time()
            objects = engine.detect(frame)
            elapsed = time.time() - t0
            
            with state.lock:
                state.objects = objects
            
            # Dynamic sleep based on performance
            sleep_ms = max(0, 0.033 - elapsed)
            time.sleep(sleep_ms)
        else:
            # Short sleep to yield when skipping frame
            time.sleep(0.005)


# ── Zoom management ───────────────────────────────────────────

class ZoomManager:
    """Handles smooth 2x face zoom for 1 second upon first detection."""
    def __init__(self):
        self.active             = False
        self.start_t            = 0.0
        self.duration           = 1.0  # Total duration is now 1 second
        self.target_track_id    = -1
        self.max_zoom           = 2.0
        self.triggered_tracks   = set()
        
        # Smoothing state
        self.current_zoom       = 1.0
        self.smoothed_cx        = 0.0
        self.smoothed_cy        = 0.0
        self.first_frame        = True

    def update(self, matches):
        now = time.time()
        
        # If no zoom active, check if we should start one
        if not self.active:
            for m in matches:
                if m.track_id not in self.triggered_tracks:
                    self.active = True
                    self.start_t = now
                    self.target_track_id = m.track_id
                    self.triggered_tracks.add(m.track_id)
                    self.first_frame = True
                    logger.info("Smooth zoom activated for track %d", m.track_id)
                    break
        
        # If zoom active, check if it should expire
        if self.active:
            elapsed = now - self.start_t
            if elapsed > self.duration:
                # Smoothly returning to 1.0 before fully deactivating
                if self.current_zoom < 1.01:
                    self.active = False
                    self.target_track_id = -1
                    logger.info("Zoom deactivated")
            else:
                # Check if target is still present
                target_present = any(m.track_id == self.target_track_id for m in matches)
                if not target_present:
                    # If target lost, smoothly zoom out then deactivate
                    self.start_t = now - (self.duration * 0.8) # force near-end state

    def apply_zoom(self, frame, matches, objects=[]):
        """Crops and resizes frame with smooth interpolation."""
        now = time.time()
        h, w = frame.shape[:2]

        if not self.active:
            # Gradually return to 1.0 zoom if we just finished
            if self.current_zoom > 1.0:
                self.current_zoom = self.current_zoom + (1.0 - self.current_zoom) * 0.15
                if self.current_zoom < 1.001: self.current_zoom = 1.0
            
            if self.current_zoom == 1.0:
                return frame, matches, objects

        if self.active:
            elapsed = now - self.start_t
            progress = elapsed / self.duration
            if progress < 0.5:
                target_z = 1.0 + (self.max_zoom - 1.0) * (progress * 2)
            else:
                target_z = self.max_zoom - (self.max_zoom - 1.0) * ((progress - 0.5) * 2)
            target_z = max(1.0, target_z)
        else:
            target_z = 1.0

        self.current_zoom = self.current_zoom + (target_z - self.current_zoom) * 0.2

        if self.current_zoom <= 1.0:
            return frame, matches, objects

        target = next((m for m in matches if m.track_id == self.target_track_id), None)
        if target:
            tx, ty = target.bbox.center
            if self.first_frame:
                self.smoothed_cx, self.smoothed_cy = tx, ty
                self.first_frame = False
            else:
                self.smoothed_cx += (tx - self.smoothed_cx) * 0.15
                self.smoothed_cy += (ty - self.smoothed_cy) * 0.15

        crop_w = int(w / self.current_zoom)
        crop_h = int(h / self.current_zoom)
        
        x1 = int(max(0, self.smoothed_cx - crop_w // 2))
        y1 = int(max(0, self.smoothed_cy - crop_h // 2))
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        if x2 == w: x1 = w - crop_w
        if y2 == h: y1 = h - crop_h
        x1, y1 = max(0, x1), max(0, y1)

        cropped = frame[y1:y2, x1:x2]
        zoomed  = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        scale_x = w / crop_w
        scale_y = h / crop_h
        
        # Transform Faces
        zoomed_matches = []
        for m in matches:
            new_bbox = BBox(
                int((m.bbox.x1 - x1) * scale_x),
                int((m.bbox.y1 - y1) * scale_y),
                int((m.bbox.x2 - x1) * scale_x),
                int((m.bbox.y2 - y1) * scale_y)
            )
            m_zoomed = FaceMatch(
                bbox=new_bbox,
                embedding=m.embedding,
                person_id=m.person_id,
                name=m.name,
                department=m.department,
                role=m.role,
                confidence=m.confidence,
                is_known=m.is_known,
                track_id=m.track_id,
                last_seen=m.last_seen
            )
            zoomed_matches.append(m_zoomed)

        # Transform Objects
        from modules.object_engine import ObjectDetection
        zoomed_objects = []
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj.bbox
            nx1 = int((ox1 - x1) * scale_x)
            ny1 = int((oy1 - y1) * scale_y)
            nx2 = int((ox2 - x1) * scale_x)
            ny2 = int((oy2 - y1) * scale_y)
            
            # Clip to frame
            nx1, ny1 = max(0, nx1), max(0, ny1)
            nx2, ny2 = min(w, nx2), min(h, ny2)
            
            if nx2 > nx1 and ny2 > ny1:
                zoomed_objects.append(ObjectDetection(
                    label=obj.label,
                    confidence=obj.confidence,
                    bbox=[nx1, ny1, nx2, ny2],
                    class_id=obj.class_id
                ))

        return zoomed, zoomed_matches, zoomed_objects


# ── Main loop ──────────────────────────────────────────────────

def run():
    logger.info("=== JARVIS Face Recognition System starting ===")

    # Init DB
    db = PersonDB()
    if not db.is_connected():
        logger.warning("MongoDB unavailable — face data will not persist.")
        enrolled = db.list_persons()
    else:
        enrolled = db.list_persons()
        logger.info("Persons in database: %d", len(enrolled))

    # Init face engine
    engine = FaceEngine(db)
    
    # Init object engine
    obj_engine = ObjectEngine()

    # Init camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

    if not cap.isOpened():
        logger.error("Cannot open camera index %d", config.CAMERA_INDEX)
        sys.exit(1)

    # Shared state
    state = SharedState()
    zoom_mgr = ZoomManager()

    # Start recognition thread
    worker = threading.Thread(
        target=recognition_worker,
        args=(state, engine),
        daemon=True,
        name="RecognitionWorker"
    )
    worker.start()

    # Start object detection thread (YOLO)
    obj_worker_thread = threading.Thread(
        target=object_worker,
        args=(state, obj_engine),
        daemon=True,
        name="ObjectWorker"
    )
    obj_worker_thread.start()
    
    logger.info("Advanced background threads started (Face + YOLO).")

    # Window setup
    win = "JARVIS"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if config.FULLSCREEN:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    fullscreen = config.FULLSCREEN

    logger.info("Entering main loop. Press Q/ESC to quit.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed — retrying…")
            time.sleep(0.05)
            continue

        with state.lock:
            state.frame = raw_frame
            matches     = list(state.matches)
            objects     = list(state.objects)
            fps         = state.fps

        # Update zoom state
        zoom_mgr.update(matches)
        
        # Apply zoom if active
        display, display_matches = zoom_mgr.apply_zoom(raw_frame.copy(), matches)

        # Draw HUD on display frame
        draw_hud(display, display_matches, objects, fps)

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):             # Q / ESC → quit
            logger.info("Quit signal received.")
            break

        elif key == ord("f"):                 # F → toggle fullscreen
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, prop)

        elif key == ord("s"):                 # S → screenshot
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshot_dir, f"jarvis_{ts}.png")
            cv2.imwrite(path, display)
            logger.info("Screenshot saved: %s", path)

        elif key == ord("r"):                 # R → reload persons
            engine._known_ts = 0.0            # force cache refresh
            logger.info("Person cache invalidated — will reload on next frame.")

    # Shutdown
    state.running = False
    cap.release()
    cv2.destroyAllWindows()
    db.close()
    logger.info("JARVIS shutdown complete.")


if __name__ == "__main__":
    run()
