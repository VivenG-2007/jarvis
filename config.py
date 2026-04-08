"""
config.py — Central configuration loader.
Reads all settings from .env and exposes them as typed constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def _rgb(env_key: str, default: str) -> tuple[int, int, int]:
    raw = os.getenv(env_key, default)
    r, g, b = [int(x.strip()) for x in raw.split(",")]
    return (b, g, r)   # OpenCV uses BGR internally

# ── MongoDB ──────────────────────────────────────────────────
MONGO_URI             = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME         = os.getenv("MONGO_DB_NAME", "jarvis_db")
MONGO_COLLECTION_PERSONS = os.getenv("MONGO_COLLECTION_PERSONS", "persons")
MONGO_COLLECTION_LOGS = os.getenv("MONGO_COLLECTION_LOGS", "recognition_logs")

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", 0))
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH", 1280))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", 720))
CAMERA_FPS    = int(os.getenv("CAMERA_FPS", 30))

# ── Face Recognition ─────────────────────────────────────────
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.45))
FACE_DETECTION_MODEL       = os.getenv("FACE_DETECTION_MODEL", "buffalo_s")
FACE_EMBED_SIZE            = int(os.getenv("FACE_EMBED_SIZE", 512))

# ── Target Lock ──────────────────────────────────────────────
TARGET_LOCK_DURATION  = int(os.getenv("TARGET_LOCK_DURATION", 10))
ZOOM_SMOOTHING        = float(os.getenv("ZOOM_SMOOTHING", 0.15))

# ── UI / HUD ─────────────────────────────────────────────────
HUD_COLOR_PRIMARY = _rgb("HUD_COLOR_PRIMARY", "0,255,70")      # BGR
HUD_COLOR_ACCENT  = _rgb("HUD_COLOR_ACCENT",  "0,180,255")
HUD_COLOR_ALERT   = _rgb("HUD_COLOR_ALERT",   "255,60,60")
HUD_FONT_SCALE    = float(os.getenv("HUD_FONT_SCALE", 0.55))
FULLSCREEN        = os.getenv("FULLSCREEN", "false").lower() == "true"

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
LOG_FILE    = os.getenv("LOG_FILE", "logs/jarvis.log")
