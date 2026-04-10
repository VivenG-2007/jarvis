from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_bgr(name: str, default: str) -> tuple[int, int, int]:
    raw = os.getenv(name, default)
    r, g, b = [int(part.strip()) for part in raw.split(",")]
    return (b, g, r)


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = ROOT_DIR / "logs"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"

for directory in [DATA_DIR, KNOWN_FACES_DIR, SAMPLE_DATA_DIR, MEMORY_DIR, LOGS_DIR, SCREENSHOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db_name: str = os.getenv("MONGO_DB_NAME", "jarvis_db")
    mongo_collection_persons: str = os.getenv("MONGO_COLLECTION_PERSONS", "persons")
    mongo_collection_logs: str = os.getenv("MONGO_COLLECTION_LOGS", "recognition_logs")
    mongo_collection_objects: str = os.getenv("MONGO_COLLECTION_OBJECTS", "objects")

    camera_index: int = _get_int("CAMERA_INDEX", 0)
    camera_width: int = _get_int("CAMERA_WIDTH", 960)
    camera_height: int = _get_int("CAMERA_HEIGHT", 540)
    camera_fps: int = _get_int("CAMERA_FPS", 30)
    mirror_camera: bool = _get_bool("MIRROR_CAMERA", True)
    enable_screen_input: bool = _get_bool("ENABLE_SCREEN_INPUT", False)
    screen_monitor_index: int = _get_int("SCREEN_MONITOR_INDEX", 1)
    fusion_mode: str = os.getenv("FUSION_MODE", "camera")

    face_detection_model: str = os.getenv("FACE_DETECTION_MODEL", "buffalo_s")
    face_recognition_threshold: float = _get_float("FACE_RECOGNITION_THRESHOLD", 0.40)
    face_cache_refresh_sec: float = _get_float("FACE_CACHE_REFRESH_SEC", 5.0)
    face_min_size: int = _get_int("FACE_MIN_SIZE", 60)
    face_frame_skip: int = _get_int("FACE_FRAME_SKIP", 2)
    face_process_max_width: int = _get_int("FACE_PROCESS_MAX_WIDTH", 512)

    object_model_path: str = os.getenv("OBJECT_MODEL_PATH", str(ROOT_DIR / "yolov8s.pt"))
    object_conf_threshold: float = _get_float("OBJECT_CONF_THRESHOLD", 0.35)
    object_db_conf_threshold: float = _get_float("OBJECT_DB_CONF_THRESHOLD", 0.50)
    object_imgsz: int = _get_int("OBJECT_IMGSZ", 320)
    object_frame_skip: int = _get_int("OBJECT_FRAME_SKIP", 3)
    vision_target_fps: float = _get_float("VISION_TARGET_FPS", 8.0)
    object_memory_refresh_sec: float = _get_float("OBJECT_MEMORY_REFRESH_SEC", 2.0)

    target_lock_duration: float = _get_float("TARGET_LOCK_DURATION", 10.0)
    target_lock_name: str = os.getenv("TARGET_LOCK_NAME", "")
    spotlight_dim_alpha: float = _get_float("SPOTLIGHT_DIM_ALPHA", 0.82)
    spotlight_padding: int = _get_int("SPOTLIGHT_PADDING", 85)
    zoom_strength: float = _get_float("ZOOM_STRENGTH", 0.18)
    zoom_smoothing: float = _get_float("ZOOM_SMOOTHING", 0.55)

    fullscreen: bool = _get_bool("FULLSCREEN", False)
    hud_intro_zoom_enabled: bool = _get_bool("HUD_INTRO_ZOOM_ENABLED", False)
    hud_spotlight_enabled: bool = _get_bool("HUD_SPOTLIGHT_ENABLED", True)
    hud_color_primary: tuple[int, int, int] = _get_bgr("HUD_COLOR_PRIMARY", "0,255,120")
    hud_color_accent: tuple[int, int, int] = _get_bgr("HUD_COLOR_ACCENT", "0,180,255")
    hud_color_alert: tuple[int, int, int] = _get_bgr("HUD_COLOR_ALERT", "255,80,80")

    audio_enabled: bool = _get_bool("AUDIO_ENABLED", True)
    microphone_device_index: int | None = _get_optional_int("MICROPHONE_DEVICE_INDEX")
    whisper_language: str = os.getenv("WHISPER_LANGUAGE", "en")
    whisper_realtime: bool = _get_bool("WHISPER_REALTIME", True)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    groq_api_base: str = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
    groq_stt_model: str = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    groq_stt_timeout_sec: float = _get_float("GROQ_STT_TIMEOUT_SEC", 20.0)
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
    groq_chat_timeout_sec: float = _get_float("GROQ_CHAT_TIMEOUT_SEC", 12.0)
    audio_min_rms: float = _get_float("AUDIO_MIN_RMS", 120.0)
    audio_energy_threshold: int = _get_int("AUDIO_ENERGY_THRESHOLD", 120)
    audio_pause_threshold: float = _get_float("AUDIO_PAUSE_THRESHOLD", 0.8)
    audio_non_speaking_duration: float = _get_float("AUDIO_NON_SPEAKING_DURATION", 0.4)
    audio_listen_timeout_sec: float = _get_float("AUDIO_LISTEN_TIMEOUT_SEC", 1.5)
    audio_phrase_time_limit_sec: float = _get_float("AUDIO_PHRASE_TIME_LIMIT_SEC", 8.0)
    wake_word: str = os.getenv("WAKE_WORD", "jarvis")
    command_timeout_sec: float = _get_float("COMMAND_TIMEOUT_SEC", 7.0)

    tts_enabled: bool = _get_bool("TTS_ENABLED", False)
    tts_voice: str = os.getenv("TTS_VOICE", "David")
    tts_rate: int = _get_int("TTS_RATE", 1)
    tts_volume: int = _get_int("TTS_VOLUME", 100)
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
    ollama_think: bool = _get_bool("OLLAMA_THINK", False)
    ollama_temperature: float = _get_float("OLLAMA_TEMPERATURE", 0.4)
    ollama_top_p: float = _get_float("OLLAMA_TOP_P", 0.95)
    ollama_top_k: int = _get_int("OLLAMA_TOP_K", 64)
    ollama_num_ctx: int = _get_int("OLLAMA_NUM_CTX", 8192)
    ollama_vision_enabled: bool = _get_bool("OLLAMA_VISION_ENABLED", True)
    ollama_vision_max_width: int = _get_int("OLLAMA_VISION_MAX_WIDTH", 960)
    ollama_vision_jpeg_quality: int = _get_int("OLLAMA_VISION_JPEG_QUALITY", 80)
    reasoning_timeout_sec: float = _get_float("REASONING_TIMEOUT_SEC", 20.0)
    name_match_timeout_sec: float = _get_float("NAME_MATCH_TIMEOUT_SEC", 2.0)

    memory_registry_path: Path = MEMORY_DIR / "persons.json"
    memory_events_path: Path = MEMORY_DIR / "events.jsonl"
    log_file: Path = LOGS_DIR / "jarvis.log"


APP = AppConfig()
