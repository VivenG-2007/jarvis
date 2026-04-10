from __future__ import annotations

import logging
import queue
import re
import threading
import time

import config
import requests

logger = logging.getLogger("jarvis.audio")

try:
    import speech_recognition as sr

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


_HALLUCINATION_PATTERNS = re.compile(
    r"^\s*("
    r"thank you[.\s]*|"
    r"thanks[.\s]*|"
    r"you[.\s]*|"
    r"\.+|"
    r"\s+"
    r")\s*$",
    re.IGNORECASE,
)

_ARTIFACT_PATTERNS = [
    re.compile(r"\[.*?\]", re.IGNORECASE),
    re.compile(r"\(.*?\)", re.IGNORECASE),
]

_JARVIS_PROMPT = (
    "Jarvis open close scan face who is this identify person "
    "stop shutdown search show camera hello lock unlock status "
    "take photo record start stop quit exit restart"
)


class WhisperEngine:
    def __init__(self, model_size: str | None = None):
        self.running = False
        self.transcription_queue: queue.Queue[str] = queue.Queue()
        self.recognizer = None
        self._speech_until = 0.0
        self._last_audio_level = 0.0
        self._api_url = f"{config.APP.groq_api_base}/audio/transcriptions"
        self._model_name = model_size or config.APP.groq_stt_model
        self._session = requests.Session()

        if not AUDIO_AVAILABLE or not config.APP.audio_enabled:
            logger.warning("Audio pipeline is disabled because dependencies are missing or AUDIO_ENABLED=false.")
            return
        if not config.APP.groq_api_key:
            logger.warning("Audio pipeline is disabled because GROQ_API_KEY is not set.")
            return

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = config.APP.audio_energy_threshold
        self.recognizer.pause_threshold = config.APP.audio_pause_threshold
        self.recognizer.non_speaking_duration = config.APP.audio_non_speaking_duration

        logger.info(
            "Audio pipeline ready. provider=groq model=%s language=%s mic_index=%s",
            self._model_name,
            config.APP.whisper_language,
            config.APP.microphone_device_index,
        )

    def start_background_listening(self) -> None:
        if self.recognizer is None or self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._listening_worker, daemon=True, name="AudioWorker")
        thread.start()

    def stop(self) -> None:
        self.running = False
        self._session.close()

    def get_latest_command(self) -> str:
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return ""

    @property
    def voice_active(self) -> bool:
        return time.time() < self._speech_until

    @property
    def audio_level(self) -> float:
        return self._last_audio_level

    def _listening_worker(self) -> None:
        assert self.recognizer is not None
        microphone = self._open_microphone()
        if microphone is None:
            self._speech_until = 0.0
            self._last_audio_level = 0.0
            logger.error("Microphone startup failed: no usable input device could be opened.")
            return
        source, cleanup = microphone
        try:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                logger.info("Microphone calibrated. energy_threshold=%s", self.recognizer.energy_threshold)
                while self.running:
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=config.APP.audio_listen_timeout_sec,
                            phrase_time_limit=config.APP.audio_phrase_time_limit_sec,
                        )
                        wav_bytes = audio.get_wav_data()
                        rms = self._calculate_rms(wav_bytes)
                        self._last_audio_level = min(1.0, rms / max(max(config.APP.audio_min_rms, 1.0) * 2.0, 1.0))
                        logger.info("Captured audio clip rms=%s", rms)

                        if config.APP.audio_min_rms > 0 and rms < config.APP.audio_min_rms:
                            logger.info(
                                "Skipping low-volume audio clip below rms threshold %.1f",
                                config.APP.audio_min_rms,
                            )
                            self._last_audio_level *= 0.35
                            continue

                        self._speech_until = time.time() + 0.9
                        text = self._transcribe_audio(wav_bytes)
                        if text:
                            logger.info("Heard: %s", text)
                            self.transcription_queue.put(text)
                        else:
                            logger.info("Transcription was empty for the latest audio clip.")
                    except sr.WaitTimeoutError:
                        self._speech_until = 0.0
                        self._last_audio_level *= 0.75
                    except Exception as exc:
                        self._speech_until = 0.0
                        self._last_audio_level = 0.0
                        logger.warning("Audio worker issue: %s", exc)
            finally:
                cleanup()
        except Exception as exc:
            self._speech_until = 0.0
            self._last_audio_level = 0.0
            logger.exception("Microphone startup failed: %s", exc)

    def _transcribe_audio(self, wav_bytes: bytes) -> str:
        headers = {"Authorization": f"Bearer {config.APP.groq_api_key}"}
        data = {
            "model": self._model_name,
            "language": config.APP.whisper_language,
            "prompt": _JARVIS_PROMPT,
            "temperature": "0",
            "response_format": "verbose_json",
        }
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}

        response = self._session.post(
            self._api_url,
            headers=headers,
            data=data,
            files=files,
            timeout=config.APP.groq_stt_timeout_sec,
        )
        response.raise_for_status()
        body = response.json()
        raw = str(body.get("text", "")).strip()
        return self._clean_transcription(raw)

    @staticmethod
    def _clean_transcription(text: str) -> str:
        for pattern in _ARTIFACT_PATTERNS:
            text = re.sub(pattern, "", text)
        text = text.strip()
        if _HALLUCINATION_PATTERNS.match(text):
            return ""
        return text

    @staticmethod
    def _calculate_rms(wav_bytes: bytes) -> int:
        pcm = memoryview(wav_bytes)[44:]
        if not pcm:
            return 0
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return 0
        return int(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))

    @staticmethod
    def _get_microphone_name(device_index: int | None) -> str:
        if not AUDIO_AVAILABLE:
            return "audio unavailable"
        try:
            names = sr.Microphone.list_microphone_names()
            if device_index is None:
                return "default microphone"
            if 0 <= device_index < len(names):
                return f"{device_index}: {names[device_index]}"
            return f"{device_index}: unavailable device index"
        except Exception:
            return "unknown microphone"

    def _open_microphone(self):
        assert self.recognizer is not None
        candidates = [config.APP.microphone_device_index]
        if config.APP.microphone_device_index is not None:
            candidates.append(None)
        for device_index in candidates:
            source_name = self._get_microphone_name(device_index)
            logger.info("Opening microphone: %s", source_name)
            microphone = sr.Microphone(device_index=device_index)
            try:
                source = microphone.__enter__()
                if getattr(source, "stream", None) is None:
                    raise RuntimeError("microphone stream was not opened")
                return source, lambda mic=microphone: self._close_microphone(mic)
            except Exception as exc:
                logger.warning("Unable to open microphone %s: %s", source_name, exc)
                self._close_microphone(microphone)
        return None

    @staticmethod
    def _close_microphone(microphone) -> None:
        try:
            if getattr(microphone, "stream", None) is not None:
                microphone.__exit__(None, None, None)
        except Exception:
            return
