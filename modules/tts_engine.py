from __future__ import annotations

import base64
import logging
import queue
import shutil
import subprocess
import threading
import time

import config

logger = logging.getLogger("jarvis.tts")

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class TTSEngine:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self._last_enqueued = ""
        self._last_spoken = ""
        self._last_spoken_at = 0.0
        self._backend = self._detect_backend() if self.enabled else "disabled"
        if self.enabled and self._backend == "disabled":
            self.enabled = False
            logger.warning("TTS is enabled in config, but no working backend is available.")
        elif self.enabled:
            logger.info("TTS backend selected: %s", self._backend)

    def start(self) -> None:
        if not self.enabled or self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._worker, daemon=True, name="TTSWorker")
        thread.start()

    def speak(self, text: str) -> None:
        if not self.enabled:
            return
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return
        now = time.time()
        if cleaned == self._last_enqueued:
            return
        if cleaned == self._last_spoken and now - self._last_spoken_at < 1.5:
            return
        self._last_enqueued = cleaned
        self.queue.put(cleaned)

    def stop(self) -> None:
        self.running = False

    def _detect_backend(self) -> str:
        if shutil.which("powershell.exe"):
            return "powershell_sapi"
        if PYTTSX3_AVAILABLE:
            return "pyttsx3"
        return "disabled"

    def _worker(self) -> None:
        engine = None
        if self._backend == "pyttsx3":
            engine = self._init_pyttsx3_engine()
            if engine is None and shutil.which("powershell.exe"):
                self._backend = "powershell_sapi"
                logger.warning("Falling back from pyttsx3 to PowerShell SAPI.")

        while self.running:
            try:
                text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                logger.info("Speaking response with TTS via %s.", self._backend)
                if self._backend == "powershell_sapi":
                    self._speak_with_powershell(text)
                elif self._backend == "pyttsx3" and engine is not None:
                    engine.say(text)
                    engine.runAndWait()
                else:
                    continue
                self._last_spoken = text
                self._last_spoken_at = time.time()
                self._last_enqueued = ""
            except Exception as exc:
                logger.exception("TTS playback failed: %s", exc)

    @staticmethod
    def _init_pyttsx3_engine():
        if not PYTTSX3_AVAILABLE:
            return None
        try:
            engine = pyttsx3.init()
            try:
                voices = engine.getProperty("voices")
                requested = config.APP.tts_voice.strip().lower()
                if requested:
                    for voice in voices:
                        label = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
                        if requested in label:
                            engine.setProperty("voice", voice.id)
                            break
            except Exception:
                pass
            engine.setProperty("rate", max(-10, min(10, config.APP.tts_rate)))
            try:
                engine.setProperty("volume", max(0.0, min(1.0, config.APP.tts_volume / 100.0)))
            except Exception:
                pass
            return engine
        except Exception as exc:
            logger.exception("pyttsx3 initialization failed: %s", exc)
            return None

    @staticmethod
    def _speak_with_powershell(text: str) -> None:
        preferred_voice = config.APP.tts_voice.strip()
        safe_text = text.replace("'", "''")
        safe_voice = preferred_voice.replace("'", "''")
        script = (
            "$voice = New-Object -ComObject SAPI.SpVoice; "
            f"$voice.Rate = {max(-10, min(10, config.APP.tts_rate))}; "
            f"$voice.Volume = {max(0, min(100, config.APP.tts_volume))}; "
            "$voices = $voice.GetVoices(); "
            f"$preferred = '{safe_voice}'; "
            "if ($preferred) { "
            "foreach ($v in $voices) { "
            "$desc = $v.GetDescription(); "
            "if ($desc -like ('*' + $preferred + '*')) { $voice.Voice = $v; break } "
            "} "
            "} "
            f"$null = $voice.Speak('{safe_text}')"
        )
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
