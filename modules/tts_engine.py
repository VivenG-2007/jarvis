from __future__ import annotations

import logging
import queue
import threading

logger = logging.getLogger("jarvis.tts")

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class TTSEngine:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled and TTS_AVAILABLE
        self.queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.engine = pyttsx3.init() if self.enabled else None
        if self.engine:
            self.engine.setProperty("rate", 185)

    def start(self) -> None:
        if not self.enabled or self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._worker, daemon=True, name="TTSWorker")
        thread.start()

    def speak(self, text: str) -> None:
        if self.enabled and text.strip():
            self.queue.put(text.strip())

    def stop(self) -> None:
        self.running = False

    def _worker(self) -> None:
        assert self.engine is not None
        while self.running:
            try:
                text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.engine.say(text)
            self.engine.runAndWait()
