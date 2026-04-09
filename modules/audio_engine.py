from __future__ import annotations

import logging
import queue
import threading
import time

import config
import numpy as np

logger = logging.getLogger("jarvis.audio")

try:
    import speech_recognition as sr
    from faster_whisper import WhisperModel

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


class WhisperEngine:
    def __init__(self, model_size: str | None = None):
        self.running = False
        self.transcription_queue: queue.Queue[str] = queue.Queue()
        self.model = None
        self.recognizer = None
        self._speech_until = 0.0
        self._last_audio_level = 0.0

        if AUDIO_AVAILABLE and config.APP.audio_enabled:
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = config.APP.audio_energy_threshold
            self.recognizer.pause_threshold = config.APP.audio_pause_threshold
            self.recognizer.non_speaking_duration = config.APP.audio_non_speaking_duration
            self.model = WhisperModel(model_size or config.APP.whisper_model_size, device="cpu", compute_type="int8")
            logger.info(
                "Audio pipeline ready. model=%s language=%s beam=%s vad=%s retry_without_vad=%s mic_index=%s",
                model_size or config.APP.whisper_model_size,
                config.APP.whisper_language,
                config.APP.whisper_beam_size,
                config.APP.whisper_use_vad,
                config.APP.whisper_retry_without_vad,
                config.APP.microphone_device_index,
            )
        else:
            logger.warning("Audio pipeline is disabled because dependencies are missing or AUDIO_ENABLED=false.")

    def start_background_listening(self) -> None:
        if self.model is None or self.recognizer is None or self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._listening_worker, daemon=True, name="AudioWorker")
        thread.start()

    def stop(self) -> None:
        self.running = False

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
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
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
                        self._last_audio_level = min(1.0, rms / max(config.APP.audio_min_rms * 2.0, 1.0))
                        logger.info("Captured audio clip rms=%s", rms)
                        if rms < config.APP.audio_min_rms:
                            logger.info("Skipping low-volume audio clip below rms threshold %.1f", config.APP.audio_min_rms)
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
        assert self.model is not None
        audio_array = self._load_audio_array(wav_bytes)
        text = self._run_transcription(audio_array, vad_filter=config.APP.whisper_use_vad)
        if text or not config.APP.whisper_retry_without_vad or not config.APP.whisper_use_vad:
            return text
        logger.info("VAD removed the clip; retrying transcription without VAD.")
        return self._run_transcription(audio_array, vad_filter=False)

    def _run_transcription(self, audio_array: np.ndarray, vad_filter: bool) -> str:
        assert self.model is not None
        language = config.APP.whisper_language.strip() or None
        segments, _ = self.model.transcribe(
            audio_array,
            beam_size=config.APP.whisper_beam_size,
            best_of=config.APP.whisper_best_of,
            language=language,
            vad_filter=vad_filter,
            condition_on_previous_text=not config.APP.whisper_realtime,
            without_timestamps=config.APP.whisper_realtime,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _load_audio_array(self, wav_bytes: bytes) -> np.ndarray:
        pcm = np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32)
        if pcm.size == 0:
            return np.zeros(0, dtype=np.float32)
        return pcm / 32768.0

    @staticmethod
    def _calculate_rms(wav_bytes: bytes) -> int:
        pcm = np.frombuffer(wav_bytes[44:], dtype=np.int16)
        if pcm.size == 0:
            return 0
        return int(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))

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
