"""
modules/audio_engine.py — Phase 3 Vocal Command Engine

Implementation of ultra-low latency local Voice-to-Text inference using Faster-Whisper
and background Voice Activity Detection (VAD).

Requires: 
pip install SpeechRecognition pyaudio faster-whisper
"""

import logging
import threading
import queue

try:
    import speech_recognition as sr
    from faster_whisper import WhisperModel
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


logger = logging.getLogger("jarvis.audio")


class WhisperEngine:
    """
    Continuous background listening engine optimized for Voice-to-Text inference 
    without blocking the primary vision pipeline.
    """
    def __init__(self, model_size: str = "tiny.en"):
        """
        model_size options: 'tiny.en', 'base.en' (Optimize for English speed)
        'tiny.en' requires less than 1GB of VRAM/RAM and inferces almost instantly.
        """
        self.running = False
        self.model = None
        self.transcription_queue = queue.Queue()
        self.voice_active = False # New state for HUD animation

        if AUDIO_AVAILABLE:
            self.recognizer = sr.Recognizer()
            # Optimize recognizer for ambient noise and fast cutoff
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 300  # More sensitive base
            self.recognizer.pause_threshold = 0.5   # Cut faster for quicker response
            self.recognizer.non_speaking_duration = 0.4
            
            logger.info(f"Loading Faster-Whisper '{model_size}' into memory...")
            # Enforce int8 computation to guarantee extremely fast CPU processing
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Faster-Whisper model active.")
        else:
            logger.warning("Required audio libraries missing. Install SpeechRecognition, Whisper, and PyAudio.")

    def listen_and_transcribe(self) -> str:
        """
        Synchronous, one-off command listening. Listens until speech stops, 
        then transcribes and returns the user query string.
        """
        if not self.model:
            return ""

        with sr.Microphone() as source:
            logger.info("Calibrating ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logger.info("Listening for command...")
            
            try:
                # Capture mic input
                audio_data = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # We save it temporarily as a WAV byte stream to pass squarely into the Whisper tensor
                with open("temp_buffer.wav", "wb") as f:
                    f.write(audio_data.get_wav_data())
                    
                # Run the inference via Faster-Whisper
                segments, info = self.model.transcribe("temp_buffer.wav", beam_size=5)
                
                # Faster-Whisper returns a generator, so we join the text segments
                text = " ".join([segment.text for segment in segments]).strip()
                return text

            except sr.WaitTimeoutError:
                return ""
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                return ""

    # ── Non-Blocking Integration Example ────────────────────────────
    
    def start_background_listening(self):
        """Spins up a daemon thread so voice doesn't block facial recognition."""
        if not AUDIO_AVAILABLE:
            return
            
        self.running = True
        t = threading.Thread(target=self._listening_worker, daemon=True, name="AudioWorker")
        t.start()

    def _listening_worker(self):
        """Continuously loads text commands into the thread-safe queue."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.running:
                try:
                    self.voice_active = True
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    self.voice_active = False 
                    
                    with open("temp_buffer.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    
                    segments, info = self.model.transcribe("temp_buffer.wav", beam_size=1) 
                    text = " ".join([segment.text for segment in segments]).strip()
                    if text:
                        self.transcription_queue.put(text)
                except sr.WaitTimeoutError:
                    self.voice_active = False
                except Exception:
                    self.voice_active = False

    def get_latest_command(self) -> str:
        """Pulls the most recent command out of the buffer, if available."""
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return ""

# ── Example Usage / Main Loop Integration ────────────────────────────────
if __name__ == "__main__":
    audio = WhisperEngine("tiny.en")

    print("\n--- SYNCHRONOUS TEST ---")
    command = audio.listen_and_transcribe()
    print(f"You said: {command}")

    print("\n--- ASYNCHRONOUS BACKGROUND TEST ---")
    audio.start_background_listening()
    
    try:
        # This simulates the existing JARVIS face/object frame loop
        print("Main Camera Loop Running. Say something!")
        while True:
            # Check for voice commands instantly without blocking the frame drawing
            background_cmd = audio.get_latest_command()
            if background_cmd:
                print(f"JARVIS OVERHEARD: {background_cmd}")
            
            # The 'WaitKey' time equivalent for cv2 loops
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Disconnecting Microphones...")
        audio.running = False
