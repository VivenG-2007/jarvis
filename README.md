# JARVIS Local Assistant

Local-first Jarvis-style assistant in Python with:

- live webcam input
- optional screen capture fusion
- InsightFace recognition
- YOLO object detection
- Faster-Whisper speech-to-text
- Ollama + Gemma 4 grounded reasoning
- HUD overlays, target lock, and crowd focus mode

## Project structure

```text
.
|-- config.py
|-- main.py
|-- enroll.py
|-- requirements.txt
|-- data/
|   |-- known_faces/
|   |-- memory/
|   |   |-- persons.json
|   |   `-- events.jsonl
|   `-- sample/
|-- logs/
|-- screenshots/
`-- modules/
    |-- audio_engine.py
    |-- context_builder.py
    |-- db.py
    |-- face_engine.py
    |-- hud_overlay.py
    |-- object_engine.py
    |-- reasoning_engine.py
    |-- rules_engine.py
    |-- screen_capture.py
    `-- tts_engine.py
```

## What it does

- detects and tracks multiple people in real time
- recognizes enrolled faces and pulls local profile data
- detects objects and keeps object memory with last-seen timestamps
- listens for the wake word `Jarvis`
- answers grounded questions from live context plus local DB memory
- can isolate one named person from a crowd with focus mode
- sends a live scene snapshot to Gemma 4 for stronger local scene understanding

## Recommended local brain

This repo is now configured for Ollama with `gemma4:e4b` by default.

- default local model: `gemma4:e4b`
- larger workstation option: `gemma4:26b`
- highest local quality if your machine can handle it: `gemma4:31b`

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install Python dependencies.

```powershell
pip install -r requirements.txt
```

3. Install and start Ollama, then pull Gemma 4.

```powershell
ollama pull gemma4:e4b
ollama serve
```

4. Copy env settings if needed.

```powershell
Copy-Item .env.example .env
```

The repo already includes Gemma 4 defaults in `config.py` and `.env.example`.

5. Enroll known people.

```powershell
python enroll.py --image path\to\person.jpg --id EMP-101 --name "Ava Rao" --department Research --role "Engineer"
```

You can also enroll from the webcam:

```powershell
python enroll.py --webcam --id EMP-102 --name "Viven" --department Engineering --role "Developer"
```

6. Run the assistant.

```powershell
python main.py
```

## Runtime usage

- `Jarvis who is that`
- `Jarvis what is happening`
- `Jarvis what should I do`
- `Jarvis spot Viven`
- `Jarvis focus on Ava`
- `Jarvis clear focus`
- `Jarvis how many tv are there`
- `Jarvis list objects in the database`

Keyboard shortcuts:

- `Q` or `Esc`: quit
- `F`: toggle fullscreen
- `S`: save screenshot

## Enrollment commands

Enroll from an image:

```powershell
python enroll.py --image path\to\person.jpg --id EMP-101 --name "Ava Rao" --department Research --role "Engineer" --notes "Main profile"
```

Enroll from the webcam:

```powershell
python enroll.py --webcam --id EMP-102 --name "Viven" --department Engineering --role "Developer"
```

List enrolled people:

```powershell
python enroll.py --list
```

Delete an enrolled person:

```powershell
python enroll.py --delete EMP-102
```

## Gemma 4 notes

- reasoning uses Ollama chat API locally
- visual reasoning can include the latest camera or fused frame snapshot
- person identity and DB history come from structured JSON context
- image input is used for scene layout, motion, and visual grounding
- `OLLAMA_THINK=false` is the default for lower latency and cleaner HUD output
- speech recognition now defaults to `base.en` with VAD enabled for better voice capture accuracy

## Important env settings

```env
OLLAMA_MODEL=gemma4:e4b
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_KEEP_ALIVE=15m
OLLAMA_THINK=false
OLLAMA_TEMPERATURE=0.4
OLLAMA_TOP_P=0.95
OLLAMA_TOP_K=64
OLLAMA_NUM_CTX=8192
OLLAMA_VISION_ENABLED=true
OLLAMA_VISION_MAX_WIDTH=960
OLLAMA_VISION_JPEG_QUALITY=80
REASONING_TIMEOUT_SEC=20
WHISPER_MODEL_SIZE=base.en
WHISPER_LANGUAGE=en
WHISPER_BEAM_SIZE=4
WHISPER_USE_VAD=true
WHISPER_RETRY_WITHOUT_VAD=true
AUDIO_MIN_RMS=140
```

## Performance tips

- use `gemma4:e4b` for lower latency on consumer hardware
- switch `OBJECT_FRAME_SKIP` above `1` if CPU usage is high
- keep `OLLAMA_VISION_MAX_WIDTH` at `960` or lower if responses are slow
- use `WHISPER_MODEL_SIZE=base.en` for better recognition accuracy
- drop to `tiny.en` only if you need the absolute lowest latency
- disable screen capture unless you need it

## Notes

- object memory is stored by label with count and last seen time
- face memory is stored locally and can also sync to MongoDB
- if Ollama is unavailable, the assistant falls back to rules-based reasoning
