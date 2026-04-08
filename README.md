# JARVIS — Face Recognition & HUD Display System

> Real-time face recognition with a Jarvis-style HUD overlay, MongoDB-backed person database, and spotlight target-lock effect.

---

## Features

| Feature | Details |
|---|---|
| Face Detection | InsightFace `buffalo_s` (falls back to Haar cascade) |
| Face Recognition | 512-dim cosine similarity against MongoDB embeddings |
| HUD Overlay | Animated brackets, data cards, scan lines, status bars |
| Target Lock | Spotlight dim + zoom pulse when a known person is detected |
| Database | MongoDB — person registry + recognition event log |
| Multi-threaded | Camera capture and recognition run in parallel |

---

## Quick Start

### 1 · Prerequisites

- Python 3.10+
- MongoDB (local or Atlas)
- Webcam

### 2 · Clone / unzip and install dependencies

```bash
cd jarvis-face-id
pip install -r requirements.txt
```

> **GPU users:** replace `onnxruntime` with `onnxruntime-gpu` in requirements.txt

### 3 · Configure environment

Edit `.env` — at minimum set your MongoDB URI:

```
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=jarvis_db
```

For MongoDB Atlas:
```
MONGO_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/
```

### 4 · Seed sample data (optional)

Inserts four placeholder persons so you can test DB connectivity and HUD labels immediately:

```bash
python seed_sample_data.py
```

### 5 · Enroll real faces

```bash
# From an image file
python enroll.py --image path/to/photo.jpg \
                 --id EMP-001 --name "Alice Chen" \
                 --department Engineering --role "Senior Engineer"

# From webcam (press SPACE to capture)
python enroll.py --webcam \
                 --id EMP-002 --name "Bob Singh" \
                 --department Security

# List enrolled persons
python enroll.py --list

# Remove a person
python enroll.py --delete EMP-001
```

### 6 · Run JARVIS

```bash
python main.py
```

**Runtime controls:**

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `F` | Toggle fullscreen |
| `S` | Save screenshot |
| `R` | Reload persons from DB |

---

## Project Structure

```
jarvis-face-id/
├── .env                  ← environment config (edit this)
├── .env.example          ← reference template
├── config.py             ← typed config loader
├── main.py               ← application entry point
├── enroll.py             ← CLI enrollment tool
├── seed_sample_data.py   ← insert test records into MongoDB
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── db.py             ← MongoDB interface (PersonDB)
│   ├── face_engine.py    ← InsightFace detection + matching
│   └── hud.py            ← Jarvis HUD renderer
├── data/
│   ├── known_faces/      ← place enrollment photos here
│   └── sample/
├── logs/
└── screenshots/
```

---

## MongoDB Schema

### `persons` collection

```json
{
  "person_id":   "EMP-001",
  "name":        "Alice Chen",
  "department":  "Engineering",
  "role":        "Senior Engineer",
  "embedding":   [0.012, -0.341, ...],   // 512 floats
  "enrolled_at": "2024-01-01T10:00:00Z",
  "photo_path":  "data/known_faces/alice.jpg"
}
```

### `recognition_logs` collection

```json
{
  "person_id":  "EMP-001",
  "confidence": 94.2,
  "bbox":       [120, 80, 280, 310],
  "timestamp":  "2024-01-15T14:32:11Z"
}
```

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `jarvis_db` | Database name |
| `CAMERA_INDEX` | `0` | Webcam index |
| `FACE_RECOGNITION_THRESHOLD` | `0.45` | Match distance (lower = stricter) |
| `FACE_DETECTION_MODEL` | `buffalo_s` | InsightFace model name |
| `TARGET_LOCK_DURATION` | `10` | Seconds to hold spotlight lock |
| `SPOTLIGHT_DIM_ALPHA` | `0.75` | Background dim level (0–1) |
| `HUD_COLOR_PRIMARY` | `0,255,70` | Main HUD colour (R,G,B) |
| `FULLSCREEN` | `false` | Start in fullscreen mode |

---

## Troubleshooting

**InsightFace model download fails on first run**
The model is downloaded automatically to `~/.insightface/`. Ensure internet access on first run, then it works offline.

**MongoDB connection refused**
Check that `mongod` is running: `sudo systemctl start mongod`

**Low FPS**
- Reduce `CAMERA_WIDTH`/`CAMERA_HEIGHT` in `.env`
- Use `buffalo_sc` (small/fast) instead of `buffalo_s`
- Install `onnxruntime-gpu` if you have a CUDA GPU
"# jarvis" 
