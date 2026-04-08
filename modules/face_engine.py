"""
modules/face_engine.py — InsightFace-powered face detection and recognition.

Provides:
  - FaceEngine   : detect faces, extract embeddings, match against known DB
  - FaceMatch    : dataclass returned per detected face
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config
from modules.db import PersonDB

logger = logging.getLogger("jarvis.face_engine")

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not installed — using OpenCV fallback detector.")


# ── Data types ────────────────────────────────────────────────

@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int

    @property
    def width(self):  return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def center(self): return ((self.x1+self.x2)//2, (self.y1+self.y2)//2)

    def as_list(self): return [self.x1, self.y1, self.x2, self.y2]
    def scale(self, sx, sy):
        return BBox(int(self.x1*sx), int(self.y1*sy),
                    int(self.x2*sx), int(self.y2*sy))


@dataclass
class FaceMatch:
    bbox:        BBox
    embedding:   Optional[np.ndarray] = None
    # Recognition result
    person_id:   Optional[str]   = None
    name:        Optional[str]   = None
    department:  Optional[str]   = None
    role:        Optional[str]   = None
    confidence:  float           = 0.0
    is_known:    bool            = False
    # Tracking state
    track_id:    int             = -1
    last_seen:   float           = field(default_factory=time.time)


# ── Main engine ───────────────────────────────────────────────

class FaceEngine:
    """
    Detects faces in each frame, extracts 512-dim embeddings,
    and matches against the persons stored in MongoDB.
    """

    def __init__(self, db: PersonDB):
        self.db = db
        self._known: list[dict] = []          # cached person records
        self._known_ts: float   = 0.0         # last cache refresh
        self._next_track_id     = 0
        self._tracker: dict[int, FaceMatch] = {}

        self._init_model()
        self._refresh_known()

    # ── Model initialisation ──────────────────────────────────

    def _init_model(self):
        if INSIGHTFACE_AVAILABLE:
            logger.info("Loading InsightFace model '%s'…", config.FACE_DETECTION_MODEL)
            
            # Advanced ONNX Optimization & Noise Suppression
            import onnxruntime as ort
            ort.set_default_logger_severity(3) # Suppress C++ backend spam for missing CUDA dlls
            
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Try to use CUDA if available, fallback to CPU
            # We explicitly check for cublas errors mentioned in logs by trying to load
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            
            self.app = FaceAnalysis(
                name=config.FACE_DETECTION_MODEL,
                providers=providers,
                provider_options=[{}, {}] # Can be tuned further
            )
            # ctx_id=0 uses GPU; -1 uses CPU
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            
            # Verify which provider was actually used
            actual_providers = []
            for model in self.app.models.values():
                if hasattr(model, 'session'):
                    p = model.session.get_providers()
                    actual_providers.extend(p)
            
            p_set = set(actual_providers)
            logger.info("InsightFace initialized. Active Providers: %s ✓", p_set)
        else:
            # Fallback: Haar cascade (no embeddings, no recognition)
            logger.warning("Using Haar cascade fallback — no recognition available.")
            self.app = None
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)

    # ── Known-persons cache ───────────────────────────────────

    def _refresh_known(self, interval: float = 30.0):
        """Reload person embeddings from DB every `interval` seconds."""
        now = time.time()
        if now - self._known_ts > interval:
            self._known = self.db.get_all_persons()
            self._known_ts = now
            logger.debug("Known persons refreshed: %d entries", len(self._known))

    # ── Core inference ────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[FaceMatch]:
        """
        Run detection + recognition on one BGR frame.
        Returns list of FaceMatch objects (one per detected face).
        """
        self._refresh_known()
        h, w = frame.shape[:2]

        if self.app is not None:
            return self._process_insightface(frame, w, h)
        else:
            return self._process_haar(frame, w, h)

    # ── InsightFace path ──────────────────────────────────────

    def _process_insightface(self, frame: np.ndarray, w: int, h: int) -> list[FaceMatch]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        matches = []
        for face in faces:
            bbox = self._to_bbox(face.bbox, w, h)
            emb  = face.embedding.astype(np.float32)
            match = FaceMatch(bbox=bbox, embedding=emb, last_seen=time.time())
            self._match_identity(match)
            self._assign_track(match)
            matches.append(match)
        return matches

    # ── Haar cascade fallback ─────────────────────────────────

    def _process_haar(self, frame: np.ndarray, w: int, h: int) -> list[FaceMatch]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        matches = []
        for (x, y, fw, fh) in rects:
            bbox  = BBox(x, y, x+fw, y+fh)
            match = FaceMatch(bbox=bbox, last_seen=time.time())
            self._assign_track(match)
            matches.append(match)
        return matches

    # ── Identity matching ─────────────────────────────────────

    def _match_identity(self, match: FaceMatch):
        """Cosine-similarity search against known embeddings."""
        if not self._known or match.embedding is None:
            return

        q = match.embedding / (np.linalg.norm(match.embedding) + 1e-6)
        best_score = -1.0
        best_person = None

        for person in self._known:
            ref = person["embedding"]
            ref = ref / (np.linalg.norm(ref) + 1e-6)
            score = float(np.dot(q, ref))
            if score > best_score:
                best_score = score
                best_person = person

        # Similarity → distance (1 - sim) and compare to threshold
        distance = 1.0 - best_score
        if best_person and distance < config.FACE_RECOGNITION_THRESHOLD:
            match.person_id  = best_person["person_id"]
            match.name       = best_person["name"]
            match.department = best_person["department"]
            match.role       = best_person.get("role", "")
            match.confidence = round(best_score * 100, 1)
            match.is_known   = True
            self.db.log_recognition(
                match.person_id, match.confidence, match.bbox.as_list()
            )

    # ── Tracking ──────────────────────────────────────────────

    def _assign_track(self, match: FaceMatch):
        """Assign a persistent track_id based on IoU with previous frame."""
        best_id, best_iou = -1, 0.35   # min IoU to link
        for tid, prev in self._tracker.items():
            iou = self._iou(match.bbox, prev.bbox)
            if iou > best_iou:
                best_iou = iou
                best_id  = tid

        if best_id == -1:
            best_id = self._next_track_id
            self._next_track_id += 1

        match.track_id = best_id
        self._tracker[best_id] = match

        # Prune stale tracks (>2s old)
        now = time.time()
        self._tracker = {
            k: v for k, v in self._tracker.items()
            if now - v.last_seen < 2.0
        }

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        union = a.width*a.height + b.width*b.height - inter
        return inter / union

    @staticmethod
    def _to_bbox(raw, w, h) -> BBox:
        x1, y1, x2, y2 = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
        return BBox(
            max(0, x1), max(0, y1),
            min(w, x2), min(h, y2)
        )
