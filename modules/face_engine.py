from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config
from modules.db import PersonDB

logger = logging.getLogger("jarvis.face")

try:
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class FaceMatch:
    bbox: BBox
    embedding: Optional[np.ndarray] = None
    track_id: int = -1
    last_seen: float = field(default_factory=time.time)
    person_id: Optional[str] = None
    name: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    confidence: float = 0.0
    is_known: bool = False
    lock_candidate: bool = False


class FaceEngine:
    def __init__(self, db: PersonDB):
        self.db = db
        self._known: list[dict] = []
        self._known_ts = 0.0
        self._next_track_id = 1
        self._tracks: dict[int, FaceMatch] = {}
        self._frame_index = 0
        self._cached_results: list[FaceMatch] = []
        self._init_model()
        self._refresh_known(force=True)

    def _init_model(self) -> None:
        if INSIGHTFACE_AVAILABLE:
            providers = ["CPUExecutionProvider"]
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                providers = ["CPUExecutionProvider"]
            self.app = FaceAnalysis(name=config.APP.face_detection_model, providers=providers)
            ctx = 0 if providers[0] == "CUDAExecutionProvider" else -1
            self.app.prepare(ctx_id=ctx, det_size=(640, 640))
        else:
            self.app = None
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            logger.warning("InsightFace unavailable; running with OpenCV fallback and no embeddings.")

    def _refresh_known(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self._known_ts > config.APP.face_cache_refresh_sec:
            self._known = self.db.get_all_persons()
            self._known_ts = now

    def process_frame(self, frame: np.ndarray, focus_person_id: str | None = None) -> list[FaceMatch]:
        self._refresh_known()
        self._frame_index += 1
        skip = max(1, config.APP.face_frame_skip)
        if self._frame_index % skip != 0 and self._cached_results:
            cached = [self._clone_match(match) for match in self._cached_results]
            self._mark_focus_candidates(cached, focus_person_id=focus_person_id)
            return cached
        if self.app is not None:
            results = self._process_insightface(frame, focus_person_id=focus_person_id)
        else:
            results = self._process_fallback(frame, focus_person_id=focus_person_id)
        self._cached_results = [self._clone_match(match) for match in results]
        return results

    def _process_insightface(self, frame: np.ndarray, focus_person_id: str | None = None) -> list[FaceMatch]:
        detection_frame, scale_x, scale_y = self._prepare_detection_frame(frame)
        rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        detections = self.app.get(rgb)
        height, width = frame.shape[:2]
        results: list[FaceMatch] = []
        for detected_face in detections:
            bbox = self._to_bbox(detected_face.bbox, width, height, scale_x=scale_x, scale_y=scale_y)
            match = FaceMatch(bbox=bbox, embedding=detected_face.embedding.astype(np.float32))
            self._assign_identity(match)
            self._assign_track(match)
            results.append(match)
        self._mark_focus_candidates(results, focus_person_id=focus_person_id)
        return results

    def _process_fallback(self, frame: np.ndarray, focus_person_id: str | None = None) -> list[FaceMatch]:
        detection_frame, scale_x, scale_y = self._prepare_detection_frame(frame)
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
        min_size = max(16, int(config.APP.face_min_size / max(scale_x, scale_y)))
        rects = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
        results: list[FaceMatch] = []
        for x, y, w, h in rects:
            match = FaceMatch(
                bbox=BBox(
                    int(x * scale_x),
                    int(y * scale_y),
                    int((x + w) * scale_x),
                    int((y + h) * scale_y),
                )
            )
            self._assign_track(match)
            results.append(match)
        self._mark_focus_candidates(results, focus_person_id=focus_person_id)
        return results

    def _assign_identity(self, match: FaceMatch) -> None:
        if match.embedding is None or not self._known:
            return
        query = match.embedding / (np.linalg.norm(match.embedding) + 1e-6)
        best_score = -1.0
        best_person = None
        for person in self._known:
            reference = person["embedding"]
            if reference.ndim != 1 or reference.shape[0] != query.shape[0]:
                continue
            if float(np.linalg.norm(reference)) == 0.0:
                continue
            reference = reference / (np.linalg.norm(reference) + 1e-6)
            score = float(np.dot(query, reference))
            if score > best_score:
                best_score = score
                best_person = person
        distance = 1.0 - best_score
        if best_person and distance <= config.APP.face_recognition_threshold:
            match.person_id = best_person["person_id"]
            match.name = best_person["name"]
            match.department = best_person.get("department", "")
            match.role = best_person.get("role", "")
            match.confidence = round(best_score * 100, 1)
            match.is_known = True
            self.db.log_recognition(match.person_id, match.confidence, match.bbox.as_list())

    def _assign_track(self, match: FaceMatch) -> None:
        best_track = -1
        best_iou = 0.25
        for track_id, previous in self._tracks.items():
            iou = self._iou(match.bbox, previous.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track_id
        if best_track == -1:
            best_track = self._next_track_id
            self._next_track_id += 1
        if best_track in self._tracks and self._tracks[best_track].is_known and not match.is_known:
            previous = self._tracks[best_track]
            match.person_id = previous.person_id
            match.name = previous.name
            match.department = previous.department
            match.role = previous.role
            match.confidence = max(match.confidence, previous.confidence * 0.96)
            match.is_known = previous.is_known
        match.track_id = best_track
        match.last_seen = time.time()
        self._tracks[best_track] = match
        self._tracks = {track_id: tracked for track_id, tracked in self._tracks.items() if time.time() - tracked.last_seen < 2.0}

    def _mark_focus_candidates(self, matches: list[FaceMatch], focus_person_id: str | None = None) -> None:
        requested_person_id = (focus_person_id or "").strip()
        target_name = config.APP.target_lock_name.strip().lower()
        for match in matches:
            if requested_person_id:
                match.lock_candidate = match.is_known and (match.person_id or "").strip() == requested_person_id
            elif target_name:
                match.lock_candidate = match.is_known and (match.name or "").strip().lower() == target_name
            else:
                match.lock_candidate = match.is_known

    def _prepare_detection_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = frame.shape[:2]
        max_width = max(160, config.APP.face_process_max_width)
        if width <= max_width:
            return frame, 1.0, 1.0
        scale = max_width / float(width)
        resized = cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        return resized, width / float(resized.shape[1]), height / float(resized.shape[0])

    @staticmethod
    def _clone_match(match: FaceMatch) -> FaceMatch:
        embedding = None if match.embedding is None else match.embedding.copy()
        return FaceMatch(
            bbox=BBox(match.bbox.x1, match.bbox.y1, match.bbox.x2, match.bbox.y2),
            embedding=embedding,
            track_id=match.track_id,
            last_seen=match.last_seen,
            person_id=match.person_id,
            name=match.name,
            department=match.department,
            role=match.role,
            confidence=match.confidence,
            is_known=match.is_known,
            lock_candidate=match.lock_candidate,
        )

    @staticmethod
    def _to_bbox(raw: np.ndarray, width: int, height: int, scale_x: float = 1.0, scale_y: float = 1.0) -> BBox:
        x1, y1, x2, y2 = [int(value) for value in raw[:4]]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        return BBox(max(0, x1), max(0, y1), min(width, x2), min(height, y2))

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0:
            return 0.0
        union = (a.width * a.height) + (b.width * b.height) - inter_area
        return inter_area / max(union, 1)
