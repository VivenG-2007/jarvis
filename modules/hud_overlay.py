from __future__ import annotations

import math
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

import config


@dataclass
class LockState:
    track_id: int
    bbox: list[int]
    started_at: float
    name: str


class HUDOverlay:
    def __init__(self):
        self.smooth_boxes: dict[int, tuple[float, float, float, float]] = {}
        self.active_lock: Optional[LockState] = None
        self.zoomed_people: set[str] = set()
        self.intro_zoom: Optional[dict[str, Any]] = None
        self.scan_offset = 0
        self.frame_tick = 0
        self.last_response_text = ""
        self.response_changed_at = time.time()

    def render(
        self,
        frame: np.ndarray,
        faces: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        alerts: list[str],
        context_summary: str = "",
        heard_text: str = "",
        is_listening: bool = False,
        wake_active: bool = False,
        audio_level: float = 0.0,
        focus_mode_active: bool = False,
    ) -> np.ndarray:
        canvas = frame
        self.frame_tick += 1
        self._update_lock(faces)
        self._update_intro_zoom(faces)
        canvas = self._apply_intro_zoom(canvas, faces)
        if focus_mode_active and self.active_lock is not None:
            canvas = self._apply_spotlight(canvas, self.active_lock)
        self._draw_background_fx(canvas)
        self._draw_objects(canvas, objects)
        self._draw_faces(canvas, faces, focus_mode_active)
        self._draw_header(canvas, len(faces), len(objects), is_listening, wake_active, audio_level)
        self._draw_alerts(canvas, alerts)
        self._draw_transcript_panel(canvas, heard_text, is_listening, wake_active)
        self._draw_reasoning_panel(canvas, context_summary, is_listening, wake_active, audio_level)
        return canvas

    def _update_lock(self, faces: list[dict[str, Any]]) -> None:
        now = time.time()
        if self.active_lock and now - self.active_lock.started_at > config.APP.target_lock_duration:
            self.active_lock = None
        if self.active_lock:
            for face in faces:
                if face.get("id") == self.active_lock.track_id:
                    self.active_lock.bbox = face["bbox"]
                    return
        candidate = next((face for face in faces if face.get("lock_candidate")), None)
        if candidate:
            self.active_lock = LockState(
                track_id=candidate["id"],
                bbox=list(candidate["bbox"]),
                started_at=now,
                name=candidate.get("name", "TARGET"),
            )

    def _update_intro_zoom(self, faces: list[dict[str, Any]]) -> None:
        if not config.APP.hud_intro_zoom_enabled:
            self.intro_zoom = None
            return
        now = time.time()
        identified_faces = [
            face
            for face in faces
            if face.get("is_known") and str(face.get("person_id") or "").strip() and str(face.get("person_id")) not in self.zoomed_people
        ]

        if identified_faces and self.intro_zoom is None:
            target = max(identified_faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))
            person_id = str(target["person_id"]).strip()
            self.zoomed_people.add(person_id)
            self.intro_zoom = {"person_id": person_id, "started_at": now}

        if self.intro_zoom and now - float(self.intro_zoom["started_at"]) > 2.0:
            self.intro_zoom = None

    def _apply_intro_zoom(self, frame: np.ndarray, faces: list[dict[str, Any]]) -> np.ndarray:
        if not config.APP.hud_intro_zoom_enabled or not self.intro_zoom:
            return frame

        target = next(
            (
                face
                for face in faces
                if face.get("is_known") and str(face.get("person_id") or "").strip() == str(self.intro_zoom["person_id"])
            ),
            None,
        )
        if target is None:
            return frame

        progress = min(1.0, max(0.0, (time.time() - float(self.intro_zoom["started_at"])) / 2.0))
        curve = math.sin(progress * math.pi)
        zoom = 1.0 + curve
        return self._zoom_into_bbox(frame, target["bbox"], zoom)

    def _zoom_into_bbox(self, frame: np.ndarray, bbox: list[int], zoom: float) -> np.ndarray:
        if zoom <= 1.001:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        center_x = max(0, min(w - 1, (x1 + x2) // 2))
        center_y = max(0, min(h - 1, (y1 + y2) // 2))

        crop_w = max(1, int(w / zoom))
        crop_h = max(1, int(h / zoom))
        left = max(0, min(w - crop_w, center_x - crop_w // 2))
        top = max(0, min(h - crop_h, center_y - crop_h // 2))
        cropped = frame[top : top + crop_h, left : left + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def _apply_spotlight(self, frame: np.ndarray, lock: LockState) -> np.ndarray:
        if not config.APP.hud_spotlight_enabled:
            return frame
        overlay = (frame.astype(np.float32) * (1.0 - config.APP.spotlight_dim_alpha)).astype(np.uint8)
        x1, y1, x2, y2 = lock.bbox
        pad = config.APP.spotlight_padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(70, (x2 - x1) // 2), max(100, (y2 - y1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (41, 41), 0)
        mask_float = (mask / 255.0)[..., None]
        mixed = (overlay.astype(np.float32) * (1.0 - mask_float) + frame.astype(np.float32) * mask_float).astype(np.uint8)
        return mixed

    def _draw_background_fx(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self._blend_rect(frame, 0, 0, w, 64, (10, 24, 18), 0.10)
        self._blend_rect(frame, 0, h - 96, w, 96, (6, 14, 18), 0.10)

        self.scan_offset = (self.scan_offset + 3) % max(h, 1)
        sweep_y = (self.frame_tick * 7) % max(h + 180, 1) - 90
        if 0 <= sweep_y < h:
            y1 = max(0, sweep_y - 2)
            y2 = min(h, sweep_y + 3)
            self._blend_rect(frame, 0, y1, w, y2 - y1, (30, 90, 70), 0.05)

    def _draw_objects(self, frame: np.ndarray, objects: list[dict[str, Any]]) -> None:
        color = config.APP.hud_color_accent
        for obj in objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            self._draw_target_box(frame, x1, y1, x2, y2, color, locked=False, corner_length=14, border_alpha=0.0)
            label = f"{obj['label'].upper()}  {int(obj.get('confidence', 0) * 100)}%"
            if obj.get("db_count"):
                label += f"  DB {int(obj.get('db_count', 0))}"
            self._draw_chip(frame, x1, max(24, y1 - 18), label, color, align_top=True, compact=True)

    def _draw_faces(self, frame: np.ndarray, faces: list[dict[str, Any]], focus_mode_active: bool = False) -> None:
        active_ids = set()
        for face in faces:
            face_id = face["id"]
            active_ids.add(face_id)
            if face.get("suppressed_overlay"):
                continue
            bbox = self._smooth_box(face_id, face["bbox"])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            is_target = self.active_lock is not None and face_id == self.active_lock.track_id
            if focus_mode_active and not is_target:
                subdued = (92, 92, 92)
                self._draw_target_box(frame, x1, y1, x2, y2, subdued, False, corner_length=14, border_alpha=0.0)
                continue
            color = config.APP.hud_color_primary if face.get("is_known") else config.APP.hud_color_alert
            self._draw_target_box(frame, x1, y1, x2, y2, color, face.get("lock_candidate", False))
            self._draw_identity_hud(frame, face, [x1, y1, x2, y2], color)
        self.smooth_boxes = {face_id: box for face_id, box in self.smooth_boxes.items() if face_id in active_ids}

    def _smooth_box(self, face_id: int, target: list[int]) -> tuple[float, float, float, float]:
        current = self.smooth_boxes.get(face_id)
        if current is None:
            smooth = tuple(float(v) for v in target)
        else:
            deltas = [abs(float(tgt) - cur) for cur, tgt in zip(current, target)]
            max_delta = max(deltas)
            factor = config.APP.zoom_smoothing
            if max_delta > 60:
                factor = 1.0
            elif max_delta > 28:
                factor = max(factor, 0.82)
            smooth = tuple(cur + (float(tgt) - cur) * factor for cur, tgt in zip(current, target))
        self.smooth_boxes[face_id] = smooth
        return smooth

    def _draw_target_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        locked: bool,
        corner_length: int = 22,
        border_alpha: float = 0.18,
    ) -> None:
        length = corner_length
        thickness = 2 if not locked else 3
        pulse = 1.0 if not locked else 0.7 + 0.3 * abs(math.sin(time.time() * 5))
        draw_color = tuple(int(channel * pulse) for channel in color)
        for start, dx, dy in [((x1, y1), 1, 1), ((x2, y1), -1, 1), ((x1, y2), 1, -1), ((x2, y2), -1, -1)]:
            sx, sy = start
            cv2.line(frame, (sx, sy), (sx + dx * length, sy), draw_color, thickness, cv2.LINE_AA)
            cv2.line(frame, (sx, sy), (sx, sy + dy * length), draw_color, thickness, cv2.LINE_AA)
        if border_alpha > 0:
            roi = frame[max(0, y1 - 1) : min(frame.shape[0], y2 + 2), max(0, x1 - 1) : min(frame.shape[1], x2 + 2)]
            if roi.size > 0:
                border = roi.copy()
                cv2.rectangle(border, (1, 1), (max(1, border.shape[1] - 2), max(1, border.shape[0] - 2)), draw_color, 1, cv2.LINE_AA)
                cv2.addWeighted(border, border_alpha, roi, 1.0 - border_alpha, 0, roi)
        if locked:
            cv2.circle(frame, ((x1 + x2) // 2, y1 - 8), 3, draw_color, -1, cv2.LINE_AA)

    def _draw_identity_hud(self, frame: np.ndarray, face: dict[str, Any], bbox: list[int], color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        anchor_x = min(frame.shape[1] - 210, x2 + 16)
        anchor_y = max(44, y1 - 4)
        if anchor_x < x1:
            anchor_x = max(10, x1 - 210)
        card_w = 196
        card_h = 84
        self._draw_panel(frame, anchor_x, anchor_y, card_w, card_h, color, alpha=0.42, accent=False)
        cv2.putText(frame, "ID OVERLAY", (anchor_x + 12, anchor_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.APP.hud_color_accent, 1, cv2.LINE_AA)
        rows = [
            ("NAME", face.get("name", "Unknown")),
            ("ID", face.get("person_id", "N/A")),
            ("DEPT", face.get("department", "N/A")),
            ("ROLE", face.get("role", "N/A")),
        ]
        for index, (label, value) in enumerate(rows):
            row_y = anchor_y + 35 + (index * 12)
            cv2.putText(frame, label, (anchor_x + 12, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 170, 160), 1, cv2.LINE_AA)
            cv2.putText(frame, str(value), (anchor_x + 58, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
        confidence = f"{face.get('confidence', 0.0):.1f}%"
        self._draw_chip(frame, anchor_x + card_w - 74, anchor_y + 8, confidence, color, width=62, compact=True)

    def _draw_header(
        self,
        frame: np.ndarray,
        face_count: int,
        object_count: int,
        listening: bool,
        wake_active: bool,
        audio_level: float,
    ) -> None:
        h, w = frame.shape[:2]
        self._draw_panel(frame, 12, 12, 260, 42, config.APP.hud_color_primary, alpha=0.36, accent=False)
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, "J.A.R.V.I.S.", (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, config.APP.hud_color_primary, 1, cv2.LINE_AA)
        cv2.putText(frame, f"TACTICAL OVERLAY  {timestamp}", (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 180, 170), 1, cv2.LINE_AA)
        mic_state = "VOICE" if listening else "ARMED" if wake_active else "STBY"
        status = f"FACES {face_count:02d}  OBJECTS {object_count:02d}  MIC {mic_state}"
        self._draw_chip(frame, w - 250, 18, status, config.APP.hud_color_accent, width=238, align_top=True, compact=True)
        mic_color = config.APP.hud_color_alert if listening else config.APP.hud_color_accent if wake_active else (70, 110, 95)
        cv2.circle(frame, (w - 34, 26), 7, mic_color, -1, cv2.LINE_AA)
        if listening or wake_active:
            pulse = 11 + int(4 * abs(math.sin(time.time() * 7)))
            cv2.circle(frame, (w - 34, 26), pulse, mic_color, 1, cv2.LINE_AA)
        self._draw_audio_meter(frame, w - 248, 44, 120, 6, audio_level, mic_color if (listening or wake_active) else (88, 118, 108))
        if self.active_lock:
            self._draw_chip(frame, 20, h - 182, f"TARGET LOCK  {self.active_lock.name}", config.APP.hud_color_alert, width=210, compact=True)

    def _draw_alerts(self, frame: np.ndarray, alerts: list[str]) -> None:
        x = frame.shape[1] - 300
        y = 72
        for alert in alerts[:4]:
            self._draw_panel(frame, x, y, 280, 24, config.APP.hud_color_alert, alpha=0.38, accent=False)
            cv2.putText(frame, alert[:36], (x + 10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.APP.hud_color_alert, 1, cv2.LINE_AA)
            y += 30

    def _draw_transcript_panel(self, frame: np.ndarray, heard_text: str, is_listening: bool, wake_active: bool) -> None:
        h, w = frame.shape[:2]
        x1, y1, width, height = max(20, w - 500), h - 128, min(460, w - 40), 104
        panel_color = config.APP.hud_color_primary if heard_text.strip() else (90, 120, 110)
        alpha = 0.38 if heard_text.strip() else 0.24
        self._draw_panel(frame, x1, y1, width, height, panel_color, alpha=alpha, accent=False)
        cv2.putText(frame, "VOICE LINK", (x1 + 12, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.APP.hud_color_primary, 1, cv2.LINE_AA)
        state_text = "LISTENING..." if is_listening else "AWAITING COMMAND" if wake_active else "LAST TRANSCRIPT"
        cv2.putText(frame, state_text, (x1 + width - 150, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (135, 172, 165), 1, cv2.LINE_AA)
        wrapped = textwrap.wrap(heard_text.strip() or "No speech captured yet.", width=36)[:4]
        text_color = (235, 245, 250) if heard_text.strip() else (150, 170, 168)
        for index, line in enumerate(wrapped):
            cv2.putText(frame, line, (x1 + 12, y1 + 42 + (index * 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, text_color, 1, cv2.LINE_AA)

    def _draw_reasoning_panel(self, frame: np.ndarray, text: str, is_listening: bool, wake_active: bool, audio_level: float) -> None:
        if text != self.last_response_text:
            self.last_response_text = text
            self.response_changed_at = time.time()

        h = frame.shape[0]
        x1, y1, width, height = 20, h - 128, 460, 104
        panel_color = config.APP.hud_color_alert if (is_listening or wake_active) else config.APP.hud_color_accent
        alpha = 0.42 if (is_listening or wake_active) else 0.34
        self._draw_panel(frame, x1, y1, width, height, panel_color, alpha=alpha, accent=False)
        cv2.putText(frame, "MISSION RESPONSE", (x1 + 12, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.APP.hud_color_accent, 1, cv2.LINE_AA)
        status_text = "SPEECH DETECTED" if is_listening else "READY FOR COMMAND" if wake_active else "MIC ON STANDBY"
        status_color = config.APP.hud_color_alert if (is_listening or wake_active) else (135, 172, 165)
        cv2.putText(frame, status_text, (x1 + width - 176, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, status_color, 1, cv2.LINE_AA)
        self._draw_audio_meter(frame, x1 + width - 132, y1 + 28, 112, 6, audio_level, status_color)

        wrapped = textwrap.wrap(text or "Awaiting grounded query.", width=46)[:4]
        animation_window = 1.0
        reveal_ratio = min(1.0, (time.time() - self.response_changed_at) / animation_window)
        reveal_chars = max(1, int(sum(len(line) for line in wrapped) * reveal_ratio))
        remaining = reveal_chars
        for index, line in enumerate(wrapped):
            visible = line[:remaining]
            remaining = max(0, remaining - len(line))
            if visible:
                cv2.putText(frame, visible, (x1 + 12, y1 + 42 + (index * 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 245, 250), 1, cv2.LINE_AA)
        if reveal_ratio < 1.0:
            caret_x = x1 + 12 + min(width - 28, int((width - 24) * reveal_ratio))
            cv2.line(frame, (caret_x, y1 + 32), (caret_x, y1 + height - 18), status_color, 1, cv2.LINE_AA)
        footer = f"TRACKS {len(self.smooth_boxes):02d}   TARGET {'LOCKED' if self.active_lock else 'SEARCHING'}"
        cv2.putText(frame, footer, (x1 + 12, y1 + height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (135, 172, 165), 1, cv2.LINE_AA)

    def _draw_audio_meter(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        audio_level: float,
        color: tuple[int, int, int],
    ) -> None:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (56, 76, 72), 1, cv2.LINE_AA)
        fill = max(0, min(width - 2, int((width - 2) * max(0.0, min(1.0, audio_level)))))
        if fill > 0:
            cv2.rectangle(frame, (x + 1, y + 1), (x + 1 + fill, y + height - 1), color, -1)

    def _draw_panel(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        border_color: tuple[int, int, int],
        alpha: float = 0.7,
        accent: bool = True,
    ) -> None:
        self._blend_rect(frame, x, y, width, height, (5, 12, 18), alpha)
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 1, cv2.LINE_AA)
        if accent:
            cv2.line(frame, (x + 10, y), (x + min(46, width - 10), y), config.APP.hud_color_accent, 1, cv2.LINE_AA)

    def _blend_rect(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + width)
        y2 = min(frame.shape[0], y + height)
        if x1 >= x2 or y1 >= y2:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = np.empty_like(roi)
        overlay[...] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    def _draw_chip(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
        width: int | None = None,
        compact: bool = False,
        align_top: bool = False,
    ) -> None:
        font_scale = 0.36 if compact else 0.42
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        chip_w = width or (text_w + 18)
        chip_h = 20 if compact else 24
        top = y if align_top else y - chip_h
        self._draw_panel(frame, x, top, chip_w, chip_h, color, alpha=0.34, accent=False)
        text_y = top + chip_h - 7
        cv2.putText(frame, text, (x + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    def _draw_dashed_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
    ) -> None:
        dash = 10
        gap = 8
        for x in range(x1, x2, dash + gap):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, 1, cv2.LINE_AA)
        for y in range(y1, y2, dash + gap):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, 1, cv2.LINE_AA)
