"""
modules/hud.py — Jarvis-style HUD overlay renderer.

Draws on top of each OpenCV frame:
  ┌── Animated corner brackets around each face
  ├── Data card (Name / ID / Department / Confidence)
  ├── Connecting HUD lines
  ├── Spotlight / dim effect for target-lock mode
  ├── Scan-line animation
  └── System status bar (top & bottom)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config
from modules.face_engine import FaceMatch


# ── Per-face animation state ───────────────────────────────────

@dataclass
class FaceAnimState:
    lock_start:   float = 0.0
    lock_active:  bool  = False
    bracket_phase: float = 0.0     # 0→1 open animation
    zoom_factor:  float = 1.0


_anim_states: dict[int, FaceAnimState] = {}


def _get_anim(track_id: int) -> FaceAnimState:
    if track_id not in _anim_states:
        _anim_states[track_id] = FaceAnimState()
    return _anim_states[track_id]


def _prune_anims(active_ids: set[int]):
    stale = [k for k in _anim_states if k not in active_ids]
    for k in stale:
        del _anim_states[k]


# ── Main draw function ─────────────────────────────────────────

def draw_hud(
    frame: np.ndarray, 
    matches: list[FaceMatch], 
    drawn_objects: list,
    all_objects: list,
    fps: float
) -> np.ndarray:
    """
    Render full JARVIS HUD onto `frame` in-place.
    """
    h, w = frame.shape[:2]
    now   = time.time()
    active_ids = {m.track_id for m in matches}
    _prune_anims(active_ids)

    # ── Scan-line overlay ──
    _draw_scanlines(frame, now, h, w)

    # ── Objects (YOLO) ──
    _draw_objects(frame, drawn_objects)

    # ── Per-face elements ──
    for m in matches:
        anim = _get_anim(m.track_id)
        _tick_anim(anim, m, now)

    # ── Face overlays ──
    for m in matches:
        anim = _get_anim(m.track_id)
        _draw_face_overlay(frame, m, anim, h, w, now)

    # ── Status bars ──
    _draw_status_bar_top(frame, w, fps, len(matches))
    _draw_status_bar_bottom(frame, h, w, matches, all_objects)

    # ── Corner decorations ──
    _draw_corner_brackets_frame(frame, h, w, now)

    return frame


# ── Object drawing (YOLO) ──────────────────────────────────────

def _draw_objects(frame: np.ndarray, objects: list):
    """Draw technical boxes and labels for YOLO detections."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = config.HUD_COLOR_ACCENT
    
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        label = f"{obj.label.upper()} [{int(obj.confidence*100)}%]"
        
        # Dashed-style corner lines for objects
        length = 15
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, 1, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, 1, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, 1, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, 1, cv2.LINE_AA)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        # Gradient or solid background for high visibility
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6), font, 0.45, color, 1, cv2.LINE_AA)


# ── Animation tick ─────────────────────────────────────────────

def _tick_anim(anim: FaceAnimState, m: FaceMatch, now: float):
    # Bracket open animation
    if anim.bracket_phase < 1.0:
        anim.bracket_phase = min(1.0, anim.bracket_phase + 0.08)

    # Target lock: activate when person is recognised
    if m.is_known and not anim.lock_active:
        anim.lock_active = True
        anim.lock_start  = now

    # Expire lock after duration
    if anim.lock_active and (now - anim.lock_start) > config.TARGET_LOCK_DURATION:
        anim.lock_active = False

    # Zoom pulse on lock
    if anim.lock_active:
        elapsed = now - anim.lock_start
        anim.zoom_factor = 1.0 + 0.04 * math.sin(elapsed * 2.5)
    else:
        anim.zoom_factor = 1.0


# ── Per-face drawing ───────────────────────────────────────────

def _draw_face_overlay(
    frame: np.ndarray, m: FaceMatch, anim: FaceAnimState,
    h: int, w: int, now: float
):
    bb    = m.bbox
    color = config.HUD_COLOR_PRIMARY if m.is_known else config.HUD_COLOR_ACCENT
    if anim.lock_active:
        pulse = 0.6 + 0.4 * abs(math.sin((now - anim.lock_start) * 4))
        color = tuple(int(c * pulse) for c in color)

    _draw_corner_brackets(frame, bb, color, anim.bracket_phase, anim.lock_active)
    if m.is_known:
        _draw_data_card(frame, m, bb, color, h, w)
    else:
        _draw_unknown_tag(frame, bb, color)


# ── Animated corner brackets ────────────────────────────────────

def _draw_corner_brackets(
    frame: np.ndarray, bb, color: tuple,
    phase: float, active: bool
):
    x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
    arm_len = int(min(bb.width, bb.height) * 0.25 * phase)
    thick   = 2 if not active else 3
    gap     = 4

    corners = [
        ((x1-gap, y1-gap), (1,  1)),
        ((x2+gap, y1-gap), (-1, 1)),
        ((x1-gap, y2+gap), (1, -1)),
        ((x2+gap, y2+gap), (-1,-1)),
    ]
    for (cx, cy), (dx, dy) in corners:
        cv2.line(frame, (cx, cy), (cx + dx*arm_len, cy),           color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy*arm_len),           color, thick, cv2.LINE_AA)

    # Thin rectangle border
    alpha_rect = min(1.0, phase * 2)
    rect_color = tuple(int(c * alpha_rect * 0.35) for c in color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 1, cv2.LINE_AA)

    # Animated side ticks on active lock
    if active:
        mid_y = (y1 + y2) // 2
        mid_x = (x1 + x2) // 2
        tick  = 10
        cv2.line(frame, (x1-gap, mid_y-tick), (x1-gap, mid_y+tick), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2+gap, mid_y-tick), (x2+gap, mid_y+tick), color, thick, cv2.LINE_AA)
        cv2.line(frame, (mid_x-tick, y1-gap), (mid_x+tick, y1-gap), color, thick, cv2.LINE_AA)


# ── Data card ──────────────────────────────────────────────────

def _draw_data_card(
    frame: np.ndarray, m: FaceMatch, bb,
    color: tuple, h: int, w: int
):
    face_cx = (bb.x1 + bb.x2) // 2
    card_x  = bb.x2 + 18
    card_y  = bb.y1

    # Keep card inside frame
    card_w = 220
    if card_x + card_w > w:
        card_x = bb.x1 - card_w - 18

    lines = [
        ("NAME",       m.name        or "—"),
        ("ID",         m.person_id   or "—"),
        ("DEPT",       m.department  or "—"),
        ("ROLE",       m.role        or "—"),
        ("CONF",       f"{m.confidence:.1f}%"),
    ]

    fs    = config.HUD_FONT_SCALE
    lh    = 22
    pad_x = 10
    pad_y = 8
    card_h = len(lines) * lh + pad_y * 2

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (card_x - pad_x, card_y - pad_y),
                  (card_x + card_w, card_y + card_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Border
    cv2.rectangle(frame,
                  (card_x - pad_x, card_y - pad_y),
                  (card_x + card_w, card_y + card_h),
                  color, 1, cv2.LINE_AA)

    # Top accent line
    cv2.line(frame,
             (card_x - pad_x, card_y - pad_y),
             (card_x - pad_x + 40, card_y - pad_y),
             config.HUD_COLOR_ACCENT, 2, cv2.LINE_AA)

    # Text rows
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (label, value) in enumerate(lines):
        ty = card_y + i * lh + lh // 2
        cv2.putText(frame, label, (card_x, ty),
                    font, fs * 0.75,
                    tuple(int(c*0.7) for c in color), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (card_x + 60, ty),
                    font, fs, color, 1, cv2.LINE_AA)

    # Connecting lines from card to face
    connect_y = card_y + card_h // 2
    end_x     = bb.x2 if card_x > bb.x2 else bb.x1
    mid_x     = (card_x - pad_x + end_x) // 2

    cv2.line(frame, (end_x, (bb.y1+bb.y2)//2), (mid_x, connect_y),
             color, 1, cv2.LINE_AA)
    cv2.line(frame, (mid_x, connect_y), (card_x - pad_x, connect_y),
             color, 1, cv2.LINE_AA)
    cv2.circle(frame, (end_x, (bb.y1+bb.y2)//2), 3, color, -1, cv2.LINE_AA)


# ── Unknown tag ────────────────────────────────────────────────

def _draw_unknown_tag(frame: np.ndarray, bb, color: tuple):
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = "UNIDENTIFIED"
    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
    tx = bb.x1 + (bb.x2 - bb.x1 - tw) // 2
    ty = bb.y2 + 18
    cv2.putText(frame, label, (tx, ty),
                font, 0.45, color, 1, cv2.LINE_AA)


# ── Scan lines ─────────────────────────────────────────────────

def _draw_scanlines(frame: np.ndarray, now: float, h: int, w: int):
    speed = 80    # px/sec
    pos   = int(now * speed) % h
    for yy in range(pos, h, 6):
        cv2.line(frame, (0, yy), (w, yy), (0, 30, 0), 1)

    # Moving bright scan band
    band_y = pos % h
    for dy in range(-2, 3):
        yy = (band_y + dy) % h
        alpha = 1.0 - abs(dy) / 3
        row = frame[yy].astype(np.float32)
        row[:, 1] = np.clip(row[:, 1] + 25 * alpha, 0, 255)
        frame[yy] = row.astype(np.uint8)


# ── Status bars ────────────────────────────────────────────────

def _draw_status_bar_top(frame: np.ndarray, w: int, fps: float, face_count: int):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = config.HUD_COLOR_PRIMARY
    bar_h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (0, bar_h), (w, bar_h), color, 1)

    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"JARVIS  //  {ts}", (12, 22),
                font, 0.45, color, 1, cv2.LINE_AA)

    status_r = f"FPS {fps:05.1f}   FACES {face_count:02d}   SYSTEM NOMINAL"
    (tw, _), _ = cv2.getTextSize(status_r, font, 0.45, 1)
    cv2.putText(frame, status_r, (w - tw - 12, 22),
                font, 0.45, config.HUD_COLOR_ACCENT, 1, cv2.LINE_AA)


def _draw_status_bar_bottom(frame: np.ndarray, h: int, w: int, matches: list[FaceMatch], all_objects: list):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = config.HUD_COLOR_PRIMARY
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), color, 1)

    known   = [m.name for m in matches if m.is_known]
    unknown = sum(1 for m in matches if not m.is_known)

    parts   = []
    if known:
        parts.append("IDENTIFIED: " + " | ".join(known))
    if unknown:
        parts.append(f"UNKNOWN: {unknown}")
    if not parts:
        parts.append("NO SUBJECTS DETECTED")

    text = "   ▸   ".join(parts)
    cv2.putText(frame, text, (12, h - 10),
                font, 0.42, color, 1, cv2.LINE_AA)


# ── Screen corner brackets ─────────────────────────────────────

def _draw_corner_brackets_frame(frame: np.ndarray, h: int, w: int, now: float):
    color  = config.HUD_COLOR_ACCENT
    arm    = 30
    thick  = 2
    margin = 8
    pulse  = 0.5 + 0.5 * abs(math.sin(now * 1.5))
    color  = tuple(int(c * pulse) for c in color)

    pts = [
        ((margin, margin), (1, 1)),
        ((w-margin, margin), (-1, 1)),
        ((margin, h-margin), (1, -1)),
        ((w-margin, h-margin), (-1, -1)),
    ]
    for (cx, cy), (dx, dy) in pts:
        cv2.line(frame, (cx, cy), (cx+dx*arm, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy+dy*arm), color, thick, cv2.LINE_AA)
