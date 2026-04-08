"""
modules/hud_overlay.py — Phase 5 Cybernetic UI Engine

Generates an advanced, low-latency OpenCV interface featuring 
neon aesthetics, connecting label lines, and semantic alert boxes.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Any

# ── Aesthetic Configurations ─────────────────────────────────────
COLOR_NEON_CYAN = (255, 255, 0)      # BGR
COLOR_NEON_MAGENTA = (200, 0, 255)
COLOR_NEON_RED = (50, 50, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (10, 10, 10)

class HUDOverlay:
    def __init__(self):
        # Tracking dictionaries to apply smooth lerping (linear interpolation) to bounding boxes
        self.smooth_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        self.pulse_frame = 0
        self.anim_speed = 0.1
        self.is_active = False # For voice pulse animation
        
    def _lerp(self, current: float, target: float, factor: float = 0.3) -> float:
        """Linear interpolation for butter-smooth visual animations."""
        return current + (target - current) * factor

    def _draw_neon_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]):
        """Draws a box with an artificial 'glow' effect by stacking thicknesses."""
        # Base Glow (Thick, Darker)
        r, g, b = color
        glow_color = (int(r * 0.4), int(g * 0.4), int(b * 0.4))
        cv2.rectangle(frame, (x1, y1), (x2, y2), glow_color, 6)
        # Core Line (Thin, Bright)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Draw sci-fi corners
        length = 15
        thickness = 2
        
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    def _draw_identity_card(self, frame: np.ndarray, bx1: int, by1: int, bx2: int, by2: int,
                             name: str, department: str, role: str, confidence: float, color: Tuple[int, int, int]):
        """Draws a premium identity card anchored below the face bounding box."""
        FONT      = cv2.FONT_HERSHEY_SIMPLEX
        FONT_MONO = cv2.FONT_HERSHEY_PLAIN
        h_frame, w_frame = frame.shape[:2]

        CARD_W   = 220
        PAD      = 10
        HDR_H    = 28   # header strip height
        ROW_H    = 19
        BAR_H    = 8
        ROWS     = 3    # dept / role / conf bar
        CARD_H   = HDR_H + PAD + ROW_H * ROWS + BAR_H + PAD * 2

        # Anchor card below the face box; clip if too low
        cx = max(0, min(bx1, w_frame - CARD_W - 2))
        cy = by2 + 8
        if cy + CARD_H > h_frame:
            cy = max(0, by1 - CARD_H - 8)

        x2c = cx + CARD_W
        y2c = cy + CARD_H

        # ── Background (dark semi-transparent fill) ──────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx, cy), (x2c, y2c), (10, 10, 12), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # ── Coloured header strip ────────────────────────────────
        r, g, b = color
        hdr_dark = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        cv2.rectangle(frame, (cx, cy), (x2c, cy + HDR_H), hdr_dark, -1)
        cv2.rectangle(frame, (cx, cy), (x2c, cy + HDR_H), color, 1)

        # Status dot in header
        dot_x = cx + PAD + 6
        dot_y = cy + HDR_H // 2
        dot_color = (0, 255, 80) if name != "Unknown" else (50, 50, 255)
        cv2.circle(frame, (dot_x, dot_y), 5, dot_color, -1)
        cv2.circle(frame, (dot_x, dot_y), 5, (255, 255, 255), 1)

        # Name in header (bold-effect via double-draw)
        name_display = name.upper() if name != "Unknown" else "UNIDENTIFIED"
        cv2.putText(frame, name_display, (dot_x + 14, cy + HDR_H - 8),
                    FONT, 0.52, (30, 30, 30), 3, cv2.LINE_AA)   # shadow
        cv2.putText(frame, name_display, (dot_x + 14, cy + HDR_H - 8),
                    FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Outer border ─────────────────────────────────────────
        cv2.rectangle(frame, (cx, cy), (x2c, y2c), color, 1)

        # Corner accents
        L = 10
        for px, py, dx, dy in [(cx,cy,1,1),(x2c,cy,-1,1),(cx,y2c,1,-1),(x2c,y2c,-1,-1)]:
            cv2.line(frame, (px, py), (px + dx*L, py), color, 2)
            cv2.line(frame, (px, py), (px, py + dy*L), color, 2)

        # ── Data rows ────────────────────────────────────────────
        ry = cy + HDR_H + PAD + ROW_H - 4

        # Department
        cv2.putText(frame, "DEPT", (cx + PAD, ry),
                    FONT, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(frame, (department or "N/A").upper(), (cx + 58, ry),
                    FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        ry += ROW_H

        # Role
        cv2.putText(frame, "ROLE", (cx + PAD, ry),
                    FONT, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(frame, (role or "N/A").upper(), (cx + 58, ry),
                    FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        ry += ROW_H

        # Confidence label
        conf_label = f"CONF  {confidence:.1f}%"
        conf_color = (0, 230, 80) if confidence >= 70 else (50, 50, 255)
        cv2.putText(frame, conf_label, (cx + PAD, ry),
                    FONT, 0.42, conf_color, 1, cv2.LINE_AA)
        ry += PAD + 2

        # Confidence bar track
        bar_x1 = cx + PAD
        bar_x2 = x2c - PAD
        cv2.rectangle(frame, (bar_x1, ry), (bar_x2, ry + BAR_H), (45, 45, 45), -1)

        # Filled portion
        fill_w = int((confidence / 100.0) * (bar_x2 - bar_x1))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, ry), (bar_x1 + fill_w, ry + BAR_H), conf_color, -1)

        # Bar border
        cv2.rectangle(frame, (bar_x1, ry), (bar_x2, ry + BAR_H), (120, 120, 120), 1)

    def _draw_ai_panel(self, frame: np.ndarray, text: str, voice_active: bool):
        """Draws a premium terminal box in the bottom left for JARVIS responses."""
        h, w = frame.shape[:2]
        pad = 20
        box_w = 400
        box_h = 100
        x1, y1 = pad, h - box_h - pad
        x2, y2 = x1 + box_w, y1 + box_h

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Neon Cyan border and corner accents
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NEON_CYAN, 1)
        cv2.line(frame, (x1, y1), (x1 + 15, y1), COLOR_NEON_CYAN, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + 15), COLOR_NEON_CYAN, 3)
        cv2.line(frame, (x2, y2), (x2 - 15, y2), COLOR_NEON_CYAN, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - 15), COLOR_NEON_CYAN, 3)

        # Header
        cv2.putText(frame, "JARVIS CORE REASONING", (x1 + 10, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_CYAN, 1, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + 25), (x2, y1 + 25), COLOR_NEON_CYAN, 1)

        # Voice Pulse Animation
        if voice_active:
            self.pulse_frame += self.anim_speed
            pulse_radius = int(8 + (np.sin(self.pulse_frame * 2) + 1) * 4)
            cv2.circle(frame, (x2 - 30, y1 + 12), pulse_radius, (0, 255, 80), 2)
            cv2.circle(frame, (x2 - 30, y1 + 12), 4, (0, 255, 80), -1)
            cv2.putText(frame, "LISTENING", (x2 - 100, y1 + 17), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80), 1)

        # Split text into lines if too long
        max_chars = 45
        wrapped_text = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        
        for idx, line in enumerate(wrapped_text[:3]): # Max 3 lines
            cv2.putText(frame, line, (x1 + 10, y1 + 45 + (idx * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    def render(self, 
               frame: np.ndarray, 
               faces: List[Dict[str, Any]], 
               objects: List[Dict[str, Any]], 
               alerts: List[str], 
               context_summary: str = "",
               is_listening: bool = False) -> np.ndarray:
        """
        Master renderer. Paints the HUD across the incoming OpenCV frame.
        
        Expected structure for faces: [{"id": 1, "bbox": [x1, y1, x2, y2], "name": "Viven", "confidence": 0.98}]
        Expected structure for objects: [{"bbox": [x1, y1, x2, y2], "label": "laptop"}]
        """
        # We copy to prevent altering the raw ML inference canvas underneath
        vis_frame = frame.copy()
        
        # 1. Render Objects (Dimmer aesthetic so they don't over-clutter)
        for obj in objects:
            ox1, oy1, ox2, oy2 = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]
            self._draw_neon_box(vis_frame, ox1, oy1, ox2, oy2, COLOR_NEON_MAGENTA)
            cv2.putText(vis_frame, obj.get("label", "Unknown"), (ox1, max(15, oy1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NEON_MAGENTA, 1)

        # 2. Render Faces with Smooth Tracking and Cyber Labels
        current_frame_ids = set()
        
        for face in faces:
            fid = face.get("id", -1)
            target_bbox = face.get("bbox", [0, 0, 0, 0])
            name = face.get("name", "Unknown")
            
            current_frame_ids.add(fid)
            
            # Smooth bounding box lerping
            if fid in self.smooth_boxes:
                cx1, cy1, cx2, cy2 = self.smooth_boxes[fid]
                tx1, ty1, tx2, ty2 = target_bbox
                bbox = [
                    self._lerp(cx1, tx1), self._lerp(cy1, ty1),
                    self._lerp(cx2, tx2), self._lerp(cy2, ty2)
                ]
            else:
                bbox = target_bbox
                
            self.smooth_boxes[fid] = tuple(bbox)
            
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            
            # Draw Face box
            face_color = COLOR_NEON_RED if name == "Unknown" else COLOR_NEON_CYAN
            self._draw_neon_box(vis_frame, bx1, by1, bx2, by2, face_color)
            
            # Draw full identity card
            self._draw_identity_card(
                vis_frame, bx1, by1, bx2, by2,
                name=name,
                department=face.get("department", ""),
                role=face.get("role", ""),
                confidence=face.get("confidence", 0.0),
                color=face_color
            )

        # Cleanup stale lerp trackers to stop memory bloat
        self.smooth_boxes = {k: v for k, v in self.smooth_boxes.items() if k in current_frame_ids}

        # 3. Render System Context & Intelligence (Bottom Left Panel)
        self._draw_ai_panel(vis_frame, context_summary, is_listening)

        # 4. Render Active Alerts (Top Right)
        h, w = vis_frame.shape[:2]
        for idx, alert in enumerate(alerts):
            text_size, _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            aw, ah = text_size
            
            pad = 10
            ax = w - aw - pad * 2 - 20
            ay = 30 + (idx * 40)
            
            # Flashing red logic
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(vis_frame, (ax, ay), (ax + aw + pad * 2, ay + ah + pad * 2), COLOR_NEON_RED, -1)
                cv2.putText(vis_frame, alert, (ax + pad, ay + ah + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
            else:
                cv2.rectangle(vis_frame, (ax, ay), (ax + aw + pad * 2, ay + ah + pad * 2), COLOR_NEON_RED, 2)
                cv2.putText(vis_frame, alert, (ax + pad, ay + ah + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_NEON_RED, 2)

        return vis_frame

# ── Example Sandbox Run ───────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    
    # Create fake blank camera feed
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = HUDOverlay()
    
    mock_faces = [
        {"id": 1, "bbox": [500, 200, 700, 450], "name": "Viven", "confidence": 0.99},
        {"id": 2, "bbox": [800, 300, 950, 480], "name": "Unknown", "confidence": 0.0}
    ]
    
    mock_objects = [
        {"bbox": [650, 420, 730, 480], "label": "cell phone"}
    ]
    
    alerts = ["WARNING: Unknown Subject Intrusion", "WARNING: Phone Detected"]
    
    # Render
    output = overlay.render(dummy_frame, mock_faces, mock_objects, alerts, 
                            context_summary="Context: 2 People, 1 Phone. Threat Level High.")
                            
    # Display Result
    cv2.imshow("JARVIS Cyber HUD", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
