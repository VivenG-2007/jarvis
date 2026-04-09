from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger("jarvis.screen")

try:
    import mss

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


class ScreenCapture:
    def __init__(self, monitor_index: Optional[int] = None):
        self.monitor_index = monitor_index or config.APP.screen_monitor_index
        self.sct = mss.mss() if MSS_AVAILABLE and config.APP.enable_screen_input else None

    def grab(self) -> Optional[np.ndarray]:
        if self.sct is None:
            return None
        monitors = self.sct.monitors
        index = min(max(self.monitor_index, 1), len(monitors) - 1)
        monitor = monitors[index]
        frame = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
