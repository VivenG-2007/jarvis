"""
modules/object_engine.py — YOLOv8-powered object detection.
"""

import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger("jarvis.object_engine")

@dataclass
class ObjectDetection:
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int

class ObjectEngine:
    """
    Handles YOLOv8n object detection with GPU support.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = None
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                # Ensure it uses GPU if available
                # Ultralytics handles this automatically but we can be explicit
                logger.info("YOLOv8 model '%s' loaded ✓", model_path)
            except Exception as e:
                logger.error("Failed to load YOLO model: %s", e)
        else:
            logger.warning("ultralytics not installed — object detection disabled.")

    def detect(self, frame: np.ndarray, conf: float = 0.25) -> List[ObjectDetection]:
        if self.model is None:
            return []

        # Run inference
        results = self.model(frame, conf=conf, verbose=False, device='0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
        
        detections = []
        if results and len(results) > 0:
            res = results[0]
            for box in res.boxes:
                # box.xyxy[0] is [x1, y1, x2, y2]
                b = box.xyxy[0].cpu().numpy().astype(int)
                detections.append(ObjectDetection(
                    label=res.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=b.tolist(),
                    class_id=int(box.cls[0])
                ))
        
        return detections
