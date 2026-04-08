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
    import torch
    torch.set_num_threads(2)  # Limit CPU thrashing with ONNX Runtime
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
    Handles YOLOv8 object detection with GPU support.
    """
    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model = None
        self.device = 'cpu'
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                # Cache device once to avoid overhead in detect()
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.device = '0'
                self.model.to(self.device)
                logger.info("YOLOv8 loaded on %s ✓", self.device)
            except Exception as e:
                logger.error("Failed to load YOLO model: %s", e)
        else:
            logger.warning("ultralytics not installed — object detection disabled.")

    def detect(self, frame: np.ndarray, conf: float = 0.20) -> List[ObjectDetection]:
        if self.model is None:
            return []

        # Run inference using 416px — 4x faster than 640 on CPU, still high accuracy
        results = self.model.predict(frame, conf=conf, imgsz=416, verbose=False, device=self.device)
        
        if not results:
            return []
            
        res = results[0]
        # Use more efficient extraction
        boxes = res.boxes
        det_list = []
        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy().astype(float)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                # Class 0 in YOLO is "person" - ignore it so only Buffalo handles humans
                if cls[i] == 0:
                    continue
                    
                det_list.append(ObjectDetection(
                    label=res.names[cls[i]],
                    confidence=conf[i],
                    bbox=xyxy[i].tolist(),
                    class_id=cls[i]
                ))
        return det_list
