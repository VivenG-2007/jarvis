from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

import config

logger = logging.getLogger("jarvis.object")

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


@dataclass
class ObjectDetection:
    label: str
    confidence: float
    bbox: list[int]
    class_id: int
    source: str = "camera"


class ObjectEngine:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.frame_index = 0
        self.cached_results: list[ObjectDetection] = []
        self.device = "cpu"
        if ULTRALYTICS_AVAILABLE:
            path = model_path or config.APP.object_model_path
            self.model = YOLO(path)
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.device = "0"
                    self.model.to(self.device)
            except Exception:
                self.device = "cpu"
        else:
            logger.warning("ultralytics is not installed; object detection will stay disabled.")

    def detect(self, frame: np.ndarray, conf: float | None = None, source: str = "camera") -> list[ObjectDetection]:
        if self.model is None:
            return []

        self.frame_index += 1
        skip = max(1, config.APP.object_frame_skip)
        if self.frame_index % skip != 0 and self.cached_results:
            return self.cached_results

        results = self.model.predict(
            frame,
            conf=conf if conf is not None else config.APP.object_conf_threshold,
            imgsz=config.APP.object_imgsz,
            verbose=False,
            device=self.device,
        )
        if not results:
            self.cached_results = []
            return []

        detections: list[ObjectDetection] = []
        result = results[0]
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            cls_list = boxes.cls.cpu().numpy().astype(int)
            conf_list = boxes.conf.cpu().numpy().astype(float)
            bbox_list = boxes.xyxy.cpu().numpy().astype(int)
            for index, class_id in enumerate(cls_list):
                if class_id == 0:
                    continue
                detections.append(
                    ObjectDetection(
                        label=result.names[class_id],
                        confidence=float(conf_list[index]),
                        bbox=bbox_list[index].tolist(),
                        class_id=int(class_id),
                        source=source,
                    )
                )
        self.cached_results = detections
        return detections
