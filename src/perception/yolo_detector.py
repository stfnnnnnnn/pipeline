"""
YOLOv8 detector wrapper for MoodPlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

import numpy as np
import torch

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


@dataclass
class Detection:
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    cls_id: int
    label: str


class YoloV8Detector:
    def __init__(self, model: str = "yolo26s.pt", device: Optional[str] = "cuda") -> None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")
        self.model = YOLO(model)
        self.device = device

    def detect(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.6,
        target_labels: Optional[Iterable[str]] = None,
    ) -> List[Detection]:
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise ValueError("image_bgr must be a numpy array in BGR format")

        target_set: Optional[Set[str]] = None
        if target_labels is not None:
            target_set = {str(label).lower() for label in target_labels}

        with torch.inference_mode():
            use_amp = bool(self.device and "cuda" in str(self.device) and torch.cuda.is_available())
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = self.model.predict(
                        source=image_bgr,
                        conf=conf,
                        iou=iou,
                        verbose=False,
                        device=self.device,
                    )[0]
            else:
                result = self.model.predict(
                    source=image_bgr,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                    device=self.device,
                )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        xyxy = boxes.xyxy
        cls_ids = boxes.cls
        confs = boxes.conf

        if hasattr(xyxy, "detach"):
            xyxy = xyxy.detach().cpu().numpy()
        else:
            xyxy = np.asarray(xyxy)

        if hasattr(cls_ids, "detach"):
            cls_ids = cls_ids.detach().cpu().numpy().astype(int)
        else:
            cls_ids = np.asarray(cls_ids, dtype=int)

        if hasattr(confs, "detach"):
            confs = confs.detach().cpu().numpy()
        else:
            confs = np.asarray(confs)

        names = result.names

        detections: List[Detection] = []
        for b, c, cid in zip(xyxy, confs, cls_ids):
            label = self._label_for_class_id(names, int(cid))
            if target_set is not None and label.lower() not in target_set:
                continue
            detections.append(
                Detection(
                    bbox_xyxy=b.astype(np.float32),
                    conf=float(c),
                    cls_id=int(cid),
                    label=label,
                )
            )
        return detections

    @staticmethod
    def _label_for_class_id(names: object, cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)):
            if 0 <= cls_id < len(names):
                return str(names[cls_id])
        return str(cls_id)