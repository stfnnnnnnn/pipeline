"""
SAM2 image segmenter wrapper for MoodPlay.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from hydra.errors import MissingConfigException
except Exception:
    MissingConfigException = None

try:
    from omegaconf.errors import ConfigAttributeError
except Exception:
    ConfigAttributeError = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    build_sam2 = None
    SAM2ImagePredictor = None


class Sam2Masker:
    def __init__(self, model_cfg: str, checkpoint_path: str, device: str = "cuda") -> None:
        if build_sam2 is None or SAM2ImagePredictor is None:
            raise ImportError("SAM2 is not available. Ensure sam2 is installed and importable.")
        sam2_model = self._build_model(model_cfg, checkpoint_path, device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def masks_from_boxes(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        *,
        multimask_output: bool = True,
        selection: str = "score",
        area_weight: float = 0.15,
    ) -> np.ndarray:
        if boxes_xyxy is None or boxes_xyxy.size == 0:
            return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_xyxy,
                multimask_output=multimask_output,
            )

        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()

        if masks.ndim == 4:
            if multimask_output and scores is not None and scores.ndim == 2:
                if selection not in {"score", "area", "score_area"}:
                    selection = "score"
                areas = masks.reshape(masks.shape[0], masks.shape[1], -1).sum(axis=2)
                if selection == "area":
                    best = np.argmax(areas, axis=1)
                elif selection == "score_area":
                    area_norm = areas / (areas.max(axis=1, keepdims=True) + 1e-6)
                    combined = scores + area_weight * area_norm
                    best = np.argmax(combined, axis=1)
                else:
                    best = np.argmax(scores, axis=1)
                masks = masks[np.arange(masks.shape[0]), best]
            else:
                masks = masks[:, 0, :, :]

        return (masks > 0).astype(bool)

    @staticmethod
    def _build_model(model_cfg: str, checkpoint_path: str, device: str):
        try:
            return build_sam2(model_cfg, checkpoint_path, device=device)
        except Exception as exc:
            cfg_path = Path(model_cfg)
            if Sam2Masker._is_missing_config(exc) and cfg_path.exists():
                return build_sam2(f"configs/sam2/{cfg_path.name}", checkpoint_path, device=device)
            if Sam2Masker._is_config_attr_error(exc) and cfg_path.exists():
                return build_sam2(f"configs/sam2/{cfg_path.name}", checkpoint_path, device=device)
            raise

    @staticmethod
    def _is_missing_config(exc: Exception) -> bool:
        if MissingConfigException is not None and isinstance(exc, MissingConfigException):
            return True
        return exc.__class__.__name__ == "MissingConfigException"

    @staticmethod
    def _is_config_attr_error(exc: Exception) -> bool:
        if ConfigAttributeError is not None and isinstance(exc, ConfigAttributeError):
            return True
        return exc.__class__.__name__ == "ConfigAttributeError"