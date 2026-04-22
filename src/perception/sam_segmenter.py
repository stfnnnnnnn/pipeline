"""
SAM2 image segmenter wrapper for MoodPlay.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
import torch

try:
    from hydra.errors import MissingConfigException
except ImportError:
    MissingConfigException = None

try:
    from omegaconf.errors import ConfigAttributeError
except ImportError:
    ConfigAttributeError = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2 = None
    SAM2ImagePredictor = None


class Sam2Masker:
    def __init__(self, model_cfg: str, checkpoint_path: str, device: str = "cuda") -> None:
        if build_sam2 is None or SAM2ImagePredictor is None:
            raise ImportError("SAM2 is not available. Ensure sam2 is installed and importable.")
        sam2_model = self._build_model(model_cfg, checkpoint_path, device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = str(device)

    def masks_from_boxes(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        *,
        multimask_output: bool = True,
        selection: str = "score",
        area_weight: float = 0.15,
        negative_points_by_box: Optional[List[np.ndarray]] = None,
        labels: Optional[List[str]] = None,
        large_stuff_labels: Optional[List[str]] = None,
        max_mask_to_box_ratio: float = 6.0,
        max_frame_area_ratio: float = 0.96,
        oversize_penalty: float = 2.5,
    ) -> np.ndarray:
        if boxes_xyxy is None or boxes_xyxy.size == 0:
            return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        has_negative = bool(negative_points_by_box and len(negative_points_by_box) == len(boxes_xyxy))

        if has_negative:
            masks, scores = self._predict_with_boxwise_prompts(
                boxes_xyxy=boxes_xyxy,
                negative_points_by_box=negative_points_by_box,
                multimask_output=multimask_output,
            )
        else:
            masks, scores = self._predict_batched(
                boxes_xyxy=boxes_xyxy,
                multimask_output=multimask_output,
            )

        masks = self._to_numpy(masks)
        scores = self._to_numpy(scores)

        if masks.ndim == 4:
            masks = self._select_multimask_candidates(
                masks=masks,
                scores=scores,
                boxes_xyxy=boxes_xyxy,
                selection=selection,
                area_weight=area_weight,
                labels=labels,
                large_stuff_labels=large_stuff_labels,
                max_mask_to_box_ratio=max_mask_to_box_ratio,
                max_frame_area_ratio=max_frame_area_ratio,
                oversize_penalty=oversize_penalty,
            )

        # Keep failed masks empty; never substitute detector boxes as pseudo-masks.
        return (masks > 0).astype(bool)

    def appearance_embeddings_from_masks(
        self,
        image_bgr: np.ndarray,
        masks: np.ndarray,
        *,
        assume_image_is_set: bool = False,
    ) -> np.ndarray:
        if masks is None or masks.size == 0:
            return np.zeros((0, 1), dtype=np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if not assume_image_is_set:
            self.predictor.set_image(image_rgb)

        feat = self._get_internal_feature_map()
        if feat is None or feat.size == 0:
            return self._rgb_fallback_embeddings(image_rgb, masks)

        c, fh, fw = feat.shape
        out: List[np.ndarray] = []
        for m in masks:
            m_small = cv2.resize((m > 0).astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
            if not np.any(m_small):
                out.append(np.zeros((c,), dtype=np.float32))
                continue
            vec = feat[:, m_small].mean(axis=1)
            denom = float(np.linalg.norm(vec) + 1e-6)
            out.append((vec / denom).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)

    def _predict_batched(self, boxes_xyxy: np.ndarray, multimask_output: bool) -> tuple[Any, Any]:
        with torch.inference_mode():
            if self.device.startswith("cuda") and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    masks, scores, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes_xyxy,
                        multimask_output=multimask_output,
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_xyxy,
                    multimask_output=multimask_output,
                )
        return masks, scores

    def _predict_with_boxwise_prompts(
        self,
        *,
        boxes_xyxy: np.ndarray,
        negative_points_by_box: List[np.ndarray],
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        masks_all: List[np.ndarray] = []
        scores_all: List[np.ndarray] = []

        for i, box in enumerate(boxes_xyxy):
            neg = np.asarray(negative_points_by_box[i], dtype=np.float32) if i < len(negative_points_by_box) else np.zeros((0, 2), dtype=np.float32)
            point_coords = None
            point_labels = None
            if neg.size > 0:
                point_coords = neg.reshape(-1, 2)
                point_labels = np.zeros((point_coords.shape[0],), dtype=np.int32)

            with torch.inference_mode():
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        m, s, _ = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=box.reshape(1, 4),
                            multimask_output=multimask_output,
                        )
                else:
                    m, s, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box.reshape(1, 4),
                        multimask_output=multimask_output,
                    )

            m = self._to_numpy(m)
            s = self._to_numpy(s)

            if m.ndim == 4:
                m = m[0]
            if s.ndim == 2:
                s = s[0]

            masks_all.append(m.astype(np.float32))
            scores_all.append(s.astype(np.float32))

        return np.stack(masks_all, axis=0), np.stack(scores_all, axis=0)

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if x is None:
            return np.array([])
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _select_multimask_candidates(
        self,
        *,
        masks: np.ndarray,
        scores: np.ndarray,
        boxes_xyxy: np.ndarray,
        selection: str,
        area_weight: float,
        labels: Optional[List[str]],
        large_stuff_labels: Optional[List[str]],
        max_mask_to_box_ratio: float,
        max_frame_area_ratio: float,
        oversize_penalty: float,
    ) -> np.ndarray:
        if selection not in {"score", "area", "score_area"}:
            selection = "score"

        if scores is None or scores.ndim != 2:
            return masks[:, 0, :, :]

        n, _, h, w = masks.shape
        areas = masks.reshape(n, masks.shape[1], -1).sum(axis=2).astype(np.float32)
        box_w = np.maximum(1.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0])
        box_h = np.maximum(1.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        box_area = (box_w * box_h).reshape(-1, 1)
        frame_area = float(max(1, h * w))

        ratio_to_box = areas / (box_area + 1e-6)
        ratio_to_frame = areas / frame_area
        area_norm = areas / (areas.max(axis=1, keepdims=True) + 1e-6)

        if selection == "area":
            base = area_norm
        elif selection == "score_area":
            base = scores + area_weight * area_norm
        else:
            base = scores

        oversize = np.maximum(0.0, ratio_to_box - float(max_mask_to_box_ratio))
        frame_oversize = np.maximum(0.0, ratio_to_frame - float(max_frame_area_ratio))
        penalties = float(oversize_penalty) * (oversize + frame_oversize)

        default_large_stuff = {"sky", "building", "road", "street", "wall", "mountain", "tree canopy"}
        configured_large_stuff = (
            {" ".join(str(x).strip().lower().split()) for x in large_stuff_labels if str(x).strip()}
            if isinstance(large_stuff_labels, list)
            else default_large_stuff
        )
        if labels is not None:
            for i in range(min(len(labels), n)):
                lbl = " ".join(str(labels[i]).strip().lower().split())
                if lbl in configured_large_stuff:
                    penalties[i, :] = 0.0

        combined = base - penalties

        valid = (ratio_to_box <= float(max_mask_to_box_ratio) * 1.5) & (ratio_to_frame <= 1.0)
        if labels is not None:
            for i in range(min(len(labels), n)):
                lbl = " ".join(str(labels[i]).strip().lower().split())
                if lbl in configured_large_stuff:
                    valid[i, :] = ratio_to_frame[i, :] <= 1.0
        combined = np.where(valid, combined, -1e9)

        best = np.argmax(combined, axis=1)
        return masks[np.arange(n), best]

    def _get_internal_feature_map(self) -> Optional[np.ndarray]:
        # Prefer SAM2 encoder activations. This keeps ReID lightweight with no extra network.
        candidates: List[Any] = []
        for attr in ("_features", "features", "_image_features"):
            if hasattr(self.predictor, attr):
                candidates.append(getattr(self.predictor, attr))

        for item in candidates:
            feat = self._extract_feature_tensor(item)
            if feat is not None:
                return feat

        if hasattr(self.predictor, "model"):
            model = getattr(self.predictor, "model")
            for attr in ("_features", "features"):
                if hasattr(model, attr):
                    feat = self._extract_feature_tensor(getattr(model, attr))
                    if feat is not None:
                        return feat
        return None

    def _extract_feature_tensor(self, obj: Any) -> Optional[np.ndarray]:
        tensor = None
        if obj is None:
            return None
        if isinstance(obj, dict):
            for key in ("image_embed", "image_embeddings", "image_embedding", "low_res_feats", "feat"):
                if key in obj:
                    tensor = obj[key]
                    break
            if tensor is None:
                for v in obj.values():
                    if hasattr(v, "shape"):
                        tensor = v
                        break
        elif isinstance(obj, (list, tuple)) and obj:
            for v in obj:
                if hasattr(v, "shape"):
                    tensor = v
                    break
        else:
            tensor = obj

        if tensor is None:
            return None
        arr = self._to_numpy(tensor)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        if arr.shape[0] > 8 and arr.shape[1] < arr.shape[0] and arr.shape[2] < arr.shape[0]:
            feat = arr
        else:
            feat = np.moveaxis(arr, -1, 0)
        return feat.astype(np.float32)

    @staticmethod
    def _rgb_fallback_embeddings(image_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        out: List[np.ndarray] = []
        img = image_rgb.astype(np.float32) / 255.0
        for m in masks:
            fg = img[m > 0]
            if fg.size == 0:
                out.append(np.zeros((6,), dtype=np.float32))
                continue
            mean = fg.mean(axis=0)
            std = fg.std(axis=0)
            vec = np.concatenate([mean, std], axis=0)
            denom = float(np.linalg.norm(vec) + 1e-6)
            out.append((vec / denom).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)

    @staticmethod
    def _build_model(model_cfg: str, checkpoint_path: str, device: str):
        if build_sam2 is None:
            raise ImportError("SAM2 builder is unavailable")
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
