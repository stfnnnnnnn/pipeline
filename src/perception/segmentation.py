"""
MoodPlay Video Colorizer - Preprocessing & Segmentation Pipeline

Sequential model swapping:
  Phase A: YOLO + SAM2 keyframe masks -> unload
  Phase B: CoTracker tracking -> unload
  Phase C: SAM2 final per-frame masks -> unload
"""
from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml

from .yolo_detector import YoloV8Detector
from .sam_segmenter import Sam2Masker
from .cotracker_wrapper import CoTrackerPersistentTracker
from .grounding_dino_detector import GroundingDinoDetector


@dataclass
class Instance:
    instance_id: int
    label: str
    first_frame: int
    last_frame: int
    keyframes: List[str] = field(default_factory=list)
    keyframe_indices: List[int] = field(default_factory=list)
    bboxes: Dict[int, List[float]] = field(default_factory=dict)
    centroids: Dict[int, List[float]] = field(default_factory=dict)
    mask_paths: Dict[int, str] = field(default_factory=dict)


@dataclass
class SimpleDetection:
    bbox_xyxy: np.ndarray
    label: str
    conf: float = 1.0


def _noop_progress(current: int, total: int, message: str) -> None:
    return


def _clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "not enough memory" in msg
        or "defaultcpuallocator" in msg
    )


def _mkdir(path: Path, clean: bool = False) -> None:
    if clean and path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _frame_name(idx: int) -> str:
    return f"frame_{idx:06d}.png"


def _resize_keep_aspect(frame: np.ndarray, max_h: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if h <= max_h:
        return frame, 1.0
    scale = max_h / float(h)
    new_w = int(round(w * scale))
    resized = cv2.resize(frame, (new_w, max_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _save_mask(mask: np.ndarray, png_path: Path, save_npy: bool = False) -> str:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), (mask > 0).astype(np.uint8) * 255)
    if save_npy:
        np.save(str(png_path.with_suffix(".npy")), (mask > 0).astype(np.uint8))
    return str(png_path)


def _sam2_chunk_size(model_cfg: str, device: str) -> int:
    name = Path(model_cfg).name.lower()
    chunk = 4 if "hiera_l" in name or "large" in name else 8
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        chunk = min(chunk, 2)
    return max(1, int(chunk))


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _mask_centroid(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def _clip_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _pad_box(box: np.ndarray, w: int, h: int, pad_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_w = bw * pad_ratio
    pad_h = bh * pad_ratio
    return _clip_box(np.array([x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h], dtype=np.float32), w, h)


def _refine_boxes(
    boxes: np.ndarray,
    w: int,
    h: int,
    pad_ratio: float = 0.04,
    min_side: float = 2.0,
) -> np.ndarray:
    refined = []
    for b in boxes:
        rb = _pad_box(b, w, h, pad_ratio)
        if (rb[2] - rb[0]) >= min_side and (rb[3] - rb[1]) >= min_side:
            refined.append(rb)
        else:
            refined.append(_clip_box(b, w, h))
    return np.stack(refined, axis=0).astype(np.float32)


def _postprocess_mask(mask: np.ndarray, min_area: int = 64, kernel: int = 3) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    m = (mask > 0).astype(np.uint8) * 255
    if kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = _fill_mask_holes(m)
    if int((m > 0).sum()) < int(min_area):
        return np.zeros_like(m, dtype=np.uint8)
    return (m > 0).astype(np.uint8)


def _constrain_mask_to_box(
    mask: np.ndarray,
    box: np.ndarray,
    w: int,
    h: int,
    pad_ratio: float = 0.2,
) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    m = (mask > 0)
    pb = _pad_box(np.asarray(box, dtype=np.float32), w, h, pad_ratio=pad_ratio)
    x1 = max(0, min(w - 1, int(np.floor(pb[0]))))
    y1 = max(0, min(h - 1, int(np.floor(pb[1]))))
    x2 = max(0, min(w - 1, int(np.ceil(pb[2]))))
    y2 = max(0, min(h - 1, int(np.ceil(pb[3]))))
    if x2 <= x1 or y2 <= y1:
        return np.zeros_like(mask, dtype=np.uint8)
    out = np.zeros_like(m, dtype=bool)
    out[y1 : y2 + 1, x1 : x2 + 1] = m[y1 : y2 + 1, x1 : x2 + 1]
    return out.astype(np.uint8)


def _mask_border_touch_ratio(mask: np.ndarray) -> float:
    m = (mask > 0)
    if m.size == 0:
        return 0.0
    h, w = m.shape[:2]
    if h == 1 or w == 1:
        return float(m.mean())
    border_count = int(
        m[0, :].sum()
        + m[h - 1, :].sum()
        + m[1 : h - 1, 0].sum()
        + m[1 : h - 1, w - 1].sum()
    )
    border_len = (2 * w) + (2 * max(0, h - 2))
    return float(border_count / max(1, border_len))


def _label_threshold_for(
    label: str,
    overrides: Dict[str, float],
    default_value: float,
) -> float:
    if not overrides:
        return float(default_value)
    name = str(label).lower().strip()
    if name in overrides:
        return float(overrides[name])
    matched = [float(v) for k, v in overrides.items() if str(k).strip() and str(k).lower() in name]
    if not matched:
        return float(default_value)
    return float(min(matched))


def _is_mask_plausible(
    mask: np.ndarray,
    box: np.ndarray,
    label: str,
    w: int,
    h: int,
    *,
    full_frame_reject_ratio: float,
    global_max_area_ratio: float,
    large_stuff_max_area_ratio: float,
    default_max_mask_to_box_ratio: float,
    large_stuff_max_mask_to_box_ratio: float,
    min_bbox_area_ratio_for_large_mask: float,
    border_touch_max_ratio: float,
    border_relief_bbox_area_ratio: float,
    large_stuff_labels: set[str],
    class_guard_overrides: Dict[str, Dict[str, float]],
) -> bool:
    m = (mask > 0)
    mask_area = float(m.sum())
    if mask_area <= 0:
        return False

    frame_area = float(max(1, w * h))
    area_ratio = mask_area / frame_area
    if area_ratio >= float(full_frame_reject_ratio):
        return False

    clipped = _clip_box(np.asarray(box, dtype=np.float32), w, h)
    box_w = max(1.0, float(clipped[2] - clipped[0]))
    box_h = max(1.0, float(clipped[3] - clipped[1]))
    box_area = box_w * box_h
    box_area_ratio = box_area / frame_area
    mask_to_box = mask_area / max(1.0, box_area)
    border_touch_ratio = _mask_border_touch_ratio(m)

    label_l = str(label).lower().strip()
    large_stuff = label_l in large_stuff_labels
    max_area_ratio = float(large_stuff_max_area_ratio if large_stuff else global_max_area_ratio)
    max_mask_to_box = float(
        large_stuff_max_mask_to_box_ratio if large_stuff else default_max_mask_to_box_ratio
    )
    local_min_bbox_area_ratio_for_large_mask = float(min_bbox_area_ratio_for_large_mask)
    local_border_relief_bbox_area_ratio = float(border_relief_bbox_area_ratio)

    class_cfg = class_guard_overrides.get(label_l)
    if class_cfg is None:
        for k, v in class_guard_overrides.items():
            key = str(k).lower().strip()
            if key and key in label_l:
                class_cfg = v
                break
    if isinstance(class_cfg, dict):
        if "max_area_ratio" in class_cfg:
            max_area_ratio = float(class_cfg["max_area_ratio"])
        if "max_mask_to_box_ratio" in class_cfg:
            max_mask_to_box = float(class_cfg["max_mask_to_box_ratio"])
        if "min_bbox_area_ratio_for_large_mask" in class_cfg:
            local_min_bbox_area_ratio_for_large_mask = float(class_cfg["min_bbox_area_ratio_for_large_mask"])
        if "border_relief_bbox_area_ratio" in class_cfg:
            local_border_relief_bbox_area_ratio = float(class_cfg["border_relief_bbox_area_ratio"])

    if area_ratio > max_area_ratio and box_area_ratio < local_min_bbox_area_ratio_for_large_mask:
        return False
    if mask_to_box > max_mask_to_box and box_area_ratio < local_min_bbox_area_ratio_for_large_mask:
        return False
    if (
        border_touch_ratio > float(border_touch_max_ratio)
        and box_area_ratio < local_border_relief_bbox_area_ratio
    ):
        return False
    return True


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    flood = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(m, inv)
    return filled


def _resolve_mask_overlaps(masks: np.ndarray) -> np.ndarray:
    if masks.size == 0:
        return masks
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    order = np.argsort(areas)
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    out = masks.copy().astype(bool)
    for idx in order:
        m = out[idx] & ~occupied
        out[idx] = m
        occupied |= m
    return out


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def process_video_for_segmentation(
    video_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    *,
    data_root: str = "data",
    target_height: int = 720,
    yolo_model: str = "models/checkpoints/yolo/yolo26s.pt",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.6,
    keyframe_stride: int = 12,
    tracker_points_per_instance: int = 24,
    max_frames_per_chunk: Optional[int] = None,
    max_total_frames: Optional[int] = None,
    clean_output_dirs: bool = True,
    save_npy_masks: bool = False,
    cotracker_checkpoint: str = "models/checkpoints/cotracker/cotracker3.pth",
    sam2_model_cfg: str = "configs/perception/sam2_hiera_l.yaml",
    sam2_checkpoint: str = "models/checkpoints/sam2/sam2_hiera_large.pt",
    target_labels: Optional[List[str]] = None,
    device: str = "cuda",
    yolo_config_path: str = "configs/perception/yolo26_s.yaml",
    cotracker_config_path: str = "configs/perception/cotracker3.yaml",
    grounding_dino_config_path: str = "configs/perception/grounding_dino.yaml",
) -> Dict[str, Any]:
    cb = progress_callback or _noop_progress
    runtime_device = str(device)

    # ---------- Config loading (YAML -> runtime args) ----------
    yolo_cfg = _load_yaml(yolo_config_path)
    cot_cfg = _load_yaml(cotracker_config_path)
    gdino_cfg = _load_yaml(grounding_dino_config_path)

    yolo_model = str(yolo_cfg.get("model_path", yolo_model))
    yolo_conf = float(yolo_cfg.get("conf_threshold", yolo_conf))
    yolo_iou = float(yolo_cfg.get("iou_threshold", yolo_iou))
    device = str(yolo_cfg.get("device", device))
    runtime_device = str(device)

    label_aliases_cfg = yolo_cfg.get("label_aliases", {})
    label_aliases: Dict[str, str] = {}
    if isinstance(label_aliases_cfg, dict):
        for k, v in label_aliases_cfg.items():
            label_aliases[str(k).lower()] = str(v)

    mask_cfg = yolo_cfg.get("mask_refine", {})
    if not isinstance(mask_cfg, dict):
        mask_cfg = {}
    box_pad_ratio = float(mask_cfg.get("box_pad_ratio", 0.06))
    constrain_to_box_pad_ratio = float(mask_cfg.get("constrain_to_box_pad_ratio", 0.2))
    mask_min_area = int(mask_cfg.get("min_area", 32))
    mask_kernel = int(mask_cfg.get("morph_kernel", 3))

    guard_cfg = yolo_cfg.get("mask_guard", {})
    if not isinstance(guard_cfg, dict):
        guard_cfg = {}
    full_frame_reject_ratio = float(guard_cfg.get("full_frame_reject_ratio", 0.95))
    global_max_area_ratio = float(guard_cfg.get("global_max_area_ratio", 0.72))
    large_stuff_max_area_ratio = float(guard_cfg.get("large_stuff_max_area_ratio", 0.9))
    default_max_mask_to_box_ratio = float(guard_cfg.get("default_max_mask_to_box_ratio", 2.6))
    large_stuff_max_mask_to_box_ratio = float(
        guard_cfg.get("large_stuff_max_mask_to_box_ratio", 4.4)
    )
    min_bbox_area_ratio_for_large_mask = float(
        guard_cfg.get("min_bbox_area_ratio_for_large_mask", 0.18)
    )
    border_touch_max_ratio = float(guard_cfg.get("border_touch_max_ratio", 0.95))
    border_relief_bbox_area_ratio = float(guard_cfg.get("border_relief_bbox_area_ratio", 0.42))
    class_guard_overrides_cfg = guard_cfg.get("class_overrides", {})
    class_guard_overrides: Dict[str, Dict[str, float]] = {}
    if isinstance(class_guard_overrides_cfg, dict):
        for cls_name, cls_cfg in class_guard_overrides_cfg.items():
            if not isinstance(cls_cfg, dict):
                continue
            clean_cfg: Dict[str, float] = {}
            for k, v in cls_cfg.items():
                if isinstance(v, (int, float)):
                    clean_cfg[str(k)] = float(v)
            if clean_cfg:
                class_guard_overrides[str(cls_name).lower().strip()] = clean_cfg
    large_stuff_labels_cfg = guard_cfg.get(
        "large_stuff_labels",
        ["road", "street", "sidewalk", "sky", "building", "house", "wall", "mountain", "water"],
    )
    if isinstance(large_stuff_labels_cfg, list):
        large_stuff_labels = {str(x).lower().strip() for x in large_stuff_labels_cfg if str(x).strip()}
    else:
        large_stuff_labels = {
            "road",
            "street",
            "sidewalk",
            "sky",
            "building",
            "house",
            "wall",
            "mountain",
            "water",
        }

    scene_cfg = yolo_cfg.get("scene_fallback", {})
    scene_enabled = False
    scene_track = False
    scene_label = "scene"
    if isinstance(scene_cfg, dict):
        scene_enabled = bool(scene_cfg.get("enabled", False))
        scene_track = bool(scene_cfg.get("track", False))
        scene_label = str(scene_cfg.get("label", scene_label))

    if target_labels is None:
        cfg_labels = yolo_cfg.get("target_labels", None)
        if isinstance(cfg_labels, list):
            target_labels = [str(x) for x in cfg_labels]

    gdino_detector = GroundingDinoDetector.from_config(grounding_dino_config_path, gdino_cfg)
    gdino_box_pad_ratio = float(gdino_cfg.get("box_pad_ratio", box_pad_ratio))
    gdino_mask_selection = str(gdino_cfg.get("mask_selection", "area"))
    gdino_area_weight = float(gdino_cfg.get("area_weight", 0.4))
    gdino_score_threshold_default = float(gdino_cfg.get("score_threshold", 0.0))
    gdino_score_thresholds_cfg = gdino_cfg.get("score_thresholds", {})
    gdino_score_thresholds: Dict[str, float] = {}
    if isinstance(gdino_score_thresholds_cfg, dict):
        for k, v in gdino_score_thresholds_cfg.items():
            if isinstance(v, (int, float)):
                gdino_score_thresholds[str(k).lower().strip()] = float(v)

    cotracker_checkpoint = str(cot_cfg.get("checkpoint", cotracker_checkpoint))
    tracker_points_per_instance = int(cot_cfg.get("points_per_instance", tracker_points_per_instance))
    keyframe_stride = int(cot_cfg.get("keyframe_stride", keyframe_stride))

    max_frames_per_chunk_cfg = cot_cfg.get("max_frames_per_chunk", None)
    if isinstance(max_frames_per_chunk_cfg, (int, float)):
        max_frames_per_chunk_cfg = int(max_frames_per_chunk_cfg)
        if max_frames_per_chunk_cfg <= 0:
            max_frames_per_chunk_cfg = None
    else:
        if max_frames_per_chunk_cfg is not None:
            max_frames_per_chunk_cfg = None

    if max_frames_per_chunk is None:
        max_frames_per_chunk = max_frames_per_chunk_cfg
    elif isinstance(max_frames_per_chunk, (int, float)):
        max_frames_per_chunk = int(max_frames_per_chunk)
        if max_frames_per_chunk <= 0:
            max_frames_per_chunk = None
    else:
        max_frames_per_chunk = None

    max_total_frames_cfg = cot_cfg.get("max_total_frames", None)
    if isinstance(max_total_frames_cfg, (int, float)):
        max_total_frames_cfg = int(max_total_frames_cfg)
        if max_total_frames_cfg <= 0:
            max_total_frames_cfg = None
    else:
        if max_total_frames_cfg is not None:
            max_total_frames_cfg = None

    if max_total_frames is None:
        max_total_frames = max_total_frames_cfg
    elif isinstance(max_total_frames, (int, float)):
        max_total_frames = int(max_total_frames)
        if max_total_frames <= 0:
            max_total_frames = None
    else:
        max_total_frames = None
    device = str(cot_cfg.get("device", device))
    runtime_device = str(device)

    if max_frames_per_chunk is None and not torch.cuda.is_available():
        max_frames_per_chunk = 8

    # COCO-safe default
    if target_labels is None:
        target_labels = ["person", "tie", "handbag", "backpack", "umbrella", "suitcase"]

    def _run_once(max_h: int) -> Dict[str, Any]:
        nonlocal runtime_device
        data_root_p = Path(data_root)
        frames_dir = data_root_p / "extracted_frames"
        masks_root = data_root_p / "intermediate" / "masks"
        meta_dir = data_root_p / "intermediate" / "metadata"

        _mkdir(frames_dir, clean=clean_output_dirs)
        _mkdir(masks_root, clean=clean_output_dirs)
        _mkdir(meta_dir, clean=False)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        est_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        frames_bgr: List[np.ndarray] = []
        frame_paths: List[str] = []

        total_hint = est_total
        if max_total_frames is not None:
            total_hint = min(total_hint, max_total_frames)
        cb(0, max(1, total_hint), f"Extracting frames at <= {max_h}p...")
        idx = 0
        while True:
            if max_total_frames is not None and idx >= max_total_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            resized, _scale = _resize_keep_aspect(frame, max_h=max_h)
            out = frames_dir / _frame_name(idx)
            cv2.imwrite(str(out), resized)
            frames_bgr.append(resized)
            frame_paths.append(str(out))
            idx += 1
            cb(min(idx, max(1, total_hint)), max(1, total_hint), f"Extracted frame {idx}")
        cap.release()

        total_frames = len(frames_bgr)
        if total_frames == 0:
            raise RuntimeError("No frames extracted from video.")

        label_set = {x.lower() for x in target_labels}
        if label_aliases:
            label_set |= set(label_aliases.keys())
        instances: Dict[int, Instance] = {}
        next_instance_id = 1

        keyframes = list(range(0, total_frames, max(1, keyframe_stride)))
        if (total_frames - 1) not in keyframes:
            keyframes.append(total_frames - 1)

        prev_key_bboxes: Dict[int, np.ndarray] = {}
        instance_seed_points: Dict[int, np.ndarray] = {}
        scene_instance_id: Optional[int] = None

        sam2_chunk = _sam2_chunk_size(sam2_model_cfg, device)

        # ---------- Phase A ----------
        cb(0, 1, "Phase A: Loading YOLO + SAM2...")
        detector = YoloV8Detector(model=yolo_model, device=device)
        sam2 = Sam2Masker(model_cfg=sam2_model_cfg, checkpoint_path=sam2_checkpoint, device=device)

        cb(0, len(keyframes), "Phase A: keyframe detection + masks...")
        for i, fidx in enumerate(keyframes):
            frame = frames_bgr[fidx]
            yolo_dets = detector.detect(frame, conf=yolo_conf, iou=yolo_iou, target_labels=label_set)
            gdino_dets: List[SimpleDetection] = []
            if gdino_detector is not None:
                gdino_raw = gdino_detector.detect(frame_paths[fidx])
                for gd in gdino_raw:
                    gdino_dets.append(
                        SimpleDetection(
                            bbox_xyxy=gd["bbox_xyxy"],
                            label=gd["label"],
                            conf=float(gd.get("score", 1.0)),
                        )
                    )

            if not yolo_dets and not gdino_dets:
                if scene_enabled:
                    h, w = frame.shape[:2]
                    scene_box = np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)
                    scene_masks = sam2.masks_from_boxes(
                        frame,
                        scene_box,
                        multimask_output=True,
                        selection="score_area",
                        area_weight=0.15,
                    )
                    if scene_masks.size == 0:
                        cb(i + 1, len(keyframes), f"Keyframe {fidx}: scene fallback empty")
                        continue

                    scene_mask = _postprocess_mask(
                        scene_masks[0].astype(np.uint8),
                        min_area=mask_min_area,
                        kernel=mask_kernel,
                    ).astype(bool)
                    if not np.any(scene_mask):
                        cb(i + 1, len(keyframes), f"Keyframe {fidx}: scene fallback empty")
                        continue

                    if scene_instance_id is None:
                        scene_instance_id = next_instance_id
                        next_instance_id += 1
                        instances[scene_instance_id] = Instance(
                            instance_id=scene_instance_id,
                            label=scene_label,
                            first_frame=fidx,
                            last_frame=fidx,
                        )

                    inst = instances[scene_instance_id]
                    inst.first_frame = min(inst.first_frame, fidx)
                    inst.last_frame = max(inst.last_frame, fidx)
                    inst.keyframes.append(frame_paths[fidx])
                    inst.keyframe_indices.append(fidx)
                    inst.bboxes[fidx] = [float(v) for v in scene_box[0].tolist()]
                    ctr = _mask_centroid(scene_mask.astype(np.uint8))
                    if ctr is not None:
                        inst.centroids[fidx] = ctr

                    frame_mask_dir = masks_root / f"frame_{fidx:06d}"
                    inst.mask_paths[fidx] = _save_mask(
                        scene_mask.astype(np.uint8),
                        frame_mask_dir / f"instance_{scene_instance_id:04d}.png",
                        save_npy=save_npy_masks,
                    )

                    if scene_track and scene_instance_id not in instance_seed_points:
                        pts = CoTrackerPersistentTracker.sample_points_from_mask(
                            scene_mask.astype(np.uint8),
                            k=min(16, tracker_points_per_instance),
                        )
                        if pts.shape[0] == 0:
                            pts = np.array(
                                [[w / 2.0, h / 2.0], [0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0]],
                                dtype=np.float32,
                            )
                        instance_seed_points[scene_instance_id] = pts

                    cb(i + 1, len(keyframes), f"Keyframe {fidx}: scene fallback mask")
                    continue

                cb(i + 1, len(keyframes), f"Keyframe {fidx}: no detections")
                continue

            if label_aliases:
                for det in yolo_dets:
                    key = det.label.lower()
                    if key in label_aliases:
                        det.label = label_aliases[key]
                for det in gdino_dets:
                    key = det.label.lower()
                    if key in label_aliases:
                        det.label = label_aliases[key]

            def _run_sam2_for_boxes(
                boxes_in: np.ndarray,
                pad_ratio_in: float,
                selection_in: str,
                area_weight_in: float,
            ) -> np.ndarray:
                if boxes_in.size == 0:
                    return np.zeros((0, frame.shape[0], frame.shape[1]), dtype=bool)
                h, w = frame.shape[:2]
                refined = _refine_boxes(boxes_in, w, h, pad_ratio=pad_ratio_in)
                masks_out: List[np.ndarray] = []
                for s in range(0, len(refined), sam2_chunk):
                    chunk_boxes = refined[s : s + sam2_chunk]
                    chunk_masks = sam2.masks_from_boxes(
                        frame,
                        chunk_boxes,
                        multimask_output=True,
                        selection=selection_in,
                        area_weight=area_weight_in,
                    )
                    if chunk_masks.size > 0:
                        chunk_masks = np.stack(
                            [
                                _postprocess_mask(
                                    m.astype(np.uint8),
                                    min_area=mask_min_area,
                                    kernel=mask_kernel,
                                )
                                for m in chunk_masks
                            ],
                            axis=0,
                        ).astype(bool)
                        chunk_masks = np.stack(
                            [
                                _constrain_mask_to_box(
                                    m.astype(np.uint8),
                                    b,
                                    w,
                                    h,
                                    pad_ratio=constrain_to_box_pad_ratio,
                                )
                                for m, b in zip(chunk_masks, chunk_boxes)
                            ],
                            axis=0,
                        ).astype(bool)
                        chunk_masks = _resolve_mask_overlaps(chunk_masks)
                    masks_out.append(chunk_masks)
                if masks_out:
                    return np.concatenate(masks_out, axis=0)
                return np.zeros((0, frame.shape[0], frame.shape[1]), dtype=bool)

            dets: List[SimpleDetection | Any] = []
            masks_list: List[np.ndarray] = []
            if yolo_dets:
                yolo_boxes = np.stack([d.bbox_xyxy for d in yolo_dets], axis=0).astype(np.float32)
                yolo_masks = _run_sam2_for_boxes(yolo_boxes, box_pad_ratio, "score_area", 0.15)
                dets.extend(yolo_dets)
                masks_list.extend(list(yolo_masks))
            if gdino_dets:
                gdino_boxes = np.stack([d.bbox_xyxy for d in gdino_dets], axis=0).astype(np.float32)
                gdino_masks = _run_sam2_for_boxes(
                    gdino_boxes,
                    gdino_box_pad_ratio,
                    gdino_mask_selection,
                    gdino_area_weight,
                )
                dets.extend(gdino_dets)
                masks_list.extend(list(gdino_masks))

            h, w = frame.shape[:2]
            if masks_list:
                masks = np.stack(masks_list, axis=0)
            else:
                masks = np.zeros((0, h, w), dtype=bool)

            assigned = []
            curr_id_to_bbox: Dict[int, np.ndarray] = {}

            for det, mask in zip(dets, masks):
                if isinstance(det, SimpleDetection):
                    gdino_min_score = _label_threshold_for(
                        det.label,
                        gdino_score_thresholds,
                        gdino_score_threshold_default,
                    )
                    if float(det.conf) < float(gdino_min_score):
                        continue
                if not _is_mask_plausible(
                    mask.astype(np.uint8),
                    det.bbox_xyxy,
                    det.label,
                    w,
                    h,
                    full_frame_reject_ratio=full_frame_reject_ratio,
                    global_max_area_ratio=global_max_area_ratio,
                    large_stuff_max_area_ratio=large_stuff_max_area_ratio,
                    default_max_mask_to_box_ratio=default_max_mask_to_box_ratio,
                    large_stuff_max_mask_to_box_ratio=large_stuff_max_mask_to_box_ratio,
                    min_bbox_area_ratio_for_large_mask=min_bbox_area_ratio_for_large_mask,
                    border_touch_max_ratio=border_touch_max_ratio,
                    border_relief_bbox_area_ratio=border_relief_bbox_area_ratio,
                    large_stuff_labels=large_stuff_labels,
                    class_guard_overrides=class_guard_overrides,
                ):
                    continue

                best_id, best_score = None, 0.0
                for iid, pb in prev_key_bboxes.items():
                    if iid in instances and instances[iid].label != det.label:
                        continue
                    s = _bbox_iou(det.bbox_xyxy, pb)
                    if s > best_score:
                        best_score, best_id = s, iid

                if best_id is not None and best_score >= 0.2 and best_id not in assigned:
                    iid = best_id
                else:
                    iid = next_instance_id
                    next_instance_id += 1
                    instances[iid] = Instance(
                        instance_id=iid,
                        label=det.label,
                        first_frame=fidx,
                        last_frame=fidx,
                    )

                inst = instances[iid]
                inst.last_frame = max(inst.last_frame, fidx)
                inst.keyframes.append(frame_paths[fidx])
                inst.keyframe_indices.append(fidx)
                inst.bboxes[fidx] = [float(v) for v in det.bbox_xyxy.tolist()]
                ctr = _mask_centroid(mask.astype(np.uint8))
                if ctr is not None:
                    inst.centroids[fidx] = ctr

                frame_mask_dir = masks_root / f"frame_{fidx:06d}"
                inst.mask_paths[fidx] = _save_mask(
                    mask.astype(np.uint8),
                    frame_mask_dir / f"instance_{iid:04d}.png",
                    save_npy=save_npy_masks,
                )

                if iid not in instance_seed_points:
                    pts = CoTrackerPersistentTracker.sample_points_from_mask(
                        mask.astype(np.uint8), k=tracker_points_per_instance
                    )
                    if pts.shape[0] == 0:
                        x1, y1, x2, y2 = det.bbox_xyxy
                        pts = np.array(
                            [[(x1 + x2) / 2, (y1 + y2) / 2], [x1, y1], [x2, y1], [x1, y2], [x2, y2]],
                            dtype=np.float32,
                        )
                    instance_seed_points[iid] = pts

                assigned.append(iid)
                curr_id_to_bbox[iid] = det.bbox_xyxy.copy()

            prev_key_bboxes = curr_id_to_bbox
            cb(i + 1, len(keyframes), f"Keyframe {fidx}: {len(assigned)} detections")

        del detector, sam2
        _clear_gpu()
        cb(1, 1, "Phase A complete")

        if not instances:
            meta_path = meta_dir / "segmentation_tracking_metadata.json"
            meta = {
                "video_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "resize_height": max_h,
                "max_total_frames": max_total_frames,
                "instances": {},
                "frames_dir": str(frames_dir),
                "masks_dir": str(masks_root),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            return {
                "video_path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "frame_paths": frame_paths,
                "instances": [],
                "mask_root": str(masks_root),
                "metadata_path": str(meta_path),
                "runtime_device": runtime_device,
            }

        # ---------- Phase B ----------
        cb(0, 1, "Phase B: Loading CoTracker...")
        tracker = CoTrackerPersistentTracker(
            checkpoint_path=cotracker_checkpoint,
            device=device,
            max_frames_per_chunk=max_frames_per_chunk,
        )
        tracked_points = tracker.track_points(frames_bgr, instance_seed_points)
        runtime_device = str(getattr(tracker, "device", runtime_device))
        del tracker
        _clear_gpu()
        cb(1, 1, "Phase B complete")

        H, W = frames_bgr[0].shape[:2]
        tracked_bboxes: Dict[int, Dict[int, List[float]]] = {iid: {} for iid in instances.keys()}
        for iid, tr in tracked_points.items():  # tr [T,K,2]
            for t in range(tr.shape[0]):
                pts = tr[t]
                xs = np.clip(pts[:, 0], 0, W - 1)
                ys = np.clip(pts[:, 1], 0, H - 1)
                if xs.size == 0:
                    continue
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue
                tracked_bboxes[iid][t] = [x1, y1, x2, y2]

        # ---------- Phase C ----------
        cb(0, 1, "Phase C: Loading SAM2...")
        sam2_final = Sam2Masker(model_cfg=sam2_model_cfg, checkpoint_path=sam2_checkpoint, device=device)

        cb(0, total_frames, "Phase C: generating masks on all frames...")
        for t in range(total_frames):
            frame = frames_bgr[t]
            frame_iids, frame_boxes = [], []

            for iid in instances.keys():
                if t in instances[iid].mask_paths:
                    continue
                if t in tracked_bboxes.get(iid, {}):
                    frame_iids.append(iid)
                    frame_boxes.append(tracked_bboxes[iid][t])

            if not frame_boxes:
                cb(t + 1, total_frames, f"Frame {t}: no masks")
                continue

            chunk = sam2_chunk
            kept_in_frame = 0
            for s in range(0, len(frame_boxes), chunk):
                chunk_iids = frame_iids[s : s + chunk]
                chunk_boxes = np.array(frame_boxes[s : s + chunk], dtype=np.float32)
                h, w = frame.shape[:2]
                refined_boxes = _refine_boxes(chunk_boxes, w, h, pad_ratio=box_pad_ratio)
                chunk_masks = sam2_final.masks_from_boxes(
                    frame,
                    refined_boxes,
                    multimask_output=True,
                    selection="score_area",
                    area_weight=0.15,
                )
                if chunk_masks.size > 0:
                    chunk_masks = np.stack(
                        [_postprocess_mask(m.astype(np.uint8), min_area=mask_min_area, kernel=mask_kernel) for m in chunk_masks],
                        axis=0,
                    ).astype(bool)
                    chunk_masks = np.stack(
                        [
                            _constrain_mask_to_box(
                                m.astype(np.uint8),
                                b,
                                w,
                                h,
                                pad_ratio=constrain_to_box_pad_ratio,
                            )
                            for m, b in zip(chunk_masks, refined_boxes)
                        ],
                        axis=0,
                    ).astype(bool)
                    chunk_masks = _resolve_mask_overlaps(chunk_masks)

                kept_in_chunk = 0
                for iid, box, mask in zip(chunk_iids, refined_boxes, chunk_masks):
                    inst = instances[iid]
                    if not _is_mask_plausible(
                        mask.astype(np.uint8),
                        box,
                        inst.label,
                        w,
                        h,
                        full_frame_reject_ratio=full_frame_reject_ratio,
                        global_max_area_ratio=global_max_area_ratio,
                        large_stuff_max_area_ratio=large_stuff_max_area_ratio,
                        default_max_mask_to_box_ratio=default_max_mask_to_box_ratio,
                        large_stuff_max_mask_to_box_ratio=large_stuff_max_mask_to_box_ratio,
                        min_bbox_area_ratio_for_large_mask=min_bbox_area_ratio_for_large_mask,
                        border_touch_max_ratio=border_touch_max_ratio,
                        border_relief_bbox_area_ratio=border_relief_bbox_area_ratio,
                        large_stuff_labels=large_stuff_labels,
                        class_guard_overrides=class_guard_overrides,
                    ):
                        continue

                    inst.bboxes[t] = [float(v) for v in box.tolist()]
                    ctr = _mask_centroid(mask.astype(np.uint8))
                    if ctr is not None:
                        inst.centroids[t] = ctr
                    inst.first_frame = min(inst.first_frame, t)
                    inst.last_frame = max(inst.last_frame, t)

                    frame_mask_dir = masks_root / f"frame_{t:06d}"
                    inst.mask_paths[t] = _save_mask(
                        mask.astype(np.uint8),
                        frame_mask_dir / f"instance_{iid:04d}.png",
                        save_npy=save_npy_masks,
                    )
                    kept_in_chunk += 1
                kept_in_frame += kept_in_chunk

            cb(t + 1, total_frames, f"Frame {t}: generated {kept_in_frame} masks")

        del sam2_final
        _clear_gpu()
        cb(1, 1, "Phase C complete")

        meta = {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "resize_height": max_h,
            "max_total_frames": max_total_frames,
            "frames_dir": str(frames_dir),
            "masks_dir": str(masks_root),
            "instances": {
                str(iid): {
                    "id": inst.instance_id,
                    "label": inst.label,
                    "first_frame": inst.first_frame,
                    "last_frame": inst.last_frame,
                    "keyframes": inst.keyframes,
                    "keyframe_indices": inst.keyframe_indices,
                    "bboxes": {str(k): v for k, v in inst.bboxes.items()},
                    "centroids": {str(k): v for k, v in inst.centroids.items()},
                    "mask_paths": {str(k): v for k, v in inst.mask_paths.items()},
                }
                for iid, inst in instances.items()
            },
        }
        meta_path = meta_dir / "segmentation_tracking_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        ui_instances = []
        for iid in sorted(instances.keys()):
            inst = instances[iid]
            sample_frames = sorted(inst.mask_paths.keys())[:6]
            ui_instances.append(
                {
                    "id": inst.instance_id,
                    "label": inst.label,
                    "first_frame": inst.first_frame,
                    "last_frame": inst.last_frame,
                    "keyframes": inst.keyframes,
                    "sample_mask_paths": [inst.mask_paths[k] for k in sample_frames],
                    "num_masks": len(inst.mask_paths),
                }
            )

        cb(1, 1, "Segmentation pipeline complete.")
        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "frame_paths": frame_paths,
            "instances": ui_instances,
            "mask_root": str(masks_root),
            "metadata_path": str(meta_path),
            "runtime_device": runtime_device,
        }

    retry_heights = [int(target_height)]
    if target_height > 480:
        retry_heights.append(480)
    if target_height > 360:
        retry_heights.append(360)

    last_err: Optional[RuntimeError] = None
    for i, max_h in enumerate(retry_heights):
        try:
            return _run_once(max_h)
        except RuntimeError as e:
            if not _is_oom_error(e) or i == (len(retry_heights) - 1):
                raise
            last_err = e
            _clear_gpu()
            cb(0, 1, f"OOM at {max_h}p. Retrying at {retry_heights[i + 1]}p...")

    if last_err is not None:
        raise last_err
    raise RuntimeError("Segmentation failed without an exception")