"""
MoodPlay Video Colorizer - Preprocessing & Segmentation Pipeline

Sequential model swapping:
  Phase A: Detection + SAM2 on keyframes
  Phase B: CoTracker tracking
  Phase C: XMem/SAM2 per-frame mask completion with selective re-detection
"""
from __future__ import annotations

import gc
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from .cotracker_wrapper import CoTrackerPersistentTracker
from .grounding_dino_detector import GroundingDinoDetector
from .sam_segmenter import Sam2Masker
from .xmem_wrapper import XMemMaskPropagator, XMemRuntimeConfig


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
    score: float
    source: str


@dataclass
class TrackMemory:
    instance_id: int
    label: str
    bbox_xyxy: np.ndarray
    appearance: np.ndarray
    velocity_xy: np.ndarray
    last_seen: int
    miss_count: int = 0


@dataclass
class AssociationParams:
    weight_iou: float = 0.70
    weight_app: float = 0.10
    weight_motion: float = 0.20
    min_iou_gate: float = 0.05
    min_app_gate: float = 0.10
    max_cost: float = 0.92
    max_missed_keyframes: int = 10
    embedding_dim: int = 64


@dataclass
class RecoveryParams:
    enable_selective_redetect: bool = True
    missing_ratio_trigger: float = 0.40
    overlap_trigger: float = 0.35
    min_redetect_gap: int = 8


SEMANTIC_DEPTH: Dict[str, int] = {
    "sky": 0,
    "skyline horizon": 0,
    "building": 1,
    "building facade": 1,
    "wall": 1,
    "road": 2,
    "road surface": 2,
    "sidewalk": 3,
    "sidewalk or pavement": 3,
    "tree": 4,
    "street sign": 4,
    "traffic light": 4,
    "car": 5,
    "bus": 5,
    "truck": 5,
    "person": 6,
    "pedestrian": 6,
    "handbag": 7,
    "backpack": 7,
}


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
    out = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(png_path), out)
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


def _bbox_iof(bg_box: np.ndarray, fg_box: np.ndarray) -> float:
    """Intersection over foreground area (small object coverage by larger box)."""
    ax1, ay1, ax2, ay2 = bg_box
    bx1, by1, bx2, by2 = fg_box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_fg = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / area_fg) if area_fg > 0 else 0.0


def _bbox_iom(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over minimum area; robust when one detector pads boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(area_a, area_b)
    return float(inter / denom) if denom > 0 else 0.0


def _bbox_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _bbox_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def _mask_centroid(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def _mask_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


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

    # Mild edge antialias before thresholding back to binary for stable downstream blending.
    m = cv2.GaussianBlur(m, (3, 3), 0)
    return (m > 127).astype(np.uint8)


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


def _safe_float(data: Dict[str, Any], key: str, default: float) -> float:
    v = data.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _project_embedding(vec: np.ndarray, dim: int) -> np.ndarray:
    if vec is None or vec.size == 0:
        return np.zeros((dim,), dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] >= dim:
        out = v[:dim]
    else:
        out = np.pad(v, (0, dim - v.shape[0]), mode="constant")
    n = float(np.linalg.norm(out) + 1e-6)
    return (out / n).astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b) / denom)


def _fuse_detections(detections: List[SimpleDetection], iom_thresh: float = 0.85) -> List[SimpleDetection]:
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: float(d.score), reverse=True)
    fused: List[SimpleDetection] = []

    for det in sorted_dets:
        duplicate = False
        det_label = _normalize_label_token(det.label)
        det_depth = SEMANTIC_DEPTH.get(det_label, 2)

        for kept in fused:
            kept_label = _normalize_label_token(kept.label)
            kept_depth = SEMANTIC_DEPTH.get(kept_label, 2)
            if det_depth != kept_depth:
                continue

            if det_label == kept_label:
                # SAME EXACT LABEL: aggressively suppress nested duplicates.
                if _bbox_iom(det.bbox_xyxy, kept.bbox_xyxy) >= iom_thresh:
                    duplicate = True
                    break
            else:
                # DIFFERENT LABELS on same depth: suppress only near-identical boxes.
                if _bbox_iou(det.bbox_xyxy, kept.bbox_xyxy) >= 0.70:
                    duplicate = True
                    break

        if not duplicate:
            fused.append(det)

    return fused


def _normalize_label_token(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def _is_large_stuff_label(label: str, large_stuff_labels: List[str]) -> bool:
    ls = {_normalize_label_token(x) for x in large_stuff_labels if str(x).strip()}
    return _normalize_label_token(label) in ls


def _suppress_negative_points_for_labels(
    negative_points_by_box: List[np.ndarray],
    labels: List[str],
    exempt_labels: List[str],
) -> List[np.ndarray]:
    if not negative_points_by_box:
        return negative_points_by_box

    exempt = {_normalize_label_token(x) for x in exempt_labels if str(x).strip()}
    out: List[np.ndarray] = []
    for i, pts in enumerate(negative_points_by_box):
        lbl = _normalize_label_token(labels[i]) if i < len(labels) else ""
        if lbl in exempt:
            out.append(np.zeros((0, 2), dtype=np.float32))
        else:
            out.append(np.asarray(pts, dtype=np.float32))
    return out


def _restore_underfilled_masks(
    *,
    masks: np.ndarray,
    provisional_masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: List[str],
    preferred_labels: List[str],
    min_mask_to_box_ratio: float,
    min_provisional_gain: float,
) -> np.ndarray:
    if masks is None or provisional_masks is None or masks.size == 0 or provisional_masks.size == 0:
        return masks

    preferred = {_normalize_label_token(x) for x in preferred_labels if str(x).strip()}
    if not preferred:
        return masks

    out = masks.copy().astype(bool)
    n = min(out.shape[0], provisional_masks.shape[0], boxes_xyxy.shape[0], len(labels))
    for i in range(n):
        lbl = _normalize_label_token(labels[i])
        if lbl not in preferred:
            continue

        x1, y1, x2, y2 = boxes_xyxy[i]
        box_area = float(max(1.0, (x2 - x1) * (y2 - y1)))

        cur = out[i] > 0
        prv = provisional_masks[i] > 0
        cur_area = float(cur.sum())
        prv_area = float(prv.sum())

        if prv_area <= 0:
            continue

        cur_ratio = cur_area / box_area
        if cur_ratio >= float(min_mask_to_box_ratio):
            continue

        if prv_area < cur_area * float(min_provisional_gain):
            continue

        out[i] = prv

    return out


def _build_targeted_prompts(base_prompts: List[str], target_labels: List[str], max_prompts: int = 10) -> List[str]:
    prompts: List[str] = []
    seen = set()

    def add(p: str) -> None:
        t = " ".join(str(p).strip().lower().split())
        if not t or t in seen:
            return
        seen.add(t)
        prompts.append(t)

    for p in base_prompts:
        add(p)

    # Keep prompts semantically stable so labels do not drift
    # (e.g., "full person" vs "person") and harm association.
    for l in target_labels:
        ll = str(l).strip().lower()
        add(ll)

    return prompts[:max_prompts]


def _canonicalize_label(
    raw_label: str,
    *,
    target_labels: Optional[List[str]] = None,
) -> str:
    s = " ".join(str(raw_label).strip().lower().replace("-", " ").split())
    if not s:
        return "object"

    changed = True
    while changed:
        changed = False
        for prefix in ("fully visible ", "visible ", "full ", "a ", "an ", "the "):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                changed = True

    if s.endswith(" object"):
        s = s[:-7].strip()

    if target_labels:
        tgt = [" ".join(str(t).strip().lower().split()) for t in target_labels if str(t).strip()]
        tgt_set = set(tgt)
        for t in sorted(tgt_set, key=len, reverse=True):
            if s == t:
                return t
            if s.startswith(f"{t} ") or s.endswith(f" {t}") or f" {t} " in f" {s} ":
                return t
        if s.endswith("s") and s[:-1] in tgt_set:
            return s[:-1]

    return s


def _detect_scene_cut_keyframes(
    frames_bgr: List[np.ndarray],
    threshold: float,
    min_gap: int,
) -> List[int]:
    if len(frames_bgr) < 2:
        return []

    cuts: List[int] = []
    last_cut = -10_000
    prev_hist = None

    for i, f in enumerate(frames_bgr):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [48, 24], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).reshape(-1)
        if prev_hist is not None:
            diff = float(cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
            if diff >= threshold and (i - last_cut) >= max(1, min_gap):
                cuts.append(i)
                last_cut = i
        prev_hist = hist

    return cuts


def _negative_points_from_overlap(
    boxes_xyxy: np.ndarray,
    masks: np.ndarray,
    labels: Optional[List[str]] = None,
    overlap_iof: float = 0.50,
) -> List[np.ndarray]:
    if boxes_xyxy.size == 0:
        return []

    areas = np.array([_bbox_area(b) for b in boxes_xyxy], dtype=np.float32)
    order = np.argsort(areas)  # small objects are treated as foreground priors

    negatives: List[List[List[float]]] = [[] for _ in range(len(boxes_xyxy))]
    centroids: List[Optional[List[float]]] = []
    for m in masks:
        centroids.append(_mask_centroid(m.astype(np.uint8)))

    use_depth = labels is not None and len(labels) == len(boxes_xyxy)
    depths: Optional[np.ndarray] = None
    if use_depth:
        depths = np.array([SEMANTIC_DEPTH.get(_normalize_label_token(lbl), 2) for lbl in labels], dtype=np.int32)

    for rank_bg in range(len(order) - 1, -1, -1):
        bg_idx = int(order[rank_bg])
        bg_box = boxes_xyxy[bg_idx]
        bg_depth = int(depths[bg_idx]) if use_depth and depths is not None else 2
        for rank_fg in range(0, rank_bg):
            fg_idx = int(order[rank_fg])
            if use_depth and depths is not None:
                fg_depth = int(depths[fg_idx])
                if fg_depth <= bg_depth:
                    continue
                # Avoid punching accessory-shaped holes directly into human prompts.
                if fg_depth == 7 and bg_depth == 6:
                    continue
            iof = _bbox_iof(bg_box, boxes_xyxy[fg_idx])
            if iof < overlap_iof:
                continue
            ctr = centroids[fg_idx]
            if ctr is None:
                continue
            negatives[bg_idx].append([float(ctr[0]), float(ctr[1])])

    out: List[np.ndarray] = []
    for pts in negatives:
        if not pts:
            out.append(np.zeros((0, 2), dtype=np.float32))
        else:
            out.append(np.asarray(pts[:8], dtype=np.float32))
    return out


def _depth_aware_subtract_masks(
    masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: Optional[List[str]] = None,
    overlap_iof: float = 0.12,
) -> np.ndarray:
    if masks.size == 0 or boxes_xyxy.size == 0:
        return masks

    out = masks.copy().astype(bool)
    areas = np.array([_bbox_area(b) for b in boxes_xyxy], dtype=np.float32)
    if labels is not None and len(labels) == len(boxes_xyxy):
        depths = np.array([SEMANTIC_DEPTH.get(_normalize_label_token(lbl), 2) for lbl in labels], dtype=np.int32)
    else:
        depths = np.full((len(boxes_xyxy),), 2, dtype=np.int32)

    # Lower depth values are farther background. Within the same depth, larger regions
    # are treated first to preserve stable carving order.
    order = np.lexsort((-areas, depths))

    for rank_bg in range(len(order)):
        bg_idx = int(order[rank_bg])
        bg_depth = int(depths[bg_idx])
        subtract = np.zeros_like(out[bg_idx], dtype=bool)

        for rank_fg in range(rank_bg + 1, len(order)):
            fg_idx = int(order[rank_fg])
            fg_depth = int(depths[fg_idx])
            if fg_depth <= bg_depth:
                continue
            if _binary_mask_iof(out[bg_idx], out[fg_idx]) < overlap_iof:
                continue
            subtract |= out[fg_idx]

        if np.any(subtract):
            candidate = out[bg_idx] & ~subtract
            if int(candidate.sum()) >= max(16, int(0.05 * out[bg_idx].sum())):
                out[bg_idx] = candidate

    return out


def _binary_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    if union <= 0:
        return 0.0
    return inter / union


def _binary_mask_iof(bg_mask: np.ndarray, fg_mask: np.ndarray) -> float:
    bg = bg_mask > 0
    fg = fg_mask > 0
    inter = float(np.logical_and(bg, fg).sum())
    fg_area = float(fg.sum())
    if fg_area <= 0:
        return 0.0
    return inter / fg_area


def _temporal_mask_smooth(prev_mask: Optional[np.ndarray], cur_mask: np.ndarray) -> np.ndarray:
    # Trust CoTracker and XMem temporal consistency; geometric rollback can freeze masks.
    return cur_mask


def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return [(int(rr), int(cc)) for rr, cc in zip(r, c)]

    # Greedy fallback if scipy is unavailable.
    pairs: List[Tuple[int, int]] = []
    used_r = set()
    used_c = set()
    flat = []
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            flat.append((float(cost[i, j]), i, j))
    flat.sort(key=lambda x: x[0])
    for _, i, j in flat:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
    return pairs


def _occlusion_overlap_score(boxes: List[np.ndarray]) -> float:
    if len(boxes) < 2:
        return 0.0
    vals = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            vals.append(_bbox_iou(boxes[i], boxes[j]))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _associate_detections(
    frame_idx: int,
    detections: List[SimpleDetection],
    embeddings: np.ndarray,
    track_bank: Dict[int, TrackMemory],
    next_instance_id: int,
    assoc_cfg: AssociationParams,
) -> Tuple[List[int], int]:
    det_to_iid = [-1] * len(detections)

    live_tracks = [
        iid
        for iid, tr in track_bank.items()
        if tr.miss_count <= assoc_cfg.max_missed_keyframes
    ]

    if not live_tracks:
        for d_idx, det in enumerate(detections):
            iid = next_instance_id
            next_instance_id += 1
            det_to_iid[d_idx] = iid
            emb = _project_embedding(embeddings[d_idx], assoc_cfg.embedding_dim)
            track_bank[iid] = TrackMemory(
                instance_id=iid,
                label=det.label,
                bbox_xyxy=det.bbox_xyxy.copy(),
                appearance=emb,
                velocity_xy=np.zeros((2,), dtype=np.float32),
                last_seen=frame_idx,
                miss_count=0,
            )
        return det_to_iid, next_instance_id

    cost = np.full((len(live_tracks), len(detections)), fill_value=1e6, dtype=np.float32)

    for ti, iid in enumerate(live_tracks):
        tr = track_bank[iid]
        for di, det in enumerate(detections):
            if tr.label != det.label:
                continue

            iou = _bbox_iou(tr.bbox_xyxy, det.bbox_xyxy)
            app = _cosine_sim(tr.appearance, _project_embedding(embeddings[di], assoc_cfg.embedding_dim))

            dt = max(1, int(frame_idx - tr.last_seen))
            pred_center = _bbox_center(tr.bbox_xyxy) + tr.velocity_xy * float(dt)
            det_center = _bbox_center(det.bbox_xyxy)
            diag = float(max(1.0, math.sqrt(_bbox_area(det.bbox_xyxy))))
            motion = min(1.0, float(np.linalg.norm(det_center - pred_center) / diag))

            c = (
                assoc_cfg.weight_iou * (1.0 - iou)
                + assoc_cfg.weight_app * (1.0 - app)
                + assoc_cfg.weight_motion * motion
            )

            if iou < assoc_cfg.min_iou_gate and app < assoc_cfg.min_app_gate:
                c += 0.5

            cost[ti, di] = c

    used_tracks = set()
    used_dets = set()
    for ti, di in _hungarian(cost):
        c = float(cost[ti, di])
        if c > assoc_cfg.max_cost:
            continue
        iid = live_tracks[ti]
        det_to_iid[di] = iid
        used_tracks.add(iid)
        used_dets.add(di)

    # Create new tracks for unmatched detections.
    for di, det in enumerate(detections):
        if di in used_dets:
            continue
        iid = next_instance_id
        next_instance_id += 1
        det_to_iid[di] = iid
        emb = _project_embedding(embeddings[di], assoc_cfg.embedding_dim)
        track_bank[iid] = TrackMemory(
            instance_id=iid,
            label=det.label,
            bbox_xyxy=det.bbox_xyxy.copy(),
            appearance=emb,
            velocity_xy=np.zeros((2,), dtype=np.float32),
            last_seen=frame_idx,
            miss_count=0,
        )

    # Update matched tracks.
    for di, iid in enumerate(det_to_iid):
        if iid <= 0:
            continue
        det = detections[di]
        emb = _project_embedding(embeddings[di], assoc_cfg.embedding_dim)
        tr = track_bank[iid]

        old_center = _bbox_center(tr.bbox_xyxy)
        new_center = _bbox_center(det.bbox_xyxy)
        dt = max(1, int(frame_idx - tr.last_seen))
        vel = (new_center - old_center) / float(dt)

        tr.velocity_xy = (0.65 * tr.velocity_xy + 0.35 * vel).astype(np.float32)
        tr.bbox_xyxy = det.bbox_xyxy.copy()
        tr.appearance = (0.7 * tr.appearance + 0.3 * emb).astype(np.float32)
        tr.appearance = _project_embedding(tr.appearance, assoc_cfg.embedding_dim)
        tr.last_seen = frame_idx
        tr.miss_count = 0

    # Keep unmatched tracks alive for short occlusion windows.
    for iid, tr in track_bank.items():
        if iid in used_tracks:
            continue
        if tr.last_seen < frame_idx:
            tr.miss_count += 1

    return det_to_iid, next_instance_id


def _update_track_geometry_only(track_bank: Dict[int, TrackMemory], iid: int, frame_idx: int, box_xyxy: np.ndarray) -> None:
    if iid not in track_bank:
        return
    tr = track_bank[iid]
    old_center = _bbox_center(tr.bbox_xyxy)
    new_center = _bbox_center(box_xyxy)
    dt = max(1, int(frame_idx - tr.last_seen))
    vel = (new_center - old_center) / float(dt)
    tr.velocity_xy = (0.7 * tr.velocity_xy + 0.3 * vel).astype(np.float32)
    tr.bbox_xyxy = box_xyxy.copy()
    tr.last_seen = frame_idx
    tr.miss_count = 0


def _build_keyframe_label_map(
    frame_shape: Tuple[int, int],
    assignments: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
    h, w = frame_shape
    label_map = np.zeros((h, w), dtype=np.int32)
    # Small-to-large order keeps small foreground instances from being overwritten.
    assignments_sorted = sorted(assignments, key=lambda x: int((x[1] > 0).sum()))
    for iid, mask in assignments_sorted:
        label_map[mask > 0] = int(iid)
    return label_map


def _commit_mask(
    *,
    iid: int,
    t: int,
    frame_path: str,
    mask: np.ndarray,
    box: Optional[np.ndarray],
    instances: Dict[int, Instance],
    masks_root: Path,
    save_npy_masks: bool,
    recent_masks: Dict[int, np.ndarray],
) -> None:
    if iid not in instances:
        return
    inst = instances[iid]

    prev = recent_masks.get(iid)
    smoothed = _temporal_mask_smooth(prev, mask.astype(np.uint8))
    recent_masks[iid] = smoothed.copy()

    if box is None:
        mb = _mask_bbox(smoothed)
        if mb is None:
            return
        box = mb

    inst.first_frame = min(inst.first_frame, t)
    inst.last_frame = max(inst.last_frame, t)
    inst.bboxes[t] = [float(v) for v in box.tolist()]

    ctr = _mask_centroid(smoothed)
    if ctr is not None:
        inst.centroids[t] = ctr

    frame_mask_dir = masks_root / f"frame_{t:06d}"
    inst.mask_paths[t] = _save_mask(
        smoothed.astype(np.uint8),
        frame_mask_dir / f"instance_{iid:04d}.png",
        save_npy=save_npy_masks,
    )


def process_video_for_segmentation(
    video_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    *,
    data_root: str = "data",
    target_height: int = 720,
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
    cotracker_config_path: str = "configs/perception/cotracker3.yaml",
    grounding_dino_config_path: str = "configs/perception/grounding_dino.yaml",
    xmem_config_path: str = "configs/perception/xmem.yaml",
) -> Dict[str, Any]:
    cb = progress_callback or _noop_progress

    # ---------- Config loading (YAML -> runtime args) ----------
    cot_cfg = _load_yaml(cotracker_config_path)
    gdino_cfg = _load_yaml(grounding_dino_config_path)
    xmem_cfg = _load_yaml(xmem_config_path)

    device = str(gdino_cfg.get("device", device))

    mask_cfg = gdino_cfg.get("mask_refine", {})
    if not isinstance(mask_cfg, dict):
        mask_cfg = {}
    box_pad_ratio = float(mask_cfg.get("box_pad_ratio", 0.06))
    mask_min_area = int(mask_cfg.get("min_area", 32))
    mask_kernel = int(mask_cfg.get("morph_kernel", 3))

    mask_guard_cfg = gdino_cfg.get("mask_guard", {})
    if not isinstance(mask_guard_cfg, dict):
        mask_guard_cfg = {}
    max_mask_to_box_ratio = float(mask_guard_cfg.get("default_max_mask_to_box_ratio", 3.2))
    max_frame_area_ratio = float(mask_guard_cfg.get("global_max_area_ratio", 0.82))
    oversize_penalty = float(mask_guard_cfg.get("oversize_penalty", 2.8))
    large_stuff_cfg = mask_guard_cfg.get(
        "large_stuff_labels",
        [
            "sky",
            "skyline horizon",
            "building",
            "building facade",
            "urban building",
            "road",
            "road surface",
            "lane markings",
            "street",
            "wall",
            "mountain",
            "tree canopy",
        ],
    )
    if not isinstance(large_stuff_cfg, list):
        large_stuff_cfg = [
            "sky",
            "skyline horizon",
            "building",
            "building facade",
            "urban building",
            "road",
            "road surface",
            "lane markings",
            "street",
            "wall",
            "mountain",
            "tree canopy",
        ]
    large_stuff_labels = [_normalize_label_token(str(x)) for x in large_stuff_cfg if str(x).strip()]
    detection_fusion_iom = float(gdino_cfg.get("detection_fusion_iom", 0.85))
    detection_fusion_iom = float(np.clip(detection_fusion_iom, 0.30, 0.95))

    vehicle_mask_cfg = gdino_cfg.get("vehicle_mask_recovery", {})
    if not isinstance(vehicle_mask_cfg, dict):
        vehicle_mask_cfg = {}
    vehicle_fullmask_labels = vehicle_mask_cfg.get(
        "labels",
        ["car", "bus", "truck", "train", "boat", "motorcycle", "bicycle"],
    )
    if not isinstance(vehicle_fullmask_labels, list):
        vehicle_fullmask_labels = ["car", "bus", "truck", "train", "boat", "motorcycle", "bicycle"]
    vehicle_disable_negative_points = bool(vehicle_mask_cfg.get("disable_negative_points", True))
    vehicle_min_mask_to_box_ratio = float(vehicle_mask_cfg.get("min_mask_to_box_ratio", 0.22))
    vehicle_min_provisional_gain = float(vehicle_mask_cfg.get("min_provisional_gain", 1.20))

    scene_cfg = gdino_cfg.get("scene_fallback", {})
    scene_enabled = False
    scene_track = False
    scene_label = "scene"
    if isinstance(scene_cfg, dict):
        scene_enabled = bool(scene_cfg.get("enabled", False))
        scene_track = bool(scene_cfg.get("track", False))
        scene_label = str(scene_cfg.get("label", scene_label))

    if target_labels is None:
        cfg_labels = gdino_cfg.get("target_labels", None)
        if isinstance(cfg_labels, list):
            target_labels = [str(x) for x in cfg_labels]

    if target_labels is None:
        target_labels = []

    target_labels = [
        _canonicalize_label(str(x), target_labels=None)
        for x in target_labels
        if str(x).strip()
    ]
    target_labels = list(dict.fromkeys(target_labels))

    gdino_detector = GroundingDinoDetector.from_config(grounding_dino_config_path, gdino_cfg)
    gdino_box_pad_ratio = float(gdino_cfg.get("box_pad_ratio", box_pad_ratio))

    gdino_runtime_cfg = gdino_cfg.get("runtime", {}) if isinstance(gdino_cfg.get("runtime", {}), dict) else {}
    gdino_preset = str(gdino_runtime_cfg.get("preset", gdino_cfg.get("default_preset", ""))).strip() or None
    gdino_clutter = str(gdino_runtime_cfg.get("clutter_level", gdino_cfg.get("default_clutter", "medium"))).lower()
    preset_prompts: List[str] = []
    env_presets = gdino_cfg.get("environment_presets", {}) if isinstance(gdino_cfg.get("environment_presets", {}), dict) else {}
    if gdino_preset and gdino_preset in env_presets:
        ent = env_presets.get(gdino_preset, {})
        if isinstance(ent, dict) and isinstance(ent.get("prompts", None), list):
            preset_prompts = [str(x) for x in ent.get("prompts", []) if str(x).strip()]
    if not preset_prompts:
        fallback_prompts = gdino_cfg.get("prompts", []) if isinstance(gdino_cfg.get("prompts", []), list) else []
        preset_prompts = [str(x) for x in fallback_prompts if str(x).strip()]

    cotracker_checkpoint = str(cot_cfg.get("checkpoint", cotracker_checkpoint))
    tracker_points_per_instance = int(cot_cfg.get("points_per_instance", tracker_points_per_instance))
    keyframe_stride = int(cot_cfg.get("keyframe_stride", keyframe_stride))
    device = str(cot_cfg.get("device", device))

    max_frames_per_chunk_cfg = cot_cfg.get("max_frames_per_chunk", None)
    if isinstance(max_frames_per_chunk_cfg, (int, float)):
        max_frames_per_chunk_cfg = int(max_frames_per_chunk_cfg)
        if max_frames_per_chunk_cfg <= 0:
            max_frames_per_chunk_cfg = None
    else:
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
        max_total_frames_cfg = None

    if max_total_frames is None:
        max_total_frames = max_total_frames_cfg
    elif isinstance(max_total_frames, (int, float)):
        max_total_frames = int(max_total_frames)
        if max_total_frames <= 0:
            max_total_frames = None
    else:
        max_total_frames = None

    if max_frames_per_chunk is None and not torch.cuda.is_available():
        max_frames_per_chunk = 8

    if not target_labels:
        target_labels = [
            _canonicalize_label(str(x), target_labels=None)
            for x in preset_prompts
            if str(x).strip()
        ]

    keyframe_cfg = cot_cfg.get("keyframe", {}) if isinstance(cot_cfg.get("keyframe", {}), dict) else {}
    scene_cut_enable = bool(keyframe_cfg.get("enable_scene_cuts", True))
    scene_cut_threshold = float(keyframe_cfg.get("scene_cut_threshold", 0.42))
    scene_cut_min_gap = int(keyframe_cfg.get("scene_cut_min_gap", max(2, keyframe_stride // 2)))

    assoc_raw = cot_cfg.get("association", {}) if isinstance(cot_cfg.get("association", {}), dict) else {}
    assoc_cfg = AssociationParams(
        weight_iou=_safe_float(assoc_raw, "weight_iou", 0.70),
        weight_app=_safe_float(assoc_raw, "weight_app", 0.10),
        weight_motion=_safe_float(assoc_raw, "weight_motion", 0.20),
        min_iou_gate=_safe_float(assoc_raw, "min_iou_gate", 0.05),
        min_app_gate=_safe_float(assoc_raw, "min_app_gate", 0.10),
        max_cost=_safe_float(assoc_raw, "max_cost", 0.92),
        max_missed_keyframes=int(assoc_raw.get("max_missed_keyframes", 10)),
        embedding_dim=int(assoc_raw.get("embedding_dim", 64)),
    )

    gdino_prompt_cap = int(gdino_cfg.get("max_prompts", 10))
    gdino_prompt_cap = max(1, min(10, gdino_prompt_cap))

    recovery_raw = cot_cfg.get("recovery", {}) if isinstance(cot_cfg.get("recovery", {}), dict) else {}
    recovery_cfg = RecoveryParams(
        enable_selective_redetect=bool(recovery_raw.get("enable_selective_redetect", True)),
        missing_ratio_trigger=_safe_float(recovery_raw, "missing_ratio_trigger", 0.40),
        overlap_trigger=_safe_float(recovery_raw, "overlap_trigger", 0.35),
        min_redetect_gap=int(recovery_raw.get("min_redetect_gap", 8)),
    )

    xmem_runtime = XMemRuntimeConfig(
        enabled=bool(xmem_cfg.get("enabled", False)),
        repo_path=str(xmem_cfg.get("repo_path", "XMem")),
        model_path=str(xmem_cfg.get("model_path", "XMem/saves/XMem.pth")),
        device=str(xmem_cfg.get("device", device)),
        mem_every=int(xmem_cfg.get("mem_every", 5)),
        deep_update_every=int(xmem_cfg.get("deep_update_every", -1)),
        max_mid_term_frames=int(xmem_cfg.get("max_mid_term_frames", 10)),
        min_mid_term_frames=int(xmem_cfg.get("min_mid_term_frames", 5)),
        max_long_term_elements=int(xmem_cfg.get("max_long_term_elements", 10000)),
        num_prototypes=int(xmem_cfg.get("num_prototypes", 128)),
        top_k=int(xmem_cfg.get("top_k", 30)),
        disable_long_term=bool(xmem_cfg.get("disable_long_term", False)),
    )

    def _run_once(max_h: int) -> Dict[str, Any]:
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

        instances: Dict[int, Instance] = {}
        track_bank: Dict[int, TrackMemory] = {}
        next_instance_id = 1

        keyframes = set(range(0, total_frames, max(1, keyframe_stride)))
        keyframes.add(total_frames - 1)
        if scene_cut_enable:
            for cut_idx in _detect_scene_cut_keyframes(
                frames_bgr,
                threshold=scene_cut_threshold,
                min_gap=scene_cut_min_gap,
            ):
                keyframes.add(int(cut_idx))
        keyframes_sorted = sorted(keyframes)
        keyframe_set = set(keyframes_sorted)

        instance_seed_points: Dict[int, np.ndarray] = {}
        instance_seed_frames: Dict[int, int] = {}
        scene_instance_id: Optional[int] = None
        keyframe_label_maps: Dict[int, np.ndarray] = {}

        sam2_chunk = _sam2_chunk_size(sam2_model_cfg, device)
        targeted_gdino_prompts = _build_targeted_prompts(
            preset_prompts,
            target_labels,
            max_prompts=gdino_prompt_cap,
        )
        large_stuff_set = {_normalize_label_token(x) for x in large_stuff_labels}

        # ---------- Phase A ----------
        cb(0, 1, "Phase A: Loading detectors + SAM2...")
        sam2 = Sam2Masker(model_cfg=sam2_model_cfg, checkpoint_path=sam2_checkpoint, device=device)

        cb(0, len(keyframes_sorted), "Phase A: keyframe detect/associate/mask...")
        for i, fidx in enumerate(keyframes_sorted):
            frame = frames_bgr[fidx]
            h, w = frame.shape[:2]

            detections: List[SimpleDetection] = []

            if gdino_detector is not None:
                gdino_raw = gdino_detector.detect(
                    frame_paths[fidx],
                    prompts_override=targeted_gdino_prompts,
                    preset=gdino_preset,
                    clutter_level=gdino_clutter,
                )
                for gd in gdino_raw:
                    lbl = _canonicalize_label(
                        str(gd.get("label", "object")),
                        target_labels=target_labels,
                    )
                    detections.append(
                        SimpleDetection(
                            bbox_xyxy=np.asarray(gd["bbox_xyxy"], dtype=np.float32),
                            label=lbl,
                            score=float(gd.get("score", 0.5)),
                            source="gdino",
                        )
                    )

            detections = _fuse_detections(detections, iom_thresh=detection_fusion_iom)

            if not detections:
                if scene_enabled:
                    scene_box = np.array([[0.0, 0.0, float(w - 1), float(h - 1)]], dtype=np.float32)
                    scene_masks = sam2.masks_from_boxes(
                        frame,
                        scene_box,
                        multimask_output=True,
                        selection="score_area",
                        area_weight=0.12,
                        labels=[scene_label],
                        large_stuff_labels=large_stuff_labels,
                        max_mask_to_box_ratio=max_mask_to_box_ratio,
                        max_frame_area_ratio=max_frame_area_ratio,
                        oversize_penalty=oversize_penalty,
                    )
                    if scene_masks.size > 0:
                        scene_mask = _postprocess_mask(
                            scene_masks[0].astype(np.uint8),
                            min_area=mask_min_area,
                            kernel=mask_kernel,
                        )
                        if np.any(scene_mask):
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
                            ctr = _mask_centroid(scene_mask)
                            if ctr is not None:
                                inst.centroids[fidx] = ctr

                            frame_mask_dir = masks_root / f"frame_{fidx:06d}"
                            inst.mask_paths[fidx] = _save_mask(
                                scene_mask.astype(np.uint8),
                                frame_mask_dir / f"instance_{scene_instance_id:04d}.png",
                                save_npy=save_npy_masks,
                            )

                            if (
                                scene_track
                                and scene_instance_id not in instance_seed_points
                                and not _is_large_stuff_label(scene_label, large_stuff_labels)
                            ):
                                pts = CoTrackerPersistentTracker.sample_points_from_mask(
                                    scene_mask.astype(np.uint8),
                                    k=min(16, tracker_points_per_instance),
                                )
                                if pts.shape[0] > 0:
                                    instance_seed_points[scene_instance_id] = pts
                                    instance_seed_frames[scene_instance_id] = fidx

                            keyframe_label_maps[fidx] = _build_keyframe_label_map((h, w), [(scene_instance_id, scene_mask)])
                            cb(i + 1, len(keyframes_sorted), f"Keyframe {fidx}: scene fallback")
                            continue

                # Keep track memories alive through short no-detection windows.
                for tr in track_bank.values():
                    tr.miss_count += 1
                cb(i + 1, len(keyframes_sorted), f"Keyframe {fidx}: no detections")
                continue

            boxes = np.stack([d.bbox_xyxy for d in detections], axis=0).astype(np.float32)
            refined = _refine_boxes(boxes, w, h, pad_ratio=box_pad_ratio)

            # Initial run without negative points to estimate foreground centroids.
            provisional_masks = sam2.masks_from_boxes(
                frame,
                refined,
                multimask_output=True,
                selection="score_area",
                area_weight=0.15,
                labels=[d.label for d in detections],
                large_stuff_labels=large_stuff_labels,
                max_mask_to_box_ratio=max_mask_to_box_ratio,
                max_frame_area_ratio=max_frame_area_ratio,
                oversize_penalty=oversize_penalty,
            )

            if provisional_masks.size == 0:
                for tr in track_bank.values():
                    tr.miss_count += 1
                cb(i + 1, len(keyframes_sorted), f"Keyframe {fidx}: masks empty")
                continue

            neg_pts = _negative_points_from_overlap(
                refined,
                provisional_masks,
                labels=[d.label for d in detections],
            )
            if vehicle_disable_negative_points:
                neg_pts = _suppress_negative_points_for_labels(
                    neg_pts,
                    [d.label for d in detections],
                    vehicle_fullmask_labels,
                )
            masks = sam2.masks_from_boxes(
                frame,
                refined,
                multimask_output=True,
                selection="score",
                area_weight=0.0,
                negative_points_by_box=neg_pts,
                labels=[d.label for d in detections],
                large_stuff_labels=large_stuff_labels,
                max_mask_to_box_ratio=max_mask_to_box_ratio,
                max_frame_area_ratio=max_frame_area_ratio,
                oversize_penalty=oversize_penalty,
            )
            masks = _restore_underfilled_masks(
                masks=masks,
                provisional_masks=provisional_masks,
                boxes_xyxy=refined,
                labels=[d.label for d in detections],
                preferred_labels=vehicle_fullmask_labels,
                min_mask_to_box_ratio=vehicle_min_mask_to_box_ratio,
                min_provisional_gain=vehicle_min_provisional_gain,
            )

            masks = np.stack(
                [
                    _postprocess_mask(m.astype(np.uint8), min_area=mask_min_area, kernel=mask_kernel)
                    for m in masks
                ],
                axis=0,
            ).astype(bool)
            masks = _depth_aware_subtract_masks(masks, refined, labels=[d.label for d in detections])
            masks = _resolve_mask_overlaps(masks)

            embeddings = sam2.appearance_embeddings_from_masks(frame, masks, assume_image_is_set=True)
            det_to_iid, next_instance_id = _associate_detections(
                frame_idx=fidx,
                detections=detections,
                embeddings=embeddings,
                track_bank=track_bank,
                next_instance_id=next_instance_id,
                assoc_cfg=assoc_cfg,
            )

            frame_assignments: List[Tuple[int, np.ndarray]] = []
            for det_idx, iid in enumerate(det_to_iid):
                if iid <= 0:
                    continue
                det = detections[det_idx]
                mask = masks[det_idx].astype(np.uint8)
                box = refined[det_idx]

                if iid not in instances:
                    instances[iid] = Instance(
                        instance_id=iid,
                        label=det.label,
                        first_frame=fidx,
                        last_frame=fidx,
                    )

                inst = instances[iid]
                inst.last_frame = max(inst.last_frame, fidx)
                inst.first_frame = min(inst.first_frame, fidx)
                inst.label = det.label
                inst.keyframes.append(frame_paths[fidx])
                inst.keyframe_indices.append(fidx)
                inst.bboxes[fidx] = [float(v) for v in box.tolist()]
                ctr = _mask_centroid(mask)
                if ctr is not None:
                    inst.centroids[fidx] = ctr

                frame_mask_dir = masks_root / f"frame_{fidx:06d}"
                inst.mask_paths[fidx] = _save_mask(
                    mask,
                    frame_mask_dir / f"instance_{iid:04d}.png",
                    save_npy=save_npy_masks,
                )

                if iid not in instance_seed_points and _normalize_label_token(det.label) not in large_stuff_set:
                    pts = CoTrackerPersistentTracker.sample_points_from_mask(
                        mask.astype(np.uint8),
                        k=tracker_points_per_instance,
                    )
                    if pts.shape[0] == 0:
                        x1, y1, x2, y2 = box
                        pts = np.array(
                            [
                                [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                                [x1, y1],
                                [x2, y1],
                                [x1, y2],
                                [x2, y2],
                            ],
                            dtype=np.float32,
                        )
                    instance_seed_points[iid] = pts
                    instance_seed_frames[iid] = fidx

                frame_assignments.append((iid, mask))

            if frame_assignments:
                keyframe_label_maps[fidx] = _build_keyframe_label_map((h, w), frame_assignments)

            cb(i + 1, len(keyframes_sorted), f"Keyframe {fidx}: {len(frame_assignments)} instances")

        del sam2
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
                "runtime_device": device,
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
                "runtime_device": device,
            }

        trackable_instance_ids = [
            iid
            for iid, inst in instances.items()
            if not _is_large_stuff_label(inst.label, large_stuff_labels)
        ]

        # Optional VOS propagation from keyframe masks (foreground/trackable only).
        xmem_label_maps: Dict[int, np.ndarray] = {}
        if xmem_runtime.enabled and trackable_instance_ids:
            cb(0, 1, "VOS: trying XMem propagation...")
            try:
                propagator = XMemMaskPropagator(xmem_runtime)
                if propagator.is_ready():
                    keyframe_label_maps_trackable: Dict[int, np.ndarray] = {}
                    for kf_idx, lbl_map in keyframe_label_maps.items():
                        filtered = np.zeros_like(lbl_map, dtype=np.int32)
                        for iid in trackable_instance_ids:
                            filtered[lbl_map == int(iid)] = int(iid)
                        if np.any(filtered > 0):
                            keyframe_label_maps_trackable[kf_idx] = filtered

                    xmem_label_maps = propagator.propagate(
                        frames_bgr=frames_bgr,
                        keyframe_label_maps=keyframe_label_maps_trackable,
                        all_instance_ids=sorted(trackable_instance_ids),
                    )
                    cb(1, 1, f"VOS: XMem propagated {len(xmem_label_maps)} frames")
                else:
                    cb(1, 1, "VOS: XMem not ready, skipping")
            except Exception as exc:
                cb(1, 1, f"VOS: XMem failed ({exc}), continuing without it")
                xmem_label_maps = {}

        # ---------- Phase B ----------
        cb(0, 1, "Phase B: Loading CoTracker...")
        tracker = CoTrackerPersistentTracker(
            checkpoint_path=cotracker_checkpoint,
            device=device,
            max_frames_per_chunk=max_frames_per_chunk,
        )
        tracked_points = tracker.track_points(frames_bgr, instance_seed_points, instance_seed_frames)
        del tracker
        _clear_gpu()
        cb(1, 1, "Phase B complete")

        H, W = frames_bgr[0].shape[:2]
        tracked_bboxes: Dict[int, Dict[int, List[float]]] = {iid: {} for iid in instances.keys()}
        for iid, tr in tracked_points.items():
            for t in range(tr.shape[0]):
                pts = tr[t]
                xs = np.clip(pts[:, 0], 0, W - 1)
                ys = np.clip(pts[:, 1], 0, H - 1)
                if xs.size == 0:
                    continue
                # Robust box estimate: drop drifting outlier points from CoTracker.
                x1, x2 = float(np.percentile(xs, 5.0)), float(np.percentile(xs, 95.0))
                y1, y2 = float(np.percentile(ys, 5.0)), float(np.percentile(ys, 95.0))
                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue
                tracked_bboxes[iid][t] = [x1, y1, x2, y2]

        # Use VOS output to improve tracked boxes and occlusion recovery memory.
        if xmem_label_maps:
            for t, lbl_map in xmem_label_maps.items():
                for iid in trackable_instance_ids:
                    m = (lbl_map == int(iid)).astype(np.uint8)
                    mb = _mask_bbox(m)
                    if mb is None:
                        continue
                    tracked_bboxes.setdefault(iid, {})[t] = [float(v) for v in mb.tolist()]

        # ---------- Phase C ----------
        cb(0, 1, "Phase C: Loading SAM2...")
        sam2_final = Sam2Masker(model_cfg=sam2_model_cfg, checkpoint_path=sam2_checkpoint, device=device)

        recent_masks: Dict[int, np.ndarray] = {}
        last_redetect_frame = -10_000

        cb(0, total_frames, "Phase C: temporal mask completion...")
        for t in range(total_frames):
            frame = frames_bgr[t]
            h, w = frame.shape[:2]

            # 1) Consume XMem propagated masks first (fast and temporally stable).
            if t in xmem_label_maps:
                lbl_map = xmem_label_maps[t]
                for iid in trackable_instance_ids:
                    if t in instances[iid].mask_paths:
                        continue
                    m = (lbl_map == int(iid)).astype(np.uint8)
                    m = _postprocess_mask(m, min_area=mask_min_area, kernel=mask_kernel)
                    if int((m > 0).sum()) < mask_min_area:
                        continue
                    mb = _mask_bbox(m)
                    if mb is None:
                        continue
                    _commit_mask(
                        iid=iid,
                        t=t,
                        frame_path=frame_paths[t],
                        mask=m,
                        box=mb,
                        instances=instances,
                        masks_root=masks_root,
                        save_npy_masks=save_npy_masks,
                        recent_masks=recent_masks,
                    )
                    _update_track_geometry_only(track_bank, iid, t, mb)

            frame_iids: List[int] = []
            frame_boxes: List[List[float]] = []
            active_boxes_for_overlap: List[np.ndarray] = []

            for iid in trackable_instance_ids:
                if t in instances[iid].mask_paths:
                    continue
                b = tracked_bboxes.get(iid, {}).get(t, None)
                if b is None:
                    continue
                frame_iids.append(iid)
                frame_boxes.append(b)
                active_boxes_for_overlap.append(np.asarray(b, dtype=np.float32))

            missing_count = 0
            for iid in trackable_instance_ids:
                if t in instances[iid].mask_paths:
                    continue
                if t not in tracked_bboxes.get(iid, {}):
                    missing_count += 1
            unresolved = sum(1 for iid in trackable_instance_ids if t not in instances[iid].mask_paths)
            missing_ratio = float(missing_count) / float(max(1, unresolved))
            overlap_score = _occlusion_overlap_score(active_boxes_for_overlap)

            needs_redetect = (
                recovery_cfg.enable_selective_redetect
                and (t not in keyframe_set)
                and (t - last_redetect_frame) >= max(1, recovery_cfg.min_redetect_gap)
                and (
                    missing_ratio >= recovery_cfg.missing_ratio_trigger
                    or overlap_score >= recovery_cfg.overlap_trigger
                )
            )

            # 2) Selective re-segmentation only when confidence drops / occlusions surge.
            if needs_redetect:
                redet_list: List[SimpleDetection] = []
                if gdino_detector is not None:
                    graw = gdino_detector.detect(
                        frame_paths[t],
                        prompts_override=targeted_gdino_prompts,
                        preset=gdino_preset,
                        clutter_level=gdino_clutter,
                    )
                    for gd in graw:
                        lbl = _canonicalize_label(
                            str(gd.get("label", "object")),
                            target_labels=target_labels,
                        )
                        if _is_large_stuff_label(lbl, large_stuff_labels):
                            continue
                        redet_list.append(
                            SimpleDetection(
                                bbox_xyxy=np.asarray(gd["bbox_xyxy"], dtype=np.float32),
                                label=lbl,
                                score=float(gd.get("score", 0.5)),
                                source="gdino",
                            )
                        )

                redet_list = _fuse_detections(redet_list, iom_thresh=detection_fusion_iom)

                if redet_list:
                    red_boxes = np.stack([d.bbox_xyxy for d in redet_list], axis=0).astype(np.float32)
                    red_boxes = _refine_boxes(red_boxes, w, h, pad_ratio=gdino_box_pad_ratio)

                    provisional = sam2_final.masks_from_boxes(
                        frame,
                        red_boxes,
                        multimask_output=True,
                        selection="score_area",
                        area_weight=0.15,
                        labels=[d.label for d in redet_list],
                        large_stuff_labels=large_stuff_labels,
                        max_mask_to_box_ratio=max_mask_to_box_ratio,
                        max_frame_area_ratio=max_frame_area_ratio,
                        oversize_penalty=oversize_penalty,
                    )
                    neg = _negative_points_from_overlap(
                        red_boxes,
                        provisional,
                        labels=[d.label for d in redet_list],
                    )
                    if vehicle_disable_negative_points:
                        neg = _suppress_negative_points_for_labels(
                            neg,
                            [d.label for d in redet_list],
                            vehicle_fullmask_labels,
                        )

                    red_masks = sam2_final.masks_from_boxes(
                        frame,
                        red_boxes,
                        multimask_output=True,
                        selection="score_area",
                        area_weight=0.15,
                        negative_points_by_box=neg,
                        labels=[d.label for d in redet_list],
                        large_stuff_labels=large_stuff_labels,
                        max_mask_to_box_ratio=max_mask_to_box_ratio,
                        max_frame_area_ratio=max_frame_area_ratio,
                        oversize_penalty=oversize_penalty,
                    )
                    red_masks = _restore_underfilled_masks(
                        masks=red_masks,
                        provisional_masks=provisional,
                        boxes_xyxy=red_boxes,
                        labels=[d.label for d in redet_list],
                        preferred_labels=vehicle_fullmask_labels,
                        min_mask_to_box_ratio=vehicle_min_mask_to_box_ratio,
                        min_provisional_gain=vehicle_min_provisional_gain,
                    )

                    red_masks = np.stack(
                        [
                            _postprocess_mask(m.astype(np.uint8), min_area=mask_min_area, kernel=mask_kernel)
                            for m in red_masks
                        ],
                        axis=0,
                    ).astype(bool)
                    red_masks = _depth_aware_subtract_masks(
                        red_masks,
                        red_boxes,
                        labels=[d.label for d in redet_list],
                    )
                    red_masks = _resolve_mask_overlaps(red_masks)

                    red_emb = sam2_final.appearance_embeddings_from_masks(frame, red_masks, assume_image_is_set=True)
                    red_assign, next_instance_id = _associate_detections(
                        frame_idx=t,
                        detections=redet_list,
                        embeddings=red_emb,
                        track_bank=track_bank,
                        next_instance_id=next_instance_id,
                        assoc_cfg=assoc_cfg,
                    )

                    for di, iid in enumerate(red_assign):
                        if iid <= 0:
                            continue
                        if iid not in instances:
                            instances[iid] = Instance(
                                instance_id=iid,
                                label=redet_list[di].label,
                                first_frame=t,
                                last_frame=t,
                            )
                        mask = red_masks[di].astype(np.uint8)
                        if int((mask > 0).sum()) < mask_min_area:
                            continue
                        box = red_boxes[di]
                        _commit_mask(
                            iid=iid,
                            t=t,
                            frame_path=frame_paths[t],
                            mask=mask,
                            box=box,
                            instances=instances,
                            masks_root=masks_root,
                            save_npy_masks=save_npy_masks,
                            recent_masks=recent_masks,
                        )
                        _update_track_geometry_only(track_bank, iid, t, box)
                        tracked_bboxes.setdefault(iid, {})[t] = [float(v) for v in box.tolist()]

                    last_redetect_frame = t

                # recompute unresolved tracked boxes after selective re-detect writes.
                frame_iids = []
                frame_boxes = []
                for iid in trackable_instance_ids:
                    if t in instances[iid].mask_paths:
                        continue
                    b = tracked_bboxes.get(iid, {}).get(t, None)
                    if b is None:
                        continue
                    frame_iids.append(iid)
                    frame_boxes.append(b)

            # 3) Final SAM2 completion for remaining tracked boxes.
            if frame_boxes:
                for s in range(0, len(frame_boxes), sam2_chunk):
                    chunk_iids = frame_iids[s : s + sam2_chunk]
                    chunk_boxes = np.array(frame_boxes[s : s + sam2_chunk], dtype=np.float32)
                    refined_boxes = _refine_boxes(chunk_boxes, w, h, pad_ratio=box_pad_ratio)
                    chunk_labels = [instances[iid].label if iid in instances else "" for iid in chunk_iids]

                    quick_masks = sam2_final.masks_from_boxes(
                        frame,
                        refined_boxes,
                        multimask_output=True,
                        selection="score_area",
                        area_weight=0.15,
                        labels=chunk_labels,
                        large_stuff_labels=large_stuff_labels,
                        max_mask_to_box_ratio=max_mask_to_box_ratio,
                        max_frame_area_ratio=max_frame_area_ratio,
                        oversize_penalty=oversize_penalty,
                    )
                    neg_pts = _negative_points_from_overlap(
                        refined_boxes,
                        quick_masks,
                        labels=chunk_labels,
                    )
                    if vehicle_disable_negative_points:
                        neg_pts = _suppress_negative_points_for_labels(
                            neg_pts,
                            chunk_labels,
                            vehicle_fullmask_labels,
                        )

                    chunk_masks = sam2_final.masks_from_boxes(
                        frame,
                        refined_boxes,
                        multimask_output=True,
                        selection="score_area",
                        area_weight=0.15,
                        negative_points_by_box=neg_pts,
                        labels=chunk_labels,
                        large_stuff_labels=large_stuff_labels,
                        max_mask_to_box_ratio=max_mask_to_box_ratio,
                        max_frame_area_ratio=max_frame_area_ratio,
                        oversize_penalty=oversize_penalty,
                    )
                    chunk_masks = _restore_underfilled_masks(
                        masks=chunk_masks,
                        provisional_masks=quick_masks,
                        boxes_xyxy=refined_boxes,
                        labels=chunk_labels,
                        preferred_labels=vehicle_fullmask_labels,
                        min_mask_to_box_ratio=vehicle_min_mask_to_box_ratio,
                        min_provisional_gain=vehicle_min_provisional_gain,
                    )

                    chunk_masks = np.stack(
                        [
                            _postprocess_mask(m.astype(np.uint8), min_area=mask_min_area, kernel=mask_kernel)
                            for m in chunk_masks
                        ],
                        axis=0,
                    ).astype(bool)
                    chunk_masks = _depth_aware_subtract_masks(
                        chunk_masks,
                        refined_boxes,
                        labels=chunk_labels,
                    )
                    chunk_masks = _resolve_mask_overlaps(chunk_masks)

                    for iid, box, mask in zip(chunk_iids, refined_boxes, chunk_masks):
                        if iid not in instances:
                            continue
                        m = mask.astype(np.uint8)
                        if int((m > 0).sum()) < mask_min_area:
                            continue
                        _commit_mask(
                            iid=iid,
                            t=t,
                            frame_path=frame_paths[t],
                            mask=m,
                            box=box,
                            instances=instances,
                            masks_root=masks_root,
                            save_npy_masks=save_npy_masks,
                            recent_masks=recent_masks,
                        )
                        _update_track_geometry_only(track_bank, iid, t, box)

            cb(
                t + 1,
                total_frames,
                f"Frame {t}: unresolved={sum(1 for iid in trackable_instance_ids if t not in instances[iid].mask_paths)}",
            )

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
            "keyframe_indices": keyframes_sorted,
            "runtime_device": device,
            "xmem_enabled": bool(xmem_runtime.enabled),
            "xmem_frames": len(xmem_label_maps),
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
            "runtime_device": device,
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
