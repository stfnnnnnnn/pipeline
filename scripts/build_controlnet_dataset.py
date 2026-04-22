#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import cv2
import numpy as np


# Stable semantic -> ID mapping for R channel
SEM_ID = {
    "background": 0,
    "sky": 20,
    "skyline horizon": 22,
    "building": 40,
    "building facade": 42,
    "urban building": 44,
    "road": 60,
    "road surface": 62,
    "street": 64,
    "sidewalk": 70,
    "sidewalk or pavement": 72,
    "wall": 80,
    "tree": 90,
    "tree canopy": 92,
    "mountain": 94,
    "person": 120,
    "pedestrian": 122,
    "car": 140,
    "bus": 142,
    "truck": 144,
    "train": 146,
    "motorcycle": 148,
    "bicycle": 150,
    "handbag": 170,
    "backpack": 172,
    "shoulder bag": 174,
}

DEFAULT_TEXT_PROMPT = "instance-guided semantic colorization"


def norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("-", " ").split())


def parse_hex_color(hex_str: str) -> Tuple[int, int, int]:
    t = hex_str.strip().lstrip("#")
    if len(t) != 6:
        return (0, 0, 0)
    r = int(t[0:2], 16)
    g = int(t[2:4], 16)
    b = int(t[4:6], 16)
    return (r, g, b)


def load_color_prompts(path: Path) -> Dict[str, Tuple[int, int, int]]:
    """
    Supports multiple possible schemas:
    1) {"car":"#ff0000","person":[255,200,180], ...}
    2) {"labels":{"car":"#ff0000", ...}}
    3) {"items":[{"label":"car","color":"#ff0000"}, ...]}
    """
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Tuple[int, int, int]] = {}

    def set_color(lbl: str, val: Any):
        k = norm_label(lbl)
        if isinstance(val, str):
            out[k] = parse_hex_color(val)
        elif isinstance(val, (list, tuple)) and len(val) >= 3:
            out[k] = (int(val[0]), int(val[1]), int(val[2]))

    if isinstance(data, dict):
        if "labels" in data and isinstance(data["labels"], dict):
            for k, v in data["labels"].items():
                set_color(k, v)
        elif "items" in data and isinstance(data["items"], list):
            for it in data["items"]:
                if isinstance(it, dict) and "label" in it and "color" in it:
                    set_color(it["label"], it["color"])
        else:
            for k, v in data.items():
                set_color(k, v)

    return out


def find_frame_path(frames_dir: Path, frame_idx: int) -> Path:
    return frames_dir / f"frame_{frame_idx:06d}.png"


def build_control_map(
    h: int,
    w: int,
    frame_idx: int,
    instances: Dict[str, Any],
    label_to_color: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    sem_r = np.zeros((h, w), dtype=np.uint8)   # R channel: semantic ID
    edge_g = np.zeros((h, w), dtype=np.uint8)  # G channel: edges
    hint_b = np.zeros((h, w), dtype=np.uint8)  # B channel: hint intensity (blue channel carrying color magnitude proxy)

    # We'll paint semantic map and hint intensity from masks.
    for iid, obj in instances.items():
        label = norm_label(obj.get("label", "background"))
        sem_id = int(SEM_ID.get(label, 10))
        mask_paths = obj.get("mask_paths", {})
        key = str(frame_idx)
        if key not in mask_paths:
            continue

        mpath = Path(mask_paths[key])
        if not mpath.exists():
            continue
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m_bin = (m > 127).astype(np.uint8)

        # R semantic id
        sem_r[m_bin > 0] = sem_id

        # G edges via morph gradient
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(m_bin * 255, cv2.MORPH_GRADIENT, k)
        edge_g = np.maximum(edge_g, grad.astype(np.uint8))

        # B hint fill (use luminance of desired color so stays single-channel)
        rgb = label_to_color.get(label, (0, 0, 0))
        hint_val = int(round(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]))
        hint_b[m_bin > 0] = max(hint_b[m_bin > 0].max(initial=0), hint_val)

    cond = np.stack([sem_r, edge_g, hint_b], axis=2)  # RGB control map
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, default=Path("data/extracted_frames"))
    ap.add_argument("--metadata-json", type=Path, default=Path("data/intermediate/metadata/segmentation_tracking_metadata.json"))
    ap.add_argument("--color-prompts-json", type=Path, default=Path("data/annotations/object_color_prompts.json"))
    ap.add_argument("--out-root", type=Path, default=Path("data/controlnet_train"))
    ap.add_argument("--text-prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    ap.add_argument("--resize", type=int, default=512)
    args = ap.parse_args()

    out_images = args.out_root / "images"
    out_conds = args.out_root / "conds"
    out_images.mkdir(parents=True, exist_ok=True)
    out_conds.mkdir(parents=True, exist_ok=True)

    meta = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    instances = meta.get("instances", {})
    total_frames = int(meta.get("total_frames", 0))

    label_to_color = load_color_prompts(args.color_prompts_json)

    jsonl_path = args.out_root / "train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i in range(total_frames):
            src = find_frame_path(args.frames_dir, i)
            if not src.exists():
                continue

            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            cond = build_control_map(h, w, i, instances, label_to_color)

            # resize to training resolution
            img_rs = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
            cond_rs = cv2.resize(cond, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST)

            out_img = out_images / f"frame_{i:06d}.png"
            out_cond = out_conds / f"frame_{i:06d}.png"
            cv2.imwrite(str(out_img), img_rs)
            cv2.imwrite(str(out_cond), cond_rs)

            rec = {
                "image": str(out_img).replace("\\", "/"),
                "conditioning_image": str(out_cond).replace("\\", "/"),
                "text": args.text_prompt,
            }
            f.write(json.dumps(rec) + "\n")

    # optional prompt file
    (args.out_root / "prompts.txt").write_text(args.text_prompt + "\n", encoding="utf-8")
    print(f"Done. Wrote dataset to: {args.out_root}")
    print(f"JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()