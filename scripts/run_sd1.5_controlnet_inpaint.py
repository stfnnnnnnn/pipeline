#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import yaml
import sys
from pathlib import Path

# Add repo root to PYTHONPATH so "src" imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diffusion.color_only_diffusion import InstanceColorizer, ColorOnlyConfig


def norm_label(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("-", " ").split())


def parse_color_prompts(path: Path) -> Dict[str, str]:
    if not path.exists():
        print(f"[warn] color prompts file not found: {path}")
        return {}

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        print(f"[warn] color prompts file is empty: {path}")
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[warn] invalid JSON in {path}: {e}")
        return {}

    out: Dict[str, str] = {}

    if isinstance(data, dict):
        if "labels" in data and isinstance(data["labels"], dict):
            for k, v in data["labels"].items():
                out[norm_label(k)] = str(v)
        elif "items" in data and isinstance(data["items"], list):
            for it in data["items"]:
                if isinstance(it, dict) and "label" in it:
                    out[norm_label(it["label"])] = str(it.get("prompt", it.get("color", "")))
        else:
            for k, v in data.items():
                out[norm_label(k)] = str(v)

    return out


def load_cfg(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_mask(mask_path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((h, w), dtype=np.uint8)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def make_prompt(label: str, color_hint: str) -> str:
    base = f"{label}"
    if color_hint.strip():
        return f"{color_hint} {base}, realistic colorization, keep structure"
    return f"natural {base} colorization, keep structure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/sd15_controlnet.yaml")
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--frame_start", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    p_cfg = cfg["paths"]
    i_cfg = cfg["inference"]
    m_cfg = cfg["model"]
    r_cfg = cfg.get("runtime", {})

    frames_dir = Path(p_cfg["frames_dir"])
    metadata_json = Path(p_cfg["metadata_json"])
    color_prompts_json = Path(p_cfg["object_color_prompts_json"])
    out_dir = Path(p_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = out_dir / "debug"
    if r_cfg.get("save_debug_control", True):
        debug_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(metadata_json.read_text(encoding="utf-8"))
    instances: Dict[str, Any] = meta.get("instances", {})
    total_frames = int(meta.get("total_frames", 0))
    color_hints = parse_color_prompts(color_prompts_json)

    dtype = m_cfg.get("torch_dtype", "fp16").lower()
    torch_dtype = {
        "fp16": __import__("torch").float16,
        "bf16": __import__("torch").bfloat16,
        "fp32": __import__("torch").float32,
    }.get(dtype, __import__("torch").float16)

    colorizer = InstanceColorizer(
        ColorOnlyConfig(
            base_model=m_cfg["base_inpaint_model"],
            controlnet_model=m_cfg["controlnet_model"],
            device=m_cfg.get("device", "cuda"),
            torch_dtype=torch_dtype,
            canny_low=int(i_cfg.get("canny_low", 100)),
            canny_high=int(i_cfg.get("canny_high", 200)),
            num_inference_steps=int(i_cfg.get("num_inference_steps", 20)),
            guidance_scale=float(i_cfg.get("guidance_scale", 7.5)),
            strength=float(i_cfg.get("strength", 0.70)),
        )
    )

    end = total_frames if args.max_frames < 0 else min(total_frames, args.frame_start + args.max_frames)

    for idx in range(args.frame_start, end):
        fpath = frames_dir / f"frame_{idx:06d}.png"
        if not fpath.exists():
            continue

        bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # sequential mode: apply each instance mask one-by-one on current frame
        current_rgb = rgb.copy()

        frame_instances = []
        for iid, obj in instances.items():
            mp = obj.get("mask_paths", {}).get(str(idx))
            if not mp:
                continue
            label = norm_label(obj.get("label", "object"))
            frame_instances.append((iid, label, Path(mp)))

        # Stable order for reproducibility
        frame_instances.sort(key=lambda x: int(x[0]))

        for iid, label, mpath in frame_instances:
            mask = read_mask(mpath, (h, w))
            if int((mask > 0).sum()) < 32:
                continue

            color_hint = color_hints.get(label, "")
            prompt = make_prompt(label, color_hint)

            result = colorizer.colorize_instance(
                current_rgb,
                mask,
                prompt=prompt,
                negative_prompt="deformed, distorted, extra limbs, blurry, low quality",
            )
            current_rgb = result["locked_rgb"]

            if r_cfg.get("save_debug_control", True):
                cv2.imwrite(str(debug_dir / f"frame_{idx:06d}_iid_{int(iid):04d}_mask.png"), mask)
                cv2.imwrite(
                    str(debug_dir / f"frame_{idx:06d}_iid_{int(iid):04d}_control.png"),
                    cv2.cvtColor(result["control_rgb"], cv2.COLOR_RGB2BGR),
                )

        out_bgr = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), out_bgr)
        print(f"[ok] frame {idx:06d}")

    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()