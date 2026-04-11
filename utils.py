from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st


def load_css(css_path: Optional[str] = None):
    if css_path is None:
        resolved_css_path = Path(__file__).parent / "assets" / "style.css"
    else:
        resolved_css_path = Path(__file__).parent / css_path

    try:
        with open(resolved_css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {resolved_css_path}. Please check your folder structure.")


def get_frame_image(video_path: str, frame_num: int) -> np.ndarray:
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return np.zeros((300, 400, 3), dtype=np.uint8)


def overlay_mask_on_frame(
    frame_rgb: np.ndarray,
    mask_path: str,
    color=(255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return frame_rgb

    # If frame/mask shape differ (because preview uses source video), resize mask to frame.
    if mask.shape[:2] != frame_rgb.shape[:2]:
        mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    out = frame_rgb.copy()
    m = mask > 0
    if np.any(m):
        out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def list_frame_instance_masks(mask_root: str, frame_idx: int) -> Dict[int, str]:
    """
    Reads:
      {mask_root}/frame_000123/instance_0007.png
    Returns:
      {7: ".../instance_0007.png", ...}
    """
    frame_dir = Path(mask_root) / f"frame_{frame_idx:06d}"
    if not frame_dir.exists():
        return {}

    out: Dict[int, str] = {}
    for p in sorted(frame_dir.glob("instance_*.png")):
        name = p.stem  # instance_0007
        try:
            iid = int(name.split("_")[1])
            out[iid] = str(p)
        except Exception:
            continue
    return out


def build_detection_events_from_result(result: Dict) -> List[Dict]:
    """
    Build events from backend result:
    [
      {"frame_idx": int, "frame_path": str, "instance_id": int, "label": str}
    ]
    Event source = instance keyframe indices.
    """
    events: List[Dict] = []
    instances = result.get("instances", [])

    for inst in instances:
        iid = int(inst["id"])
        label = str(inst.get("label", "unknown"))
        keyframes = inst.get("keyframes", [])

        # Try deriving frame idx from keyframe file names frame_000123.png
        for kf in keyframes:
            fp = Path(kf)
            stem = fp.stem  # frame_000123
            try:
                frame_idx = int(stem.split("_")[1])
            except Exception:
                continue
            events.append(
                {
                    "frame_idx": frame_idx,
                    "frame_path": str(kf),
                    "instance_id": iid,
                    "label": label,
                }
            )

    # Sort by frame then instance
    events.sort(key=lambda x: (x["frame_idx"], x["instance_id"]))
    return events