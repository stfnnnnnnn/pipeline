from pathlib import Path
from typing import Dict, List

import streamlit as st

from utils import (
    load_css,
    get_frame_image,
    list_frame_instance_masks,
    overlay_mask_on_frame,
    build_detection_events_from_result,
)

# backend entrypoint
from src.perception.segmentation import process_video_for_segmentation

st.set_page_config(page_title="MoodPlay | Object Detection", layout="wide", page_icon="")
load_css()

if "current_video_path" not in st.session_state:
    st.warning("Please upload a video first!")
    if st.button("Go to Upload"):
        st.switch_page("pages/upload.py")
    st.stop()

st.markdown("<h1 class='obj-header'>Object Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='obj-sub'>Review keyframes and processed segmentation masks for tracked instances.</p>",
    unsafe_allow_html=True
)
st.divider()


def run_backend_once():
    progress_text = st.empty()
    progress_bar = st.progress(0)

    def cb(cur: int, total: int, msg: str):
        pct = int((cur / max(total, 1)) * 100)
        progress_text.text(f"Status: {msg}")
        progress_bar.progress(max(0, min(100, pct)))

    try:
        result = process_video_for_segmentation(
            video_path=st.session_state["current_video_path"],
            progress_callback=cb,
            target_height=st.session_state.get("target_height", 720),
            keyframe_stride=st.session_state.get("keyframe_stride", 12),
        )
    finally:
        progress_text.empty()
        progress_bar.empty()

    st.session_state["segmentation_result"] = result
    st.session_state["detection_events"] = build_detection_events_from_result(result)
    st.session_state["event_index"] = 0
    st.session_state["instance_filter"] = "All"


if "segmentation_result" not in st.session_state:
    run_backend_once()

result = st.session_state["segmentation_result"]
events = st.session_state.get("detection_events", [])

if not events:
    st.warning("No detected instances found in this video.")
    if st.button("Upload another video"):
        st.switch_page("pages/upload.py")
    st.stop()

if "event_index" not in st.session_state:
    st.session_state["event_index"] = 0


def next_frame():
    if st.session_state.event_index < len(events) - 1:
        st.session_state.event_index += 1


def prev_frame():
    if st.session_state.event_index > 0:
        st.session_state.event_index -= 1


# ---------- Sidebar controls ----------
with st.sidebar:
    st.subheader("Review Controls")
    if st.button("Re-run processing"):
        st.session_state.pop("segmentation_result", None)
        st.session_state.pop("detection_events", None)
        st.session_state.pop("event_index", None)
        st.rerun()

    all_instance_ids = sorted({e["instance_id"] for e in events})
    instance_options = ["All"] + [str(i) for i in all_instance_ids]
    current_filter = st.selectbox("Filter by instance ID", instance_options, index=0)
    st.session_state["instance_filter"] = current_filter


# apply filter
filtered_events = events
if st.session_state["instance_filter"] != "All":
    iid = int(st.session_state["instance_filter"])
    filtered_events = [e for e in events if e["instance_id"] == iid]

if not filtered_events:
    st.warning("No events for selected instance.")
    st.stop()

# clamp index if filter changed
if st.session_state.event_index >= len(filtered_events):
    st.session_state.event_index = 0

current_idx = st.session_state.event_index
current_event = filtered_events[current_idx]

frame_idx = current_event["frame_idx"]
frame_path = current_event["frame_path"]
instance_id = current_event["instance_id"]
label = current_event["label"]

frame_rgb = get_frame_image(st.session_state["current_video_path"], frame_idx)

# choose mask for current frame+instance
mask_root = result["mask_root"]
mask_map = list_frame_instance_masks(mask_root, frame_idx)
mask_path = mask_map.get(instance_id)

if mask_path is not None:
    preview = overlay_mask_on_frame(frame_rgb, mask_path, color=(255, 64, 64), alpha=0.45)
    status_label = "Masked"
else:
    preview = frame_rgb
    status_label = "Detected (no mask file on this frame)"

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader(f"Frame #{frame_idx}")
    st.image(preview, use_container_width=True)
    st.caption(f"Status: {status_label}")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.button("< Prev", on_click=prev_frame, disabled=(current_idx == 0), use_container_width=True)
    with c2:
        st.markdown(
            f"<div style='text-align:center;padding-top:10px;color:#666;font-weight:600;'>"
            f"Event {current_idx + 1} / {len(filtered_events)}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.button("Next >", on_click=next_frame, disabled=(current_idx == len(filtered_events) - 1), use_container_width=True)

with col_right:
    st.subheader("Detected Instance")
    st.markdown(
        f"""
        <div class="custom-card">
            <div style="font-weight:700;font-size:1.1rem;color:#333;">Instance #{instance_id}</div>
            <div style="font-size:0.95rem;color:#555;">Label: {label}</div>
            <div style="font-size:0.95rem;color:#555;">Frame: {frame_idx}</div>
            <div style="font-size:0.95rem;color:#555;">Status: {status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # show available mask files for this frame
    frame_masks = list_frame_instance_masks(mask_root, frame_idx)
    st.markdown("**Masks in this frame**")
    if frame_masks:
        for iid, p in sorted(frame_masks.items()):
            st.code(f"instance_{iid:04d}: {p}", language="text")
    else:
        st.info("No mask files found for this frame.")

    if st.button("Confirm and Colorize", type="primary", use_container_width=True):
        st.switch_page("pages/object_colorization.py")