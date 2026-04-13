import tempfile
from pathlib import Path

import cv2
import streamlit as st

from utils import load_css

st.set_page_config(page_title="MoodPlay | Upload Video", page_icon="", layout="centered")
load_css()

st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5rem; margin-bottom: 5px;'>
        <span class='mood-text'>Mood</span><span class='play-text'>Play</span> Video Colorizer
    </h1>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi"],
    help="Constraints: \n- Max Duration: 15 seconds \n- Max Size: 200MB"
)

if uploaded_file is not None:
    MAX_SIZE_MB = 200
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_SIZE_MB:
        st.error(f"File too large! Your video is {file_size_mb:.2f}MB. Max allowed is {MAX_SIZE_MB}MB.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration > 15:
            st.error(f"Video too long!\nYour video is {duration:.1f} seconds. The maximum allowed is 15 seconds.")
        else:
            st.success("Video upload success!")
            st.video(tfile.name)

            st.markdown("### Processing Settings")
            target_height = st.selectbox("Processing resolution", [720, 480], index=0)
            keyframe_stride = st.slider("Keyframe stride", min_value=4, max_value=24, value=12, step=1)

            st.markdown("### Mask Generation")
            st.markdown("### Tracking Limits")
            keep_chunk_default = st.checkbox(
                "Keep default max frames per chunk",
                value=True,
                help="Uses the config value unless you override it here.",
            )
            max_frames_per_chunk = None
            if not keep_chunk_default:
                max_frames_per_chunk = st.slider(
                    "Max frames per chunk",
                    min_value=4,
                    max_value=64,
                    value=16,
                    step=2,
                )

            total_frames = int(frame_count) if frame_count and frame_count > 0 else 0
            keep_full_video = st.checkbox(
                "Keep full video (no cap)",
                value=True,
                help="Process all frames in the uploaded video.",
            )
            max_frames = None
            if not keep_full_video:
                min_frames = 30 if total_frames >= 30 else max(1, total_frames)
                max_frames = st.slider(
                    "Max frames to process",
                    min_value=min_frames,
                    max_value=max(min_frames, min(600, total_frames or 600)),
                    value=min(180, max(min_frames, min(600, total_frames or 600))),
                    step=10,
                )

            if st.button("Proceed", type="primary", key="proceed_btn"):
                st.session_state["current_video_path"] = tfile.name
                st.session_state["target_height"] = int(target_height)
                st.session_state["keyframe_stride"] = int(keyframe_stride)
                st.session_state["max_total_frames"] = int(max_frames) if max_frames is not None else None
                st.session_state["max_frames_per_chunk"] = (
                    int(max_frames_per_chunk) if max_frames_per_chunk is not None else None
                )

                # reset previous run cache
                st.session_state.pop("segmentation_result", None)
                st.session_state.pop("detection_events", None)
                st.session_state.pop("event_index", None)
                st.session_state.pop("instance_filter", None)

                st.switch_page("pages/object_detection.py")
                st.stop()

st.divider()