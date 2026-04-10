import streamlit as st
import time
from pathlib import Path
from utils import load_css

st.set_page_config(page_title="MoodPlay | Final Result", layout="wide", page_icon="")
load_css()

# Mock rendering processing
if "processing_complete" not in st.session_state:
    with st.spinner("Rendering final video with Style Transfer & Color Masks..."):
        time.sleep(3)
    st.session_state.processing_complete = True

if "selected_style" not in st.session_state or "master_color_map" not in st.session_state:
    st.warning("No session data found. Please start from the beginning.")
    if st.button("Go Home"):
        st.switch_page("pages/upload.py")
    st.stop()

style_name = st.session_state["selected_style"]
color_map = st.session_state["master_color_map"]

THIS_DIR = Path(__file__).resolve().parent
video_path = THIS_DIR / "colorized.mp4"

st.markdown("<h1 class='obj-header'>Final Result</h1>", unsafe_allow_html=True)
st.markdown("<p class='obj-sub'>Your video has been successfully processed!</p>", unsafe_allow_html=True)
st.divider()

col_result, col_summary = st.columns([1.5, 1], gap="large")

with col_result:
    st.subheader("Processed Video")
    if video_path.exists():
        st.video(str(video_path))
        with open(video_path, "rb") as file:
            st.download_button("Download Video", data=file, file_name="colorized.mp4", mime="video/mp4", type="primary", use_container_width=True)
    else:
        st.info("[ Video Player Placeholder - colorized.mp4 not found ]")
        st.button("Download Video", type="primary", use_container_width=True, disabled=True)

with col_summary:
    st.subheader("Processing Summary")

    st.markdown(f"""
        <div class="custom-card">
            <div style="font-weight: 600; font-size: 1rem; color: #555;">Style Applied</div>
            <div style="font-weight: 700; font-size: 1.8rem; color: #333;">{style_name if style_name else "None"}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**Objects Recolored:**")

    if len(color_map) > 0:
        for obj, color_hex in color_map.items():
            st.markdown(f"""
                <div class="custom-card" style="display: flex; align-items: center; gap: 15px;">
                    <div style="background-color: {color_hex}; width: 40px; height: 40px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.1); flex-shrink: 0;"></div>
                    <div>
                        <div style="font-weight: 700; font-size: 1.2rem; color: #333;">{obj}</div>
                        <div style="font-size: 0.95rem; color: #555;">Hex: {color_hex}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific objects were recolored.")

st.divider()

if st.button("Return Home", use_container_width=True):
    # Clear state for new video
    for key in ["current_video_path", "preprocessing_done", "processing_complete", "master_color_map", "selected_style", "event_index", "coloring_index"]:
        if key in st.session_state:
            del st.session_state[key]
    st.switch_page("pages/upload.py")