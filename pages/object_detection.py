import streamlit as st
import time
from utils import load_css, get_frame_image

st.set_page_config(page_title="MoodPlay | Object Detection", layout="wide", page_icon="")
load_css()

# If no video is uploaded, boot them back
if "current_video_path" not in st.session_state:
    st.warning("Please upload a video first!")
    if st.button("Go to Upload"):
        st.switch_page("pages/upload.py")
    st.stop()

st.markdown("<h1 class='obj-header'>Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='obj-sub'>Review the keyframes where new objects were identified.</p>", unsafe_allow_html=True)
st.divider()

# --- NEW: Simulated Backend Progress Bar ---
if "preprocessing_done" not in st.session_state:
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    jobs = [
        ("1/4: Extracting frames...", 25),
        ("2/4: Detecting objects with YOLO...", 50),
        ("3/4: Generating SAM masks...", 75),
        ("4/4: Tracking instances across frames...", 100)
    ]
    
    for text, percent in jobs:
        progress_text.text(f"Status: {text}")
        time.sleep(1.5) # Simulate backend processing time
        progress_bar.progress(percent)
        
    time.sleep(0.5)
    progress_text.empty()
    progress_bar.empty()
    st.session_state.preprocessing_done = True
# -------------------------------------------

# Mock Data
detection_events = [
    {"frame_id": 2, "new_objects": ["Person 1", "Hat 1", "Shirt 1"], "mask_id": ["0123", "4311", "4653"]},
    {"frame_id": 45, "new_objects": ["Person 1", "Hat 1", "Shirt 1"], "mask_id": ["0123", "4311", "4653"]},
    {"frame_id": 200, "new_objects": ["Person 1", "Hat 1", "Shirt 1"], "mask_id": ["0123", "4311", "4653"]}
]

if "event_index" not in st.session_state:
    st.session_state.event_index = 0

def next_frame():
    if st.session_state.event_index < len(detection_events) - 1:
        st.session_state.event_index += 1

def prev_frame():
    if st.session_state.event_index > 0:
        st.session_state.event_index -= 1

# Main UI Layout
current_idx = st.session_state.event_index
current_event = detection_events[current_idx]
frame_num = current_event["frame_id"]

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader(f"Frame #{frame_num}")
    frame_image = get_frame_image(st.session_state["current_video_path"], frame_num)
    st.image(frame_image, use_column_width=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.button("< Prev", on_click=prev_frame, disabled=(current_idx == 0), use_container_width=True)
    with c2:
        st.markdown(f"<div style='text-align: center; padding-top: 10px; color: #666; font-weight: 600;'>Set {current_idx + 1} / {len(detection_events)}</div>", unsafe_allow_html=True)
    with c3:
        st.button("Next >", on_click=next_frame, disabled=(current_idx == len(detection_events) - 1), use_container_width=True)

with col_right:
    st.subheader("New Instances Found")
    
    for obj, mask in zip(current_event["new_objects"], current_event["mask_id"]):      
        st.markdown(f"""
            <div class="custom-card">
                <div style="font-weight: 700; font-size: 1.2rem; color: #333; margin-bottom: 4px;">{obj}</div>
                <div style="font-size: 0.95rem; color: #555;">Mask ID: #{mask}</div>
            </div>
        """, unsafe_allow_html=True)

    if st.button("Confirm and Colorize", type="primary", use_container_width=True):
        st.switch_page("pages/object_colorization.py")