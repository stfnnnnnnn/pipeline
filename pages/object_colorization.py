import streamlit as st
import cv2
import numpy as np
from utils import get_frame_image, load_css

st.set_page_config(page_title="MoodPlay | Object Coloring", layout="wide", page_icon="")
load_css()

# Mock data
detection_events = [
    {"frame_id": 2, "new_objects": ["Person 1", "Hat 1", "Shirt 1"], "mask_id": ["0123", "4311", "4653"]},
    {"frame_id": 45, "new_objects": ["Person 1", "Hat 1", "Shirt 1"], "mask_id": ["0123", "4311", "4653"]},
    {"frame_id": 200, "new_objects": ["Person 1", "Hat 1", "Shirt 1"], "mask_id": ["0123", "4311", "4653"]}
]

# State Management
if "coloring_index" not in st.session_state:
    st.session_state.coloring_index = 0
if "master_color_map" not in st.session_state:
    st.session_state.master_color_map = {}

def next_frame():
    if st.session_state.coloring_index < len(detection_events) - 1:
        st.session_state.coloring_index += 1

def prev_frame():
    if st.session_state.coloring_index > 0:
        st.session_state.coloring_index -= 1

def update_color(obj_name):
    key_name = f"picker_{obj_name}"
    if key_name in st.session_state:
        st.session_state.master_color_map[obj_name] = st.session_state[key_name]

# If no video is uploaded
if "current_video_path" not in st.session_state:
    st.warning("Please upload a video first!")
    if st.button("Go to Upload"):
        st.switch_page("pages/upload.py")
    st.stop()
    
st.markdown("<h1 class='obj-header'>Object Colorization</h1>", unsafe_allow_html=True)
st.markdown("<p class='obj-sub'>Assign colors to objects frame-by-frame.</p>", unsafe_allow_html=True)
st.divider()

# Get current event data
current_idx = st.session_state.coloring_index
current_event = detection_events[current_idx]
frame_num = current_event["frame_id"]

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader(f"Frame #{frame_num}")
    
    # Extract and show the actual frame image
    video_path = st.session_state["current_video_path"]
    frame_image = get_frame_image(video_path, frame_num)
    
    if frame_image is not None:
        st.image(frame_image, use_column_width=True)
    else:
        st.error("Could not load frame from video.")

    # Navigation buttons
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.button("< Prev", on_click=prev_frame, disabled=(current_idx == 0), use_container_width=True, key="btn_prev")     
    with c2:
        st.markdown(
            f"<div style='text-align: center; padding-top: 10px; color: #666; font-weight: 600;'>"
            f"Set {current_idx + 1} / {len(detection_events)}"
            f"</div>", 
            unsafe_allow_html=True
        )
    with c3:
        st.button("Next >", on_click=next_frame, disabled=(current_idx == len(detection_events) - 1), use_container_width=True, key="btn_next")

with col_right:
    st.markdown("<h3 style='color: #f2a7c0; margin-bottom: 10px;'>Palette for this Frame</h3>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="background: rgba(242, 167, 192, 0.2); padding: 10px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #f2a7c0; color: #555; font-size: 0.9rem;">
            Assign objects for colors found in Frame.
        </div>
         """, unsafe_allow_html=True)

    for obj in current_event["new_objects"]:
        saved_color = st.session_state.master_color_map.get(obj, "#FFFFFF")
 
        with st.container(border=True):
            c_text, c_pick = st.columns([2, 1])
            
            with c_text:
                # Vertical center alignment trick
                st.markdown(f"<div style='justify-content: center; font-weight: 700; color: #333;'>{obj}</div>", unsafe_allow_html=True)
            
            with c_pick:
                st.color_picker(
                    f"Color for {obj}", 
                    value=saved_color, 
                    key=f"picker_{obj}", 
                    label_visibility="collapsed",
                    on_change=update_color,
                    args=(obj,)
                )

    st.divider()

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("Skip Custom Colorization", use_container_width=True, key="btn_skip"):
            st.switch_page("pages/styling.py")
    with b2:
        if st.button("Finish & Apply Style", type="primary", use_container_width=True, key="btn_finish"):
            st.switch_page("pages/styling.py")