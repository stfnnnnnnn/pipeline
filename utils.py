import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

def load_css(css_path: Optional[str] = None):
    """Reads a CSS file and injects it, defaulting to assets/style.css."""
    # Resolve a provided path relative to the project root (this file's folder).
    if css_path is None:
        resolved_css_path = Path(__file__).parent / "assets" / "style.css"
    else:
        resolved_css_path = Path(__file__).parent / css_path
    
    try:
        with open(resolved_css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {resolved_css_path}. Please check your folder structure.")

def get_frame_image(video_path, frame_num):
    """Safely extract a frame or return a blank placeholder if it fails."""
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        pass
    # Fallback placeholder
    return np.zeros((300, 400, 3), dtype=np.uint8)