import streamlit as st
import tempfile
import cv2     
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
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Check duration using cv2
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Safety check if fps is 0
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration > 15:
            st.error(f"Video too long!\nYour video is {duration:.1f} seconds. The maximum allowed is 15 seconds.")
        else:
            st.success(f"Video upload success!")
            st.session_state["current_video_path"] = tfile.name
            st.video(tfile.name) # Show the video preview
            
            if st.button("Proceed", type="primary", key="proceed_btn"):
                st.switch_page("pages/object_detection.py")
                st.stop()

st.divider()