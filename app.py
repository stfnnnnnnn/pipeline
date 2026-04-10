import streamlit as st
from utils import load_css

st.set_page_config(page_title="MoodPlay", layout="centered", page_icon="")
load_css()

st.divider()

st.markdown(
    """
    <h1 style='text-align: center; font-size: 3rem; margin-bottom: 10px;'>
        <span class='mood-text'>Mood</span><span class='play-text'>Play</span> Video Colorizer
    </h1>
    <p style='text-align: center;'>
        Instance-guided diffusion-based video colorization with style conditioning.
    </p>
    """,
    unsafe_allow_html=True
)
    

col1, col2, col3 = st.columns([2,1,2])

with col2:
    if st.button("Upload Video", key="landing_btn", use_container_width=True):
        st.switch_page("pages/upload.py")

st.divider()