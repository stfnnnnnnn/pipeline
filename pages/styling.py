import streamlit as st
from utils import load_css

st.set_page_config(page_title="MoodPlay | Choose Style", layout="wide", page_icon="")
load_css()

# If no video is uploaded
if "current_video_path" not in st.session_state:
    st.warning("Please upload a video first!")
    if st.button("Go to Upload"):
        st.switch_page("pages/upload.py")
    st.stop()


st.markdown("<h1 class='obj-header'>Style Studio</h1>", unsafe_allow_html=True)
st.markdown("<p class='obj-sub'>Select a mood from the gallery to apply to your video.</p>", unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("Original Video")
    video_path = st.session_state["current_video_path"]
    st.video(video_path)

with col_right:
    styles = [
        {"name": "Nostalgic", "icon": "🎞️", "desc": "Sepia tones & grain", "colors": ["#F4E4C1", "#D9C3A6", "#A68A64", "#735D48", "#8C7B70"]},
        {"name": "Neon City", "icon": "🌃", "desc": "Cyberpunk aesthetic", "colors": ["#FF0099", "#00F3FF", "#9D00FF", "#CCFF00", "#050510"]},
        {"name":"Neutral Realistic","icon":"🌧️","desc":"Desaturated blue","colors":["#B7C7D9","#8FA3B5","#6E8397","#4C6276","#2E3E4F"]},
        {"name":"Sunday Blues","icon":"📺","desc":"1950s Technicolor","colors":["#E85C5C","#2FAE8A","#4A74D6","#F0C66B","#F5E6D3"]}

    ]
    
    if "selected_style" not in st.session_state:
        st.session_state.selected_style = None
   
    st.markdown("<h3 style='color: #f2a7c0;'>Style Gallery</h3>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    
    for i, style in enumerate(styles):
        col = cols[i % 2]
        
        with col:
            is_selected = (st.session_state.selected_style == style["name"])
            btn_type = "primary" if is_selected else "secondary"
            
            # Create color circles HTML
            color_circles = "".join([f'<span class="color-circle" style="background-color: {color};"></span>' for color in style["colors"]])
            
            # Button label with emoji, name, and description
            label = f"{style['icon']} **{style['name']}**\n\n{style['desc']}"
            
            # The button with unique key
            if st.button(label, key=f"btn_style_{style['name']}", use_container_width=True, type=btn_type):
                # Toggle selection: if already selected, deselect it; otherwise select it
                if st.session_state.selected_style == style["name"]:
                    st.session_state.selected_style = None
                else:
                    st.session_state.selected_style = style["name"]
                st.rerun()
            
            # Display color circles right below the button
            st.markdown(f'<div class="mood-colors-container">{color_circles}</div>', unsafe_allow_html=True)
        
    st.divider()
   
    disable_next = st.session_state.selected_style is None
   
    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("Skip Custom Colorization", use_container_width=True, key="btn_skip"):
            st.switch_page("pages/results.py")
    with b2:
        if st.button("Finish & Apply Style", type="primary", disabled=disable_next, use_container_width=True, key="btn_finish"):
            st.switch_page("pages/results.py")