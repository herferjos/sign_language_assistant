import streamlit as st
import cv2
import numpy as np
from PIL import Image
from modules import image2text

st.title("Webcam Image Classification")

if 'cap' not in st.session_state:
    st.session_state.cap = None
    
if 'predictions' not in st.session_state:
    st.session_state.predictions = ""

if st.button("Start Webcam"):
    st.session_state.cap = cv2.VideoCapture(0)

stop_button = st.button("Stop Webcam")

while st.session_state.cap is not None and stop_button is False:
    ret, frame = st.session_state.cap.read()
    
    cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)
    
    st.session_state.predictions = st.session_state.predictions + (image2text(pil_image))

if stop_button is True:
    st.session_state.cap.release()
    st.write(st.session_state.predictions)

    if st.button("Restart"):
        st.session_state.predictions = ""
        st.rerun()
