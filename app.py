import cv2
import streamlit as st
import numpy as np
# from modules import image2text
import modules

st.title("Webcam Image Classification")

if 'cap' not in st.session_state:
    st.session_state.cap = None
    
if 'predictions' not in st.session_state:
    st.session_state.predictions = ""

if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)

stop_button = st.button("Stop Webcam")

while cap is not None and stop_button is False:
    ret, frame = cap.read()

    st.session_state.predictions = st.session_state.predictions + (image2text(frame))

if stop_button is True:
    cap.release()
    st.write(st.session_state.predictions)

    if st.button("Restart"):
        st.session_state.predictions = ""
        st.rerun()
