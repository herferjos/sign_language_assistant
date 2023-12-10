import cv2
import streamlit as st
import numpy as np
# from modules import image2text
import modules

st.title("Webcam Image Classification")

# Inicializar la webcam al hacer clic en un botón
start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

cap = None
image_list = []  # Lista para almacenar imágenes capturadas

if start_button:
    cap = cv2.VideoCapture(0)  # 0 corresponde a la webcam predeterminada

# Bucle principal
while cap is not None and stop_button is False:
    ret, frame = cap.read()

    # Almacena el frame en la lista
    image_list.append(frame)

    # Limita la lista a las últimas 20 imágenes
    image_list = image_list[-20:]

    # Muestra la imagen más reciente
    st.image(frame, channels="BGR", use_column_width=True, caption="Latest Image")

    # Realiza la clasificación de imágenes cada 10 frames
    if len(image_list) % 10 == 0:
        predictions = image2text(image_list)

        # Muestra las predicciones
        st.text("Predictions: " + "".join(predictions))

# Liberar recursos al finalizar
if cap is not None:
    cap.release()

# Botón para limpiar la lista de imágenes
if st.button("Clear Images"):
    image_list = []
