import cv2
import streamlit as st
import numpy as np
from modules import image2text

st.title("Webcam Image Classification")

# Checkbox para activar/desactivar el reconocimiento de imágenes
recognition_enabled = st.checkbox("Enable Image Recognition")

# Si la casilla de verificación está marcada, inicializar la webcam
if recognition_enabled:
    cap = cv2.VideoCapture(0)  # 0 corresponde a la webcam predeterminada
else:
    cap = None

image_list = []  # Lista para almacenar imágenes capturadas

while True:
    if recognition_enabled and cap is not None:
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

    # Botón para desactivar el reconocimiento de imágenes
    recognition_enabled = st.checkbox("Enable Image Recognition")

    # Botón para limpiar la lista de imágenes
    if st.button("Clear Images"):
        image_list = []

# Liberar recursos al finalizar
if cap is not None:
    cap.release()
