import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
from io import BytesIO
import tempfile

st.set_page_config(page_title="👋 Deteksi Kamera Streamlit", layout="centered")
st.title("👋 Deteksi Kamera (Tanpa Error Build)")

# Load model YOLO default
model = YOLO("yolov8n.pt")

# Pilih sumber kamera
st.sidebar.header("📷 Pilihan Kamera")
source = st.sidebar.selectbox("Pilih sumber video:", ["Upload file", "Gunakan kamera"])

# Jika upload video
if source == "Upload file":
    file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            results = model(frame, verbose=False)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")
        cap.release()

# Jika pakai kamera
elif source == "Gunakan kamera":
    st.write("Klik tombol di bawah untuk membuka kamera:")
    run = st.checkbox("Mulai Kamera")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak bisa membuka kamera!")
            break
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        FRAME_WINDOW.image(annotated, channels="BGR")
    cap.release()
