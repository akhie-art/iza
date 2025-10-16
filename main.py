import streamlit as st
import numpy as np
from PIL import Image
import tempfile

# Import cv2 dengan error handling
try:
    import cv2
except ImportError as e:
    st.error(f"Error importing cv2: {e}")
    st.info("Pastikan packages.txt sudah ada dan app sudah di-reboot")
    st.stop()

# Import YOLO dengan error handling
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Error importing YOLO: {e}")
    st.stop()

st.set_page_config(page_title="👁 Deteksi Kamera Streamlit", layout="centered")
st.title("👁 Deteksi Kamera (Fixed Build)")

# Load model YOLO default
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
    st.success("✅ Model YOLO berhasil dimuat!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Pilih sumber kamera
st.sidebar.header("📷 Pilihan Kamera")
source = st.sidebar.selectbox("Pilih sumber video:", ["Upload file", "Gunakan kamera"])

# Jika upload video
if source == "Upload file":
    file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(file.read())
            video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        
        st.info("🎬 Memproses video...")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Proses setiap 3 frame untuk performa lebih baik
            if frame_count % 3 == 0:
                frame = cv2.resize(frame, (640, 480))
                results = model(frame, verbose=False)
                annotated = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_rgb, use_container_width=True)
        
        cap.release()
        st.success("✅ Video selesai diproses!")

# Jika pakai kamera
elif source == "Gunakan kamera":
    st.warning("⚠️ Fitur kamera real-time tidak didukung di Streamlit Cloud")
    st.info("💡 Gunakan fitur 'Upload file' untuk mendeteksi objek dari video yang sudah direkam")
    
    st.markdown("""
    ### Alternatif:
    1. Rekam video menggunakan kamera HP/laptop
    2. Upload video tersebut menggunakan opsi 'Upload file'
    3. Sistem akan mendeteksi objek dari video yang diupload
    """)
