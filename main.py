import streamlit as st
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="👁 Deteksi Objek YOLO", layout="centered")
st.title("👁 Deteksi Objek dengan YOLO")

# Load model YOLO dengan cache
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
    st.success("✅ Model YOLO berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.header("📷 Upload Gambar/Video")
st.sidebar.info("Upload gambar untuk deteksi objek")

# Upload file
uploaded_file = st.file_uploader(
    "Pilih gambar untuk deteksi objek", 
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    # Baca gambar menggunakan PIL
    image = Image.open(uploaded_file)
    
    # Tampilkan gambar asli
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Gambar Asli")
        st.image(image, use_container_width=True)
    
    # Proses deteksi
    with st.spinner("🔍 Mendeteksi objek..."):
        # Konversi PIL ke numpy array untuk YOLO
        img_array = np.array(image)
        
        # Jalankan prediksi
        results = model(img_array, verbose=False)
        
        # Dapatkan gambar hasil deteksi
        annotated_img = results[0].plot()
        
        # Konversi BGR ke RGB (YOLO output dalam BGR)
        annotated_img_rgb = annotated_img[:, :, ::-1]
        
    with col2:
        st.subheader("🎯 Hasil Deteksi")
        st.image(annotated_img_rgb, use_container_width=True)
    
    # Tampilkan detail deteksi
    st.subheader("📊 Detail Deteksi")
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        detections = []
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            detections.append({
                "Objek": class_name,
                "Confidence": f"{confidence:.2%}"
            })
        
        st.table(detections)
        st.success(f"✅ Terdeteksi {len(boxes)} objek!")
    else:
        st.warning("⚠️ Tidak ada objek yang terdeteksi")

else:
    st.info("👆 Upload gambar untuk memulai deteksi objek")
    
    # Contoh gambar
    st.markdown("---")
    st.subheader("💡 Tips:")
    st.markdown("""
    - Upload gambar dengan format: JPG, PNG, JPEG, BMP, atau WebP
    - Model dapat mendeteksi 80 jenis objek (orang, mobil, hewan, dll)
    - Semakin jelas gambar, semakin akurat deteksinya
    - Ukuran file maksimal: 200MB
    """)
