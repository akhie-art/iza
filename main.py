import streamlit as st
import cv2
import mediapipe as mp
import time
from gtts import gTTS
import os
import numpy as np
from io import BytesIO # Digunakan untuk menyimpan MP3 di memori, bukan di disk

# --- INISIALISASI SESSION STATE ---
# Streamlit akan me-rerun script. Kita harus simpan state di session_state.
if 'last_wave_time' not in st.session_state:
    st.session_state.last_wave_time = 0.0
if 'prev_positions' not in st.session_state:
    st.session_state.prev_positions = []
if 'direction_changes' not in st.session_state:
    st.session_state.direction_changes = 0
if 'last_direction' not in st.session_state:
    st.session_state.last_direction = None
if 'wave_detected' not in st.session_state:
    st.session_state.wave_detected = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None


# --- INISIALISASI MEDIAPIPE (Di luar fungsi untuk efisiensi) ---
@st.cache_resource
def get_mediapipe_hands():
    """Inisialisasi Mediapipe Hands dan cache hasilnya"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_hands, hands, mp_hands.DrawingSpec, mp.solutions.drawing_utils

mp_hands, hands, DrawingSpec, mp_drawing = get_mediapipe_hands()


# --- KONSTANTA DETEKSI LAMBAIAN ---
WAVE_MOVEMENT_THRESHOLD = 0.15
WAVE_COOLDOWN_TIME = 3.0
POSITION_HISTORY_SIZE = 10
MIN_DIRECTION_CHANGES = 2

# --- FUNGSI BICARA (SPEAK FUNCTION) ---

def speak(text):
    """Generate teks dengan gTTS dan simpan di session state"""
    try:
        print(f"🔊 Mengucapkan: {text}")
        tts = gTTS(text=text, lang='id')
        
        # Simpan audio di memory buffer (BytesIO)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Simpan di session state agar bisa diputar di Streamlit
        st.session_state.audio_data = mp3_fp.read()
            
    except Exception as e:
        st.error(f"❌ Gagal membuat suara: {e}")
        st.session_state.audio_data = None
        
# --- FUNGSI DETEKSI LAMBAIAN UTAMA ---

def detect_wave(frame):
    """Memproses frame untuk deteksi tangan dan lambaian"""
    
    # Flip frame untuk tampilan cermin
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    rgb.flags.writeable = False
    result = hands.process(rgb)
    rgb.flags.writeable = True

    status_text = "Menunggu Tangan"
    status_color = (128, 128, 128)  # Abu-abu
    wave_detected_now = False # Flag untuk deteksi di frame ini
    
    # Ambil state dari session
    prev_positions = st.session_state.prev_positions
    direction_changes = st.session_state.direction_changes
    last_direction = st.session_state.last_direction
    wave_detected = st.session_state.wave_detected
    last_wave_time = st.session_state.last_wave_time

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
            
        # --- VISUALISASI TANGAN ---
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
            DrawingSpec(color=(0, 255, 0), thickness=2)
        )

        # Ambil posisi pergelangan tangan (wrist)
        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        
        # --- LOGIKA DETEKSI LAMBAIAN ---
        
        # Tambahkan posisi ke riwayat
        prev_positions.append(wrist_x)
        if len(prev_positions) > POSITION_HISTORY_SIZE:
            prev_positions.pop(0)

        # Deteksi perubahan arah
        if len(prev_positions) >= 3:
            recent_movement = prev_positions[-1] - prev_positions[-3]
            
            if abs(recent_movement) > 0.02:
                current_direction = "right" if recent_movement > 0 else "left"
                
                if last_direction is not None and last_direction != current_direction:
                    direction_changes += 1
                
                last_direction = current_direction
        
        # Cek jika ada gerakan lambaian
        if len(prev_positions) == POSITION_HISTORY_SIZE:
            movement = max(prev_positions) - min(prev_positions)
            
            if movement > WAVE_MOVEMENT_THRESHOLD and direction_changes >= MIN_DIRECTION_CHANGES:
                status_text = "LAMBAIAN TERDETEKSI!"
                status_color = (0, 165, 255)  # Oranye
                
                # --- EKSEKUSI SAPAAN (COOLDOWN CHECK) ---
                current_time = time.time()
                if not wave_detected and (current_time - last_wave_time) > WAVE_COOLDOWN_TIME:
                    print("\n👋 Gerakan lambaian terdeteksi!")
                    wave_detected_now = True
                    last_wave_time = current_time
                    wave_detected = True # Set wave_detected flag
                    status_text = "BERBICARA!"
                    status_color = (255, 0, 0)  # Biru
                    
                    # Panggil fungsi bicara
                    speak("Halo, saya Iza Cantik!")
                    
                    # Reset untuk deteksi berikutnya
                    prev_positions = []
                    direction_changes = 0
                    last_direction = None
            else:
                if movement > 0.05:
                    status_text = "Melambai..."
                    status_color = (0, 255, 0)  # Hijau
    
    # Reset flag wave_detected setelah cooldown
    current_time = time.time()
    if wave_detected and (current_time - last_wave_time) > WAVE_COOLDOWN_TIME:
        wave_detected = False
        
    # --- TAMPILKAN UI/DEBUG INFO (Menggunakan cv2.putText) ---
    
    # Tampilkan status
    cv2.putText(frame, 
                status_text, 
                (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                status_color, 
                3, 
                cv2.LINE_AA)
    
    # Tampilkan cooldown
    cooldown_remaining = max(0, last_wave_time + WAVE_COOLDOWN_TIME - current_time)
    if cooldown_remaining > 0:
        cv2.putText(frame, 
                    f"Cooldown: {cooldown_remaining:.1f}s", 
                    (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA)
    
    # Simpan state kembali
    st.session_state.prev_positions = prev_positions
    st.session_state.direction_changes = direction_changes
    st.session_state.last_direction = last_direction
    st.session_state.wave_detected = wave_detected
    st.session_state.last_wave_time = last_wave_time

    return frame, wave_detected_now

# --- APLIKASI STREAMLIT UTAMA ---

def main():
    st.set_page_config(page_title="Iza Cantik - Deteksi Lambaian", layout="wide")
    st.title("👋 Deteksi Lambaian Tangan - Iza Cantik")
    st.markdown("Unggah gambar tangan Anda atau gunakan Streamlit-Webcam untuk deteksi real-time (lebih direkomendasikan).")
    
    # Teks instruksi
    st.info("ℹ️ **Instruksi:** Unggah gambar tangan yang melambai atau gunakan kamera di bawah. Jika lambaian terdeteksi, Anda akan mendengar sapaan (hanya jika kamera berhasil diakses dan mikrofon diizinkan).")
    
    # 1. Input Gambar Statis (File Uploader)
    st.header("1. Unggah Gambar Tangan")
    uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Konversi file yang diunggah ke numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Proses frame
        processed_frame, wave_detected_now = detect_wave(frame)
        
        # Konversi BGR ke RGB untuk Streamlit
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("Hasil Deteksi")
        st.image(processed_frame_rgb, channels="RGB", caption="Gambar Hasil Deteksi")
        
        if wave_detected_now and st.session_state.audio_data:
            st.success("✅ Lambaian Terdeteksi! Memutar Audio...")
            st.audio(st.session_state.audio_data, format='audio/mp3')
            st.session_state.audio_data = None # Hapus audio setelah diputar
        elif wave_detected_now:
            st.warning("⚠️ Lambaian Terdeteksi, tapi audio tidak berhasil dibuat.")
    
    st.markdown("---")
    
    # 2. Input Kamera Langsung (Gunakan komponen kamera Streamlit)
    st.header("2. Ambil Foto (Deteksi Satu Frame)")
    camera_image = st.camera_input("Ambil foto tangan yang melambai")
    
    if camera_image is not None:
        # Konversi file bytes kamera ke numpy array
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Proses frame
        processed_frame, wave_detected_now = detect_wave(frame)
        
        # Konversi BGR ke RGB untuk Streamlit
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        st.subheader("Hasil Deteksi dari Kamera")
        st.image(processed_frame_rgb, channels="RGB", caption="Gambar Hasil Deteksi")
        
        if wave_detected_now and st.session_state.audio_data:
            st.success("✅ Lambaian Terdeteksi! Memutar Audio...")
            st.audio(st.session_state.audio_data, format='audio/mp3')
            st.session_state.audio_data = None # Hapus audio setelah diputar
        elif wave_detected_now:
            st.warning("⚠️ Lambaian Terdeteksi, tapi audio tidak berhasil dibuat.")


if __name__ == "__main__":
    main()

# --- CATATAN PENTING ---
# Untuk deteksi lambaian real-time yang sebenarnya (video), Anda sangat disarankan untuk menggunakan
# pustaka "streamlit-webrtc" yang dirancang untuk live video processing di Streamlit.
# Karena kode aslinya menggunakan cv2.VideoCapture loop, menggantinya dengan st.camera_input 
# adalah penggantian yang paling sederhana, namun hanya memproses satu frame saat foto diambil.
# Jika Anda ingin real-time:
# 1. Install 'streamlit-webrtc'
# 2. Ganti logika st.file_uploader dan st.camera_input dengan webrtc_streamer().
# 3. Sesuaikan fungsi callback di webrtc_streamer untuk memproses frame video.