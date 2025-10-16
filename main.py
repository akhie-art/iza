import streamlit as st
import time
import os
import numpy as np
from io import BytesIO
from gtts import gTTS
import sys

# ====================================================
# 🔍 DETEKSI ENVIRONMENT (Cloud vs Lokal)
# ====================================================
IS_CLOUD = "streamlit" in os.getcwd().lower() or "mount/src" in os.getcwd().lower()
st.write(f"🌐 Mode: {'CLOUD (Simulasi)' if IS_CLOUD else 'LOKAL (Deteksi Asli)'}")

# ====================================================
# 🧩 IMPORT LIBRARY OPENCV & MEDIAPIPE (opsional)
# ====================================================
try:
    import cv2
    import mediapipe as mp
    HAS_CV2 = True
    HAS_MEDIAPIPE = True
except ImportError:
    st.warning("⚠️ Tidak menemukan OpenCV / Mediapipe. Mengaktifkan mode simulasi.")
    import types
    HAS_CV2 = False
    HAS_MEDIAPIPE = False
    cv2 = types.SimpleNamespace(
        flip=lambda x, y: x,
        cvtColor=lambda x, y: x,
        COLOR_BGR2RGB=None,
        putText=lambda *a, **k: None,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0
    )
    mp = types.SimpleNamespace(solutions=types.SimpleNamespace(hands=None, drawing_utils=None))

# ====================================================
# 🧠 INISIALISASI SESSION STATE
# ====================================================
defaults = {
    'last_wave_time': 0.0,
    'prev_positions': [],
    'direction_changes': 0,
    'last_direction': None,
    'wave_detected': False,
    'audio_data': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ====================================================
# ✋ INISIALISASI MEDIAPIPE
# ====================================================
@st.cache_resource
def get_mediapipe_hands():
    if not HAS_MEDIAPIPE:
        return None, None, None, None
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_hands, hands, mp_hands.DrawingSpec, mp.solutions.drawing_utils

mp_hands, hands, DrawingSpec, mp_drawing = get_mediapipe_hands()

# ====================================================
# ⚙️ KONSTANTA
# ====================================================
WAVE_MOVEMENT_THRESHOLD = 0.15
WAVE_COOLDOWN_TIME = 3.0
POSITION_HISTORY_SIZE = 10
MIN_DIRECTION_CHANGES = 2

# ====================================================
# 🔊 FUNGSI SUARA
# ====================================================
def speak(text):
    try:
        tts = gTTS(text=text, lang='id')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        st.session_state.audio_data = mp3_fp.read()
    except Exception as e:
        st.error(f"❌ Gagal membuat suara: {e}")
        st.session_state.audio_data = None

# ====================================================
# 👋 FUNGSI DETEKSI LAMBAIAN
# ====================================================
def detect_wave(frame):
    """Deteksi lambaian tangan atau simulasi jika library tidak tersedia"""
    # MODE SIMULASI (Cloud tanpa cv2/mediapipe)
    if IS_CLOUD or not HAS_CV2 or not HAS_MEDIAPIPE or hands is None:
        cv2.putText(frame, "Simulasi: Lambaian Terdeteksi",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        speak("Halo, saya Iza Cantik! Mode simulasi aktif.")
        return frame, True

    # MODE LOKAL (dengan OpenCV + Mediapipe)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands.process(rgb)
    rgb.flags.writeable = True

    status_text = "Menunggu Tangan"
    status_color = (128, 128, 128)
    wave_detected_now = False

    prev_positions = st.session_state.prev_positions
    direction_changes = st.session_state.direction_changes
    last_direction = st.session_state.last_direction
    wave_detected = st.session_state.wave_detected
    last_wave_time = st.session_state.last_wave_time

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
            DrawingSpec(color=(0, 255, 0), thickness=2)
        )

        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        prev_positions.append(wrist_x)
        if len(prev_positions) > POSITION_HISTORY_SIZE:
            prev_positions.pop(0)

        if len(prev_positions) >= 3:
            recent_movement = prev_positions[-1] - prev_positions[-3]
            if abs(recent_movement) > 0.02:
                current_direction = "right" if recent_movement > 0 else "left"
                if last_direction and last_direction != current_direction:
                    direction_changes += 1
                last_direction = current_direction

        if len(prev_positions) == POSITION_HISTORY_SIZE:
            movement = max(prev_positions) - min(prev_positions)
            if movement > WAVE_MOVEMENT_THRESHOLD and direction_changes >= MIN_DIRECTION_CHANGES:
                status_text = "LAMBAIAN TERDETEKSI!"
                status_color = (0, 165, 255)
                current_time = time.time()
                if not wave_detected and (current_time - last_wave_time) > WAVE_COOLDOWN_TIME:
                    wave_detected_now = True
                    last_wave_time = current_time
                    wave_detected = True
                    speak("Halo, saya Iza Cantik!")
                    prev_positions, direction_changes, last_direction = [], 0, None
            else:
                if movement > 0.05:
                    status_text = "Melambai..."
                    status_color = (0, 255, 0)

    current_time = time.time()
    if wave_detected and (current_time - last_wave_time) > WAVE_COOLDOWN_TIME:
        wave_detected = False

    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                status_color, 3, cv2.LINE_AA)
    cooldown_remaining = max(0, last_wave_time + WAVE_COOLDOWN_TIME - current_time)
    if cooldown_remaining > 0:
        cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Simpan kembali state
    st.session_state.prev_positions = prev_positions
    st.session_state.direction_changes = direction_changes
    st.session_state.last_direction = last_direction
    st.session_state.wave_detected = wave_detected
    st.session_state.last_wave_time = last_wave_time

    return frame, wave_detected_now

# ====================================================
# 🖥️ APLIKASI STREAMLIT
# ====================================================
def main():
    st.set_page_config(page_title="Iza Cantik - Deteksi Lambaian", layout="wide")
    st.title("👋 Deteksi Lambaian Tangan - Iza Cantik")
    st.info("Unggah gambar tangan Anda atau gunakan kamera. Jika lambaian terdeteksi, Iza akan menyapa Anda!")

    # --- Upload gambar ---
    st.header("📤 Unggah Gambar")
    uploaded_file = st.file_uploader("Pilih gambar tangan...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        processed_frame, wave_detected_now = detect_wave(frame)
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                 channels="RGB", caption="Hasil Deteksi")
        if wave_detected_now and st.session_state.audio_data:
            st.audio(st.session_state.audio_data, format="audio/mp3")
            st.session_state.audio_data = None

    st.markdown("---")

    # --- Kamera ---
    st.header("📸 Ambil Foto dari Kamera")
    camera_image = st.camera_input("Ambil foto tangan melambai")

    if camera_image:
        frame = cv2.imdecode(np.frombuffer(camera_image.read(), np.uint8), 1)
        processed_frame, wave_detected_now = detect_wave(frame)
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                 channels="RGB", caption="Hasil Deteksi Kamera")
        if wave_detected_now and st.session_state.audio_data:
            st.audio(st.session_state.audio_data, format="audio/mp3")
            st.session_state.audio_data = None


if __name__ == "__main__":
    main()
