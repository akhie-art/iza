import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import time
from gtts import gTTS
from io import BytesIO

# ======================================================
# KONFIGURASI STREAMLIT
# ======================================================
st.set_page_config(page_title="👋 Deteksi Tangan Iza Cantik", layout="wide")
st.title("👋 Deteksi Tangan - Iza Cantik (Cloud Friendly)")
st.caption("Arahkan tangan ke kamera. Landmark muncul secara otomatis jika terdeteksi.")

# ======================================================
# TRY IMPORT CVZONE (JIKA ADA)
# ======================================================
try:
    from cvzone.HandTrackingModule import HandDetector
    detector_available = True
except Exception as e:
    st.warning("⚠️ Modul cvzone tidak tersedia di environment ini. Menjalankan mode simulasi.")
    detector_available = False

# ======================================================
# KONFIGURASI AUDIO
# ======================================================
def speak(text):
    """Konversi teks jadi audio (mp3 bytes)"""
    try:
        tts = gTTS(text=text, lang='id')
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

# ======================================================
# VIDEO PROCESSOR
# ======================================================
class HandProcessor(VideoProcessorBase):
    def __init__(self):
        if detector_available:
            self.detector = HandDetector(maxHands=1, detectionCon=0.7)
        else:
            self.detector = None
        self.last_wave_time = 0
        self.cooldown = 3
        self.audio_data = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        color = (128, 128, 128)
        status = "Tidak ada tangan"

        if self.detector:
            hands, img = self.detector.findHands(img)
            if hands:
                status = "Tangan terdeteksi ✋"
                color = (0, 255, 0)
                hand = hands[0]
                lmList = hand["lmList"]

                # Deteksi lambaian sederhana
                wrist_x = lmList[0][0]
                index_x = lmList[8][0]
                if abs(wrist_x - index_x) > 70:
                    if time.time() - self.last_wave_time > self.cooldown:
                        self.last_wave_time = time.time()
                        status = "👋 Lambaian Terdeteksi!"
                        color = (0, 165, 255)
                        self.audio_data = speak("Halo, saya Iza Cantik!")

        cv2.putText(img, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ======================================================
# KONFIGURASI KAMERA
# ======================================================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="iza-wave",
    mode="recvonly",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ======================================================
# PUTAR AUDIO SAAT TERDETEKSI LAMBAIAN
# ======================================================
if ctx.video_processor and ctx.video_processor.audio_data:
    st.audio(ctx.video_processor.audio_data, format="audio/mp3")
    ctx.video_processor.audio_data = None
