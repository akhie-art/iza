import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import av
import time
from gtts import gTTS
from io import BytesIO

# ==============================================
st.set_page_config(page_title="👋 Deteksi Tangan Iza Cantik", layout="wide")
st.title("👋 Deteksi Tangan - Iza Cantik (Streamlit Cloud Friendly)")
st.info("Arahkan tangan Anda ke kamera. Jika tangan terlihat, akan muncul landmark hijau ✋")

# ==============================================
def speak(text):
    """Ubah teks menjadi audio"""
    tts = gTTS(text=text, lang='id')
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

# ==============================================
class HandLandmarkProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1, detectionCon=0.7)
        self.last_wave_time = 0
        self.cooldown = 3
        self.audio_data = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        hands, img = self.detector.findHands(img)

        status_text = "Tidak ada tangan"
        color = (128, 128, 128)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]

            if len(lmList) > 0:
                status_text = "Tangan Terdeteksi ✋"
                color = (0, 255, 0)

                # Gerakan sederhana (lambaian simulasi)
                wrist_x = lmList[0][0]
                if abs(wrist_x - lmList[9][0]) > 70:
                    if time.time() - self.last_wave_time > self.cooldown:
                        self.last_wave_time = time.time()
                        status_text = "👋 Lambaian Terdeteksi!"
                        color = (0, 165, 255)
                        self.audio_data = speak("Halo, saya Iza Cantik!")

        cv2.putText(img, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==============================================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="iza-hands",
    mode="recvonly",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandLandmarkProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ==============================================
if ctx.video_processor and ctx.video_processor.audio_data:
    st.audio(ctx.video_processor.audio_data, format="audio/mp3")
    ctx.video_processor.audio_data = None
