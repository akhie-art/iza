import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
from io import BytesIO

# ======================== KONFIGURASI STREAMLIT ========================
st.set_page_config(page_title="👋 Iza Cantik - Deteksi Lambaian Tangan", layout="wide")
st.title("👋 Deteksi Lambaian Tangan - Iza Cantik (Realtime)")

# ======================== KONFIGURASI MEDIAPIPE ========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ======================== KONSTANTA LOGIKA ========================
WAVE_MOVEMENT_THRESHOLD = 0.15
WAVE_COOLDOWN_TIME = 3.0
POSITION_HISTORY_SIZE = 10
MIN_DIRECTION_CHANGES = 2

# ======================== FUNGSI BICARA ========================
def speak(text):
    tts = gTTS(text=text, lang='id')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# ======================== VIDEO PROCESSOR ========================
class HandWaveProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prev_positions = []
        self.direction_changes = 0
        self.last_direction = None
        self.last_wave_time = 0.0
        self.wave_detected = False
        self.audio_data = None

    def detect_wave(self, wrist_x):
        self.prev_positions.append(wrist_x)
        if len(self.prev_positions) > POSITION_HISTORY_SIZE:
            self.prev_positions.pop(0)

        if len(self.prev_positions) >= 3:
            recent_movement = self.prev_positions[-1] - self.prev_positions[-3]
            if abs(recent_movement) > 0.02:
                current_direction = "right" if recent_movement > 0 else "left"
                if self.last_direction and self.last_direction != current_direction:
                    self.direction_changes += 1
                self.last_direction = current_direction

        if len(self.prev_positions) == POSITION_HISTORY_SIZE:
            movement = max(self.prev_positions) - min(self.prev_positions)
            if movement > WAVE_MOVEMENT_THRESHOLD and self.direction_changes >= MIN_DIRECTION_CHANGES:
                current_time = time.time()
                if not self.wave_detected and (current_time - self.last_wave_time) > WAVE_COOLDOWN_TIME:
                    self.last_wave_time = current_time
                    self.wave_detected = True
                    self.audio_data = speak("Halo, saya Iza Cantik!")
                    self.prev_positions = []
                    self.direction_changes = 0
                    self.last_direction = None
                    return True
        return False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        status_text = "Menunggu tangan..."
        status_color = (128, 128, 128)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                wave = self.detect_wave(wrist_x)
                if wave:
                    status_text = "👋 Lambaian Terdeteksi!"
                    status_color = (0, 165, 255)
                    print("Lambaian terdeteksi!")
                else:
                    status_text = "Tangan Terdeteksi"
                    status_color = (0, 255, 0)

        cv2.putText(img, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ======================== KONFIGURASI WEBRTC ========================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ======================== STREAMING ========================
ctx = webrtc_streamer(
    key="wave-detector",
    mode="recvonly",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandWaveProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ======================== AUDIO FEEDBACK ========================
if ctx.video_processor and ctx.video_processor.audio_data:
    st.audio(ctx.video_processor.audio_data, format="audio/mp3")
    ctx.video_processor.audio_data = None
