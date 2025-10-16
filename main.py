import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
from ultralytics import YOLO
from gtts import gTTS
from io import BytesIO
import time

# ========================
# Konfigurasi Streamlit
# ========================
st.set_page_config(page_title="👋 Deteksi Tangan Iza Cantik", layout="wide")
st.title("👋 Deteksi Tangan - Iza Cantik (YOLO Streamlit Cloud)")
st.caption("Arahkan tangan Anda ke kamera untuk deteksi otomatis.")

# ========================
# Load model YOLO
# ========================
try:
    model = YOLO("yolov8n.pt")  # Model ringan bawaan YOLO
except Exception as e:
    st.error(f"❌ Gagal memuat model YOLO: {e}")
    model = None

# ========================
# Fungsi bicara
# ========================
def speak(text):
    tts = gTTS(text=text, lang='id')
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

# ========================
# Kelas Video Processor
# ========================
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_wave_time = 0
        self.cooldown = 3
        self.audio_data = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        status = "Mendeteksi..."
        color = (128, 128, 128)

        if model:
            results = model.predict(source=img, conf=0.5, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                cls = int(box.cls[0])
                if model.names[cls] in ["person"]:  # YOLO default tidak ada 'hand', pakai 'person' simulasi
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    status = "👋 Objek Terdeteksi!"
                    color = (0, 255, 0)

                    if time.time() - self.last_wave_time > self.cooldown:
                        self.last_wave_time = time.time()
                        self.audio_data = speak("Halo, saya Iza Cantik!")

        cv2.putText(img, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========================
# Konfigurasi WebRTC
# ========================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="iza-yolo",
    mode="recvonly",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ========================
# Putar suara jika deteksi
# ========================
if ctx.video_processor and ctx.video_processor.audio_data:
    st.audio(ctx.video_processor.audio_data, format="audio/mp3")
    ctx.video_processor.audio_data = None
