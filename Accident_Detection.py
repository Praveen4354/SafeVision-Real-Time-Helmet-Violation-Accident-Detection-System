import streamlit as st
import cv2
import tempfile
import os
import datetime
import torch
from ultralytics import YOLO
from PIL import Image
from twilio.rest import Client

def run():
    st.markdown("""
    <style>
        .stApp { background-color: #f4f7f6; }
        section[data-testid="stSidebar"] { background-color: #e0f2f1; }
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: #1b4332;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #2d6a4f;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #40916c;
            color: white;
        }
        .stFileUploader {
            background-color: #d8f3dc;
            border-radius: 8px;
        }
        .element-container p {
            font-size: 16px;
            color: #081c15;
        }
        .gan-label {
            color: black;
            font-weight: bold;
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)
    model = YOLO("bestw.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    account_sid = 'x'
    auth_token = 'x'
    twilio_phone_number = '+x'
    emergency_contact = '+x'
    client = Client(account_sid, auth_token)
    with st.sidebar:
        st.markdown(
        "<h1 style=' font-size: 40px; color: #4CAF50;'>🛡️ Accident Detection</h1>",
        unsafe_allow_html=True
    )
        st.markdown("""
        🚦 **Upload a video** to detect Accidents. 
                    
        📩 **Automatic SMS alerts** are sent to violators. 
                    
        🔍 **Accident is detected using YOLOV8 model**
                    
        ⚡ **Tip**: Upload a clear video for better detection.
        """)
        st.info("🔔 *Drive safe!!!*")
        st.markdown("---")

    st.markdown("<h2 style='color:#d00000;'>🚨 Accident Detection and Emergency Alert</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a video for accident detection", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_dir = tempfile.TemporaryDirectory()
        video_path = os.path.join(temp_dir.name, "accident_video.mp4")

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        accident_detected = False
        accident_frame = None
        accident_time = None

        frame_display = st.empty()
        detect_button = st.button("Start Detection")

        if detect_button:
            st.info("🚀 Detecting possible accidents...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb_frame, device=device)

                for r in results:
                    for box in r.boxes.data:
                        x1, y1, x2, y2, conf, cls = box.tolist()
                        if conf > 0.5 and not accident_detected:
                            accident_detected = True
                            accident_time = frame_count / fps

                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, f"Accident ({conf:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                            accident_frame = frame.copy()

                frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                frame_count += 1

            cap.release()

            if accident_detected:
                
                time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"🚑 Accident Detected at `{time_str}`")
                st.image('accident.jpg', caption="Accident Result")

                try:
                    message = client.messages.create(
                        body=f"🚨 Accident detected at {time_str}. Emergency services should be alerted immediately.",
                        from_=twilio_phone_number,
                        to=emergency_contact
                    )
                    st.success("📩 Emergency SMS sent successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to send SMS: {e}")
            else:
                st.info("✅ No accident detected.")

        temp_dir.cleanup()