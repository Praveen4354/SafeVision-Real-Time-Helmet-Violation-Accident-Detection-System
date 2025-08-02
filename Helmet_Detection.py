import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
import streamlit as st
import cv2
import tempfile
import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cvzone
import math
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from image_to_text import predict_number_plate
from twilio.rest import Client
import pyttsx3  
from paddleocr import PaddleOCR


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

    model = YOLO("best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classNames = ["with helmet", "without helmet", "rider", "number plate"]
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    class Discriminator(nn.Module):
        def __init__(self, img_channels=3, img_size=(64, 128), text_dim=50):
            super(Discriminator, self).__init__()
            self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
            self.fc1 = nn.Linear(256 * (img_size[0] // 8) * (img_size[1] // 8) + text_dim, 1024)
            self.fc2 = nn.Linear(1024, 1)

        def forward(self, img, text_embedding):
            x = torch.relu(self.conv1(img))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = torch.cat([x, text_embedding], dim=1)
            x = torch.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))

    def get_text_embedding(text, vocab_size=50):
        text_embedding = torch.zeros(vocab_size)
        for i, char in enumerate(text[:vocab_size]):
            text_embedding[i] = ord(char) % vocab_size
        return text_embedding

    def preprocess_image(image: Image.Image):
        transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0).to(device)


    disc = Discriminator().to(device)
    disc.load_state_dict(torch.load("discriminator_epoch_0.pth", map_location=device))
    disc.eval()


    database = pd.read_csv("vehicle_database.csv")
    account_sid = 'x'
    auth_token = 'x'
    twilio_phone_number = '+x'
    client = Client(account_sid, auth_token)


    with st.sidebar:
        st.markdown(
        "<h1 style=' font-size: 40px; color: #4CAF50;'>🛡️ SmartVision</h1>",
        unsafe_allow_html=True
    )
        st.markdown("""
        🚦 **Upload a video** to detect helmet violations and recognize number plates. 
                    
        📩 **Automatic SMS alerts** are sent to violators. 
                    
        🔍 **GAN verifies plate authenticity.**
                    
        🔊 **Voice alerts** will play when violations are detected.
                    
        ⚡ **Tip**: Upload a clear video for better detection.
        """)
        st.info("🔔 *Stay safe, wear your helmet!*")
        st.markdown("---")

    st.markdown("<h2 style=' color: #4CAF50;'>Helmet Violation Detection, Plate Recognition and GAN Validation with SMS & Voice Alerts</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_dir = tempfile.TemporaryDirectory()
        video_path = os.path.join(temp_dir.name, "uploaded_video.mp4")

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = os.path.join(temp_dir.name, "output.mp4")
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        detection_results = set()
        detected_vehicles = []
        plate_images = []
        riders_total = 0
        riders_without_helmet = 0
        processed_plates = set()

        seen_riders = []
        seen_violators = []

        def is_unique(center, seen_list, threshold=50):
            for c in seen_list:
                if abs(center[0] - c[0]) < threshold and abs(center[1] - c[1]) < threshold:
                    return False
            return True

        frame_display = st.empty()
        stop_button = st.button("Results")
        stop_processing = False

        while cap.isOpened():
            success, img = cap.read()
            if not success or stop_processing:
                break

            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(new_img, stream=True, device=device)
            rider_box = []

            for r in results:
                boxes = r.boxes
                xy = boxes.xyxy
                confidences = boxes.conf
                classes = boxes.cls
                new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)

                try:
                    new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
                    rows = new_boxes[torch.where(new_boxes[:, -1] == 2)]
                    for box in rows:
                        x1, y1, x2, y2 = map(int, box[:4])
                        rider_box.append((x1, y1, x2, y2))
                except:
                    pass

                for box in new_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box[4] * 100)) / 100
                    cls = int(box[5])

                    if classNames[cls] in classNames and conf >= 0.5:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        center = (cx, cy)

                        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                        cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10),
                                        scale=1.5, offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

                        if classNames[cls] == "rider":
                            if is_unique(center, seen_riders):
                                seen_riders.append(center)
                                riders_total += 1
                                rider_box.append((x1, y1, x2, y2))

                        elif classNames[cls] == "without helmet":
                            for rider in rider_box:
                                if x1 + 10 >= rider[0] and y1 + 10 >= rider[1] and x2 <= rider[2] and y2 <= rider[3]:
                                    if is_unique(center, seen_violators):
                                        seen_violators.append(center)
                                        riders_without_helmet += 1

                        elif classNames[cls] == "number plate":
                            crop = img[y1:y2, x1:x2]
                            try:
                                vehicle_number, conf_ocr = predict_number_plate(crop, ocr)
                                if vehicle_number:
                                    vehicle_number = vehicle_number.upper().strip()
                                    if len(vehicle_number) == 10 and vehicle_number not in processed_plates:
                                        processed_plates.add(vehicle_number)
                                        detected_vehicles.append(vehicle_number)
                                        plate_images.append((vehicle_number, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                                        cvzone.putTextRect(img, f"{vehicle_number} {round(conf_ocr*100, 2)}%", (x1, y1 - 50),
                                                        scale=1.5, offset=10, thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
                                        detection_results.add(f"Detected: {vehicle_number}")
                            except Exception as e:
                                print(e)

            output.write(img)
            frame_display.image(img, channels="RGB")

            if stop_button:
                stop_processing = True
                break

        cap.release()
        output.release()
        del output


        st.subheader("Final Detection Results:")
        for result in sorted(detection_results):
            st.write(result)


        st.subheader("🧠 Detection Summary")
        st.write(f"👥 {riders_total} rider(s) detected. {riders_without_helmet//2} rider(s) were without helmet(s). SMS alerts were sent to respective owner.")

        if plate_images:
            st.markdown("<h3 style='color:black;'>🔍 GAN Validation</h3>", unsafe_allow_html=True)
        
            unique_plates = {}
            for plate, plate_img in plate_images:
                if plate not in unique_plates:
                    unique_plates[plate] = plate_img
            

            for plate, plate_img in unique_plates.items():
                img_tensor = preprocess_image(plate_img)
                text_embedding = get_text_embedding(plate).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = disc(img_tensor, text_embedding).item()
                    label = "REAL ✅" if pred > 0.5 else "FAKE ⚠️"
                    st.image(plate_img, width=300)
                    st.markdown(f"<h3 style='color:black;'> {plate} → {label} (Confidence: {pred:.2f}) </h3>", unsafe_allow_html=True)

        st.subheader("📩 Sending SMS Alerts")
        for plate in set(detected_vehicles):
            row = database[database['VehicleNumber'] == plate]
            if not row.empty:
                phone = str(row.iloc[0]['PhoneNumber']).strip()
                if not phone.startswith('+'):
                    phone = '+91' + phone
                try:
                    pay = "https://thirushika24.github.io/paynowapp/"
                    message = client.messages.create(
                        body=f"Alert! Rider of vehicle {vehicle_number} detected without helmet. As per traffic regulations, a fine will be imposed for this violation. Please ensure safety.Pay Link {pay}",
                        from_=twilio_phone_number,
                        to=phone
                    )
                    st.success(f"SMS sent to {phone} for {plate}")
                except Exception as e:
                    st.error(f"Failed to send SMS to {phone}: {e}")
            else:
                st.warning(f"No phone number found for {plate}")

        summary_text = f"{riders_total} riders detected. {riders_without_helmet} without helmets. SMS alerts were sent to respective owner."
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(summary_text)
        engine.runAndWait()

        temp_dir.cleanup()