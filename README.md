# SafeVision: Real-Time Helmet Violation & Accident Detection System

SafeVision is a Streamlit-based web application designed to enhance road safety by detecting helmet violations and accidents in real-time using computer vision and machine learning. It leverages YOLOv8 for object detection, PaddleOCR for number plate recognition, and a GAN-based discriminator for plate authenticity verification. The system sends SMS alerts to vehicle owners or emergency contacts via Twilio and provides voice alerts for detection summaries using `pyttsx3`.

## Features
- **Helmet Violation Detection**:
  - Detects riders, helmets, and number plates in video streams using YOLOv8.
  - Identifies riders without helmets and extracts vehicle number plates using PaddleOCR.
  - Verifies number plate authenticity using a GAN-based discriminator.
  - Sends SMS alerts to vehicle owners with a payment link for violations via Twilio.
  - Provides voice alerts summarizing detection results using `pyttsx3`.
- **Accident Detection**:
  - Detects accidents in video streams using YOLOv8.
  - Sends SMS alerts to emergency contacts via Twilio when accidents are detected.
- **Streamlit Interface**:
  - User-friendly web interface for uploading videos and viewing detection results.
  - Displays detected objects, number plates, GAN validation results, and detection summaries.
- **Database Integration**:
  - Queries `vehicle_database.csv` to map vehicle numbers to phone numbers for SMS alerts.

## Project Structure
```
SafeVision/
├── home.py                   # Main entry point for the Streamlit app
├── helmet_detection.py       # Helmet violation detection, OCR, GAN, and alerts
├── accident_detection.py     # Accident detection and emergency alerts
├── image_to_text.py          # OCR for number plate recognition
├── data/
│   └── vehicle_database.csv  # Database of vehicle numbers and phone numbers
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── accident.jpg             # Static image for accident detection (optional)
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Praveen4354/SafeVision.git
   cd SafeVision
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Additional Dependencies**:
   - On Windows, install `pyttsx3` dependencies:
     ```bash
     pip install pywin32
     pip install pypiwin32
     ```
   - Ensure `tesseract-ocr` is installed for PaddleOCR (if required):
     - Windows: Download and install from [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
     - Linux: `sudo apt-get install tesseract-ocr`
     - macOS: `brew install tesseract`

5. **Provide Model Files**:
   - You need to provide the following model files and place them in a `models/` directory:
     - `best.pt`: YOLOv8 model for helmet violation detection.
     - `bestw.pt`: YOLOv8 model for accident detection.
     - `discriminator_epoch_0.pth`: GAN Discriminator for number plate validation.
   - These models are not included due to their large size. Obtain or train compatible YOLOv8 and GAN models and place them in `SafeVision/models/`.

6. **Prepare Data File**:
   - Place `vehicle_database.csv` in the `data/` directory with columns `VehicleNumber` and `PhoneNumber` (e.g., `+91xxxxxxxxxx` format).

7. **Set Up Twilio**:
   - Update `helmet_detection.py` and `accident_detection.py` with your Twilio `account_sid`, `auth_token`, and `twilio_phone_number`.
   - Alternatively, store credentials in a `.env` file (add `.env` to `.gitignore`):
     ```plaintext
     TWILIO_ACCOUNT_SID=your_account_sid
     TWILIO_AUTH_TOKEN=your_auth_token
     TWILIO_PHONE_NUMBER=your_twilio_phone_number
     ```
   - Update scripts to use environment variables:
     ```python
     import os
     account_sid = os.getenv("TWILIO_ACCOUNT_SID")
     auth_token = os.getenv("TWILIO_AUTH_TOKEN")
     twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
     ```

## Usage
1. **Run the Application**:
   ```bash
   streamlit run home.py
   ```
   This opens the Streamlit app in your browser.

2. **Navigate the App**:
   - Use the sidebar to select "Helmet Detection" or "Accident Detection."
   - Upload a video file (`.mp4`, `.avi`, or `.mov`).
   - For Helmet Detection: Click "Results" to stop processing and view detected plates, GAN validation, and summaries.
   - For Accident Detection: Click "Start Detection" to process the video and view results.

3. **View Results**:
   - Helmet Detection: Displays detected riders, violations, number plates, GAN validation (REAL/FAKE), and SMS/voice alert statuses.
   - Accident Detection: Displays accident locations and SMS alert statuses.

## Dependencies
Listed in `requirements.txt`:
```
streamlit
ultralytics
opencv-python
paddleocr
cvzone
torch
torchvision
pandas
numpy
Pillow
twilio
pyttsx3
```

## File Descriptions
- **home.py**: Entry point for the Streamlit app, providing navigation between helmet and accident detection.
- **helmet_detection.py**: Detects riders without helmets, recognizes number plates, validates plates with GAN, and sends SMS/voice alerts.
- **accident_detection.py**: Detects accidents and sends emergency SMS alerts.
- **image_to_text.py**: Provides OCR functionality for number plate recognition using PaddleOCR.
- **vehicle_database.csv**: Maps vehicle numbers to phone numbers for SMS alerts.
- **accident.jpg**: Static image for accident detection results (optional, may need dynamic generation).

## Notes
- **Model Files**: You must provide `best.pt`, `bestw.pt`, and `discriminator_epoch_0.pth` in `models/`. These are not included due to their size. Ensure compatibility with YOLOv8 and the GAN Discriminator architecture in `helmet_detection.py`.
- **Hardware Compatibility**: The app supports CUDA, CPU, or MPS (Apple Silicon). Ensure the `device` in scripts matches your hardware.
- **Twilio Configuration**: Replace placeholder Twilio credentials in `helmet_detection.py` and `accident_detection.py` with your own, or use environment variables for security.
- **Database**: `vehicle_database.csv` must have `VehicleNumber` and `PhoneNumber` columns in the correct format.
- **Accident Image**: `accident_detection.py` references `accident.jpg`. Either include it or modify the script to generate it dynamically:
  ```python
  if accident_detected:
      cv2.imwrite(os.path.join(temp_dir.name, "accident.jpg"), accident_frame)
      st.image(os.path.join(temp_dir.name, "accident.jpg"), caption="Accident Result")
  ```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, features, or improvements.

## License
This project is licensed under the MIT License.
