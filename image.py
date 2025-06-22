import cv2
from ultralytics import YOLO
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Running on GPU:", torch.cuda.get_device_name(0))

model = YOLO("best.pt")
print(model.names)

CAMERA_ID = "Photo Test"
camera_locations = {
    "Photo Test": {
        "name": "Static Image Test",
        "gps_available": False,
        "lat": None,
        "lon": None
    }
}

def get_camera_location_info(camera_id):
    loc = camera_locations.get(camera_id, {})
    if loc.get("gps_available") and loc.get("lat") and loc.get("lon"):
        lat, lon = loc["lat"], loc["lon"]
        maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        return f"Camera: {camera_id} ({loc.get('name')})\nLocation: {lat}, {lon}\nMap: {maps_link}"
    else:
        return f"Camera: {camera_id} ({loc.get('name')})\nNo GPS. Static camera position."

SENDER_EMAIL = "email"
SENDER_PASSWORD = "password"
RECEIVER_EMAIL = "email"

def send_email(subject, body, image_path):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, 'rb') as img_file:
        img = MIMEBase('application', 'octet-stream')
        img.set_payload(img_file.read())
        encoders.encode_base64(img)
        img.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
        msg.attach(img)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("✅ Email with image sent successfully.")
    except Exception as e:
        print("❗ Error sending email:", e)

TARGET_CLASSES = ['Debris', 'Garbage']
INPUT_IMAGE = "./159.jpg"
ANNOTATED_IMAGE = "annotated_image.jpg"

def debris_detected(results):
    detected_any = False
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        class_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        print(f"Detected: {class_name} | Confidence: {conf:.2f}")
        if class_name in TARGET_CLASSES:
            detected_any = True
    return detected_any

frame = cv2.imread(INPUT_IMAGE)
if frame is None:
    print(f"❗ Error: Could not load image {INPUT_IMAGE}")
else:
    results = model.predict(frame)
    annotated_frame = results[0].plot()

    if debris_detected(results):
        print("🚨 Garbage or debris detected in the image!")

        cv2.imwrite(ANNOTATED_IMAGE, annotated_frame)

        subject = f"🚨 Garbage Detected Alert from {CAMERA_ID} (Single Image)"
        body = f"Garbage has been detected in the provided photo.\n\n"
        body += get_camera_location_info(CAMERA_ID)

        send_email(subject, body, ANNOTATED_IMAGE)

        cv2.imshow("Detection Result", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if os.path.exists(ANNOTATED_IMAGE):
            os.remove(ANNOTATED_IMAGE)
    else:
        print("✅ No debris or garbage detected in the image.")

