import cv2
import time
import json
import threading
from datetime import datetime, timedelta
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import torch


SENDER_EMAIL = "email"
SENDER_PASSWORD = "code"
EMAIL_RECIPIENTS = [
    "email1",
    "email2",
    "email3"
]
NOTIFICATION_DELAY = 10
REMINDER_INTERVAL = 30
TODO_FILE = "todo_list.json"
IMAGE_FILENAME = "detected_frame.jpg"

CAMERA_ID = "Camera 1"
camera_locations = {
    "Camera 1": {
        "name": "Testing Zone",
        "gps_available": False,
        "lat": None,
        "lon": None
    }
}

model = YOLO("best.pt")
TARGET_CLASSES = ['Debris', 'Garbage']
CONFIDENCE_THRESHOLD = 0.5

print("CUDA Available:", torch.cuda.is_available())

if os.path.exists(TODO_FILE):
    with open(TODO_FILE, 'r') as f:
        todo_list = json.load(f)
else:
    todo_list = {}

def save_todo():
    with open(TODO_FILE, 'w') as f:
        json.dump(todo_list, f, indent=2)

def get_camera_location_info():
    loc = camera_locations.get(CAMERA_ID, {})
    if loc.get("gps_available") and loc.get("lat") and loc.get("lon"):
        lat, lon = loc["lat"], loc["lon"]
        maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        return f"Camera: {CAMERA_ID} ({loc.get('name')})\nLocation: {lat}, {lon}\nMap: {maps_link}"
    else:
        return f"Camera: {CAMERA_ID} ({loc.get('name')})\nNo GPS. Static camera position."

def send_email_to_multiple(subject, body, image_path):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(EMAIL_RECIPIENTS)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, 'rb') as img_file:
        img = MIMEBase('application', 'octet-stream')
        img.set_payload(img_file.read())
        encoders.encode_base64(img)
        img.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
        msg.attach(img)

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.sendmail(SENDER_EMAIL, EMAIL_RECIPIENTS, msg.as_string())
    server.quit()
    print("✅ Garbage was detected TODO marked Pending and email sent.")

def reminder_thread(todo_id):
    time.sleep(REMINDER_INTERVAL)
    if todo_list[todo_id]['status'] == "Pending":
        subject = f"❗ Reminder: Garbage Still Not Cleared - {CAMERA_ID}"
        body = f"Garbage was detected {todo_list[todo_id]['detected_time']} and is still pending cleanup.\n\n" + get_camera_location_info()
        send_email_to_multiple(subject, body, IMAGE_FILENAME)

def debris_detected(results):
    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        class_name = model.names[cls_id]
        if class_name in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
            return True
    return False

def add_todo():
    todo_id = str(int(time.time()))
    todo_list[todo_id] = {
        "camera": CAMERA_ID,
        "detected_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Pending"
    }
    save_todo()
    threading.Thread(target=reminder_thread, args=(todo_id,), daemon=True).start()
    return todo_id

def check_and_complete_todo():
    for todo_id, todo in todo_list.items():
        if todo["camera"] == CAMERA_ID and todo["status"] == "Pending":
            todo["status"] = "Completed"
            todo["cleared_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_todo()

            subject = f"✅ Cleared: Garbage Removed - {CAMERA_ID}"
            body = f"The previously detected garbage at {todo['detected_time']} has now been cleared.\n\n" + get_camera_location_info()
            send_email_to_multiple(subject, body, IMAGE_FILENAME)
            print("✅ TODO marked completed and email sent.")

video_path = "./345.mp4"
cap = cv2.VideoCapture(video_path)

start_detection_time = None
notification_sent = False
current_todo_id = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    if debris_detected(results):
        if start_detection_time is None:
            start_detection_time = time.time()
        else:
            elapsed = time.time() - start_detection_time
            if elapsed >= NOTIFICATION_DELAY and not notification_sent:
                cv2.imwrite(IMAGE_FILENAME, annotated_frame)

                subject = f"🚨 Garbage Detected at {CAMERA_ID}"
                body = f"Garbage has been detected for {NOTIFICATION_DELAY} seconds.\n\n" + get_camera_location_info()
                send_email_to_multiple(subject, body, IMAGE_FILENAME)

                current_todo_id = add_todo()
                notification_sent = True
    else:
        if notification_sent:
            check_and_complete_todo()
        start_detection_time = None
        notification_sent = False

    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if os.path.exists(IMAGE_FILENAME):
    os.remove(IMAGE_FILENAME)
save_todo()
