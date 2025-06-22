import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from ultralytics import YOLO

model = YOLO("best.pt") 
CONFIDENCE_THRESHOLD = 0.5
CLASS_NAMES = ['Debris', 'Garbage']

TEST_IMAGES_DIR = './test/images'
TEST_LABELS_DIR = './test/labels'

y_true = []
y_pred = []

def get_ground_truth_label(label_file_path):
    if not os.path.exists(label_file_path):
        return None
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None
        first_class = int(lines[0].split()[0])
        return first_class

for img_file in os.listdir(TEST_IMAGES_DIR):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    label_file = img_file.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(TEST_LABELS_DIR, label_file)

    true_class = get_ground_truth_label(label_path)
    if true_class is None:
        continue

    img = cv2.imread(img_path)
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD)
    
    predicted_class = None
    max_conf = 0.0

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if conf >= CONFIDENCE_THRESHOLD and conf > max_conf:
            predicted_class = cls_id
            max_conf = conf

    if predicted_class is None:
        predicted_class = -1  # No Detection

    y_true.append(true_class)
    y_pred.append(predicted_class)

# ✅ Adjust class names for No Detection
display_labels = CLASS_NAMES + ['No Detection'] if -1 in y_pred else CLASS_NAMES

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1,-1] if -1 in y_pred else [0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Multiclass Confusion Matrix")
plt.show()

# ✅ Classification Report
labels_for_report = [0,1]
if -1 in y_pred:
    labels_for_report.append(-1)
print(classification_report(y_true, y_pred, labels=labels_for_report, target_names=display_labels))
