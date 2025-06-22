import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from ultralytics import YOLO

MODEL_PATH = "best.pt"
TEST_IMAGES_DIR = "./test/images"
TEST_LABELS_DIR = "./test/labels"
CONFIDENCE_THRESHOLD = 0.25

model = YOLO(MODEL_PATH)

y_true = []
y_scores = []

image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(TEST_IMAGES_DIR, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(TEST_LABELS_DIR, label_file)

    true_label = 1 if os.path.exists(label_path) and os.path.getsize(label_path) > 0 else 0

    img = cv2.imread(img_path)
    results = model.predict(img, conf=CONFIDENCE_THRESHOLD)

    max_conf = 0.0
    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf > max_conf:
            max_conf = conf

    y_true.append(true_label)
    y_scores.append(max_conf)

y_true_bin = label_binarize(y_true, classes=[0, 1])
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), np.array(y_scores))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Debris/Garbage Detection)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"Final AUC Score: {roc_auc:.3f}")
