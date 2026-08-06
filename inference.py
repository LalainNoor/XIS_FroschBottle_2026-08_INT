import os
import re
import cv2
import easyocr
from rfdetr import RFDETRMedium

IMAGE_DIR = "/home/xisai/Workspace/lalain/frosch/Frosch bottle 5.v6i.coco-segmentation/test"
CHECKPOINT = "runs/frosch_medium/checkpoint_best_regular.pth"
THRESHOLD = 0.5

VALID_CAPACITIES = {100, 300, 500}

reader = easyocr.Reader(['en'], gpu=True)
model = RFDETRMedium(pretrain_weights=CHECKPOINT)

def extract_capacity(text):
    text = text.strip()
    digits = re.sub(r"\D", "", text)

    if not digits:
        return None

    # take first 3 digits only
    digits = digits[:3]

    value = int(digits)

    if value in VALID_CAPACITIES:
        return value

    if digits.startswith('5'):
        return 500
    if digits.startswith('3'):
        return 300
    if digits.startswith('1'):
        return 100

    return None

for img_file in sorted(os.listdir(IMAGE_DIR)):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_file)
    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    detections = model.predict(img_path, threshold=THRESHOLD)

    print(f"\n--- {img_file} ---")

    found = False

    for i, class_name in enumerate(detections.data["class_name"]):
        if class_name != "capacity":
            continue

        x1, y1, x2, y2 = map(int, detections.xyxy[i])

        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imwrite(f"crop_{img_file}", gray)

        results = reader.readtext(gray, allowlist="0123456789", detail=1, paragraph=False)

        for (_, text, conf) in results:
            print(f"  Raw OCR: '{text}' conf: {conf:.2f}")

            if conf < 0.2:
                continue

            capacity = extract_capacity(text)

            if capacity is not None:
                print(f"Capacity: {capacity} ml")
                found = True
                break

        if found:
            break

    if not found:
        print("Capacity: not detected")
