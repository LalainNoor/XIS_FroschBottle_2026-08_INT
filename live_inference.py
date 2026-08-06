import cv2
import re
import numpy as np
import easyocr
from rfdetr import RFDETRMedium
from harvesters.core import Harvester

CHECKPOINT = "runs/frosch_medium/checkpoint_best_regular.pth"
CTI_PATH = "/home/xisai/Downloads/VimbaX_2026-2/cti/VimbaCameraSimulatorTL.cti"
THRESHOLD = 0.5
VALID_CAPACITIES = {100, 300, 500}
IOU_THRESHOLD = 0.4

reader = easyocr.Reader(['en'], gpu=True)
model = RFDETRMedium(pretrain_weights=CHECKPOINT)

def extract_capacity(text):
    digits = re.sub(r"\D", "", text.strip())[:3]
    if not digits:
        return None
    value = int(digits)
    if value in VALID_CAPACITIES:
        return value
    if digits.startswith('5'): return 500
    if digits.startswith('3'): return 300
    if digits.startswith('1'): return 100
    return None

def run_ocr(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    pad = 20
    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
    cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = reader.readtext(gray, allowlist="0123456789", detail=1, paragraph=False)
    for (_, text, conf) in results:
        if conf < 0.2:
            continue
        cap = extract_capacity(text)
        if cap:
            return cap
    return None

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)

# tracked bottles: list of {box, result, done}
tracked = []

h_cam = Harvester()
h_cam.add_file(CTI_PATH)
h_cam.update()
print(f"Devices found: {len(h_cam.device_info_list)}")

ia = h_cam.create(1)
ia.start()
print("Streaming... Press 'q' to stop.")

try:
    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            width = component.width
            height = component.height
            pixel_format = component.data_format
            data = component.data

            if pixel_format == "Mono8":
                frame = data.reshape(height, width)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif pixel_format == "RGB8":
                frame = data.reshape(height, width, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif pixel_format == "BGR8":
                frame = data.reshape(height, width, 3)
            elif pixel_format in ("BayerRG8","BayerGB8","BayerGR8","BayerBG8"):
                bayer_map = {
                    "BayerRG8": cv2.COLOR_BayerRG2BGR,
                    "BayerGB8": cv2.COLOR_BayerGB2BGR,
                    "BayerGR8": cv2.COLOR_BayerGR2BGR,
                    "BayerBG8": cv2.COLOR_BayerBG2BGR,
                }
                frame = cv2.cvtColor(data.reshape(height, width), bayer_map[pixel_format])
            else:
                continue

            detections = model.predict(frame, threshold=THRESHOLD)

            # get bottle boxes
            bottle_boxes = []
            capacity_boxes = {}

            for i, cls in enumerate(detections.data['class_name']):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                if cls == 'bottle':
                    bottle_boxes.append((x1, y1, x2, y2))
                elif cls == 'capacity':
                    capacity_boxes[i] = (x1, y1, x2, y2)

            # match detections to tracked bottles
            for box in bottle_boxes:
                matched = False
                for t in tracked:
                    if iou(box, t['box']) > IOU_THRESHOLD:
                        t['box'] = box
                        matched = True
                        if not t['done']:
                            # find capacity for this bottle
                            for cap_box in capacity_boxes.values():
                                cap = run_ocr(frame, *cap_box)
                                if cap:
                                    t['capacity'] = cap
                                    t['done'] = True
                                    break
                        break
                if not matched:
                    tracked.append({'box': box, 'done': False, 'capacity': None})

            # draw results
            display = frame.copy()
            for i, cls in enumerate(detections.data['class_name']):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                color = (0, 255, 0)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, cls, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for t in tracked:
                if t['done'] and t['capacity']:
                    x1, y1, x2, y2 = t['box']
                    cv2.putText(display, f"{t['capacity']} ml", (x1, y2+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Frosch Inference", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Stopped.")

finally:
    ia.stop()
    ia.destroy()
    h_cam.reset()
    cv2.destroyAllWindows()
