import cv2
import re
import easyocr
from rfdetr import RFDETRMedium
from harvesters.core import Harvester

import csv
from datetime import datetime

LOG_FILE = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Bottle', 'Capacity', 'Orientation', 'H_Center', 'V_Center', 'Defects', 'Timestamp'])
# -----------------------------
# Configuration
# -----------------------------
CHECKPOINT = "runs/frosch_medium/checkpoint_best_regular.pth"
CTI_PATH = "/home/xisai/Downloads/VimbaX_2026-2/cti/VimbaCameraSimulatorTL.cti"
THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MAX_MISSING_FRAMES = 20
H_CENTRICITY_THRESH = 0.15
V_CENTRICITY_THRESH = 0.15

reader = easyocr.Reader(['en'], gpu=True)
model = RFDETRMedium(pretrain_weights=CHECKPOINT)

bottle_count = 0
next_track_id = 0
tracked = []


# -----------------------------
# Helpers
# -----------------------------
def extract_capacity(text):
    digits = re.sub(r"\D", "", text.strip())[:3]
    if not digits:
        return None
    if digits.startswith('5'):
        return 500
    if digits.startswith('3'):
        return 300
    if digits.startswith('1'):
        return 100
    value = int(digits)
    if value in {100, 300, 500}:
        return value
    return None


def run_ocr(image, box):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = reader.readtext(gray, allowlist="0123456789", paragraph=False)
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
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def check_orientation(bottle_box):
    w = bottle_box[2] - bottle_box[0]
    h = bottle_box[3] - bottle_box[1]
    return "PASS" if h >= w else "FAIL"


def check_centricity(bottle_box, label_box):
    bx, by = box_center(bottle_box)
    lx, ly = box_center(label_box)
    bw = bottle_box[2] - bottle_box[0]
    bh = bottle_box[3] - bottle_box[1]
    h_ok = abs(lx - bx) / bw <= H_CENTRICITY_THRESH
    v_ok = abs(ly - by) / bh <= V_CENTRICITY_THRESH
    return ("PASS" if h_ok else "FAIL"), ("PASS" if v_ok else "FAIL")


def create_track(box):
    global next_track_id
    track = {
        "id": next_track_id,
        "box": box,
        "capacity": None,
        "defects": [],
        "orientation": None,
        "h_center": None,
        "v_center": None,
        "missing": 0,
    }
    next_track_id += 1
    return track


def analyze_bottle(frame, bottle_box, capacity_boxes, label_boxes, damage_boxes, bump_boxes):
    """Run OCR, centricity and defect checks once when bottle first appears."""
    result = {
        "capacity": None,
        "defects": [],
        "orientation": check_orientation(bottle_box),
        "h_center": None,
        "v_center": None,
    }

    # OCR
    for cap_box in capacity_boxes:
        if cap_box[0] >= bottle_box[0] and cap_box[2] <= bottle_box[2]:
            cap = run_ocr(frame, cap_box)
            if cap:
                result["capacity"] = cap
                break

    # Centricity
    for lbl_box in label_boxes:
        if lbl_box[0] >= bottle_box[0] and lbl_box[2] <= bottle_box[2]:
            h, v = check_centricity(bottle_box, lbl_box)
            result["h_center"] = h
            result["v_center"] = v
            break

    # Defects
    defects = []
    for dmg in damage_boxes:
        if dmg[0] >= bottle_box[0] and dmg[2] <= bottle_box[2]:
            defects.append("damage")
    for bump in bump_boxes:
        if bump[0] >= bottle_box[0] and bump[2] <= bottle_box[2]:
            defects.append("bump")
    result["defects"] = sorted(set(defects))

    return result


# -----------------------------
# Camera
# -----------------------------
h_cam = Harvester()
h_cam.add_file(CTI_PATH)
h_cam.update()
print(f"Devices found: {len(h_cam.device_info_list)}")

ia = h_cam.create(1)
ia.start()
print("Streaming... Press q to exit.")


# -----------------------------
# Main Loop
# -----------------------------
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
            elif pixel_format in ("BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8"):
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

            bottle_boxes = []
            capacity_boxes = []
            label_boxes = []
            damage_boxes = []
            bump_boxes = []

            display = frame.copy()

            for i, cls in enumerate(detections.data["class_name"]):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                if cls == "bottle":
                    bottle_boxes.append((x1, y1, x2, y2))
                elif cls == "capacity":
                    capacity_boxes.append((x1, y1, x2, y2))
                elif cls == "label":
                    label_boxes.append((x1, y1, x2, y2))
                elif cls == "damage":
                    damage_boxes.append((x1, y1, x2, y2))
                elif cls == "bump":
                    bump_boxes.append((x1, y1, x2, y2))

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, cls, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Age all tracks
            for track in tracked:
                track["missing"] += 1

            # Match / create tracks
            for bottle in bottle_boxes:
                matched = False

                for track in tracked:
                    if iou(bottle, track["box"]) > IOU_THRESHOLD:
                        # Existing track — just update position
                        track["box"] = bottle
                        track["missing"] = 0
                        matched = True
                        # Retry OCR if not yet detected
                        if not track["capacity"]:
                            for cap_box in capacity_boxes:
                                if cap_box[0] >= bottle[0] and cap_box[2] <= bottle[2]:
                                    cap = run_ocr(frame, cap_box)
                                    if cap:
                                        track["capacity"] = cap
                                        print(f"Bottle #{track['id']+1} capacity updated: {cap} ml")
                                        with open(LOG_FILE, 'a', newline='') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([
                                                track['id']+1,
                                                cap,
                                                track['orientation'],
                                                track['h_center'] or 'N/A',
                                                track['v_center'] or 'N/A',
                                                ', '.join(track['defects']) or 'None',
                                                datetime.now().strftime('%H:%M:%S')
                                            ])
                                    break
                        break

                if not matched:
                    # New bottle — analyze once
                    bottle_count += 1
                    track = create_track(bottle)
                    result = analyze_bottle(
                        frame, bottle,
                        capacity_boxes, label_boxes,
                        damage_boxes, bump_boxes
                    )
                    track["capacity"] = result["capacity"]
                    track["defects"] = result["defects"]
                    track["orientation"] = result["orientation"]
                    track["h_center"] = result["h_center"]
                    track["v_center"] = result["v_center"]

                    # Print to terminal once
                    print("=" * 40)
                    print(f"Bottle #{bottle_count}")
                    print(f"Capacity    : {track['capacity'] or 'Not detected'} ml")
                    print(f"Orientation : {track['orientation']}")
                    print(f"H Center    : {track['h_center'] or 'N/A'}")
                    print(f"V Center    : {track['v_center'] or 'N/A'}")
                    print(f"Defects     : {', '.join(track['defects']) or 'None'}")
                    print("=" * 40)

                     # CSV logging
                    with open(LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            bottle_count,
                            track['capacity'] or 'Not detected',
                            track['orientation'],
                            track['h_center'] or 'N/A',
                            track['v_center'] or 'N/A',
                            ', '.join(track['defects']) or 'None',
                            datetime.now().strftime('%H:%M:%S')
                        ])

                    tracked.append(track)

            # Remove lost tracks
            tracked = [t for t in tracked if t["missing"] < MAX_MISSING_FRAMES]

            # Draw overlays
            for track in tracked:
                x1, y1, x2, y2 = track["box"]

                cv2.putText(display, f"Bottle #{track['id'] + 1}",
                            (x1, y1 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if track["capacity"]:
                    cv2.putText(display, f"{track['capacity']} ml",
                                (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if track["orientation"]:
                    col = (0, 255, 0) if track["orientation"] == "PASS" else (0, 0, 255)
                    cv2.putText(display, f"Orient: {track['orientation']}",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                if track["h_center"]:
                    col = (0, 255, 0) if track["h_center"] == "PASS" else (0, 0, 255)
                    cv2.putText(display, f"H Center: {track['h_center']}",
                                (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                if track["v_center"]:
                    col = (0, 255, 0) if track["v_center"] == "PASS" else (0, 0, 255)
                    cv2.putText(display, f"V Center: {track['v_center']}",
                                (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                if track["defects"]:
                    cv2.putText(display, "Defects: " + ", ".join(track["defects"]),
                                (x1, y2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Total bottle count overlay
            cv2.putText(display, f"Total Bottles: {bottle_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            display_resized = cv2.resize(display, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow("Frosch Inference", display_resized)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    ia.stop()
    ia.destroy()
    h_cam.reset()
    cv2.destroyAllWindows()
