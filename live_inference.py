import cv2
import re
import easyocr
from rfdetr import RFDETRMedium
from harvesters.core import Harvester

# -----------------------------
# Configuration
# -----------------------------
CHECKPOINT = "runs/frosch_medium/checkpoint_best_regular.pth"
CTI_PATH = "/home/xisai/Downloads/VimbaX_2026-2/cti/VimbaCameraSimulatorTL.cti"

THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

VALID_CAPACITIES = {100, 300, 500}
MAX_MISSING_FRAMES = 20

reader = easyocr.Reader(['en'], gpu=True)
model = RFDETRMedium(pretrain_weights=CHECKPOINT)

# -----------------------------
# OCR
# -----------------------------
def extract_capacity(text):
    digits = re.sub(r"\D", "", text)

    if not digits:
        return None

    if digits.startswith("5"):
        return 500

    if digits.startswith("3"):
        return 300

    if digits.startswith("1"):
        return 100

    value = int(digits)

    if value in VALID_CAPACITIES:
        return value

    return None


def run_ocr(image, box):
    x1, y1, x2, y2 = box

    pad = 20

    h, w = image.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    results = reader.readtext(
        gray,
        allowlist="0123456789",
        paragraph=False,
    )

    for _, text, conf in results:

        if conf < 0.2:
            continue

        capacity = extract_capacity(text)

        if capacity:
            return capacity

    return None


# -----------------------------
# IoU
# -----------------------------
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


# -----------------------------
# Tracking
# -----------------------------
tracked = []

next_track_id = 0


def create_track(box):
    global next_track_id

    track = {
        "id": next_track_id,
        "box": box,
        "capacity": None,
        "defects": [],
        "ocr_done": False,
        "printed": False,
        "missing": 0,
    }

    next_track_id += 1

    return track


# -----------------------------
# Camera
# -----------------------------
h_cam = Harvester()

h_cam.add_file(CTI_PATH)

h_cam.update()

print(f"Devices found: {len(h_cam.device_info_list)}")

for i, device in enumerate(h_cam.device_info_list):
    print(f"Device {i}: {device}")

ia = h_cam.create(1)

ia.start()
cv2.namedWindow("Frosch Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frosch Inference", 1280, 720)

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

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_GRAY2BGR,
                )

            elif pixel_format == "RGB8":

                frame = data.reshape(height, width, 3)

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR,
                )

            elif pixel_format == "BGR8":

                frame = data.reshape(height, width, 3)

            else:
                continue

            detections = model.predict(
                frame,
                threshold=THRESHOLD,
            )

            bottle_boxes = []
            capacity_boxes = []
            damage_boxes = []
            bump_boxes = []

            display = frame.copy()

            # -----------------------------------
            # Separate detections by class
            # -----------------------------------
            for i, cls in enumerate(detections.data["class_name"]):

                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                if cls == "bottle":
                    bottle_boxes.append((x1, y1, x2, y2))

                elif cls == "capacity":
                    capacity_boxes.append((x1, y1, x2, y2))

                elif cls == "damage":
                    damage_boxes.append((x1, y1, x2, y2))

                elif cls == "bump":
                    bump_boxes.append((x1, y1, x2, y2))

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display,
                    cls,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # -----------------------------------
            # Age existing tracks
            # -----------------------------------
            for track in tracked:
                track["missing"] += 1

            # -----------------------------------
            # Match bottles to tracks
            # -----------------------------------
            for bottle in bottle_boxes:

                matched = False

                for track in tracked:

                    if iou(bottle, track["box"]) > IOU_THRESHOLD:

                        matched = True

                        track["box"] = bottle
                        track["missing"] = 0

                        if not track["ocr_done"]:

                            # ----------------------
                            # Capacity OCR
                            # ----------------------
                            capacity_found = False
                            for cap_box in capacity_boxes:

                                if iou(cap_box, bottle) > 0.2:

                                    capacity = run_ocr(frame, cap_box)

                                    if capacity:

                                        track["capacity"] = capacity
                                        track["ocr_done"] = True
                                        capacity_found = True
                                        break
                            
                            # ----------------------
                            # Damage
                            # ----------------------
                            defects = []

                            for dmg in damage_boxes:

                                if iou(dmg, bottle) > 0.2:
                                    defects.append("damage")

                            for bump in bump_boxes:

                                if iou(bump, bottle) > 0.2:
                                    defects.append("bump")

                            track["defects"] = sorted(list(set(defects)))


                            if not capacity_found:
                                continue

                            if not track["printed"]:

                                print("=" * 40)
                                print(f"Bottle {track['id']}")

                                if track["capacity"]:
                                    print(f"Capacity : {track['capacity']} ml")
                                else:
                                    print("Capacity : Not detected")

                                if track["defects"]:
                                    print(
                                        "Defects  :",
                                        ", ".join(track["defects"])
                                    )
                                else:
                                    print("Defects  : None")

                                print("=" * 40)

                                track["printed"] = True

                        break

                if not matched:

                    tracked.append(create_track(bottle))

            # -----------------------------------
            # Remove disappeared bottles
            # -----------------------------------
            tracked = [
                t for t in tracked
                if t["missing"] < MAX_MISSING_FRAMES
            ]

            # -----------------------------------
            # Draw results
            # -----------------------------------
            for track in tracked:

                x1, y1, x2, y2 = track["box"]

                if track["capacity"]:

                    cv2.putText(
                        display,
                        f"{track['capacity']} ml",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                if track["defects"]:

                    cv2.putText(
                        display,
                        ",".join(track["defects"]),
                        (x1, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("Frosch Inference", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:

    ia.stop()
    ia.destroy()

    h_cam.reset()

    cv2.destroyAllWindows()
