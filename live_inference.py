import cv2
import numpy as np
import re
import os
import argparse
import easyocr
from rfdetr import RFDETRMedium, RFDETRSegMedium
from harvesters.core import Harvester

import csv
from datetime import datetime
from collections import Counter

# -----------------------------
# Runtime arguments / configuration
# -----------------------------
def parse_runtime_args():
    parser = argparse.ArgumentParser(
        description="Frosch bottle live/folder inference pipeline"
    )
    parser.add_argument(
        "--input",
        "--input-mode",
        dest="input_mode",
        choices=("camera", "folder"),
        default="camera",
        help="Input source: camera (default) or folder",
    )
    parser.add_argument(
        "--frame-dir",
        default="/home/xisai/Downloads/Capture_2026-07-15_07h53m14s",
        help="Folder containing recorded frames when --input folder is used",
    )
    parser.add_argument(
        "--expected-capacity",
        type=int,
        choices=(100, 300, 500),
        default=100,
        help="Expected bottle capacity for the currently tested bottle type",
    )
    return parser.parse_args()

ARGS = parse_runtime_args()

LOG_FILE = f"results_{ARGS.expected_capacity}ml.csv"
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Bottle",
        "Capacity",
        "Orientation",
        "H_Center",
        "V_Center",
        "Defects",
        "Timestamp",
    ])

# -----------------------------
# Configuration
# -----------------------------
INPUT_MODE = ARGS.input_mode
FRAME_DIR = ARGS.frame_dir

CHECKPOINT = "runs/frosch_medium/checkpoint_best_regular.pth"
SEG_CHECKPOINT = "runs/frosch_seg_medium/checkpoint_best_total.pth"
SEG_THRESHOLD = 0.30

# Per-class confidence thresholds (overrides universal THRESHOLD / SEG_THRESHOLD).
# Lower  → more recall, more false positives.
# Higher → fewer false positives, more misses.
PER_CLASS_CONF = {
    "bottle":   0.70,   # was 0.40 - filters person FP (conf~0.43)
    "label":    0.35,
    "capacity": 0.35,
    "bump":     0.50,   # was 0.25, and than 0.35 - raise to cut 500ml #17 FP
    "damage":   0.30,   # was 0.45 - lower to catch 100ml #17 missed
    "scratch":  0.30,   # undertrained – lower to catch more
}

BOTTLE_CLASS_NAME = "bottle"
ORIENTATION_MAX_ANGLE_DEG = 45.0
CTI_PATH = "/home/xisai/Downloads/VimbaX_2026-2/cti/VimbaCameraSimulatorTL.cti"
THRESHOLD = 0.4
IOU_THRESHOLD = 0.4
MAX_MISSING_FRAMES = 20

# Trigger line for bottle finalization
TRIGGER_LINE_X_RATIO = 0.40
TRIGGER_LINE_TOLERANCE = 20

# A centricity measurement is considered reliable only when
# the bottle is sufficiently away from the left/right frame edges.
COMPLETE_X_MARGIN = 20

H_CENTRICITY_THRESH = 0.15

# Expected V centricity for each bottle type.
BOTTLE_TYPE1_EXPECTED_V = 0.01   # 500 ml
BOTTLE_TYPE2_EXPECTED_V = 0.12   # 100 ml
BOTTLE_TYPE3_EXPECTED_V = 0.07   # 300 ml

V_DEVIATION_THRESH = 0.05

# Expected capacity for the bottle type currently being tested.
EXPECTED_CAPACITY = ARGS.expected_capacity

# Preserve the existing camera/folder workflow while selecting
# the correct V reference for each bottle capacity.
if EXPECTED_CAPACITY == 500:
    BOTTLE_TYPE = 1
    EXPECTED_V = BOTTLE_TYPE1_EXPECTED_V

elif EXPECTED_CAPACITY == 100:
    BOTTLE_TYPE = 2
    EXPECTED_V = BOTTLE_TYPE2_EXPECTED_V

elif EXPECTED_CAPACITY == 300:
    BOTTLE_TYPE = 3
    EXPECTED_V = BOTTLE_TYPE3_EXPECTED_V

else:
    raise ValueError(
        f"Unsupported bottle capacity: {EXPECTED_CAPACITY}"
    )

LABEL_CONTAINMENT_TOLERANCE = 10

# Centricity stabilization:
# relative label position is expected to remain spatially consistent
# while the bottle moves through the camera frame.
CENTRICITY_SPATIAL_TOLERANCE = 0.08
CENTRICITY_MIN_HISTORY = 3
CENTRICITY_HISTORY_WINDOW = 5

# Minimum number of reliable complete-frame observations
# required before H/V centricity can determine the result.
MIN_RELIABLE_CENTRICITY_MEASUREMENTS = 3
# -----------------------------
# Classification Rules
# -----------------------------
# GOOD      : orientation PASS, h_center PASS, v_center PASS, no defects, label present
# DEFECTIVE : any check FAIL OR damage/bump detected inside bottle mask
# INCOMPLETE: any required measurement still Pending (label never detected, mask unavailable)
DEFECT_OVERLAP_THRESH = 0.3   # defect box must overlap bottle mask by this fraction to count
# A defect must be detected in consecutive frames before it is accepted.
# This reduces one-frame false positives from the defect detector.
DEFECT_CONFIRMATION_FRAMES = 1

# Allow a confirmed candidate to survive a short detector miss.
# This handles intermittent bump/damage detections between frames.
DEFECT_MAX_MISSING_FRAMES = 1
# FIX 1: minimum bottle area to avoid saving spurious/partial detections
MIN_BOTTLE_AREA = 20000

SAVE_ROOT_DIR = os.path.join(os.getcwd(), "saved_bottles")
SAVE_DIR = os.path.join(
    SAVE_ROOT_DIR,
    f"{EXPECTED_CAPACITY}ml",
)

SAVE_PADDING = 20

os.makedirs(SAVE_DIR, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=True)
model = RFDETRMedium(pretrain_weights=CHECKPOINT)
seg_model = RFDETRSegMedium(pretrain_weights=SEG_CHECKPOINT)

bottle_count = 0
completed_count = 0
good_count = 0
defective_count = 0
incomplete_count = 0

next_track_id = 0
tracked = []


# -----------------------------
# Helpers
# -----------------------------
def extract_capacity(text):
    digits = re.sub(r"\D", "", text.strip())
    if digits in {"100", "300", "500"}:
        return int(digits)

    return None

def run_ocr(image, box):
    """
    Run capacity OCR using several lightweight preprocessing variants.

    The detector box and capacity classes are unchanged. We only make
    the OCR step more tolerant to low contrast, blur, and label glare.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]

    pad_x = 24
    pad_y = 24

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale once and reuse the same image for all OCR variants.
    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)

    variants = [
        gray,
        enhanced,
        cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1],
        cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        ),
    ]

    candidates = []

    for variant in variants:
        results = reader.readtext(
            variant,
            allowlist="0123456789",
            paragraph=False,
        )

        for (_, text_value, conf) in results:
            if conf < 0.15:
                continue

            cap = extract_capacity(text_value)

            if cap in {100, 300, 500}:
                candidates.append((cap, float(conf)))

    if not candidates:
        return None

    # Prefer the capacity supported by the largest number of OCR
    # variants; use highest confidence as the tie-breaker.
    counts = Counter(cap for cap, _ in candidates)

    best_capacity = max(
        counts,
        key=lambda cap: (
            counts[cap],
            max(
                confidence
                for candidate, confidence in candidates
                if candidate == cap
            ),
        ),
    )

    return best_capacity


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


def stable_capacity(values):
    """Return the most frequently observed valid capacity."""
    clean = [
        int(v)
        for v in values
        if v in {100, 300, 500}
    ]

    if not clean:
        return None

    counts = Counter(clean)
    return max(counts, key=counts.get)


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def bottle_track_match(current_box, previous_box):
    """
    Match the current bottle to an existing track.

    IoU remains the primary/strict match. If the bottle moves enough that
    IoU drops below the normal threshold, allow a conservative center-based
    fallback. This is intended for normal frame-to-frame motion, not for
    unrelated bottles.
    """
    if iou(current_box, previous_box) > IOU_THRESHOLD:
        return True

    cx, cy = box_center(current_box)
    px, py = box_center(previous_box)

    current_w = max(1.0, current_box[2] - current_box[0])
    current_h = max(1.0, current_box[3] - current_box[1])
    previous_w = max(1.0, previous_box[2] - previous_box[0])
    previous_h = max(1.0, previous_box[3] - previous_box[1])

    max_w = max(current_w, previous_w)
    max_h = max(current_h, previous_h)

    horizontal_distance = abs(cx - px)
    vertical_distance = abs(cy - py)

    # Bottles mainly move horizontally through the inspection view.
    # Keep the vertical gate tighter to avoid merging nearby bottles.
    return (
        horizontal_distance <= max_w * 0.75
        and vertical_distance <= max_h * 0.45
    )


def get_mask_orientation(mask):
    """Calculate bottle orientation from actual bottle-mask pixels."""
    if mask is None:
        return None

    mask_u8 = mask.astype(np.uint8) * 255
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 20:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)

    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    norm = float(np.linalg.norm(major))
    if norm < 1e-6:
        return None

    major /= norm
    if major[1] < 0:
        major = -major

    minor = np.array([-major[1], major[0]], dtype=np.float32)

    angle_deg = float(
        np.degrees(
            np.arctan2(
                abs(float(major[0])),
                abs(float(major[1])),
            )
        )
    )

    status = "PASS" if angle_deg <= ORIENTATION_MAX_ANGLE_DEG else "FAIL"

    projections = centered @ major
    half_length = max(20.0, float(np.max(np.abs(projections))))

    # Keep the external contour so the actual segmentation mask can be
    # visualized later on both the live frame and the saved annotated image.
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = None
    if contours:
        mask_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(mask_contour) < 20:
            mask_contour = None

    return {
        "status": status,
        "angle_deg": angle_deg,
        "center": (float(center[0]), float(center[1])),
        "major": (float(major[0]), float(major[1])),
        "minor": (float(minor[0]), float(minor[1])),
        "half_length": half_length,
        "mask_contour": mask_contour.reshape(-1, 2).tolist() if mask_contour is not None else None,
    }


def find_bottle_mask(bottle_box, segmentation_detections):
    """Match a detector bottle to its segmentation bottle mask."""
    if segmentation_detections is None or segmentation_detections.mask is None:
        return None

    best_mask = None
    best_score = 0.0
    class_names = segmentation_detections.data.get("class_name", [])

    for i, cls in enumerate(class_names):
        if cls != BOTTLE_CLASS_NAME:
            continue
        if float(segmentation_detections.confidence[i]) < PER_CLASS_CONF.get(BOTTLE_CLASS_NAME, SEG_THRESHOLD):
            continue

        seg_box = tuple(map(int, segmentation_detections.xyxy[i]))
        score = iou(bottle_box, seg_box)

        if score > best_score:
            best_score = score
            best_mask = segmentation_detections.mask[i]

    return best_mask

def get_mask_centroid(mask):
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))

def find_label_mask(bottle_box, seg_detections, frame):
    if seg_detections is None or seg_detections.mask is None:
        return None
    best_mask, best_score = None, 0.0
    for i, cls in enumerate(seg_detections.data.get("class_name", [])):
        if cls != "label":
            continue
        if float(seg_detections.confidence[i]) < PER_CLASS_CONF.get("label", SEG_THRESHOLD):
            continue
        score = iou(bottle_box, tuple(map(int, seg_detections.xyxy[i])))
        if score > best_score:
            best_score = score
            best_mask = seg_detections.mask[i]
    if best_mask is None:
        return None
    best_mask = resize_mask_to_frame(best_mask, frame)
    return constrain_mask_to_box(best_mask, bottle_box)

def resize_mask_to_frame(mask, frame):
    """Resize seg model mask to match frame resolution."""
    if mask is None:
        return None
    fh, fw = frame.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) == (fh, fw):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def constrain_mask_to_box(mask, box):
    """
    Zero out mask pixels outside the detection bounding box.
    Fixes letterbox/padding misalignment: even if the seg model mask is in
    a padded coordinate space, we trust the detection box and only keep
    mask pixels inside it.
    """
    if mask is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    mh, mw = mask.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(mw, x2); y2 = min(mh, y2)
    constrained = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        constrained[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return constrained


def draw_mask_contour(image, orientation_data, color=(0, 255, 0), thickness=3, fill=False):
    """Draw the actual RF-DETR bottle segmentation mask outline."""
    if orientation_data is None:
        return

    contour = orientation_data.get("mask_contour")
    if contour is None or len(contour) < 3:
        return

    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    pts = np.round(pts).astype(np.int32)

    if fill:
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        image[:] = cv2.addWeighted(overlay, 0.14, image, 0.86, 0)

    cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)


def draw_mask_orientation(image, orientation_data, color=(255, 0, 0), thickness=3):
    """Draw orientation axes derived from bottle-mask geometry."""
    if orientation_data is None:
        return

    cx, cy = orientation_data["center"]
    dx, dy = orientation_data["major"]
    mx, my = orientation_data["minor"]
    half_length = orientation_data["half_length"]

    cx, cy = int(round(cx)), int(round(cy))

    p1 = (
        int(round(cx - dx * half_length)),
        int(round(cy - dy * half_length)),
    )
    p2 = (
        int(round(cx + dx * half_length)),
        int(round(cy + dy * half_length)),
    )
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

    minor_half = max(15.0, half_length * 0.25)
    q1 = (
        int(round(cx - mx * minor_half)),
        int(round(cy - my * minor_half)),
    )
    q2 = (
        int(round(cx + mx * minor_half)),
        int(round(cy + my * minor_half)),
    )
    cv2.line(image, q1, q2, color, max(1, thickness - 1), cv2.LINE_AA)
    cv2.drawMarker(image, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)

def is_reliable_bottle_frame(box, frame_shape):
    """
    Return True only when the bottle is sufficiently inside
    the camera frame to provide a reliable centricity measurement.
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)

    return (
        x1 >= COMPLETE_X_MARGIN
        and x2 <= frame_w - COMPLETE_X_MARGIN
        and y1 >= 0
        and y2 <= frame_h
    )

def check_centricity(
    bottle_box,
    label_box,
    bottle_mask=None,
    label_mask=None,
):
    """
    Calculate centricity from segmentation-mask centroids.

    Masks are the primary source for bottle/label centers so the same
    geometry is used for both bottle types. Bounding-box centers are only
    used as a fallback when a corresponding mask is unavailable.
    """
    bottle_c = get_mask_centroid(bottle_mask)
    label_c = get_mask_centroid(label_mask)

    bx, by = bottle_c if bottle_c is not None else box_center(bottle_box)
    lx, ly = label_c if label_c is not None else box_center(label_box)

    bw = max(1.0, float(bottle_box[2] - bottle_box[0]))
    bh = max(1.0, float(bottle_box[3] - bottle_box[1]))

    h_offset = (lx - bx) / bw
    v_offset = (ly - by) / bh

    h_ok = abs(h_offset) <= H_CENTRICITY_THRESH

    v_ok = (
        abs(v_offset - EXPECTED_V)
        <= V_DEVIATION_THRESH
    )

    return (
        "PASS" if h_ok else "FAIL",
        "PASS" if v_ok else "FAIL",
        h_offset,
        v_offset,
        (bx, by),
        (lx, ly),
    )

def has_crossed_trigger_line(track, box, frame_width):
    """
    Detect whether the tracked bottle center crossed the trigger line.
    This only controls bottle finalization.
    """
    line_x = int(frame_width * TRIGGER_LINE_X_RATIO)

    current_center_x = (box[0] + box[2]) / 2.0
    previous_center_x = track.get("previous_center_x")

    crossed = False

    if previous_center_x is not None:
        previous_side = previous_center_x - line_x
        current_side = current_center_x - line_x

        if (
            previous_side * current_side <= 0
            and abs(current_center_x - previous_center_x) >= TRIGGER_LINE_TOLERANCE
        ):
            crossed = True

    track["previous_center_x"] = current_center_x

    if crossed:
        track["trigger_crossed"] = True
        print(
            f"Bottle #{track['id'] + 1} crossed trigger line "
            f"at x={line_x}"
        )

    return crossed


def create_track(box):
    global next_track_id
    track = {
        "id": next_track_id,
        "box": box,
        "capacity": None,
        "defects": [],
        "defect_streaks": {
            "damage": 0,
            "bump": 0,
        },
        "defect_missing_frames": {
            "damage": 0,
            "bump": 0,
        },
        "orientation": None,
        "orientation_data": None,
        "_current_bottle_mask": None,
        "best_complete_orientation_data": None,
        "best_defect_orientation_data": None,
        "h_center": None,
        "v_center": None,
        "missing": 0,
        "saved": False,
        "frames_seen": 0,
        "previous_center_x": None,
        "trigger_crossed": False,
        "best_box": box,
        "best_frame": None,
        "best_complete_box": None,
        "best_complete_frame": None,
        "best_complete_label_box": None,
        "best_complete_damage_boxes": [],
        "best_complete_bump_boxes": [],
        "best_defect_frame": None,
        "best_defect_box": None,
        "best_defect_label_box": None,
        "best_defect_damage_boxes": [],
        "best_defect_bump_boxes": [],

        # Best confirmed-defect snapshot where the complete bottle is
        # fully inside the source frame. Used only for saved annotation.
        "best_complete_defect_frame": None,
        "best_complete_defect_box": None,
        "best_complete_defect_label_box": None,
        "best_complete_defect_damage_boxes": [],
        "best_complete_defect_bump_boxes": [],
        "best_complete_defect_orientation_data": None,

        # Defect boxes normalized to the bottle box. Used only as a
        # visualization fallback when the saved frame differs from
        # the original confirmed-defect frame.
        "best_defect_damage_relative": [],
        "best_defect_bump_relative": [],

        "best_valid_box": None,
        "best_valid_frame": None,
        "annotation_label_box": None,
        "annotation_damage_boxes": [],
        "annotation_bump_boxes": [],
        "best_valid_h_center": None,
        "best_valid_v_center": None,
        "best_valid_centricity_error": None,
        # Signed normalized label-to-bottle center offsets.
        # These are frame-motion invariant and are used to reject
        # occasional label-association/box-jitter outliers.
        "centricity_offset_history": [],
        "h_history": [],
        "v_history": [],
        "orientation_history": [],
        # Numeric measurements retained separately from PASS/FAIL histories.
        # H/V values are absolute normalized label-to-bottle center offsets.
        "h_value_history": [],
        "v_value_history": [],
        "orientation_angle_history": [],
        "final_h_value": None,
        "final_v_value": None,
        "final_orientation_angle": None,
        "capacity_history": [],
        "finalized": False,
        "final_status": None,
    }
    next_track_id += 1
    return track


def majority_result(values):
    """Return the majority PASS/FAIL value, or None if there is no measurement."""
    clean = [v for v in values if v in {"PASS", "FAIL"}]
    if not clean:
        return None
    passes = clean.count("PASS")
    fails = clean.count("FAIL")
    return "PASS" if passes >= fails else "FAIL"


def stable_result(values):
    """Return a stable majority result once enough measurements exist."""
    result = majority_result(values)
    if result is None:
        return None
    return result


def stable_numeric(values):
    """Return a robust representative numeric measurement."""
    clean = [
        float(v)
        for v in values
        if v is not None and np.isfinite(float(v))
    ]

    if not clean:
        return None

    return float(np.median(clean))


def format_measurement(value, status, suffix=""):
    """Format a numeric measurement with its existing PASS/FAIL status."""
    if value is None:
        return status or "N/A"

    if status in {"PASS", "FAIL"}:
        return f"{value:.3f}{suffix} ({status})"

    return f"{value:.3f}{suffix} ({status or 'N/A'})"


def finalize_measurements(track):
    """Freeze the final bottle state from accumulated observations."""
    h_values = [
        abs(float(x))
        for x in track.get("h_value_history", [])
        if x is not None and np.isfinite(float(x))
    ]

    v_values = [
        abs(float(x))
        for x in track.get("v_value_history", [])
        if x is not None and np.isfinite(float(x))
    ]

    if h_values:
        h = (
            "PASS"
            if float(np.median(h_values)) <= H_CENTRICITY_THRESH
            else "FAIL"
        )
    else:
        h = "Pending"

    if v_values:
        v = (
            "PASS"
            if abs(float(np.median(v_values)) - EXPECTED_V)
            <= V_DEVIATION_THRESH
            else "FAIL"
        )
    else:
        v = "Pending"

    # Fallback: if still Pending but a saved complete label box exists,
    # compute a one-shot centricity so the bottle is not left INCOMPLETE
    # solely because the label was never detected in a "reliable" frame window.
    if h == "Pending" or v == "Pending":
        fallback_box = track.get("best_complete_box")
        fallback_label = track.get("best_complete_label_box")
        if fallback_box is not None and fallback_label is not None:
            bx_c = (fallback_box[0] + fallback_box[2]) / 2.0
            by_c = (fallback_box[1] + fallback_box[3]) / 2.0
            lx_c = (fallback_label[0] + fallback_label[2]) / 2.0
            ly_c = (fallback_label[1] + fallback_label[3]) / 2.0
            bw_c = max(1.0, float(fallback_box[2] - fallback_box[0]))
            bh_c = max(1.0, float(fallback_box[3] - fallback_box[1]))
            h_off_c = (lx_c - bx_c) / bw_c
            v_off_c = (ly_c - by_c) / bh_c
            if h == "Pending":
                h = "PASS" if abs(h_off_c) <= H_CENTRICITY_THRESH else "FAIL"
                track["final_h_value"] = abs(h_off_c)
                print(
                    f"Bottle #{track['id'] + 1} H fallback → "
                    f"{h_off_c:.3f} ({h})"
                )
            if v == "Pending":
                v = "PASS" if abs(v_off_c - EXPECTED_V) <= V_DEVIATION_THRESH else "FAIL"
                track["final_v_value"] = abs(v_off_c)
                print(
                    f"Bottle #{track['id'] + 1} V fallback → "
                    f"{v_off_c:.3f} ({v})"
                )

    track["h_center"] = h
    track["v_center"] = v

    orientation = majority_result(
        track.get("orientation_history", [])
    )
    # Freeze representative numeric measurements independently from the
    # categorical PASS/FAIL majority result.
    # Preserve fallback values set above; only overwrite if history data exists.
    h_numeric = stable_numeric([abs(v) for v in track.get("h_value_history", [])])
    v_numeric = stable_numeric([abs(v) for v in track.get("v_value_history", [])])
    if h_numeric is not None:
        track["final_h_value"] = h_numeric
    if v_numeric is not None:
        track["final_v_value"] = v_numeric
    track["final_orientation_angle"] = stable_numeric(
        track.get("orientation_angle_history", [])
    )

    # If a measurement was never available, keep it as Pending/N/A rather
    # than silently converting missing data into FAIL.
    if track["h_center"] is None:
        track["h_center"] = "Pending"
    if track["v_center"] is None:
        track["v_center"] = "Pending"
    if track["orientation"] is None:
        track["orientation"] = "Pending"

    # If capacity was never read by OCR, fall back to the expected capacity
    # passed via --expected-capacity. All bottles in a single run are the
    # same type, so this is always correct for the current run.
    if track.get("capacity") is None:
        track["capacity"] = EXPECTED_CAPACITY

    track["finalized"] = True

    # Missing measurements are INCOMPLETE, not DEFECTIVE.
    missing_measurement = any(
        track[key] == "Pending"
        for key in ("orientation", "h_center", "v_center")
    )

    if missing_measurement:
        track["final_status"] = "INCOMPLETE"
    elif bool(track["defects"]) or any(
        track[key] == "FAIL"
        for key in ("orientation", "h_center", "v_center")
    ):
        track["final_status"] = "DEFECTIVE"
    else:
        track["final_status"] = "GOOD"
    return track["final_status"]


def analyze_bottle(frame, bottle_box, capacity_boxes, label_boxes, damage_boxes, bump_boxes):
    result = {
        "capacity": None,
        "defects": [],
        "orientation": None,
        "h_center": None,
        "v_center": None,
    }

    _cap_tol = 15
    for cap_box in capacity_boxes:
        if (
            cap_box[0] >= bottle_box[0] - _cap_tol
            and cap_box[2] <= bottle_box[2] + _cap_tol
            and cap_box[1] >= bottle_box[1] - _cap_tol
            and cap_box[3] <= bottle_box[3] + _cap_tol
        ):
            cap = run_ocr(frame, cap_box)
            if cap:
                result["capacity"] = cap
                break

    # H/V centricity is intentionally not calculated here.
    # It is calculated through update_centricity() so the live path uses
    # the same mask-based centroids and expected-V logic for every bottle type.

    # Defects are evaluated only after the bottle segmentation
    # mask is available.
    result["defects"] = []

    return result


def draw_bottle_annotation(image, bottle_box, track, label_box=None, damage_boxes=None, bump_boxes=None, orientation_data=None):
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = map(int, bottle_box)

    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    bottle_color = (0, 255, 0)
    label_color = (255, 0, 255)
    axis_color = (255, 255, 0)
    centricity_color = (0, 165, 255)
    orientation_color = (255, 0, 0)
    defect_color = (0, 0, 255)
    text_color = (255, 255, 255)

    cv2.rectangle(image, (x1, y1), (x2, y2), bottle_color, 3)
    cv2.putText(image, "BOTTLE", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bottle_color, 2, cv2.LINE_AA)

    bottle_cx = int(round((x1 + x2) / 2))
    bottle_cy = int(round((y1 + y2) / 2))

    if orientation_data is not None:
        local_orientation = dict(orientation_data)

        # Draw actual bottle segmentation mask
        draw_mask_contour(
            image,
            local_orientation,
            color=bottle_color,
            thickness=3,
            fill=False
        )

        # Draw orientation axes calculated from the mask
        draw_mask_orientation(
            image,
            local_orientation,
            orientation_color,
            3
        )

    if label_box is not None:
        lx1, ly1, lx2, ly2 = map(int, label_box)
        lx1 = max(0, min(lx1, w_img - 1))
        ly1 = max(0, min(ly1, h_img - 1))
        lx2 = max(0, min(lx2, w_img - 1))
        ly2 = max(0, min(ly2, h_img - 1))

        if lx2 > lx1 and ly2 > ly1:
            cv2.rectangle(image, (lx1, ly1), (lx2, ly2), label_color, 3)
            cv2.putText(image, "LABEL", (lx1, max(20, ly1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)

            label_cx = int(round((lx1 + lx2) / 2))
            label_cy = int(round((ly1 + ly2) / 2))

            cv2.drawMarker(image, (label_cx, label_cy), label_color, cv2.MARKER_CROSS, 20, 2)
            cv2.line(image, (bottle_cx, bottle_cy), (label_cx, bottle_cy), centricity_color, 3)
            cv2.line(image, (label_cx, bottle_cy), (label_cx, label_cy), centricity_color, 3)
            cv2.line(image, (bottle_cx, bottle_cy), (label_cx, label_cy), (255, 255, 255), 1)

    for defect_box in damage_boxes or []:
        dx1, dy1, dx2, dy2 = map(int, defect_box)
        dx1 = max(0, min(dx1, w_img - 1))
        dy1 = max(0, min(dy1, h_img - 1))
        dx2 = max(0, min(dx2, w_img - 1))
        dy2 = max(0, min(dy2, h_img - 1))
        if dx2 > dx1 and dy2 > dy1:
            cv2.rectangle(image, (dx1, dy1), (dx2, dy2), defect_color, 3)
            cv2.putText(image, "DAMAGE", (dx1, max(20, dy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

    for defect_box in bump_boxes or []:
        bx1, by1, bx2, by2 = map(int, defect_box)
        bx1 = max(0, min(bx1, w_img - 1))
        by1 = max(0, min(by1, h_img - 1))
        bx2 = max(0, min(bx2, w_img - 1))
        by2 = max(0, min(by2, h_img - 1))
        if bx2 > bx1 and by2 > by1:
            cv2.rectangle(image, (bx1, by1), (bx2, by2), defect_color, 3)
            cv2.putText(image, "BUMP", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

    is_defective = (
        bool(track["defects"])
        or track["orientation"] == "FAIL"
        or track["h_center"] == "FAIL"
        or track["v_center"] == "FAIL"
    )
    status = "DEFECTIVE" if is_defective else "GOOD"
    status_color = defect_color if is_defective else bottle_color

    lines = [
        f"Bottle #{track['id'] + 1}",
        f"Status: {status}",
        f"Capacity: {track['capacity']} ml" if track["capacity"] else "Capacity: N/A",
        "Orientation: "
        + (
            format_measurement(
                track.get("final_orientation_angle"),
                track.get("orientation"),
                " deg",
            )
            if track.get("final_orientation_angle") is not None
            else (track.get("orientation") or "N/A")
        ),
        "H Center: "
        + (
            format_measurement(
                track.get("final_h_value"),
                track.get("h_center"),
            )
            if track.get("final_h_value") is not None
            else (track.get("h_center") or "Pending")
        ),
        "V Center: "
        + (
            format_measurement(
                track.get("final_v_value"),
                track.get("v_center"),
            )
            if track.get("final_v_value") is not None
            else (track.get("v_center") or "Pending")
        ),
        f"Defects: {', '.join(track['defects']) if track['defects'] else 'None'}",
        "Blue = orientation axes",
        "Orange = centricity",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 2
    line_height = 22
    panel_width = min(w_img - 10, 360)
    panel_height = min(h_img - 10, line_height * len(lines) + 20)

    if panel_width > 20 and panel_height > 20:
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
        image[:] = cv2.addWeighted(overlay, 0.68, image, 0.32, 0)
        for i, line in enumerate(lines):
            color = status_color if i == 1 else text_color
            cv2.putText(image, line, (12, 25 + i * line_height), font, font_scale, color, thickness, cv2.LINE_AA)


def save_bottle_images(frame, track):
    global completed_count, good_count, defective_count, incomplete_count

    if track["saved"]:
        return False

    # Saving is only allowed for a finalized track.
    if not track.get("finalized", False):
        return False

    # If a real defect was detected, save a frame that actually contains
    # its bounding box. Otherwise use the best complete bottle frame.
    # Always initialize this because a GOOD bottle may have no defect snapshot.
    saved_orientation_data = None

    if (
        track.get("best_complete_defect_frame") is not None
        and track.get("defects")
    ):
        # Prefer a confirmed-defect frame that also contains the
        # complete bottle. Defect boxes and mask data come from
        # this exact same frame.
        frame = track["best_complete_defect_frame"]
        box = track["best_complete_defect_box"]
        saved_label_box = track.get(
            "best_complete_defect_label_box"
        )
        saved_damage_boxes = track.get(
            "best_complete_defect_damage_boxes",
            []
        )
        saved_bump_boxes = track.get(
            "best_complete_defect_bump_boxes",
            []
        )
        saved_orientation_data = track.get(
            "best_complete_defect_orientation_data"
        )

    elif track["best_complete_frame"] is not None:
        # For GOOD bottles, or when no complete defect snapshot exists,
        # use the best complete bottle frame.
        frame = track["best_complete_frame"]
        box = track["best_complete_box"]
        saved_label_box = track.get("best_complete_label_box")
        saved_damage_boxes = track.get(
            "best_complete_damage_boxes",
            []
        )
        saved_bump_boxes = track.get(
            "best_complete_bump_boxes",
            []
        )
        saved_orientation_data = track.get(
            "best_complete_orientation_data"
        )
        # Keep the frozen final measurements. The selected frame is only
        # used for the visual annotation; it must not change the final state.

    elif track.get("best_defect_frame") is not None and track.get("defects"):
        # Last-resort fallback when no complete defect snapshot exists.
        frame = track["best_defect_frame"]
        box = track["best_defect_box"]
        saved_label_box = track.get("best_defect_label_box")
        saved_damage_boxes = track.get("best_defect_damage_boxes", [])
        saved_bump_boxes = track.get("best_defect_bump_boxes", [])
        saved_orientation_data = track.get("best_defect_orientation_data")

    else:
        print(
            f"WARNING: Bottle #{track['id'] + 1} has no complete "
            f"in-frame detection; skipping save."
        )
        return False

    # If the chosen frame does not carry the defect boxes directly,
    # reproject the confirmed defect location from bottle-relative
    # coordinates onto this exact saved bottle box.
    bx1, by1, bx2, by2 = map(int, box)
    bw = max(1.0, float(bx2 - bx1))
    bh = max(1.0, float(by2 - by1))

    if "bump" in track["defects"] and not saved_bump_boxes:
        saved_bump_boxes = track.get("best_complete_bump_boxes", [])

        if not saved_bump_boxes:
            saved_bump_boxes = [
                (
                    int(round(bx1 + rx1 * bw)),
                    int(round(by1 + ry1 * bh)),
                    int(round(bx1 + rx2 * bw)),
                    int(round(by1 + ry2 * bh)),
                )
                for rx1, ry1, rx2, ry2
                in track.get("best_defect_bump_relative", [])
            ]

    if "damage" in track["defects"] and not saved_damage_boxes:
        saved_damage_boxes = track.get(
            "best_complete_damage_boxes",
            []
        )

        if not saved_damage_boxes:
            saved_damage_boxes = [
                (
                    int(round(bx1 + rx1 * bw)),
                    int(round(by1 + ry1 * bh)),
                    int(round(bx1 + rx2 * bw)),
                    int(round(by1 + ry2 * bh)),
                )
                for rx1, ry1, rx2, ry2
                in track.get("best_defect_damage_relative", [])
            ]

    x1, y1, x2, y2 = map(int, box)
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, x1 - SAVE_PADDING)
    y1 = max(0, y1 - SAVE_PADDING)
    x2 = min(frame_w, x2 + SAVE_PADDING)
    y2 = min(frame_h, y2 + SAVE_PADDING)

    if x2 <= x1 or y2 <= y1:
        print(f"WARNING: Could not save Bottle #{track['id'] + 1}: invalid crop.")
        return False

    original_crop = frame[y1:y2, x1:x2].copy()
    if original_crop.size == 0:
        print(f"WARNING: Could not save Bottle #{track['id'] + 1}: empty crop.")
        return False

    annotated_crop = original_crop.copy()

    local_box = (int(box[0] - x1), int(box[1] - y1), int(box[2] - x1), int(box[3] - y1))

    def to_local_box(saved_box):
        if saved_box is None:
            return None
        return (int(saved_box[0] - x1), int(saved_box[1] - y1), int(saved_box[2] - x1), int(saved_box[3] - y1))

    local_label_box = to_local_box(saved_label_box)
    local_damage_boxes = [to_local_box(b) for b in saved_damage_boxes]
    local_bump_boxes = [to_local_box(b) for b in saved_bump_boxes]

    local_orientation_data = saved_orientation_data

    if local_orientation_data is not None:
        contour = local_orientation_data.get("mask_contour")
        # Convert full-frame orientation_data → crop-local coordinates HERE,
        # so draw_bottle_annotation receives coords already in the crop space.
        local_orientation_data = dict(local_orientation_data)
        ocx, ocy = local_orientation_data["center"]
        local_orientation_data["center"] = (ocx - x1, ocy - y1)

        if contour is not None and len(contour) > 0:
            pts = np.asarray(contour, dtype=np.float32)
            local_pts = pts.copy()
            local_pts[:, 0] -= x1
            local_pts[:, 1] -= y1
            local_orientation_data["mask_contour"] = local_pts.tolist()

            print(
                f"DEBUG SAVE Bottle #{track['id'] + 1}: "
                f"local_box={local_box}, "
                f"local_mask_bbox=("
                f"{local_pts[:, 0].min():.0f}, "
                f"{local_pts[:, 1].min():.0f}, "
                f"{local_pts[:, 0].max():.0f}, "
                f"{local_pts[:, 1].max():.0f})"
            )
            
    draw_bottle_annotation(
        annotated_crop,
        local_box,
        track,
        label_box=local_label_box,
        damage_boxes=local_damage_boxes,
        bump_boxes=local_bump_boxes,
        orientation_data=local_orientation_data,
    )

    final_status = track.get("final_status")
    if final_status is None:
        missing_measurement = any(
            track[key] in {None, "Pending"}
            for key in ("orientation", "h_center", "v_center")
        )
        if missing_measurement:
            final_status = "INCOMPLETE"
        elif bool(track["defects"]) or any(
            track[key] == "FAIL"
            for key in ("orientation", "h_center", "v_center")
        ):
            final_status = "DEFECTIVE"
        else:
            final_status = "GOOD"
        track["final_status"] = final_status

    category = final_status.lower()

    print("=" * 40)
    print(f"Bottle #{track['id'] + 1} FINAL RESULT")
    orientation_text = (
        format_measurement(
            track.get("final_orientation_angle"),
            track.get("orientation"),
            " deg",
        )
        if track.get("final_orientation_angle") is not None
        else (track.get("orientation") or "N/A")
    )
    h_center_text = (
        format_measurement(
            track.get("final_h_value"),
            track.get("h_center"),
        )
        if track.get("final_h_value") is not None
        else (track.get("h_center") or "Pending")
    )
    v_center_text = (
        format_measurement(
            track.get("final_v_value"),
            track.get("v_center"),
        )
        if track.get("final_v_value") is not None
        else (track.get("v_center") or "Pending")
    )

    print(f"Orientation : {orientation_text}")
    print(f"H Center    : {h_center_text}")
    print(f"V Center    : {v_center_text}")
    print(f"Defects     : {', '.join(track['defects']) if track['defects'] else 'None'}")
    print(f"Status      : {final_status}")
    print("=" * 40)

    category_dir = os.path.join(SAVE_DIR, category)
    bottle_dir = os.path.join(category_dir, f"bottle_{track['id'] + 1:03d}")
    os.makedirs(bottle_dir, exist_ok=True)

    original_path = os.path.join(bottle_dir, "original.jpg")
    annotated_path = os.path.join(bottle_dir, "annotated.jpg")

    original_saved = cv2.imwrite(original_path, original_crop)
    annotated_saved = cv2.imwrite(annotated_path, annotated_crop)

    if original_saved and annotated_saved:
        track["saved"] = True
        track["finalized"] = True
        track["final_status"] = final_status

        completed_count += 1
        if final_status == "DEFECTIVE":
            defective_count += 1
        elif final_status == "INCOMPLETE":
            incomplete_count += 1
        else:
            good_count += 1

        print(f"Saved Bottle #{track['id'] + 1}:")
        print(f"  Original  : {original_path}")
        print(f"  Annotated : {annotated_path}")
        print(
            f"Final counts -> Total: {completed_count} | "
            f"Good: {good_count} | Defective: {defective_count} | "
            f"Incomplete: {incomplete_count}"
        )
        return True

    print(f"WARNING: Failed to save both images for Bottle #{track['id'] + 1}.")
    return False


def get_matching_label_box(bottle_box, label_boxes):
    """
    Select the label that is spatially most consistent with the bottle.

    Previously the first overlapping label was returned. If more than one
    label candidate was present, that could associate the bottle with a
    label from another region of the frame and produce a false H/V result.

    The existing 10 px containment tolerance is preserved, and the selected
    label is clipped to the bottle boundary before centricity is calculated.
    """
    bx, by = box_center(bottle_box)
    candidates = []

    for lbl_box in label_boxes:
        label_overlaps_bottle = (
            lbl_box[2] > bottle_box[0] - LABEL_CONTAINMENT_TOLERANCE
            and lbl_box[0] < bottle_box[2] + LABEL_CONTAINMENT_TOLERANCE
            and lbl_box[3] > bottle_box[1] - LABEL_CONTAINMENT_TOLERANCE
            and lbl_box[1] < bottle_box[3] + LABEL_CONTAINMENT_TOLERANCE
        )

        if not label_overlaps_bottle:
            continue

        clipped_label_box = (
            max(lbl_box[0], bottle_box[0]),
            max(lbl_box[1], bottle_box[1]),
            min(lbl_box[2], bottle_box[2]),
            min(lbl_box[3], bottle_box[3]),
        )

        if (
            clipped_label_box[2] <= clipped_label_box[0]
            or clipped_label_box[3] <= clipped_label_box[1]
        ):
            continue

        lx, ly = box_center(clipped_label_box)
        bw = max(1.0, float(bottle_box[2] - bottle_box[0]))
        bh = max(1.0, float(bottle_box[3] - bottle_box[1]))

        # Normalize the distance so it remains stable as the bottle
        # changes apparent size while moving through the frame.
        distance = (
            ((lx - bx) / bw) ** 2
            + ((ly - by) / bh) ** 2
        ) ** 0.5

        area = (
            clipped_label_box[2] - clipped_label_box[0]
        ) * (
            clipped_label_box[3] - clipped_label_box[1]
        )

        candidates.append((distance, -area, clipped_label_box))

    if not candidates:
        return None

    # Closest label center wins; clipped area is the deterministic tie-breaker.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]

def update_centricity(track, bottle_box, label_boxes, seg_detections=None, bottle_mask=None, frame=None):
    clipped_label_box = get_matching_label_box(bottle_box, label_boxes)
    if clipped_label_box is None:
        return False

    label_mask = (
        find_label_mask(bottle_box, seg_detections, frame)
        if seg_detections is not None and frame is not None
        else None
    )

    (
        _h_status,
        _v_status,
        h_offset,
        v_offset,
        (bx, by),
        (lx, ly),
    ) = check_centricity(
        bottle_box,
        clipped_label_box,
        bottle_mask=bottle_mask,
        label_mask=label_mask,
    )

    history = track.get("centricity_offset_history", [])
    if len(history) >= CENTRICITY_MIN_HISTORY:
        previous_h, previous_v = history[-1]
        if (abs(h_offset - previous_h) > CENTRICITY_SPATIAL_TOLERANCE or
                abs(v_offset - previous_v) > CENTRICITY_SPATIAL_TOLERANCE):
            print(f"Bottle #{track['id'] + 1} centricity sudden jump ignored: "
                  f"offset=({h_offset:.3f}, {v_offset:.3f}), "
                  f"previous=({previous_h:.3f}, {previous_v:.3f})")
            return False

    history.append((h_offset, v_offset))
    if len(history) > CENTRICITY_HISTORY_WINDOW:
        del history[:-CENTRICITY_HISTORY_WINDOW]
    track["centricity_offset_history"] = history

    h = "PASS" if abs(h_offset) <= H_CENTRICITY_THRESH else "FAIL"

    v = (
        "PASS"
        if abs(v_offset - EXPECTED_V) <= V_DEVIATION_THRESH
        else "FAIL"
    )
    track["h_center"] = h
    track["v_center"] = v

    print(f"Bottle #{track['id'] + 1} mask centricity updated: H={h}, V={v}, "
          f"mask_offset=({h_offset:.3f}, {v_offset:.3f}), "
          f"bottle_centroid=({bx:.1f}, {by:.1f}), "
          f"label_centroid=({lx:.1f}, {ly:.1f})"
        )

    return True

def defect_on_bottle(defect_box, bottle_mask):
    """
    Return True only if a defect box overlaps the actual
    bottle segmentation mask sufficiently.
    """

    if bottle_mask is None:
        return False

    dx1, dy1, dx2, dy2 = [int(v) for v in defect_box]

    mh, mw = bottle_mask.shape[:2]

    dx1 = max(0, dx1)
    dy1 = max(0, dy1)
    dx2 = min(mw, dx2)
    dy2 = min(mh, dy2)

    if dx2 <= dx1 or dy2 <= dy1:
        return False

    roi = bottle_mask[dy1:dy2, dx1:dx2]

    overlap_ratio = float(roi.sum()) / max(1, roi.size)

    return overlap_ratio >= DEFECT_OVERLAP_THRESH


def update_defects(
    track,
    bottle_box,
    damage_boxes,
    bump_boxes,
    frame=None,
    label_box=None,
    orientation_data=None,
    bottle_mask=None,
):
    """
    Confirm defects only after they are detected in consecutive frames.

    A single-frame defect prediction is treated as a possible false positive.
    The defect is added to the bottle only after it remains valid for
    DEFECT_CONFIRMATION_FRAMES consecutive frames.
    """

    defects = set(track["defects"])

    current_mask = bottle_mask

    if current_mask is None:
        current_mask = track.get("_current_bottle_mask")
    # ---------------------------------------------------------
    # Check whether each defect type is valid in this frame
    # ---------------------------------------------------------
    damage_valid = any(
        defect_on_bottle(dmg, current_mask)
        for dmg in damage_boxes
    )

    bump_valid = any(
        defect_on_bottle(bump, current_mask)
        for bump in bump_boxes
    )

    # ---------------------------------------------------------
    # Update defect confirmation streaks
    # Allow a short detector miss without immediately resetting
    # the confirmation streak.
    # ---------------------------------------------------------

    # DAMAGE
    if damage_valid:
        track["defect_streaks"]["damage"] += 1
        track["defect_missing_frames"]["damage"] = 0
    else:
        if track["defect_streaks"]["damage"] > 0:
            track["defect_missing_frames"]["damage"] += 1

            if (
                track["defect_missing_frames"]["damage"]
                > DEFECT_MAX_MISSING_FRAMES
            ):
                track["defect_streaks"]["damage"] = 0
                track["defect_missing_frames"]["damage"] = 0


    # BUMP
    if bump_valid:
        track["defect_streaks"]["bump"] += 1
        track["defect_missing_frames"]["bump"] = 0
    else:
        if track["defect_streaks"]["bump"] > 0:
            track["defect_missing_frames"]["bump"] += 1

            if (
                track["defect_missing_frames"]["bump"]
                > DEFECT_MAX_MISSING_FRAMES
            ):
                track["defect_streaks"]["bump"] = 0
                track["defect_missing_frames"]["bump"] = 0

    # ---------------------------------------------------------
    # Accept defect only after confirmation
    # ---------------------------------------------------------
    if (
        track["defect_streaks"]["damage"]
        >= DEFECT_CONFIRMATION_FRAMES
    ):
        defects.add("damage")

    if (
        track["defect_streaks"]["bump"]
        >= DEFECT_CONFIRMATION_FRAMES
    ):
        defects.add("bump")

    track["defects"] = sorted(defects)

    # ---------------------------------------------------------
    # Preserve a frame that actually contains a CONFIRMED defect
    # ---------------------------------------------------------
    confirmed_defect = (
        damage_valid
        and track["defect_streaks"]["damage"]
        >= DEFECT_CONFIRMATION_FRAMES
    ) or (
        bump_valid
        and track["defect_streaks"]["bump"]
        >= DEFECT_CONFIRMATION_FRAMES
    )

    if confirmed_defect and frame is not None:
        track["best_defect_frame"] = frame.copy()
        track["best_defect_box"] = bottle_box
        track["best_defect_label_box"] = label_box

        # Only store the defect boxes that are actually valid
        # against the bottle mask.
        track["best_defect_damage_boxes"] = [
            dmg for dmg in damage_boxes
            if defect_on_bottle(dmg, current_mask)
        ]

        track["best_defect_bump_boxes"] = [
            bump for bump in bump_boxes
            if defect_on_bottle(bump, current_mask)
        ]

        # Keep normalized defect coordinates relative to the bottle box.
        # This lets the annotation remain correct if the best saved
        # complete frame is different from the exact defect frame.
        bx1, by1, bx2, by2 = map(int, bottle_box)
        bw = max(1.0, float(bx2 - bx1))
        bh = max(1.0, float(by2 - by1))

        track["best_defect_damage_relative"] = [
            (
                (dmg[0] - bx1) / bw,
                (dmg[1] - by1) / bh,
                (dmg[2] - bx1) / bw,
                (dmg[3] - by1) / bh,
            )
            for dmg in track["best_defect_damage_boxes"]
        ]

        track["best_defect_bump_relative"] = [
            (
                (bump[0] - bx1) / bw,
                (bump[1] - by1) / bh,
                (bump[2] - bx1) / bw,
                (bump[3] - by1) / bh,
            )
            for bump in track["best_defect_bump_boxes"]
        ]

        track["best_defect_orientation_data"] = orientation_data

        # Also keep a confirmed-defect snapshot only when the complete
        # bottle is inside the source frame. The defect boxes and mask
        # data are then guaranteed to belong to the same frame.
        frame_h, frame_w = frame.shape[:2]
        dx1, dy1, dx2, dy2 = map(int, bottle_box)

        bottle_is_complete = (
            dx1 > 0
            and dy1 > 0
            and dx2 < frame_w
            and dy2 < frame_h
        )

        if bottle_is_complete:
            current_area = max(0, dx2 - dx1) * max(0, dy2 - dy1)

            if track["best_complete_defect_box"] is None:
                should_store_complete_defect = True
            else:
                cx1, cy1, cx2, cy2 = map(
                    int,
                    track["best_complete_defect_box"]
                )
                previous_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)

                should_store_complete_defect = (
                    current_area > previous_area
                )

            if should_store_complete_defect:
                track["best_complete_defect_frame"] = frame.copy()
                track["best_complete_defect_box"] = bottle_box
                track["best_complete_defect_label_box"] = label_box

                track["best_complete_defect_damage_boxes"] = [
                    dmg for dmg in damage_boxes
                    if defect_on_bottle(dmg, current_mask)
                ]

                track["best_complete_defect_bump_boxes"] = [
                    bump for bump in bump_boxes
                    if defect_on_bottle(bump, current_mask)
                ]

                if orientation_data is not None:
                    saved_orientation = dict(orientation_data)

                    if orientation_data.get("mask_contour") is not None:
                        saved_orientation["mask_contour"] = [
                            list(point)
                            for point in orientation_data["mask_contour"]
                        ]

                    track[
                        "best_complete_defect_orientation_data"
                    ] = saved_orientation
                else:
                    track[
                        "best_complete_defect_orientation_data"
                    ] = None

def update_best_complete_detection(
    track,
    box,
    frame,
    label_box=None,
    damage_boxes=None,
    bump_boxes=None,
    force=False,
    orientation_data=None,
):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    if not is_reliable_bottle_frame(box, frame.shape):
        return

    current_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Skip spurious/tiny detections.
    if current_area < MIN_BOTTLE_AREA:
        return

    # A complete saved snapshot must keep the frame and its
    # orientation/mask data together.
    #
    # If there is already a valid complete snapshot and the current
    # frame has no segmentation mask, do not replace the existing
    # snapshot with a frame that cannot provide mask visualization.
    if orientation_data is None and track.get("best_complete_frame") is not None:
        return

    if track["best_complete_box"] is None:
        should_update = True
    else:
        bx1, by1, bx2, by2 = map(int, track["best_complete_box"])
        best_area = max(0, bx2 - bx1) * max(0, by2 - by1)

        should_update = force or (
            current_area > best_area
            or (
                current_area == best_area
                and track.get("best_complete_label_box") is None
                and label_box is not None
            )
        )

    if not should_update:
        return

    # Store the frame and everything drawn on that frame together.
    track["best_complete_box"] = box
    track["best_complete_frame"] = frame.copy()
    track["best_complete_label_box"] = label_box
    track["best_complete_damage_boxes"] = list(damage_boxes or [])
    track["best_complete_bump_boxes"] = list(bump_boxes or [])

    if orientation_data is not None:
        saved_orientation = dict(orientation_data)

        if orientation_data.get("mask_contour") is not None:
            saved_orientation["mask_contour"] = [
                list(point)
                for point in orientation_data["mask_contour"]
            ]

        track["best_complete_orientation_data"] = saved_orientation
    else:
        track["best_complete_orientation_data"] = None

def update_best_valid_detection(
    track,
    box,
    frame,
    label_box=None,
    damage_boxes=None,
    bump_boxes=None,
    force=False,
    seg_detections=None,
    bottle_mask=None,
):
    if label_box is None:
        return

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    if not is_reliable_bottle_frame(box, frame.shape):
        return

    label_mask = (
        find_label_mask(box, seg_detections, frame)
        if seg_detections is not None
        else None
    )

    (
        h,
        v,
        h_offset,
        v_offset,
        _bottle_center,
        _label_center,
    ) = check_centricity(
        box,
        label_box,
        bottle_mask=bottle_mask,
        label_mask=label_mask,
    )

    h_error = abs(h_offset)
    v_error = abs(v_offset - EXPECTED_V)

    centricity_error = h_error + v_error

    current_area = max(0, x2 - x1) * max(0, y2 - y1)
    previous_error = track.get("best_valid_centricity_error")

    if force or track["best_valid_box"] is None:
        should_update = True
    elif previous_error is None:
        should_update = True
    else:
        bx1, by1, bx2, by2 = map(int, track["best_valid_box"])
        best_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        should_update = (
            centricity_error < previous_error - 1e-6
            or (abs(centricity_error - previous_error) <= 1e-6 and current_area > best_area)
        )

    if should_update:
        track["best_valid_box"] = box
        # Do not retain another full-resolution frame per bottle.
        track["best_valid_frame"] = None
        track["annotation_label_box"] = label_box
        track["annotation_damage_boxes"] = list(damage_boxes or [])
        track["annotation_bump_boxes"] = list(bump_boxes or [])
        track["best_valid_h_center"] = h
        track["best_valid_v_center"] = v
        track["best_valid_centricity_error"] = centricity_error


def should_save_bottle(track, frame_shape, force=False):
    """A bottle is finalized only after it has disappeared from the stream."""
    if track["saved"] or track.get("finalized", False):
        return False

    if not force and track["missing"] < MAX_MISSING_FRAMES:
        return False

    if track["best_complete_box"] is None or track["best_complete_frame"] is None:
        return False

    finalize_measurements(track)
    return True


# -----------------------------
# Folder Frame Source
# -----------------------------

class FolderFrameBuffer:
    def __init__(self, frame):
        self.frame = frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class FolderFrameSource:
    def __init__(self, frame_dir):
        self.frame_dir = frame_dir

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }

        self.frames = [
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if os.path.splitext(name)[1].lower() in valid_extensions
        ]

        # Natural filename sorting:
        # frame_2 comes before frame_10
        self.frames.sort(
            key=lambda path: [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", os.path.basename(path))
            ]
        )

        self.index = 0

        print(f"Frame folder: {frame_dir}")
        print(f"Frames found: {len(self.frames)}")

        if not self.frames:
            raise RuntimeError(
                f"No image frames found in folder: {frame_dir}"
            )

    def start(self):
        pass

    def stop(self):
        pass

    def fetch(self):
        if self.index >= len(self.frames):
            raise StopIteration

        frame_path = self.frames[self.index]
        self.index += 1

        frame = cv2.imread(frame_path)

        if frame is None:
            raise RuntimeError(
                f"Unable to read frame: {frame_path}"
            )

        return FolderFrameBuffer(frame)

    def destroy(self):
        pass

# -----------------------------
# Camera
# -----------------------------
# -----------------------------
# Camera / Frame Source
# -----------------------------

h_cam = None

if INPUT_MODE == "folder":

    ia = FolderFrameSource(FRAME_DIR)
    ia.start()

    print("Reading frames from folder...")
    print("Press q to exit.")

else:

    h_cam = Harvester()
    h_cam.add_file(CTI_PATH)
    h_cam.update()

    print(f"Devices found: {len(h_cam.device_info_list)}")
    print(f"Bottle images will be saved to: {SAVE_DIR}")

    ia = h_cam.create(1)
    ia.start()

    print("Streaming... Press q to exit.")


# -----------------------------
# Main Loop
# -----------------------------
try:
    while True:

        if INPUT_MODE == "folder":

            try:
                with ia.fetch() as buffer:
                    frame = buffer.frame

            except StopIteration:
                print("All frames have been processed.")
                break

        else:

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
                        cv2.COLOR_GRAY2BGR
                    )

                elif pixel_format == "RGB8":
                    frame = data.reshape(height, width, 3)
                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_RGB2BGR
                    )

                elif pixel_format == "BGR8":
                    frame = data.reshape(height, width, 3)

                elif pixel_format in (
                    "BayerRG8",
                    "BayerGB8",
                    "BayerGR8",
                    "BayerBG8",
                ):
                    bayer_map = {
                        "BayerRG8": cv2.COLOR_BayerRG2BGR,
                        "BayerGB8": cv2.COLOR_BayerGB2BGR,
                        "BayerGR8": cv2.COLOR_BayerGR2BGR,
                        "BayerBG8": cv2.COLOR_BayerBG2BGR,
                    }

                    frame = cv2.cvtColor(
                        data.reshape(height, width),
                        bayer_map[pixel_format],
                    )

                else:
                    continue

        detections = model.predict(frame, threshold=min(PER_CLASS_CONF.values()))

        # Segmentation is used to obtain bottle/label masks for orientation and centricity.
        seg_detections = seg_model.predict(frame, threshold=min(PER_CLASS_CONF.values()))

        bottle_boxes = []
        capacity_boxes = []
        label_boxes = []
        damage_boxes = []
        bump_boxes = []

        display = frame.copy()

        for i, cls in enumerate(detections.data["class_name"]):
            if float(detections.confidence[i]) < PER_CLASS_CONF.get(cls, THRESHOLD):
                continue
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            print(f"  [BOTTLE DETECT] cls={cls} conf={float(detections.confidence[i]):.3f}")
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
            cv2.putText(display, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for track in tracked:
            track["missing"] += 1
            track["frames_seen"] += 1

        for bottle in bottle_boxes:
            matched = False

            for track in tracked:
                if track.get("finalized", False) or track.get("saved", False):
                    continue
                if bottle_track_match(bottle, track["box"]):
                    track["box"] = bottle
                    track["missing"] = 0
                    matched = True

                    current_area = max(0, bottle[2] - bottle[0]) * max(0, bottle[3] - bottle[1])
                    best_area = max(0, track["best_box"][2] - track["best_box"][0]) * max(0, track["best_box"][3] - track["best_box"][1])
                    if current_area > best_area:
                        track["best_box"] = bottle
                        # Avoid retaining an additional full-resolution frame.
                        track["best_frame"] = None

                    bottle_mask = find_bottle_mask(bottle, seg_detections)

                    if bottle_mask is not None:
                        bottle_mask = resize_mask_to_frame(bottle_mask, frame)
                        bottle_mask = constrain_mask_to_box(bottle_mask, bottle)

                        # Store the latest valid mask.
                        track["_current_bottle_mask"] = bottle_mask

                        orientation_data = get_mask_orientation(bottle_mask)

                        if orientation_data is not None:
                            track["orientation_data"] = orientation_data
                            track["orientation"] = orientation_data["status"]

                    else:
                        # Segmentation missed this frame.
                        # Keep the previous valid mask and orientation data.
                        bottle_mask = track.get("_current_bottle_mask")
                        orientation_data = track.get("orientation_data")

                    bx1, by1, bx2, by2 = bottle
                    fh, fw = frame.shape[:2]
                    centricity_updated = False

                    centricity_updated = update_centricity(
                        track,
                        bottle,
                        label_boxes,
                        seg_detections=seg_detections,
                        bottle_mask=bottle_mask,
                        frame=frame,
                    )

                    if centricity_updated:
                        if track["h_center"] in {"PASS", "FAIL"}:
                            track["h_history"].append(track["h_center"])

                        if track["v_center"] in {"PASS", "FAIL"}:
                            track["v_history"].append(track["v_center"])

                    if track["orientation"] in {"PASS", "FAIL"}:
                        track["orientation_history"].append(track["orientation"])
                    
                    # Keep the numeric measurements that correspond to the
                    # accepted observations used for final reporting.
                    if centricity_updated and track.get("centricity_offset_history"):
                        latest_h, latest_v = track["centricity_offset_history"][-1]
                        track["h_value_history"].append(float(latest_h))
                        track["v_value_history"].append(float(latest_v))

                    if orientation_data is not None:
                        angle_deg = orientation_data.get("angle_deg")
                        if angle_deg is not None:
                            track["orientation_angle_history"].append(
                                float(angle_deg)
                            )

                    current_label_box = get_matching_label_box(bottle, label_boxes)
                    current_damage_boxes = [dmg for dmg in damage_boxes if dmg[0] >= bottle[0] and dmg[2] <= bottle[2] and dmg[1] >= bottle[1] and dmg[3] <= bottle[3]]
                    current_bump_boxes = [bump for bump in bump_boxes if bump[0] >= bottle[0] and bump[2] <= bottle[2] and bump[1] >= bottle[1] and bump[3] <= bottle[3]]
                    # TEMPORARY DEBUG — Bottle #15 only
                    if track["id"] + 1 == 15:
                        print("\n========== BOTTLE #15 BUMP DEBUG ==========")
                        print(f"Raw bump boxes: {len(bump_boxes)}")
                        print(f"Current bump boxes: {len(current_bump_boxes)}")
                        print(
                            f"Bottle mask: "
                            f"{track.get('_current_bottle_mask') is not None}"
                        )

                        for i, bump in enumerate(current_bump_boxes):
                            valid = defect_on_bottle(
                                bump,
                                track.get("_current_bottle_mask")
                            )
                            print(
                                f"Bump {i + 1}: "
                                f"box={tuple(map(int, bump))}, "
                                f"valid_on_bottle={valid}"
                            )

                        print(
                            f"Bump streak: "
                            f"{track['defect_streaks']['bump']}"
                        )
                        print("============================================")
                    
                    previous_defects = set(track["defects"])
                    update_defects(
                        track,
                        bottle,
                        current_damage_boxes,
                        current_bump_boxes,
                        frame=frame,
                        label_box=current_label_box,
                        orientation_data=orientation_data,
                        bottle_mask=bottle_mask,
                    )
                    defect_changed = set(track["defects"]) != previous_defects

                    update_best_complete_detection(track, bottle, frame, label_box=current_label_box, damage_boxes=current_damage_boxes, bump_boxes=current_bump_boxes, force=defect_changed, orientation_data=orientation_data)
                    update_best_valid_detection(
                        track,
                        bottle,
                        frame,
                        label_box=current_label_box,
                        damage_boxes=current_damage_boxes,
                        bump_boxes=current_bump_boxes,
                        force=defect_changed,
                        seg_detections=seg_detections,
                        bottle_mask=bottle_mask,
                    )

                    # Keep trying OCR while the bottle is visible.
                    # Capacity is stabilized from repeated observations rather
                    # than permanently locking on the first successful read.
                    observed_capacity = None
                    _cap_tol = 15  # px tolerance for capacity box containment

                    for cap_box in capacity_boxes:
                        if (
                            cap_box[0] >= bottle[0] - _cap_tol
                            and cap_box[2] <= bottle[2] + _cap_tol
                            and cap_box[1] >= bottle[1] - _cap_tol
                            and cap_box[3] <= bottle[3] + _cap_tol
                        ):
                            cap = run_ocr(frame, cap_box)

                            if cap in {100, 300, 500}:
                                observed_capacity = cap
                                break

                    if observed_capacity is not None:
                        track["capacity_history"].append(observed_capacity)

                        stable_cap = stable_capacity(
                            track["capacity_history"]
                        )

                        if stable_cap != track.get("capacity"):
                            track["capacity"] = stable_cap

                            print(
                                f"Bottle #{track['id'] + 1} capacity updated: "
                                f"{stable_cap} ml"
                            )

                            with open(LOG_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    track['id'] + 1,
                                    stable_cap,
                                    (
                                        format_measurement(
                                            track.get("orientation_angle_history", [])[-1]
                                            if track.get("orientation_angle_history")
                                            else None,
                                            track.get("orientation"),
                                            " deg",
                                        )
                                        if track.get("orientation_angle_history")
                                        else track.get("orientation", "Pending")
                                    ),
                                    (
                                        format_measurement(
                                            abs(track["centricity_offset_history"][-1][0]),
                                            track.get("h_center"),
                                        )
                                        if track.get("centricity_offset_history")
                                        else track.get("h_center", "Pending")
                                    ),
                                    (
                                        format_measurement(
                                            abs(track["centricity_offset_history"][-1][1]),
                                            track.get("v_center"),
                                        )
                                        if track.get("centricity_offset_history")
                                        else track.get("v_center", "Pending")
                                    ),
                                    ', '.join(track['defects']) or 'None',
                                    datetime.now().strftime('%H:%M:%S')
                                ])

                    trigger_crossed = has_crossed_trigger_line(
                        track,
                        bottle,
                        frame.shape[1]
                    )

                    break

            if not matched:
                bottle_count += 1
                track = create_track(bottle)
                track["best_frame"] = frame.copy()

                result = analyze_bottle(frame, bottle, capacity_boxes, label_boxes, damage_boxes, bump_boxes)
                track["capacity"] = result["capacity"]
                track["defects"] = result["defects"]

                bottle_mask = find_bottle_mask(bottle, seg_detections)

                if bottle_mask is not None:
                    bottle_mask = resize_mask_to_frame(bottle_mask, frame)
                    bottle_mask = constrain_mask_to_box(
                        bottle_mask,
                        bottle
                    )

                    track["_current_bottle_mask"] = bottle_mask

                    orientation_data = get_mask_orientation(bottle_mask)

                    if orientation_data is not None:
                        track["orientation_data"] = orientation_data
                        track["orientation"] = orientation_data["status"]

                else:
                    # No segmentation mask was available when the bottle first appeared.
                    track["_current_bottle_mask"] = None
                    orientation_data = None

                initial_label_box = get_matching_label_box(bottle, label_boxes)

                centricity_updated = False
                centricity_updated = update_centricity(
                    track,
                    bottle,
                    label_boxes,
                    seg_detections=seg_detections,
                    bottle_mask=bottle_mask,
                    frame=frame,
                )

                if centricity_updated:
                    if track["h_center"] in {"PASS", "FAIL"}:
                        track["h_history"].append(track["h_center"])

                    if track["v_center"] in {"PASS", "FAIL"}:
                        track["v_history"].append(track["v_center"])

                    if centricity_updated and track.get("centricity_offset_history"):
                        latest_h, latest_v = track["centricity_offset_history"][-1]
                        track["h_value_history"].append(float(latest_h))
                        track["v_value_history"].append(float(latest_v))

                if track["orientation"] in {"PASS", "FAIL"}:
                    track["orientation_history"].append(track["orientation"])

                if orientation_data is not None:
                    angle_deg = orientation_data.get("angle_deg")
                    if angle_deg is not None:
                        track["orientation_angle_history"].append(
                            float(angle_deg)
                        )

                if track["capacity"] is not None:
                    track["capacity_history"].append(track["capacity"])

                initial_damage_boxes = [
                    dmg for dmg in damage_boxes
                    if dmg[0] >= bottle[0]
                    and dmg[2] <= bottle[2]
                    and dmg[1] >= bottle[1]
                    and dmg[3] <= bottle[3]
                ]

                initial_bump_boxes = [
                    bump for bump in bump_boxes
                    if bump[0] >= bottle[0]
                    and bump[2] <= bottle[2]
                    and bump[1] >= bottle[1]
                    and bump[3] <= bottle[3]
                ]

                # Evaluate initial defects against the actual
                # bottle segmentation mask.
                update_defects(
                    track,
                    bottle,
                    initial_damage_boxes,
                    initial_bump_boxes,
                    frame=frame,
                    label_box=initial_label_box,
                    orientation_data=orientation_data,
                )

                update_best_complete_detection(
                    track,
                    bottle,
                    frame,
                    label_box=initial_label_box,
                    damage_boxes=initial_damage_boxes,
                    bump_boxes=initial_bump_boxes,
                    orientation_data=orientation_data
                )
                update_best_valid_detection(
                    track,
                    bottle,
                    frame,
                    label_box=initial_label_box,
                    damage_boxes=initial_damage_boxes,
                    bump_boxes=initial_bump_boxes,
                    seg_detections=seg_detections,
                    bottle_mask=bottle_mask,
                )

                print("=" * 40)
                print(f"Bottle #{bottle_count}")
                print(f"Capacity    : {track['capacity'] or 'Not detected'} ml")
                print(f"Orientation : {track['orientation']}")
                print(f"H Center    : {track['h_center'] or 'Pending'}")
                print(f"V Center    : {track['v_center'] or 'Pending'}")
                print(f"Defects     : {', '.join(track['defects']) or 'None'}")
                print("=" * 40)

                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        bottle_count,
                        track['capacity'] or 'Not detected',
                        (
                            format_measurement(
                                track.get("orientation_angle_history", [])[-1]
                                if track.get("orientation_angle_history")
                                else None,
                                track.get("orientation"),
                                " deg",
                            )
                            if track.get("orientation_angle_history")
                            else track.get("orientation", "Pending")
                        ),
                        (
                            format_measurement(
                                abs(track["h_value_history"][-1]),
                                track.get("h_center"),
                            )
                            if track.get("h_value_history")
                            else track.get("h_center", "Pending")
                        ),
                        (
                            format_measurement(
                                abs(track["v_value_history"][-1]),
                                track.get("v_center"),
                            )
                            if track.get("v_value_history")
                            else track.get("v_center", "Pending")
                        ),
                        ", ".join(track["defects"]) or "None",
                        datetime.now().strftime("%H:%M:%S"),
                    ])

                tracked.append(track)

        # Finalize ONLY after the bottle has disappeared for the configured
        # number of frames. While visible, its measurements continue to update.
        remaining_tracks = []

        for track in tracked:

            if track.get("trigger_crossed", False) and not track["saved"]:
                if should_save_bottle(track, frame.shape, force=True):
                    print(
                        f"Bottle #{track['id'] + 1} crossed trigger line; "
                        f"finalizing stable result: {track['final_status']}"
                    )
                    save_bottle_images(frame, track)

            elif track["missing"] >= MAX_MISSING_FRAMES and not track["saved"]:
                if should_save_bottle(track, frame.shape, force=True):
                    print(
                        f"Bottle #{track['id'] + 1} disappeared; "
                        f"finalizing stable result: {track['final_status']}"
                    )
                    save_bottle_images(frame, track)

            # Once finalized/saved, remove the track immediately
            # so no later frame can modify its final result.
            if not track["saved"]:
                remaining_tracks.append(track)

        tracked = remaining_tracks

        for track in tracked:
            if track["missing"] > 0:
                continue
            x1, y1, x2, y2 = map(int, track["box"])

            bottle_color = (0, 255, 0)
            label_color = (255, 0, 255)
            axis_color = (255, 0, 0)
            centricity_color = (0, 165, 255)
            defect_color = (0, 0, 255)

            cv2.rectangle(display, (x1, y1), (x2, y2), bottle_color, 3)
            cv2.putText(display, f"Bottle #{track['id'] + 1}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bottle_color, 2, cv2.LINE_AA)

            bottle_cx = int((x1 + x2) / 2)
            bottle_cy = int((y1 + y2) / 2)

            if track.get("orientation_data") is not None and track["missing"] == 0:
                draw_mask_contour(
                    display,
                    track["orientation_data"],
                    (0, 255, 0),
                    3,
                    fill=False,
                )
                draw_mask_orientation(
                    display,
                    track["orientation_data"],
                    axis_color,
                    3,
                )

            live_label_box = get_matching_label_box(track["box"], label_boxes)
            if live_label_box is not None:
                lx1, ly1, lx2, ly2 = map(int, live_label_box)
                cv2.rectangle(display, (lx1, ly1), (lx2, ly2), label_color, 3)
                cv2.putText(display, "LABEL", (lx1, max(20, ly1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)

                label_cx = int((lx1 + lx2) / 2)
                label_cy = int((ly1 + ly2) / 2)
                cv2.drawMarker(display, (label_cx, label_cy), label_color, cv2.MARKER_CROSS, 22, 2)
                cv2.line(display, (bottle_cx, bottle_cy), (label_cx, bottle_cy), centricity_color, 3)
                cv2.line(display, (label_cx, bottle_cy), (label_cx, label_cy), centricity_color, 3)
                cv2.line(display, (bottle_cx, bottle_cy), (label_cx, label_cy), (255, 255, 255), 2)

            live_damage_boxes = [dmg for dmg in damage_boxes if dmg[0] >= x1 and dmg[2] <= x2 and dmg[1] >= y1 and dmg[3] <= y2]
            live_bump_boxes = [bump for bump in bump_boxes if bump[0] >= x1 and bump[2] <= x2 and bump[1] >= y1 and bump[3] <= y2]

            for defect_box in live_damage_boxes:
                dx1, dy1, dx2, dy2 = map(int, defect_box)
                cv2.rectangle(display, (dx1, dy1), (dx2, dy2), defect_color, 3)
                cv2.putText(display, "DAMAGE", (dx1, max(20, dy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

            for defect_box in live_bump_boxes:
                bx1, by1, bx2, by2 = map(int, defect_box)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), defect_color, 3)
                cv2.putText(display, "BUMP", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

            if track.get("final_status") in {"GOOD", "DEFECTIVE", "INCOMPLETE"}:
                status = track["final_status"]
            elif any(
                track[key] in {None, "Pending"}
                for key in ("orientation", "h_center", "v_center")
            ):
                status = "PENDING"
            elif bool(track["defects"]) or any(
                track[key] == "FAIL"
                for key in ("orientation", "h_center", "v_center")
            ):
                status = "DEFECTIVE"
            else:
                status = "GOOD"

            status_color = (
                (0, 0, 255) if status == "DEFECTIVE"
                else (0, 165, 255) if status in {"PENDING", "INCOMPLETE"}
                else (0, 255, 0)
            )
            text_y = min(display.shape[0] - 20, max(25, y1 + 25))

            live_orientation_angle = (
                track["orientation_angle_history"][-1]
                if track.get("orientation_angle_history")
                else None
            )
            live_h_value = (
                abs(track["centricity_offset_history"][-1][0])
                if track.get("centricity_offset_history")
                else None
            )
            live_v_value = (
                abs(track["centricity_offset_history"][-1][1])
                if track.get("centricity_offset_history")
                else None
            )

            info_lines = [
                f"Status: {status}",
                f"Capacity: {track['capacity']} ml" if track["capacity"] else "Capacity: N/A",
                "Orientation: "
                + (
                    format_measurement(
                        live_orientation_angle,
                        track.get("orientation"),
                        " deg",
                    )
                    if live_orientation_angle is not None
                    else (track["orientation"] or "N/A")
                ),
                "H Center: "
                + (
                    format_measurement(
                        live_h_value,
                        track.get("h_center"),
                    )
                    if live_h_value is not None
                    else (track["h_center"] or "Pending")
                ),
                "V Center: "
                + (
                    format_measurement(
                        live_v_value,
                        track.get("v_center"),
                    )
                    if live_v_value is not None
                    else (track["v_center"] or "Pending")
                ),
                f"Defects: {', '.join(track['defects']) if track['defects'] else 'None'}",
            ]

            for line_index, line in enumerate(info_lines):
                line_color = status_color if line_index == 0 else (255, 255, 255)
                cv2.putText(display, line, (x1 + 8, text_y + line_index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)

        cv2.putText(
            display,
            f"Completed: {completed_count} | Good: {good_count} | "
            f"Defective: {defective_count} | Incomplete: {incomplete_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        display_resized = cv2.resize(display, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Frosch Inference", display_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    for track in tracked:
        if not track["saved"] and track.get("best_complete_frame") is not None:
            if should_save_bottle(track, (0, 0), force=True):
                save_bottle_images(track["best_complete_frame"], track)

    print("=" * 40)
    print("FINAL COUNTS")
    print(
        f"Total: {completed_count} | Good: {good_count} | "
        f"Defective: {defective_count} | Incomplete: {incomplete_count}"
    )
    print("=" * 40)

    if ia is not None:
        ia.stop()

    if h_cam is not None:
        h_cam.reset()

    cv2.destroyAllWindows()
