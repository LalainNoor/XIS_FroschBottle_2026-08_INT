# XIS_FroschBottle_2026-08_INT

RF-DETR Medium based bottle inspection pipeline for Frosch bottles. The project performs bottle detection, defect detection, capacity region detection, OCR-based capacity recognition, label centricity checks, orientation checks, bottle tracking, and live inference using the Allied Vision Vimba Camera Simulator.

---

## Classes

The trained model contains the following classes:

- bottle
- label
- capacity
- bump
- damage
- scratch

---

## Pipeline

1. Train RF-DETR Medium on the Roboflow COCO Segmentation dataset.
2. Export the trained model to ONNX.
3. Convert the ONNX model to a TensorRT engine.
4. Perform image inference with EasyOCR for capacity recognition.
5. Run live inference using the Allied Vision Vimba Camera Simulator.
6. Detect and track bottles across consecutive frames using IoU-based tracking.
7. Perform OCR, orientation, label centricity, and defect checks when a new bottle is detected.
8. Retry capacity OCR on subsequent frames if the capacity was not detected initially.
9. Display bottle inspection results on the live visual overlay.
10. Maintain cumulative bottle, Good, and Defective counts.
11. Log inspection results to a timestamped CSV file.

---

## Project Files

| File | Description |
|---|---|
| `train.py` | RF-DETR model training |
| `export_onnx.py` | Export trained model to ONNX |
| `convert_trt.py` | Convert ONNX model to TensorRT |
| `inference.py` | Static image inference with OCR |
| `live_inference.py` | Live inference with bottle tracking, OCR, and inspection checks |
| `FROSCH_TRAINING_REPORT.md` | Full training and evaluation report |
| `requirements.txt` | Python dependencies |

---

## Features

### Detection

- RF-DETR Medium instance segmentation
- Detection of:
  - bottle
  - label
  - capacity
  - bump
  - damage
  - scratch

### Capacity Recognition

- EasyOCR-based capacity recognition
- Supports:
  - 100 ml
  - 300 ml
  - 500 ml
- OCR is performed on the detected capacity region
- OCR preprocessing includes:
  - grayscale conversion
  - image resizing
  - CLAHE enhancement
  - Otsu thresholding
- OCR is retried on subsequent frames if the capacity is not detected initially

### Bottle Inspection

Each newly tracked bottle is evaluated for:

- Bottle orientation
- Horizontal label centricity
- Vertical label centricity
- Detected defects

Inspection results are stored with the bottle track and displayed in the terminal and visual overlay.

### Good / Defective Classification

A bottle is classified as:

- **Good** — no detected `damage` or `bump`
- **Defective** — one or more `damage` or `bump` detections are associated with the bottle

The Good/Defective status is displayed on the live visual overlay.

### Centricity Checks

Horizontal and vertical label centricity are checked using the relative position of the label center to the bottle center.

Results are reported as:

```text
H: PASS / FAIL
V: PASS / FAIL
```

### Orientation Check

Bottle orientation is checked using the detected bottle bounding-box dimensions.

Results are reported as:
```text
    O: PASS / FAIL
```
### Bottle Tracking

- IoU-based tracking is used to associate bottles across consecutive frames.
- Each detected bottle receives a unique tracking ID.
- Bottle analysis is performed when the bottle first appears.
- Tracks are retained for a configurable number of missing frames.
- OCR can be retried for an existing track when the capacity was not initially detected.

### Live Visual Overlay

The live inference display provides inspection information for each tracked bottle:

    Good
    Bottle #2
    500 ml
    O: PASS
    H: PASS
    V: PASS

Where:

- `O` = Bottle orientation
- `H` = Horizontal label centricity
- `V` = Vertical label centricity

The status is displayed in green for Good/PASS results and red for Defective/FAIL results.

The top-left corner displays cumulative inspection counts:

    Total: 10 | Good: 8 | Defective: 2

Where:

- `Total` = Total bottles processed
- `Good` = Cumulative bottles without detected damage/bump defects
- `Defective` = Cumulative bottles with detected damage/bump defects

The counters remain cumulative even after previously detected bottles leave the camera view.

---

## Results

| Metric | Value |
|---|---:|
| Best mAP (EMA) | **0.6146** |
| Capacity Class AP | **0.7459** |
| Static OCR Detection Rate | **~78%** |
| Live OCR Detection Rate | **~75%** (with retry logic) |

---

## Requirements

- Python 3.10+
- CUDA-enabled GPU
- Allied Vision Vimba X SDK
- RF-DETR
- EasyOCR
- OpenCV

Install dependencies:

    pip install -r requirements.txt

---

## Run Image Inference

Run static image inference with:

    python inference.py

---

## Run Live Inference

### 1. Configure Vimba Camera Simulator

Configure the Allied Vision Vimba Camera Simulator with the required BMP frames path in:

    VimbaCameraSimulatorTL.xml

### 2. Run Live Inference

    python live_inference.py

The live inference window will display the detected bottles and their inspection results.

Results are also saved to a timestamped CSV file in the same directory.

---

## Live Inference Output

Each newly detected bottle is analyzed and printed once in the terminal:

    ========================================
    Bottle #1
    Capacity    : 500 ml
    Orientation : PASS
    H Center    : PASS
    V Center    : PASS
    Defects     : None
    ========================================

If capacity is not detected initially, OCR is retried on subsequent frames:

    Bottle #1 capacity updated: 500 ml

---

## Live Visual Overlay

The live inference display provides per-bottle inspection information:

    Good
    Bottle #2
    500 ml
    O: PASS
    H: PASS
    V: PASS

Where:

- `O` = Bottle orientation
- `H` = Horizontal label centricity
- `V` = Vertical label centricity

The top-left corner displays cumulative counts:

    Total: 10 | Good: 8 | Defective: 2

Where:

- `Total` = Total bottles processed
- `Good` = Cumulative bottles without detected damage/bump defects
- `Defective` = Cumulative bottles with detected damage/bump defects

---

## CSV Logging

Inspection results are saved automatically to a timestamped CSV file:

    results_YYYYMMDD_HHMMSS.csv

The CSV contains:

    Bottle,Capacity,Orientation,H_Center,V_Center,Defects,Timestamp

Example:

    Bottle,Capacity,Orientation,H_Center,V_Center,Defects,Timestamp
    1,500,PASS,PASS,PASS,None,11:39:23
    2,500,PASS,PASS,PASS,None,11:39:31

The CSV records the inspection results associated with each bottle when it is analyzed.

---

## Inspection Logic

For each newly detected bottle, the pipeline performs the following checks:

    Bottle Detection
           |
           v
    Bottle Tracking
           |
           +----> Capacity Region ---> EasyOCR
           |
           +----> Label Region ------> H/V Centricity
           |
           +----> Bottle Box --------> Orientation
           |
           +----> Damage/Bump -------> Defect Check
           |
           v
    Good / Defective Classification
           |
           +----> Visual Overlay
           |
           +----> Terminal Output
           |
           +----> CSV Logging

---

## Configuration

The main live inference configuration parameters include:

    THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    MAX_MISSING_FRAMES = 20
    H_CENTRICITY_THRESH = 0.15
    V_CENTRICITY_THRESH = 0.15

These control:

- Detection confidence threshold
- Bottle tracking IoU threshold
- Maximum number of missing frames before removing a track
- Horizontal centricity tolerance
- Vertical centricity tolerance

---

## Notes

- Bottle inspection results are generated once when a bottle first appears.
- Capacity OCR can be retried for an existing bottle track if the first OCR attempt fails.
- Bottle tracking prevents the same physical bottle from being counted repeatedly across consecutive frames.
- Good/Defective counters are cumulative and are not based only on bottles currently visible in the camera frame.
- The current Good/Defective decision is based on detected `damage` and `bump` defects.
- Orientation and centricity results are reported separately from the Good/Defective status.
- The `scratch` class is included in the trained detection classes, but the current Good/Defective decision logic does not classify a bottle as defective based on `scratch`.
