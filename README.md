# XIS_FroschBottle_2026-08_INT

RF-DETR Medium based bottle inspection pipeline for Frosch bottles. The project performs bottle detection, defect detection, capacity region detection, OCR-based capacity recognition, label centricity checks, orientation checks, and live inference using the Allied Vision Vimba Camera Simulator.

---

## Classes
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
5. Run live inference using the Vimba Camera Simulator.
6. Track detected bottles and display all inspection results once per bottle.

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

- RF-DETR Medium instance segmentation
- Detection of bottle, label, capacity, bump, damage, scratch
- EasyOCR-based capacity reading (100ml / 300ml / 500ml)
- OCR restricted to capacity region crop
- Horizontal and vertical label centricity check (PASS/FAIL)
- Bottle orientation check (PASS/FAIL)
- Defect detection (bump, damage)
- Live inference via Allied Vision Camera Simulator
- IoU-based bottle tracking across frames
- OCR retry on subsequent frames if not detected on first appearance
- Results displayed once per tracked bottle
- CSV logging of all inspection results with timestamps

---

## Results

| Metric | Value |
|---|---|
| Best mAP (EMA) | **0.6146** |
| Capacity Class AP | **0.7459** |
| Static OCR Detection Rate | **~78%** |
| Live OCR Detection Rate | **~75%** (with retry logic) |

---

## Requirements
- Python 3.10+
- CUDA-enabled GPU
- Allied Vision Vimba X SDK

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Run Image Inference
```bash
python inference.py
```

---

## Run Live Inference

1. Configure Vimba Camera Simulator with BMP frames path in `VimbaCameraSimulatorTL.xml`
2. Run:
```bash
python live_inference.py
```

Results are saved to a timestamped CSV file in the same folder.

---

## Live Inference Output

Each detected bottle is printed once:
```
========================================
Bottle #1
Capacity    : 500 ml
Orientation : PASS
H Center    : PASS
V Center    : PASS
Defects     : None
========================================
```

And saved to CSV:
```
Bottle, Capacity, Orientation, H_Center, V_Center, Defects, Timestamp
1, 500, PASS, PASS, PASS, None, 11:39:23
```
