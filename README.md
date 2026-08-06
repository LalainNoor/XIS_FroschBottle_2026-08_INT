# XIS_FroschBottle_2026-08_INT

RF-DETR Medium based bottle inspection pipeline for Frosch bottles. The project performs bottle detection, defect detection, capacity region detection, and OCR-based capacity recognition using EasyOCR. It also supports live inference using the Allied Vision Vimba Camera Simulator.

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
6. Track detected bottles and display OCR and defect results only once per bottle.

---

## Project Files

- `train.py` – RF-DETR model training
- `export_onnx.py` – Export trained model to ONNX
- `convert_trt.py` – Convert ONNX model to TensorRT
- `inference.py` – Image inference with OCR
- `live_inference.py` – Live inference with bottle tracking and OCR
- `FROSCH_TRAINING_REPORT.md` – Training report and evaluation results

---

## Features

- RF-DETR Medium instance segmentation
- Bottle, label, capacity, bump, damage, and scratch detection
- EasyOCR-based capacity recognition
- OCR restricted to capacity region
- Live inference using Allied Vision Camera Simulator
- Bottle tracking across frames
- OCR result displayed only once per bottle
- Live visualization of detections and OCR results

---

## Results

| Metric | Value |
|--------|-------|
| Best mAP | **0.6146** |
| Capacity Class AP | **0.7459** |
| OCR Detection Rate | **~78%** |

---

## Requirements

- Python 3.10+
- CUDA-enabled GPU
- RF-DETR
- EasyOCR
- OpenCV
- Harvester
- TensorRT
- NumPy

---

## Run Image Inference

```bash
python inference.py
```

---

## Run Live Inference

```bash
python live_inference.py
```

---

## Live Inference Workflow

- Read frames from the Vimba Camera Simulator.
- Detect bottles and defects using RF-DETR.
- Detect the capacity region.
- Extract the capacity value using EasyOCR.
- Track bottles across frames.
- Display the capacity and defect information only once for each tracked bottle.
