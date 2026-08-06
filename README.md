# XIS_FroschBottle_2026-08_INT

RF-DETR Medium based instance segmentation pipeline for Frosch bottle defect detection.

## Classes
bottle, label, capacity, bump, damage, scratch

## Pipeline
1. Train RF-DETR Medium on Roboflow COCO segmentation dataset
2. Export to ONNX
3. Convert to TensorRT
4. Run inference with EasyOCR on capacity class

## Files
- `train.py` — model training script
- `export_onnx.py` — ONNX export
- `convert_trt.py` — TensorRT conversion
- `inference.py` — segmentation + OCR inference
- `FROSCH_TRAINING_REPORT.md` — full training and results report

## Results
- Best mAP: 0.6146 (EMA, epoch 49)
- Capacity class AP: 0.7459
- OCR detection rate: ~78%

## Requirements
- Python 3.10+
- rfdetr, easyocr, opencv, tensorrt
