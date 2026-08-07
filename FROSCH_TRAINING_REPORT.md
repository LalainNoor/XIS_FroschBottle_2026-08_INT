# Frosch Bottle 5 — RF-DETR Medium Training Report

**Date:** 2026-08-05  
**Intern:** Lalain  
**Project:** Frosch Bottle Defect Detection  
**Model:** RF-DETR Medium (Instance Segmentation)  

---

## 1. Dataset

| Property | Value |
|---|---|
| Source | Roboflow — Frosch bottle 5 (v6) |
| Format | COCO JSON Segmentation |
| Total Images | 1,046 |
| Splits | Train + Validation |
| Classes | 6 |

### Class Distribution

| Class | Count |
|---|---|
| bottle | 1,035 |
| label | 1,030 |
| capacity | 955 |
| bump | 209 |
| damage | 72 |
| scratch | 65 |

---

## 2. Training Configuration

| Parameter | Value |
|---|---|
| Model | RF-DETR Medium |
| Epochs | 50 |
| Batch Size | 4 |
| Input Resolution | 576 (train), 736 (multi-scale) |
| Optimizer | AdamW |
| AMP | bfloat16 |
| EMA | Enabled |
| Hardware | NVIDIA RTX 5080 (16 GB VRAM) |
| Framework | rfdetr 1.9.1, PyTorch 2.13.0, PyTorch Lightning 2.6.5 |

---

## 3. Results

### Overall Best Metrics (Epoch 50 / Best EMA Epoch 49)

| Metric | Value |
|---|---|
| mAP 50:95 (best regular) | 0.6050 |
| mAP 50:95 (best EMA) | **0.6146** |
| mAP@50 | 0.7772 |
| mAP@75 | 0.6184 |
| F1 | 0.7849 |
| Precision | 0.8569 |
| Recall | 0.7547 |

### Per-Class Metrics (Final Epoch)

| Class | AP 50:95 | AR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| bottle | 0.9885 | 0.9961 | 0.9885 | 0.9837 | 0.9934 |
| label | 0.9878 | 0.9921 | 0.9885 | 0.9837 | 0.9934 |
| capacity | **0.7459** | 0.7894 | 0.9642 | 0.9439 | 0.9853 |
| damage | 0.5863 | 0.6917 | 0.8333 | 0.8333 | 0.8333 |
| bump | 0.2067 | 0.4047 | 0.5660 | 0.7143 | 0.4688 |
| scratch | 0.0780 | 0.2821 | 0.2857 | 0.7143 | 0.1786 |

---

## 4. Observations

- **bottle** and **label** classes achieved near-perfect AP (~0.99), consistent across all epochs.
- **capacity** (primary focus class) reached AP 0.7459 with high F1 (0.9642) and recall (0.9853).
- **damage** showed steady improvement, reaching AP 0.5863 by epoch 50.
- **bump** and **scratch** performed poorly due to limited training samples (209 and 65 respectively).
- Model converged steadily; EMA mAP improved from 0.0018 at epoch 0 to 0.6146 at epoch 49.

---

## 5. Saved Checkpoints

| Checkpoint | Path |
|---|---|
| Best Regular | `runs/frosch_medium/checkpoint_best_regular.pth` |
| Best EMA | `runs/frosch_medium/` (saved at epoch 49) |

---

## 6. Model Export & TensorRT Conversion

| Step | Output |
|---|---|
| ONNX Export | `runs/frosch_medium/rfdetr-medium.onnx` |
| TensorRT Engine | `runs/frosch_medium/rfdetr-medium.engine` |
| TensorRT Version | 10.14.1.48 |
| Precision | FP16 |

---

## 7. OCR Pipeline

**Library:** EasyOCR (GPU)  
**Target Class:** `capacity` only  
**Input:** Cropped capacity region from segmentation output  

### Preprocessing Steps
- BGR to Grayscale conversion
- 4x upscaling (INTER_CUBIC)
- CLAHE contrast enhancement (clipLimit=3.0)
- Otsu binarization
- Digit-only allowlist (`0123456789`)

### Capacity Extraction Logic
- First 3 digits extracted from OCR output
- Matched against valid values: `{100, 300, 500}`
- Prefix-based fallback for partial reads (`3xx→300`, `5xx→500`, `1xx→100`)

### Static Image Test Results (23 Images)

| Bottle Type | Total | Detected |
|---|---|---|
| 300 ml | 12 | 9 |
| 500 ml | 7 | 7 |
| 100 ml | 4 | 2 |

**Overall detection rate:** ~78%  
**Failure reason:** Partial occlusion or extreme angle causing capacity region to be cut off.

---

## 8. Live Inference Pipeline

**Camera:** Allied Vision Vimba Camera Simulator  
**CTI:** VimbaCameraSimulatorTL.cti  
**Script:** `live_inference.py`  

### Features
- Real-time bottle detection and tracking using IoU-based tracker
- OCR on capacity region — runs once on first appearance, retries on subsequent frames if not detected
- Results logged to timestamped CSV file
- Visual overlay on live feed

### Inspection Checks Per Bottle

| Check | Method |
|---|---|
| Capacity | EasyOCR on capacity crop |
| Orientation | PASS if bottle height ≥ width, else FAIL |
| H Center | PASS if label horizontal offset ≤ 15% of bottle width |
| V Center | PASS if label vertical offset ≤ 15% of bottle height |
| Defects | bump/damage detected via containment check |

### Live Test Results (500ml bottles, 28 bottles)

| Check | Result |
|---|---|
| Capacity detection rate | ~75% (improved with retry logic) |
| Orientation | All PASS |
| H Center | Mixed PASS/FAIL |
| V Center | Mostly PASS |
| Defect detection | bump detected on bottle #13 |

### CSV Log Output
Results saved to `results_YYYYMMDD_HHMMSS.csv` with columns:  
`Bottle, Capacity, Orientation, H_Center, V_Center, Defects, Timestamp`

---

## 9. Saved Files

| File | Path |
|---|---|
| Best Regular Checkpoint | `runs/frosch_medium/checkpoint_best_regular.pth` |
| Best EMA Checkpoint | `runs/frosch_medium/` (epoch 49) |
| ONNX Model | `runs/frosch_medium/rfdetr-medium.onnx` |
| TensorRT Engine | `runs/frosch_medium/rfdetr-medium.engine` |
| Image Inference Script | `inference.py` |
| Live Inference Script | `live_inference.py` |

---

## 10. Recommendations

- Re-annotate `bump` and `scratch` classes with more samples to improve recall.
- Improve camera angle/positioning to ensure capacity text is always fully visible.
- Fine-tune OCR on capacity region crops for better accuracy on dark/grainy images.
