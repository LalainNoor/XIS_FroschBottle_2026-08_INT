# Setup & Installation Guide

## Verified Environment

See `ENVIRONMENT.md` for the complete version table.

Key values:

- Ubuntu 22.04.5 LTS
- Python 3.10.20
- NVIDIA GeForce RTX 5080, 16 GB
- NVIDIA driver 595.84
- CUDA Toolkit 12.9.41
- PyTorch 2.13.0+cu130
- TensorRT 10.14.1.48.post1
- RF-DETR 1.9.1
- EasyOCR 1.7.2
- OpenCV 4.10.0
- Harvester 1.4.3

## Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required Model Artifacts

The runtime expects:

```text
output/rfdetr-medium.trt
output/rfdetr-seg-medium.trt
```

## Folder-Based Inference

```bash
python live_inference.py     --input folder     --frame-dir /path/to/frames     --expected-capacity 500
```

The expected-capacity value can be `100`, `300`, or `500`.

Examples:

```bash
python live_inference.py --input folder --frame-dir /path/to/frames --expected-capacity 100
python live_inference.py --input folder --frame-dir /path/to/frames --expected-capacity 300
python live_inference.py --input folder --frame-dir /path/to/frames --expected-capacity 500
```

## Camera / Simulator

The runtime uses Harvester for GenICam/Vimba X acquisition. The CTI path on the development machine is machine-specific.

See `Camera Simulator - Setup Guide.md` for simulator installation, frame extraction and playback.

## Outputs

The runtime produces:

- finalized bottle status,
- CSV result,
- original bottle crop,
- annotated bottle crop.

## Troubleshooting

### TensorRT engine loading

Check GPU/driver availability, TensorRT installation, engine path and engine/environment compatibility.

### Camera not detected

Check Vimba X installation, CTI path, transport-layer availability and simulator/device visibility.

### No bottles detected from a folder

Check that the folder exists, frames are readable and ordered, the engines load successfully, and the expected capacity is correct.

### OCR failure

OCR may fail to produce a supported capacity. The configured expected capacity can be used as a fallback.
