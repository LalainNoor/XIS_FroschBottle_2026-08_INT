# Frosch Bottle Inspection — Live Inference

## Overview

This repository contains the live computer-vision inference pipeline for Frosch bottle inspection.

The current pipeline combines:

- RF-DETR Medium object detection
- RF-DETR Medium instance segmentation
- EasyOCR for bottle-capacity reading
- Bottle tracking across camera frames
- Bottle-mask-based orientation estimation
- Label-based horizontal and vertical centricity checks
- Damage and bump detection
- Defect confirmation across consecutive frames
- Good / Defective / Incomplete final classification
- Complete-bottle image saving
- CSV result logging
- Live annotated display through OpenCV
- GenICam camera acquisition through Harvester

The current implementation is designed around a VimbaX/GenICam camera interface and the Frosch bottle inspection dataset.

## Current Pipeline

```text
GenICam / VimbaX camera
        |
        v
   Frame acquisition
        |
        +-----------------------------+
        |                             |
        v                             v
RF-DETR Medium detection      RF-DETR Seg Medium
        |                             |
        |                             +--> bottle mask
        |                                  |
        |                                  +--> orientation
        |
        +--> bottle
        +--> capacity ---------> EasyOCR
        +--> label ------------> H/V centricity
        +--> damage -----------> defect validation
        +--> bump -------------> defect validation
        |
        v
       Tracking
        |
        +--> temporal H/V/orientation history
        +--> defect confirmation
        +--> trigger-line / disappearance finalization
        |
        v
   Final classification
   GOOD / DEFECTIVE / INCOMPLETE
        |
        +--> CSV result log
        |
        +--> saved_bottles/
              ├── good/
              ├── defective/
              └── incomplete/
```

## Models

### Detection model

The live script loads:

```text
runs/frosch_medium/checkpoint_best_regular.pth
```

The detection model is instantiated as:

```python
RFDETRMedium(pretrain_weights=CHECKPOINT)
```

The runtime detection threshold is `0.40`.

### Segmentation model

The live script loads:

```text
runs/frosch_seg_medium/checkpoint_best_total.pth
```

The segmentation model is instantiated as:

```python
RFDETRSegMedium(pretrain_weights=SEG_CHECKPOINT)
```

The runtime segmentation threshold is `0.30`.

The segmentation model is used to obtain the actual bottle mask. The mask is resized to the camera-frame resolution and constrained to the detector bounding box before orientation is calculated.

## Detection Classes Used by the Live Script

The detection output is separated into these classes:

| Class | Purpose |
|---|---|
| `bottle` | Main bottle detection and tracking |
| `capacity` | Region containing bottle capacity text |
| `label` | Bottle-label region used for H/V centricity |
| `damage` | Damage defect detection |
| `bump` | Bump defect detection |

The runtime checkpoint has been reported as containing 7 classes. The current inference script explicitly consumes the five class names above; other checkpoint classes are not used by this script.

## Capacity OCR

EasyOCR is initialized with English text recognition and GPU execution:

```python
reader = easyocr.Reader(['en'], gpu=True)
```

The capacity crop is:

1. padded by 20 pixels,
2. converted to grayscale,
3. enlarged by 4×,
4. enhanced with CLAHE,
5. thresholded using Otsu,
6. passed to EasyOCR with a numeric allowlist.

The parser maps detected numeric text to the supported capacities:

- `100 ml`
- `300 ml`
- `500 ml`

OCR confidence below `0.20` is ignored.

## Bottle Tracking

A bottle is tracked across frames using:

1. IoU matching with threshold `0.40`;
2. a conservative center-distance fallback when IoU becomes too small.

The fallback is intended for the normal horizontal movement of bottles through the inspection area while keeping the vertical gate tighter to reduce track merging.

Each track maintains:

- bottle ID
- current bounding box
- capacity/history
- orientation/history
- H/V centricity history
- current segmentation mask
- confirmed defects
- best complete frame
- best defect frame
- final status
- saved/finalized state

## Orientation Check

Orientation is calculated from the actual segmentation-mask pixels rather than only from the detector box.

The mask pixels are used to calculate the principal axis through covariance/eigenvector analysis.

The resulting angle is compared against:

```text
ORIENTATION_MAX_ANGLE_DEG = 45.0
```

Therefore:

```text
angle <= 45°  -> PASS
angle > 45°   -> FAIL
```

The live display also draws the segmentation contour and the major/minor orientation axes.

## H/V Centricity

The label is matched to the bottle using spatial consistency rather than simply taking the first overlapping label.

The label is clipped to the bottle boundary before centricity is calculated.

Thresholds:

```text
H_CENTRICITY_THRESH = 0.15
V_CENTRICITY_THRESH = 0.15
```

The normalized offsets are:

```text
H error = |label_center_x - bottle_center_x| / bottle_width
V error = |label_center_y - bottle_center_y| / bottle_height
```

A value within the corresponding threshold is a PASS.

### Centricity stabilization

Because the bottle moves through the camera, the label-to-bottle offset can change gradually.

The current implementation:

- keeps a short history of normalized offsets,
- waits for a warm-up period,
- rejects sudden spatial jumps,
- retains recent history within a five-sample window.

Current values:

```text
CENTRICITY_SPATIAL_TOLERANCE = 0.08
CENTRICITY_MIN_HISTORY       = 3
CENTRICITY_HISTORY_WINDOW    = 5
```

## Defect Validation

Damage and bump detections are not accepted solely because their bounding boxes overlap the detector bottle box.

A defect box must overlap the actual RF-DETR bottle segmentation mask by at least:

```text
DEFECT_OVERLAP_THRESH = 0.30
```

In addition, a defect must remain valid for:

```text
DEFECT_CONFIRMATION_FRAMES = 2
```

consecutive frames before it is added to the bottle's confirmed defect list.

This is intended to reduce one-frame false positives.

## Final Classification

The final status is frozen when the bottle is finalized.

### GOOD

A bottle is GOOD when:

- orientation is PASS,
- H centricity is PASS,
- V centricity is PASS,
- no confirmed defects exist.

### DEFECTIVE

A bottle is DEFECTIVE when:

- orientation is FAIL, or
- H centricity is FAIL, or
- V centricity is FAIL, or
- a confirmed `damage` defect exists, or
- a confirmed `bump` defect exists.

### INCOMPLETE

A bottle is INCOMPLETE when a required measurement remains unavailable/Pending.

Missing measurements are not silently converted into FAIL.

## Bottle Finalization

The script supports two finalization paths.

### 1. Trigger-line finalization

The bottle center is monitored against:

```text
TRIGGER_LINE_X_RATIO = 0.40
TRIGGER_LINE_TOLERANCE = 20 px
```

When the tracked bottle crosses the trigger line, its accumulated measurements are finalized and the bottle is saved.

### 2. Missing-frame finalization

If a bottle has not been detected for:

```text
MAX_MISSING_FRAMES = 20
```

frames, the track is finalized as a fallback.

This allows the pipeline to finalize bottles even when the trigger-line condition is not reached.

## Saved Bottle Images

The output directory is created relative to the current working directory:

```text
saved_bottles/
├── good/
│   └── bottle_XXX/
│       ├── original.jpg
│       └── annotated.jpg
├── defective/
│   └── bottle_XXX/
│       ├── original.jpg
│       └── annotated.jpg
└── incomplete/
    └── bottle_XXX/
        ├── original.jpg
        └── annotated.jpg
```

Each successfully saved bottle produces exactly two image files.

`original.jpg` is the clean crop.

`annotated.jpg` contains the inspection visualization, including the bottle ID, final status, capacity, orientation, H/V centricity, defect information, segmentation contour, and relevant inspection graphics.

A 20-pixel padding is applied around the selected bottle crop:

```text
SAVE_PADDING = 20
```

A track can only be saved once.

## CSV Logging

A timestamped CSV file is created when the script starts:

```text
results_YYYYMMDD_HHMMSS.csv
```

The columns are:

```text
Bottle
Capacity
Orientation
H_Center
V_Center
Defects
Timestamp
```

The CSV is updated as bottles are detected and capacity information becomes available.

## Camera Acquisition

The current script uses Harvester:

```python
from harvesters.core import Harvester
```

The CTI path is currently configured as:

```text
/home/xisai/Downloads/VimbaX_2026-2/cti/VimbaCameraSimulatorTL.cti
```

The script creates the camera interface and starts streaming.

The current implementation accepts:

- Mono8
- RGB8
- BGR8
- BayerRG8
- BayerGB8
- BayerGR8
- BayerBG8

The Bayer formats are converted to BGR using OpenCV.

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a conda environment:

```bash
conda create -n frosch python=3.10
conda activate frosch
pip install -r requirements.txt
```

The exact Python/PyTorch/CUDA combination should match the RF-DETR installation supported by the target machine.

## Required Model Files

Before running the live pipeline, the following checkpoints must exist:

```text
runs/
├── frosch_medium/
│   └── checkpoint_best_regular.pth
└── frosch_seg_medium/
    └── checkpoint_best_total.pth
```

Large model weights should be managed with Git LFS or approved external storage rather than committed directly to the normal Git tree.

## Running the Pipeline

From the repository root:

```bash
python live_inference.py
```

The window title is:

```text
Frosch Inference
```

Press:

```text
q
```

to stop the live display.

On shutdown, the camera interface is stopped/destroyed and OpenCV windows are closed.

## Configuration Summary

| Parameter | Current value | Purpose |
|---|---:|---|
| Detection threshold | `0.40` | RF-DETR detection confidence |
| Segmentation threshold | `0.30` | RF-DETR segmentation confidence |
| Tracking IoU | `0.40` | Primary bottle matching |
| Missing frames | `20` | Track finalization fallback |
| Trigger line | `40%` of frame width | Bottle finalization |
| Trigger tolerance | `20 px` | Minimum movement across trigger |
| H centricity | `0.15` | Horizontal centricity threshold |
| V centricity | `0.15` | Vertical centricity threshold |
| Centricity jump tolerance | `0.08` | Reject sudden normalized-offset jumps |
| Orientation limit | `45°` | Orientation PASS limit |
| Defect mask overlap | `0.30` | Required defect/mask overlap |
| Defect confirmation | `2 frames` | Consecutive-frame confirmation |
| Minimum bottle area | `20000 px²` | Prevent tiny/partial saves |
| Save padding | `20 px` | Crop padding |

## Important Current Limitations

The following points describe the current script rather than future assumptions:

1. The script does not currently call `cv2.undistort()`. Camera calibration/undistortion therefore needs to be handled upstream if required by the deployment setup.
2. The script does not explicitly set RF-DETR inference to FP16. Runtime logs have indicated that the loaded RF-DETR models are not optimized for inference, so latency can be higher than an optimized deployment.
3. The CTI path is machine-specific and must be changed on another system.
4. The current code assumes the expected detector class names are present.
5. The final Good/Defective/Incomplete counts are runtime inspection counts, not model accuracy metrics. A proper accuracy evaluation requires a ground-truth test set and sample-level comparison.
6. The current script records capacity as soon as OCR succeeds, but the final status does not require a valid capacity value.
7. The segmentation model is used to obtain the bottle mask and orientation. Detector boxes remain the main source for bottle tracking and the class-specific defect/label candidates.

## Training Documentation

RF-DETR segmentation training is documented separately in:

```text
FROSCH_TRAINING_REPORT.md
```

The training report records the segmentation training configuration that is available from the training script. Model evaluation metrics should only be added after the actual training/evaluation artifacts are available.

## Results Documentation

Runtime observations and validation results are documented in:

```text
RESULTS.md
```

The image-saving behavior is documented in:

```text
FROSHC_BOTTLE_IMAGE_SAVING.md
```

## Repository Hygiene

See `.gitignore` for the repository rules.

## Current Status

### Implemented

- [x] RF-DETR bottle detection
- [x] RF-DETR bottle segmentation
- [x] Bottle tracking
- [x] Capacity OCR
- [x] Label association
- [x] H centricity
- [x] V centricity
- [x] Mask-based orientation
- [x] Damage detection
- [x] Bump detection
- [x] Defect confirmation
- [x] Good / Defective / Incomplete classification
- [x] Trigger-line finalization
- [x] Missing-frame finalization
- [x] Original bottle saving
- [x] Annotated bottle saving
- [x] Separate output directories
- [x] CSV result logging
- [x] Live inspection visualization

