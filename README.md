# Frosch Bottle Inspection — Live Inference

## Overview

This repository contains the live computer-vision inference pipeline for Frosch bottle inspection.

The current pipeline combines:

- RF-DETR Medium object detection
- RF-DETR Medium instance segmentation
- EasyOCR for bottle-capacity reading
- Bottle tracking across camera frames
- Bottle-mask-based orientation estimation
- Mask-based horizontal and vertical centricity checks
- Damage and bump detection
- Per-class detector confidence thresholds
- Defect confirmation across frames
- GOOD / DEFECTIVE / INCOMPLETE final classification
- Capacity-specific H/V centricity references for 100 ml, 300 ml, and 500 ml bottles
- Complete-bottle image saving
- CSV result logging
- Live annotated display through OpenCV
- GenICam camera acquisition through Harvester

The current implementation supports both live camera input and recorded frame-folder testing.

## Current Pipeline

```text
GenICam / VimbaX camera or recorded frame folder
                    |
                    v
              Frame acquisition
                    |
          +---------+---------+
          |                   |
          v                   v
 RF-DETR Medium detection   RF-DETR Seg Medium
          |                   |
          |                   +--> bottle mask
          |                         |
          |                         +--> orientation
          |
          +--> bottle
          +--> capacity ------> EasyOCR
          +--> label ---------> H/V centricity
          +--> damage --------> defect validation
          +--> bump ----------> defect validation
          |
          v
       Tracking
          |
          +--> temporal H/V/orientation history
          +--> defect confirmation
          +--> trigger-line / missing-frame finalization
          |
          v
   Final classification
   GOOD / DEFECTIVE / INCOMPLETE
          |
          +--> CSV result log
          |
          +--> saved_bottles/
                +--> 100ml/
                |    +--> good/
                |    +--> defective/
                |    +--> incomplete/
                +--> 300ml/
                |    +--> good/
                |    +--> defective/
                |    +--> incomplete/
                +--> 500ml/
                     +--> good/
                     +--> defective/
                     +--> incomplete/
```

## Models

### Detection model

The live script loads:

```text
runs/frosch_medium/checkpoint_best_regular.pth
```

The model is instantiated as:

```python
RFDETRMedium(pretrain_weights=CHECKPOINT)
```

The live pipeline uses per-class confidence thresholds instead of relying on one universal threshold:

| Class | Confidence |
|---|---:|
| `bottle` | `0.70` |
| `label` | `0.35` |
| `capacity` | `0.35` |
| `bump` | `0.50` |
| `damage` | `0.30` |
| `scratch` | `0.30` |

The current live inspection logic explicitly consumes the bottle, label, capacity, damage, and bump classes.

### Segmentation model

The live script loads:

```text
runs/frosch_seg_medium/checkpoint_best_total.pth
```

The model is instantiated as:

```python
RFDETRSegMedium(pretrain_weights=SEG_CHECKPOINT)
```

The runtime segmentation threshold is:

```text
SEG_THRESHOLD = 0.30
```

The best bottle segmentation mask is matched to the detector bottle using IoU, resized to the frame resolution when needed, and constrained to the detector box before mask-based geometry is calculated.

## Capacity OCR

EasyOCR is initialized with GPU execution:

```python
reader = easyocr.Reader(['en'], gpu=True)
```

The detected capacity region is padded, converted to grayscale, enlarged 5×, enhanced with CLAHE, and evaluated with multiple preprocessing variants including Otsu and adaptive thresholding.

OCR results are restricted to:

- `100 ml`
- `300 ml`
- `500 ml`

OCR candidates below confidence `0.15` are ignored.

The final run also supports an expected bottle capacity supplied through `--expected-capacity`. If OCR never produces a valid capacity, the finalization logic falls back to that configured expected capacity for the current run.

## Bottle Tracking

Bottle tracks use:

- IoU matching with threshold `0.40`
- conservative center-distance fallback for normal horizontal motion
- up to `20` missing frames before fallback finalization

Each track maintains capacity history, orientation history, H/V centricity history, segmentation state, defect state, selected complete frames, selected defect frames, and final status.

## Orientation Check

Orientation is calculated from actual bottle-mask pixels rather than only from the detector bounding box.

The mask pixels are analyzed using covariance/eigenvector geometry to obtain the bottle's principal axis.

```text
ORIENTATION_MAX_ANGLE_DEG = 45.0
```

Therefore:

```text
angle <= 45°  -> PASS
angle > 45°   -> FAIL
```

## H/V Centricity

Bottle and label centroids are primarily obtained from segmentation masks. Bounding-box centers are used only as fallback when the corresponding mask is unavailable.

The normalized offsets are:

```text
H offset = (label_center_x - bottle_center_x) / bottle_width
V offset = (label_center_y - bottle_center_y) / bottle_height
```

### H centricity

```text
H_CENTRICITY_THRESH = 0.15
```

### Capacity-specific V centricity

The expected normalized V position depends on bottle capacity:

| Capacity | Expected V |
|---|---:|
| 500 ml | `0.01` |
| 100 ml | `0.12` |
| 300 ml | `0.07` |

The allowed deviation is:

```text
V_DEVIATION_THRESH = 0.05
```

### Centricity stabilization

The pipeline stabilizes measurements across multiple observations:

```text
CENTRICITY_SPATIAL_TOLERANCE = 0.08
CENTRICITY_MIN_HISTORY = 3
CENTRICITY_HISTORY_WINDOW = 5
MIN_RELIABLE_CENTRICITY_MEASUREMENTS = 3
```

Sudden spatial jumps are ignored rather than immediately replacing an established measurement.

## Defect Validation

Damage and bump detections are validated against the actual bottle segmentation mask.

A defect must overlap the bottle mask by at least:

```text
DEFECT_OVERLAP_THRESH = 0.30
```

The current final pipeline accepts a defect after:

```text
DEFECT_CONFIRMATION_FRAMES = 1
```

valid detection, with:

```text
DEFECT_MAX_MISSING_FRAMES = 1
```

This preserves short-lived real detections while rejecting detections that do not overlap the actual bottle mask.

## Final Classification

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
- a confirmed damage defect exists, or
- a confirmed bump defect exists.

### INCOMPLETE

A bottle is INCOMPLETE when one or more required measurements remain `Pending`.

Missing measurements are kept distinct from actual FAIL conditions.

## Bottle Finalization

The script supports two finalization paths.

### Trigger-line finalization

```text
TRIGGER_LINE_X_RATIO = 0.40
TRIGGER_LINE_TOLERANCE = 20 px
```

When the tracked bottle crosses the trigger line, accumulated measurements are finalized.

### Missing-frame finalization

If a tracked bottle is absent for:

```text
MAX_MISSING_FRAMES = 20
```

frames, the track is finalized through the missing-frame fallback.

## Saved Bottle Images

The final implementation saves bottles by capacity and final status:

```text
saved_bottles/
├── 100ml/
│   ├── good/
│   │   └── bottle_XXX/
│   │       ├── original.jpg
│   │       └── annotated.jpg
│   ├── defective/
│   └── incomplete/
├── 300ml/
│   ├── good/
│   ├── defective/
│   └── incomplete/
└── 500ml/
    ├── good/
    ├── defective/
    └── incomplete/
```

Each successfully saved bottle produces:

```text
1 bottle -> original.jpg + annotated.jpg
```

`original.jpg` is the clean selected bottle crop.

`annotated.jpg` contains the relevant inspection visualization, including the bottle bounding box, segmentation contour, orientation graphics, label information, centricity guides, defect boxes, and final result information.

The crop uses:

```text
MIN_BOTTLE_AREA = 20000 px²
SAVE_PADDING = 20 px
```

## CSV Logging

The CSV filename is capacity-specific:

```text
results_100ml.csv
results_300ml.csv
results_500ml.csv
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

## Camera Acquisition

The current script uses Harvester:

```python
from harvesters.core import Harvester
```

The configured CTI path is machine-specific:

```text
/home/xisai/Downloads/VimbaX_2026-2/cti/VimbaCameraSimulatorTL.cti
```

The camera path should be changed on another machine as required.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A conda environment can also be used:

```bash
conda create -n frosch python=3.10
conda activate frosch
pip install -r requirements.txt
```

The Python/PyTorch/CUDA combination must be compatible with the RF-DETR installation on the target system.

## Required Model Files

Before inference, the following checkpoints must be available:

```text
runs/
├── frosch_medium/
│   └── checkpoint_best_regular.pth
└── frosch_seg_medium/
    └── checkpoint_best_total.pth
```

Model checkpoints are ignored by the repository `.gitignore` and should be managed through Git LFS or approved external storage.

## Running the Pipeline

### Live camera

```bash
python live_inference.py --input camera --expected-capacity 500
```

Replace `500` with `100` or `300` for the corresponding bottle type.

### Recorded folder

```bash
python live_inference.py --input folder     --frame-dir /path/to/frames     --expected-capacity 500
```

The supported capacities are:

```text
100
300
500
```

Example validation commands:

```bash
python live_inference.py --input folder     --frame-dir /path/to/100ml_frames     --expected-capacity 100

python live_inference.py --input folder     --frame-dir /path/to/300ml_frames     --expected-capacity 300

python live_inference.py --input folder     --frame-dir /path/to/500ml_frames     --expected-capacity 500
```

Press `q` to stop the live display.

## Configuration Summary

| Parameter | Final value | Purpose |
|---|---:|---|
| Bottle confidence | `0.70` | Filters low-confidence bottle detections |
| Label confidence | `0.35` | Label association |
| Capacity confidence | `0.35` | Capacity candidate detection |
| Bump confidence | `0.50` | Reduces bump false positives |
| Damage confidence | `0.30` | Improves damage recall |
| Scratch confidence | `0.30` | Low threshold for the undertrained class |
| Segmentation threshold | `0.30` | Bottle-mask confidence |
| Tracking IoU | `0.40` | Primary bottle matching |
| Missing frames | `20` | Track finalization fallback |
| Trigger line | `40%` of frame width | Bottle finalization |
| Trigger tolerance | `20 px` | Minimum trigger-line movement |
| Complete X margin | `20 px` | Reliable complete-frame measurements |
| H centricity | `0.15` | Horizontal offset limit |
| 500 ml expected V | `0.01` | Capacity-specific V reference |
| 100 ml expected V | `0.12` | Capacity-specific V reference |
| 300 ml expected V | `0.07` | Capacity-specific V reference |
| V deviation | `0.05` | Allowed V deviation |
| Centricity jump tolerance | `0.08` | Reject sudden normalized-offset jumps |
| Minimum reliable centricity observations | `3` | H/V stabilization |
| Orientation limit | `45°` | Orientation PASS limit |
| Defect mask overlap | `0.30` | Required defect/mask overlap |
| Defect confirmation | `1 frame` | Valid defect acceptance |
| Defect missing-frame tolerance | `1 frame` | Short detector gaps |
| Minimum bottle area | `20000 px²` | Prevent partial/spurious saves |
| Save padding | `20 px` | Crop padding |

## Validation Results

Final recorded frame-folder validation was performed for all three bottle capacities on 21 August 2026. Runtime counts are validation observations, not formal precision/recall/F1/mAP metrics.

See `RESULTS.md` for the detailed run summary.

## Training Documentation

Training configuration is documented separately in:

```text
FROSCH_TRAINING_REPORT.md
```

Training metrics are only documented when they are available from actual training/evaluation artifacts.

## Bottle Image Saving Documentation

Detailed saving behavior is documented in:

```text
FROSCH_BOTTLE_IMAGE_SAVING.md
```

## Repository Hygiene

See `.gitignore` for ignored runtime outputs, model artifacts, camera SDK files, local configuration, and temporary files.

## Demo
 
A live inference recording across all three bottle types (100 ml, 300 ml, 500 ml) is available here:
 
[Live Inference Demo — 21 August 2026](https://drive.google.com/file/d/1o3kBGbjst0_tstIQagQg7lkvzcRYC_23/view?usp=sharing)

## Current Status

- [x] RF-DETR bottle detection
- [x] RF-DETR bottle segmentation
- [x] Per-class confidence thresholds
- [x] Bottle tracking
- [x] Capacity OCR
- [x] Capacity-specific expected V references
- [x] Mask-based orientation
- [x] Mask-based H/V centricity
- [x] Centricity stabilization
- [x] Damage validation
- [x] Bump validation
- [x] Defect confirmation
- [x] GOOD / DEFECTIVE / INCOMPLETE classification
- [x] Trigger-line finalization
- [x] Missing-frame finalization
- [x] Capacity-specific bottle saving
- [x] Original bottle saving
- [x] Annotated bottle saving
- [x] CSV result logging
- [x] Live camera input
- [x] Recorded folder input
- [x] Final validation across 100 ml, 300 ml, and 500 ml bottle runs
