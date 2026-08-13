# Frosch RF-DETR Training Report

## 1. Purpose

This document records the RF-DETR models used by the Frosch bottle live-inspection pipeline.

The live inference system uses two separate RF-DETR checkpoints:

1. RF-DETR Medium for object detection.
2. RF-DETR Medium instance segmentation for bottle-mask generation.

The segmentation model is used during live inference to obtain the actual bottle mask and calculate bottle orientation.

## 2. Models Used by Live Inference

| Model | Class | Checkpoint |
|---|---|---|
| Detection | `RFDETRMedium` | `runs/frosch_medium/checkpoint_best_regular.pth` |
| Segmentation | `RFDETRSegMedium` | `runs/frosch_seg_medium/checkpoint_best_total.pth` |

The current inference script loads both checkpoints independently.

## 3. Detection Model

The detection checkpoint is loaded with:

```python
model = RFDETRMedium(
    pretrain_weights="runs/frosch_medium/checkpoint_best_regular.pth"
)
```

The live detection threshold is:

```text
0.40
```

The detector provides the class-specific bounding boxes used by the inspection logic.

The classes explicitly consumed by the current live script are:

- `bottle`
- `capacity`
- `label`
- `damage`
- `bump`

Runtime logs reported that the detection checkpoint contains 7 classes. The current inference code only uses the five class names above.

### Detection metrics

The available project material does not contain a verified final detection metric table for the checkpoint used by the current live script.

Therefore, the following should not be invented:

- mAP@0.50
- mAP@0.50:0.95
- precision
- recall
- F1
- validation loss
- test-set accuracy

These values should be added only from the actual RF-DETR training/evaluation output.

## 4. RF-DETR Segmentation Training

### Training script

The segmentation training script is designed specifically for the Frosch bottle COCO-segmentation dataset.

It uses:

```python
from rfdetr import RFDETRSegMedium
```

and initializes:

```python
model = RFDETRSegMedium()
```

The segmentation model intentionally starts from its own COCO-pretrained segmentation weights.

It does **not** load the detection checkpoint:

```text
runs/frosch_medium/checkpoint_best_regular.pth
```

This is important because the detection and segmentation models use different model architectures/checkpoint structures.

### Training configuration

The available training script defines these defaults:

| Parameter | Value |
|---|---:|
| Architecture | RF-DETR Medium segmentation |
| Epochs | `50` |
| Batch size | `4` |
| Gradient accumulation | `4` |
| Training resolution | `432` |
| Learning rate | `1e-4` |
| Dataset root | `.` by default |
| Output directory | `runs/frosch_seg_medium` |

The corresponding training call is:

```python
model.train(
    dataset_dir=str(dataset),
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    resolution=432,
    output_dir="runs/frosch_seg_medium",
)
```

### Reproducible command

From the repository root, the training configuration can be reproduced with:

```bash
python train_frosch_segmentation.py \
    --dataset . \
    --output runs/frosch_seg_medium \
    --epochs 50 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --resolution 432 \
    --lr 1e-4
```

## 5. Segmentation Checkpoint Used in Inference

The current live script expects:

```text
runs/frosch_seg_medium/checkpoint_best_total.pth
```

and loads it using:

```python
seg_model = RFDETRSegMedium(
    pretrain_weights=SEG_CHECKPOINT
)
```

with:

```text
SEG_THRESHOLD = 0.30
```

## 6. How the Segmentation Output Is Used

The segmentation model is not used to replace the detector.

Instead:

1. RF-DETR detection identifies the bottle bounding box.
2. RF-DETR segmentation provides candidate bottle masks.
3. The best segmentation mask is matched to the detector bottle box using IoU.
4. The mask is resized to the full camera-frame resolution if necessary.
5. The mask is constrained to the detector bounding box.
6. The mask pixels are used to calculate bottle orientation.
7. The same mask is used to validate whether damage/bump detections are actually on the bottle.

This separation keeps tracking/class association based on the detector while using segmentation where pixel-level geometry is required.


### Status

| Metric / Artifact | Status |
|---|---|
| Training architecture | Available |
| Training hyperparameters | Available |
| Output directory | Available |
| Inference checkpoint path | Available |
| Train/validation loss curves | Pending training artifact |
| mAP@0.50 | Pending evaluation artifact |
| mAP@0.50:0.95 | Pending evaluation artifact |
| IoU | Pending evaluation artifact |
| Precision | Pending evaluation artifact |
| Recall | Pending evaluation artifact |
| F1 | Pending evaluation artifact |
| Held-out test visualizations | Pending artifact |

**Do not replace the Pending values with estimates.**

## 7. Runtime Observations

When loading the current checkpoints, RF-DETR runtime warnings have been observed concerning:

- different positional encodings from DINOv2,
- patch-size differences,
- checkpoint/model class-count configuration,
- missing `num_queries` / `group_detr` checkpoint arguments,
- inference optimization.

These warnings should be documented as runtime/environment observations rather than interpreted as model-quality metrics.

The runtime also reported that the loaded RF-DETR model was not optimized for inference and suggested FP16 inference for higher GPU throughput.

## 8. Relationship to the Inspection Pipeline

The training outputs support the following inspection functions:

```text
Detection checkpoint
    |
    +--> bottle
    +--> capacity
    +--> label
    +--> damage
    +--> bump

Segmentation checkpoint
    |
    +--> bottle mask
            |
            +--> orientation
            |
            +--> defect-on-bottle validation
            |
            +--> annotated visualization
---

