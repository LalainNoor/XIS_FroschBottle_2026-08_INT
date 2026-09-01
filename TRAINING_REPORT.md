# Model Training Report

## 1. Model Selection

RF-DETR was selected rather than Roboflow-hosted models or Ultralytics YOLO. The project uses separate detection and instance-segmentation models.

## 2. Detection Model

Checkpoint:

```text
runs/frosch_medium/checkpoint_best_regular.pth
```

Runtime engine:

```text
output/rfdetr-medium.trt
```

Input:

```text
1 x 3 x 576 x 576
```

Outputs:

```text
dets:   1 x 300 x 4
labels: 1 x 300 x 8
```

### Confidence Thresholds

| Class | Threshold |
|---|---:|
| bottle | 0.70 |
| label | 0.35 |
| capacity | 0.35 |
| bump | 0.50 |
| damage | 0.30 |
| scratch | 0.30 |

## 3. Detection Evaluation

| Class | Precision | Recall | F1 | AP50 | n_gt |
|---|---:|---:|---:|---:|---:|
| bottle | 0.9869 | 0.9934 | 0.9902 | 0.9091 | 304 |
| bump | 0.4487 | 0.5469 | 0.4930 | 0.4130 | 64 |
| capacity | 0.9189 | 0.9963 | 0.9561 | 0.9072 | 273 |
| damage | 0.3607 | 0.9167 | 0.5176 | 0.8455 | 24 |
| label | 0.9869 | 0.9934 | 0.9901 | 0.9088 | 303 |
| scratch | 0.3333 | 0.2143 | 0.2609 | 0.2061 | 28 |

The evaluation confusion matrix uses an **IoU matching threshold of 0.50**.

## 4. Confusion Matrix

| Ground truth \ Predicted | bottle | bump | capacity | damage | label | scratch | missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 302 | 0 | 0 | 0 | 0 | 0 | 2 |
| bump | 3 | 68 | 4 | 10 | 3 | 2 | 29 |
| capacity | 0 | 0 | 273 | 0 | 0 | 1 | 1 |
| damage | 0 | 0 | 2 | 41 | 0 | 0 | 2 |
| label | 0 | 10 | 15 | 4 | 301 | 1 | 2 |
| scratch | 1 | 0 | 0 | 6 | 1 | 14 | 22 |

This is model-level evaluation and must not be interpreted as a bottle-level GOOD/DEFECTIVE/INCOMPLETE confusion matrix.

## 5. Segmentation Model

Checkpoint:

```text
runs/frosch_seg_medium/checkpoint_best_total.pth
```

Runtime engine:

```text
output/rfdetr-seg-medium.trt
```

Input:

```text
1 x 3 x 432 x 432
```

Outputs:

```text
dets:   1 x 200 x 4
labels: 1 x 200 x 8
masks:  1 x 200 x 108 x 108
```

Segmentation threshold: `0.30`.

## 6. Segmentation Training Configuration

| Parameter | Value |
|---|---:|
| Architecture | RF-DETR Medium segmentation |
| Epochs | 50 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Resolution | 432 |
| Learning rate | 1e-4 |
| Output | `runs/frosch_seg_medium` |

## 7. Loss Curves and Aggregate Metrics

The final retained project artifacts do not contain epoch-by-epoch train/validation loss history or a loss-curve file.

The following aggregate metrics were also not retained in verified artifacts:

- mAP@0.50 aggregate
- mAP@0.50:0.95
- aggregate IoU
- held-out test metrics

These values are therefore not fabricated.

## 8. Interpretation

Bottle, capacity and label show strong evaluation performance. Bump, damage and especially scratch are weaker. The live pipeline therefore combines detector thresholds with segmentation-based validation, tracking and finalization logic rather than using raw detector output as the final inspection result.
