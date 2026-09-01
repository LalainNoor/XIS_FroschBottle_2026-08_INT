# Assumptions & Limitations

## Assumptions

- Bottle motion is sufficiently compatible with the configured tracking logic.
- Expected-capacity input corresponds to the bottles in a capacity-specific run.
- TensorRT engines are available at the configured paths.
- Deployment environment is compatible with the serialized engines.

## Limitations

### Calibration

No intrinsic camera calibration was performed and no undistortion stage is used.

### Metric Measurement

No pixel-to-mm measurement is implemented.

### Dataset

The available v6 export contains train and validation only; there is no separate test split.

### Training History

Epoch-by-epoch loss history was not retained.

### Defect Classes

Bump, damage and scratch have weaker evaluation performance than bottle, capacity and label.

### OCR

Capacity OCR can fail; expected-capacity fallback is available.

### Incomplete Cases

A bottle may be classified INCOMPLETE when required information cannot be established before finalization.

### TensorRT Portability

Serialized TensorRT engines are environment dependent.

### Data Provenance

Images were provided by the team lead. Physical bottle dimensions and the original collection protocol were not recorded.
