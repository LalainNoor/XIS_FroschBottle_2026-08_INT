# Results

## Model

RF-DETR Medium

---

## OCR

EasyOCR

---

## Live Pipeline

The live inference pipeline was tested using the Allied Vision Camera Simulator.

The system successfully:

- detected bottles
- detected capacity regions
- detected damage
- detected bumps
- performed OCR on capacity labels
- tracked bottles across frames
- displayed OCR results only once per bottle

---

## Observations

- OCR accuracy depends on the visibility of the capacity label.
- The digit allowlist significantly reduced incorrect OCR predictions.
- Tracking prevented repeated OCR for the same bottle.
- Camera simulator successfully emulated live streaming for testing.
