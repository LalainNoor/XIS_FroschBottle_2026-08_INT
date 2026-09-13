**FROSCH BOTTLE INSPECTION PIPELINE**

**A08 — Evaluation Record**

*Evidence-based reconstruction from existing project records • September
2026*

| **Artifact ID**      | A08-FROSCH-EVAL-v1                | **Project**                | Frosch Bottle Inspection Pipeline                                          |
|----------------------|-----------------------------------|----------------------------|----------------------------------------------------------------------------|
| **Parent artifacts** | A02, A03, A04, A05, A06, A07      | **Primary sources**        | Consolidated Technical Report; TRAINING_REPORT.md; RESULTS/runtime records |
| **Evaluation basis** | Existing documented evidence only | **Re-execution**           | Not performed                                                              |
| **Overall status**   | PARTIALLY COMPLETE                | **Release interpretation** | Evaluation record; not a calibrated metrology validation                   |

**STATUS: PARTIALLY COMPLETE** Model-level detection evidence and
runtime status observations are available; several handbook evaluation
gates cannot be completed because physical ground truth, calibrated mm
measurement, repeatability data, and a separately sealed test set were
not retained.

# 1. Purpose and Evaluation Scope

This artifact records the available Frosch evaluation evidence and maps
it to the handbook evaluation expectations. It intentionally separates
model-level detection metrics from bottle-level runtime inspection
outcomes. The consolidated project report states that the 52-bottle
validation is a runtime status-classification exercise, not a formal
model-accuracy study.

The evaluation therefore covers: recorded detector metrics, the retained
confusion matrix, supported bottle-capacity groups, runtime
GOOD/DEFECTIVE/INCOMPLETE counts, normalized geometry behavior, and
observed active/idle FPS. It does not claim calibrated physical
measurement accuracy or repeatability where the source record does not
contain those studies.

# 2. Evaluation Evidence Inventory

| **Evidence item**               | **Available record**                                               | **Use in A08**                       | **Status**    |
|---------------------------------|--------------------------------------------------------------------|--------------------------------------|---------------|
| Detection evaluation table      | Per-class precision, recall, F1, AP50; IoU matching threshold 0.50 | Model-level evaluation               | CONFIRMED     |
| Detection confusion matrix      | Retained matrix by class, including missed detections              | Error interpretation                 | CONFIRMED     |
| Segmentation aggregate metrics  | mAP/IoU/held-out test metrics not retained                         | Required by handbook but unavailable | NOT EVIDENCED |
| Runtime validation              | 31 Aug 2026 frame-folder validation; 52 finalized records          | Operational status evidence only     | CONFIRMED     |
| Physical measurement validation | No calibrated reference, physical GT, MAE/MPE, or mm study         | Cannot evaluate physical accuracy    | NOT EVIDENCED |
| Repeatability study             | No repeated-measurement study retained                             | Cannot evaluate repeatability        | NOT EVIDENCED |
| Per-size runtime groups         | 100 ml, 300 ml, 500 ml counts recorded                             | Group-level operational evidence     | CONFIRMED     |
| Latency distribution            | Only observed FPS range retained                                   | No p50/p95/p99 distribution          | PARTIAL       |

# 3. Per-Criterion Evaluation

| **Criterion**                     | **Required evidence**                                 | **Recorded Frosch evidence**                                                                              | **Result**                | **Boundary / reason**                                                        |
|-----------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|---------------------------|------------------------------------------------------------------------------|
| Model-level detection performance | Precision/recall/F1/AP and documented matching basis  | Per-class metrics and IoU=0.50 matching table are retained.                                               | PASS — EVIDENCE AVAILABLE | This is detector-object evaluation, not end-to-end bottle decision accuracy. |
| Segmentation performance          | mAP, IoU, precision, recall, F1 and held-out evidence | Training configuration and checkpoint are retained, but aggregate metrics/test evidence are not retained. | NOT EVIDENCED             | Do not infer segmentation quality from runtime behavior.                     |
| Per-group evaluation              | Performance across supported sizes/groups             | Runtime outcomes recorded separately for 100/300/500 ml.                                                  | PARTIAL                   | These are runtime status counts, not statistical model accuracy by size.     |
| Physical mm accuracy              | Calibrated measurement against physical ground truth  | Calibration and pixel-to-mm workflow were not implemented.                                                | NOT EVALUATED             | width_mm, height_mm, MAE and MPE are N/A.                                    |
| Repeatability                     | Repeated same-feature measurements / spread           | No repeatability study retained.                                                                          | NOT EVIDENCED             | No claim made.                                                               |
| Bias / spread                     | Error distribution versus reference                   | No physical reference measurements available.                                                             | NOT EVALUATED             | Cannot calculate bias/spread without ground truth.                           |
| Error vs size                     | Measurement error plotted by physical size            | No physical error data retained.                                                                          | NOT EVALUATED             | Runtime counts by capacity are not an error-vs-size plot.                    |
| Runtime performance               | Measured FPS / latency evidence                       | Idle 83–96 FPS; active full pipeline 22–35 FPS.                                                           | PARTIAL                   | Only range is retained; percentile latency statistics are unavailable.       |
| M1 / formal evaluation gate       | Handbook measurement/evaluation acceptance gate       | Formal calibrated physical-measurement acceptance cannot be assessed.                                     | NOT EVALUATED             | No supported evidence for a metrology pass/fail claim.                       |

# 4. Detection Evaluation Results

The consolidated Frosch report records the following detection
evaluation table. The evaluation uses an IoU matching threshold of 0.50.
The values are reported as per-class detector metrics and should not be
interpreted as bottle-level GOOD/DEFECTIVE/INCOMPLETE accuracy.

| **Class** | **Precision** | **Recall** | **F1** | **AP50** | **n_gt** |
|-----------|---------------|------------|--------|----------|----------|
| bottle    | 0.9869        | 0.9934     | 0.9902 | 0.9091   | 304      |
| bump      | 0.4487        | 0.5469     | 0.4930 | 0.4130   | 64       |
| capacity  | 0.9189        | 0.9963     | 0.9561 | 0.9072   | 273      |
| damage    | 0.3607        | 0.9167     | 0.5176 | 0.8455   | 24       |
| label     | 0.9869        | 0.9934     | 0.9901 | 0.9088   | 303      |
| scratch   | 0.3333        | 0.2143     | 0.2609 | 0.2061   | 28       |

Interpretation: bottle, capacity and label show the strongest recorded
detector performance. Bump, damage and especially scratch are materially
weaker. The project documentation explicitly identifies these defect
classes as risk areas and therefore combines detector thresholds with
segmentation-based validation, tracking and finalization rather than
treating raw detector outputs as the final inspection decision.

# 5. Detection Confusion Matrix

| **GT \\ Pred** | **bottle** | **bump** | **capacity** | **damage** | **label** | **scratch** | **missed** |
|----------------|------------|----------|--------------|------------|-----------|-------------|------------|
| bottle         | 302        | 0        | 0            | 0          | 0         | 0           | 2          |
| bump           | 3          | 68       | 4            | 10         | 3         | 2           | 29         |
| capacity       | 0          | 0        | 273          | 0          | 0         | 1           | 1          |
| damage         | 0          | 0        | 2            | 41         | 0         | 0           | 2          |
| label          | 0          | 10       | 15           | 4          | 301       | 1           | 2          |
| scratch        | 1          | 0        | 0            | 6          | 1         | 14          | 22         |

The consolidated report explicitly cautions that this matrix is a
model-level class matching matrix and must not be interpreted as a
confusion matrix for the final bottle-level GOOD/DEFECTIVE/INCOMPLETE
decision.

# 6. Runtime Group Results

Final recorded frame-folder validation was documented on 31 August 2026
for all three supported capacities. These are runtime inspection counts,
not formal model-accuracy metrics.

| **Capacity** | **Frames** | **GOOD** | **DEFECTIVE** | **INCOMPLETE** | **Finalized** | **Interpretation**                                         |
|--------------|------------|----------|---------------|----------------|---------------|------------------------------------------------------------|
| 500 ml       | 2,760      | 25       | 2             | 2              | 29            | Two INCOMPLETE cases were not treated as physical defects. |
| 300 ml       | 1,270      | 6        | 4             | 0              | 10            | Runtime status results.                                    |
| 100 ml       | 1,292      | 9        | 4             | 0              | 13            | Runtime status results.                                    |
| Overall      | 5,322      | 40       | 10            | 2              | 52            | Operational validation total.                              |

# 7. Geometry and Measurement Evaluation Boundary

Orientation is calculated from bottle-mask pixels with a maximum
accepted angle of 45 degrees. Horizontal centricity is H =
(label_center_x - bottle_center_x) / bottle_width with a documented
threshold of 0.15. Vertical centricity is V = (label_center_y -
bottle_center_y) / bottle_height, with expected values of 0.12, 0.07 and
0.01 for 100 ml, 300 ml and 500 ml respectively and an allowed deviation
of 0.05.

These are normalized image-space quantities. They are not millimetres.
The project record states that camera calibration, undistortion,
calibrated reference-object derivation, pixels-per-mm conversion,
physical width/height validation, MAE and MPE were not completed.
Accordingly, no ±mm accuracy pass/fail result is claimed in A08.

# 8. Runtime Performance Evidence

| **Condition**                          | **Observed performance** | **A08 interpretation**                                                                                     |
|----------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------|
| Idle — no bottle in frame              | 83–96 FPS                | Recorded range only; no latency percentile distribution.                                                   |
| Active — bottle present, full pipeline | 22–35 FPS                | Includes dual TensorRT inference, mask geometry, centricity, defect validation, tracking and image saving. |
| Earlier synchronous-OCR versions       | Approximately 4–10 FPS   | Historical comparison from optimization record; not used as final performance.                             |

# 9. Known Evaluation Gaps

- No separate sealed test-set evaluation is retained for the supplied
  Frosch dataset; the project export contains train and validation
  splits only.

- No calibrated physical ground truth is available. Physical bottle
  dimensions were not retained, and the runtime measurement remains
  image-space.

- No repeatability study, bias/spread calculation, or error-vs-size plot
  is supported by the retained evidence.

- Segmentation aggregate metrics and held-out test metrics were not
  retained in the available project documentation.

- Runtime performance is retained as FPS ranges rather than p50/p95/p99
  or worst-case latency distributions.

- The final runtime count is from the later 31 August 2026 consolidated
  validation record; an earlier runtime record exists elsewhere in the
  project history and is not silently merged into this final count.

# 10. Evaluation Conclusion

A08 is PARTIALLY COMPLETE. The available record supports a documented
detector evaluation, per-class performance interpretation,
confusion-matrix review, capacity-group runtime status results, and
observed runtime FPS. The evidence does not support formal claims of
calibrated physical measurement accuracy, repeatability, bias/spread, or
a complete held-out test evaluation.

The appropriate downstream interpretation is therefore: the Frosch
system has documented model-level and runtime evaluation evidence within
its implemented image-space inspection scope, while handbook-grade
metrology and complete evaluation gates remain unevidenced or out of
scope for the recorded implementation.

# 11. Traceability to Parent Artifacts

| **Parent**                 | **Relationship to A08**                                                    | **Status**        |
|----------------------------|----------------------------------------------------------------------------|-------------------|
| A02 — Measurable Objective | Defines measurement/accuracy expectations that A08 evaluates against.      | Referenced        |
| A03 — Dataset Inventory    | Provides dataset identity, counts, provenance and split state.             | Referenced        |
| A04 — Annotation Schema    | Defines class/geometry semantics used by evaluation.                       | Referenced        |
| A05 — Annotation QA        | Provides annotation QA status and limitations.                             | Referenced        |
| A06 — Dataset Version      | Provides dataset version and split/test-set traceability.                  | Referenced        |
| A07 — Training Run         | Provides model/checkpoint/training provenance and known traceability gaps. | Direct dependency |

# 12. Source Basis

- Frosch_Bottle_Inspection_Consolidated_Technical_Report.docx — Sections
  5–8 and 11; Appendix A.

- FROSCH_TRAINING_REPORT.md — model roles, checkpoint paths, training
  configuration, segmentation metric limitations.

- Existing Frosch runtime records / RESULTS documentation — recorded
  bottle outcomes and runtime behavior.

- A07_FROSCH_Training_Run_v1 — reconstructed training traceability and
  retained-artifact status.

- A06_FROSCH_Dataset_Version_v1 — dataset-version and held-out-test
  traceability context.

# Document Control

| **Prepared for**      | XIS AI Departmental Handbook compliance    | **Version**          | v1                          |
|-----------------------|--------------------------------------------|----------------------|-----------------------------|
| **Preparation basis** | Existing project documents only            | **Reporting period** | September 2026              |
| **Re-execution**      | None                                       | **Evidence policy**  | No unsupported values added |
| **Approval**          | Pending project lead/reviewer confirmation | **Status**           | PARTIALLY COMPLETE          |
