# FROSCH BOTTLE INSPECTION PIPELINE
## A02 — Measurable Objective
*Evidence-based objective definition, acceptance boundaries, and measurement gaps*

| Document Control | Value |
|---|---|
| Document ID | A02-FROSCH-MO-v1 |
| Artifact | A02 — Measurable Objective |
| Project | Frosch Bottle Inspection Pipeline |
| Status | Evidence-based project objective; unestablished requirements explicitly identified |
| Parent artifact | A01-FROSCH-QS-v2 |
| Scope | 100 ml, 300 ml, and 500 ml bottle inspection |
| Version | v1.0 |

> **Objective rule: only values supported by the project evidence are treated as established. Physical accuracy requirements are not invented where no Team Lead/client requirement or ground truth exists.**

---

## 1. Objective Statement

The Frosch Bottle Inspection Pipeline is intended to inspect 100 ml, 300 ml, and 500 ml bottles using camera-based computer vision and to produce inspection decisions based on bottle/label positioning and detected bottle-related conditions. The implemented objective is primarily image-space inspection rather than calibrated physical metrology.

Within the available evidence, the measurable objective can therefore be stated in terms of the implemented image-space criteria and observed inspection outcomes. A production-grade physical measurement objective in millimetres cannot be claimed because physical ground truth, bottle dimensions, camera calibration, and an approved ±mm tolerance were not provided or established.

---

## 2. Objective Components

| Objective component | Established project definition | Status |
|---|---|---|
| Subject | 100 ml, 300 ml, and 500 ml Frosch bottles. | CONFIRMED |
| Observable | Bottle/label position, orientation, and detected inspection conditions. | IMPLEMENTED |
| Measurement domain | Image-space / normalized positional measurements and detection outputs. | ESTABLISHED |
| Physical unit | No validated physical millimetre output. | NOT AVAILABLE |
| Accuracy requirement | No approved ±mm requirement was provided. | NOT ESTABLISHED |
| Acceptance yield | No approved percentage-within-tolerance requirement was provided. | NOT ESTABLISHED |
| Throughput target | Formal production line rate / bottle throughput target was not recorded. | NOT ESTABLISHED |
| Validation basis | Manual/status validation exists; it is not a physical metrology study. | AVAILABLE, LIMITED |

---

## 3. Current Implemented Measurement Criteria

The project documentation records the following image-space criteria:

| Criterion | Definition | Established boundary |
|---|---|---|
| Maximum orientation | Bottle orientation must be within 45°. | ≤ 45° |
| Horizontal position H | (label_center_x − bottle_center_x) / bottle_width | Threshold 0.15 |
| Vertical position V | (label_center_y − bottle_center_y) / bottle_height | Size-specific values below |

| Bottle size | Expected V | Allowed deviation | Interpretation |
|---|---|---|---|
| 100 ml | 0.12 | 0.05 | Normalized image-space criterion |
| 300 ml | 0.07 | 0.05 | Normalized image-space criterion |
| 500 ml | 0.01 | 0.05 | Normalized image-space criterion |

> **These H/V thresholds are normalized image-space criteria. They are not equivalent to ±mm physical accuracy.**

---

## 4. Detection / Inspection Acceptance Context

The implemented pipeline also uses model detections for bottle, label, capacity, bump, damage, and scratch-related inspection. The documented inference thresholds are implementation parameters, not customer-level accuracy acceptance criteria.

| Detection class | Recorded threshold |
|---|---|
| Bottle | 0.70 |
| Label | 0.35 |
| Capacity | 0.35 |
| Bump | 0.50 |
| Damage | 0.30 |
| Scratch | 0.30 |

Model-level precision, recall, F1, and AP values are evaluation metrics and should not be substituted for the physical measurement acceptance criterion required for a millimetre-based metrology objective.

---

## 5. Validation Evidence Available

| Variant | Frames recorded | GOOD | DEFECTIVE | INCOMPLETE | Reviewed total |
|---|---|---|---|---|---|
| 100 ml | 1,292 | 9 | 4 | 0 | 13 |
| 300 ml | 1,270 | 6 | 4 | 0 | 10 |
| 500 ml | 2,760 | 25 | 2 | 2 | 29 |
| Overall | 5,322 | 40 | 10 | 2 | 52 |

The 52 reviewed bottles provide evidence about status classification under the recorded inspection workflow. They do not establish physical dimensional accuracy, MAE, MPE, bias, spread, or percentage within a ±mm tolerance.

---

## 6. Latency and Throughput Objective

| Metric | Project evidence | A02 interpretation |
|---|---|---|
| Observed idle FPS | Approximately 83–96 FPS | Runtime observation; not a production acceptance target. |
| Observed active FPS | Approximately 22–35 FPS | Runtime observation; not equivalent to confirmed bottle/min throughput. |
| Formal production line rate | Not recorded | Cannot assess line-rate compliance. |
| Formal throughput target | Not established | No pass/fail throughput boundary exists in project evidence. |
| Latency acceptance boundary | No formal p95/p99/worst-latency target established in A02 evidence. | Not established. |

> **Do not convert observed FPS into a claimed production bottle throughput without a documented line rate, trigger/spacing assumptions, and end-to-end throughput measurement.**

---

## 7. Physical Accuracy Objective — Status

| Handbook-relevant requirement | Current project status | Why no value is reported |
|---|---|---|
| ±mm accuracy requirement | NOT ESTABLISHED | No approved physical tolerance was supplied. |
| Ground-truth reference | NOT PROVIDED | No physical ground-truth measurements were provided. |
| Physical bottle dimensions | NOT PROVIDED | Dimensions were not supplied with the project inputs. |
| Camera calibration | NOT PERFORMED | Team Lead confirmed calibration was not performed. |
| Pixel-to-mm conversion | NOT IMPLEMENTED | No validated scale/reference exists. |
| % within tolerance | NOT ESTABLISHED / NOT CALCULABLE | Requires an approved tolerance and ground-truth values. |
| Repeatability | NOT PERFORMED / NOT DOCUMENTED | No formal repeated physical measurement series exists. |

---

## 8. Operating Conditions and Boundaries

| Condition / boundary | Recorded status |
|---|---|
| Supported bottle variants | 100 ml, 300 ml, 500 ml |
| Camera input configuration | 2048 × 2448 recorded in project runtime configuration |
| Trigger configuration | 40% recorded |
| Image-space tolerance | 20 px recorded in runtime configuration |
| Centricity tolerance | 0.08 recorded |
| Required measurement unavailable | Runtime can classify the bottle as INCOMPLETE |
| Confirmed defect / failed measurement | Can contribute to DEFECTIVE status |
| Physical disposition of DEFECTIVE bottle | Not recorded |
| Production line operating rate | Not recorded |

---

## 9. Pass/Fail Boundary Matrix

| Objective area | Boundary | Pass/Fail assessment possible now? |
|---|---|---|
| Orientation | ≤45° | Yes, for implemented image-space logic |
| Horizontal position | H threshold 0.15 | Yes, for implemented image-space logic |
| Vertical position | Size-specific expected V ±0.05 | Yes, for implemented image-space logic |
| Physical dimensional accuracy | ±mm | No — requirement and GT absent |
| Percentage within physical tolerance | Approved percentage target | No — target and GT absent |
| Production throughput | Formal line-rate target | No — target absent |
| End-to-end physical rejection | Documented disposition criterion | No — physical process not recorded |

---

## 10. Objective Limitations

- Physical millimetre accuracy cannot be claimed from the available project evidence.
- Image-space thresholds should not be presented as physical tolerances.
- The available 52-bottle validation is status-oriented and is not a traceable metrology validation.
- Observed FPS is not a substitute for a formal production line-rate or throughput acceptance target.
- No physical bottle dimension values are introduced because they were not provided.
- No calibration result, pixel-to-mm factor, MAE, MPE, bias, or repeatability statistic is introduced because the underlying studies were not performed.

---

## 11. Required Re-entry to Establish a Production-Grade Physical Objective

| Step | Required evidence | Owner / approval needed |
|---|---|---|
| 1 | Define the physical feature and measurement axis. | Team Lead / project stakeholder |
| 2 | Provide traceable ground-truth measurements and instrument/operator information. | Measurement owner / project team |
| 3 | Provide actual physical dimensions or reference geometry where applicable. | Team Lead / project stakeholder |
| 4 | Perform camera calibration and document calibration parameters. | Engineering |
| 5 | Validate pixel-to-mm conversion against physical reference measurements. | Engineering + validation owner |
| 6 | Approve ±mm tolerance and percentage-within-tolerance target. | Team Lead / client |
| 7 | Conduct formal repeatability/reproducibility validation. | Engineering + validation owner |
| 8 | Establish formal line rate, throughput target, and physical DEFECTIVE disposition. | Operations / Team Lead |

---

## 12. Traceability

| Parent / source | Relationship to A02 |
|---|---|
| A01-FROSCH-QS-v2 | Defines the evidence baseline, measurement gaps, and project scope inherited by A02. |
| Frosch consolidated technical documentation | Source for implemented H/V criteria, runtime configuration, validation evidence, and calibration/physical-measurement limitations. |
| MEASUREMENT_REPORT.md | Source for the documented absence of pixel-to-mm implementation and physical measurement validation. |
| Team Lead confirmation | Source for confirmation that calibration was not performed and ground-truth measurements were not provided. |

---

## 13. A02 Conclusion

> **The measurable objective that can be legitimately claimed is an image-space bottle inspection objective for 100 ml, 300 ml, and 500 ml bottles, using the documented orientation and normalized H/V criteria plus model-based inspection outputs.**

A physical ±mm accuracy objective, percentage-within-tolerance requirement, and formal production throughput objective were not established in the project evidence. Accordingly, A02 records these as open requirements rather than assigning unsupported numerical values. This preserves traceability between what the project implemented and what would be required for future physical measurement validation.

---
*Frosch Bottle Inspection Pipeline | A02-FROSCH-MO-v1 | Evidence-based documentation*
