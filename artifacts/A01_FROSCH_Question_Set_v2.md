# FROSCH BOTTLE INSPECTION PIPELINE
## A01 — Question Set & Measurement Definition
*Project evidence baseline and unresolved operational / measurement requirements*

| Document Control | Value |
|---|---|
| Document ID | A01-FROSCH-QS-v2 |
| Artifact | A01 — Question Set & Measurement Definition |
| Project | Frosch Bottle Inspection Pipeline |
| Status | Evidence-based baseline / gaps explicitly recorded |
| Prepared for | AI Departmental Handbook v1.0 artifact set |
| Scope | 100 ml, 300 ml, and 500 ml bottle inspection |
| Primary evidence | Project technical documentation, runtime/configuration records, dataset documentation, and Team Lead confirmations |
| Version | v2.0 |

> **Evidence rule: this document records what was actually provided, implemented, measured, or confirmed. Missing evidence is not replaced with assumptions.**

---

## 1. Purpose and Scope

This artifact establishes the measurement and operational question set for the Frosch Bottle Inspection Pipeline. It defines the inspection scope, currently implemented image-space criteria, available project inputs, measurement evidence, and unresolved requirements needed for physical measurement validation and production acceptance.

The purpose of A01 is not to invent acceptance criteria or retrospectively claim experiments that were not performed. Where the project evidence does not contain a required value or study, the item is explicitly marked as Not Provided, Not Recorded, Not Established, or Not Performed.

### 1.1 Supported Inspection Scope

| Item | Current project scope / evidence |
|---|---|
| Bottle variants | 100 ml, 300 ml, 500 ml |
| Inspection context | Camera-based bottle inspection pipeline |
| Primary outputs | Inspection status and bottle/label/capacity/defect-related detections and measurements |
| Measurement domain | Image-space inspection and normalized positional criteria |
| Physical metrology | Not established in the supplied project evidence |

---

## 2. Executive Evidence Summary

| Question / Requirement | Status | Evidence-based conclusion |
|---|---|---|
| What bottles are in scope? | CONFIRMED | 100 ml, 300 ml, and 500 ml were provided for the project. |
| What physical bottle dimensions are available? | NOT PROVIDED | Physical dimensions were not provided with the project inputs. |
| What is the physical ground truth? | NOT PROVIDED | No ground-truth measurement set was supplied. |
| Was camera calibration performed? | NOT PERFORMED | Team Lead confirmed calibration was not performed; project documentation also records this gap. |
| Was pixel-to-mm conversion implemented? | NOT IMPLEMENTED | Runtime inspection remains image-space; no validated pixels-per-mm conversion exists. |
| What is the ±mm accuracy requirement? | NOT ESTABLISHED | No agreed physical accuracy tolerance was supplied. |
| What percentage must be within tolerance? | NOT ESTABLISHED | Cannot be established without an approved tolerance and physical ground truth. |
| Was repeatability formally studied? | NOT PERFORMED / NOT DOCUMENTED | No formal physical repeatability study or 30-measurement ground-truth series exists. |
| What is the formal production line rate? | NOT RECORDED | No formal line-rate value was supplied; observed runtime FPS must not be treated as line throughput. |
| How are defective bottles physically handled? | NOT RECORDED | Software DEFECTIVE status exists, but physical handling/disposition was not documented. |

---

## 3. Measurement Definition — What the System Currently Measures

The implemented Frosch pipeline uses image-space inspection logic. The available documentation records normalized orientation/centricity criteria rather than a calibrated physical millimetre measurement.

### 3.1 Orientation and Position Criteria

| Criterion | Definition / implementation | Current status |
|---|---|---|
| Orientation | Bottle orientation must be within a maximum of 45°. | Implemented / documented |
| Horizontal position H | H = (label_center_x − bottle_center_x) / bottle_width; threshold = 0.15. | Implemented / documented |
| Vertical position V | V = (label_center_y − bottle_center_y) / bottle_height. | Implemented / documented |

### 3.2 Size-Specific V Criteria

| Bottle size | Expected V | Allowed deviation | Measurement domain |
|---|---|---|---|
| 100 ml | 0.12 | 0.05 | Normalized image-space |
| 300 ml | 0.07 | 0.05 | Normalized image-space |
| 500 ml | 0.01 | 0.05 | Normalized image-space |

> **Important:** these values are normalized image-space criteria. They are not millimetre tolerances and must not be reported as physical accuracy requirements.

---

## 4. Physical Measurement and Calibration Questions

The following questions are required to establish a defensible physical measurement objective. The current project evidence does not contain the inputs needed to answer them quantitatively.

| Question | Current answer | Implication |
|---|---|---|
| What instrument provides ground truth? | Not provided / not recorded. | No traceable physical reference exists for accuracy calculations. |
| Who is the measurement operator? | Not provided / not recorded. | Operator-related reproducibility cannot be assessed. |
| What are the physical bottle dimensions? | Not provided. | Image measurements cannot be converted to physical dimensions using known bottle geometry. |
| Was camera calibration completed? | No. | Lens distortion / camera geometry were not calibrated for this project. |
| Is there a calibrated reference object? | No evidence available. | Pixels-per-mm cannot be derived from a validated physical reference. |
| Was pixel-to-mm conversion implemented? | No. | Physical width/height in mm cannot be reported from the current runtime. |
| What is the physical ±mm requirement? | Not established. | Pass/fail physical accuracy cannot be evaluated. |
| What percentage must be within tolerance? | Not established. | Acceptance rate cannot be calculated against an agreed physical requirement. |
| Was formal repeatability tested? | No formal study documented. | Repeatability / reproducibility statistics are unavailable. |

> **Team Lead confirmation: calibration was not performed and no ground-truth measurements were provided with the project inputs.**

---

## 5. Input and Acquisition Conditions

| Parameter | Recorded project information |
|---|---|
| Camera acquisition | Project runtime uses Vimba X / camera acquisition infrastructure. |
| Runtime image configuration | 2048 × 2448 recorded in the consolidated project documentation. |
| Trigger configuration | 40% in the recorded runtime configuration. |
| Image-space tolerance | 20 px in the recorded runtime configuration. |
| Recorded dataset validation | Final frame-folder validation is documented for the 100 ml, 300 ml, and 500 ml variants. |
| Production line rate | Not recorded. |
| Formal throughput acceptance target | Not established. |
| Physical bottle handling/disposition | Not recorded. |

Observed runtime FPS values are implementation observations and should not be substituted for a formal production line-rate or bottle-throughput acceptance target.

---

## 6. Dataset and Validation Evidence Available at A01

| Variant | Recorded validation frames | GOOD | DEFECTIVE | INCOMPLETE | Total reviewed |
|---|---|---|---|---|---|
| 100 ml | 1,292 | 9 | 4 | 0 | 13 |
| 300 ml | 1,270 | 6 | 4 | 0 | 10 |
| 500 ml | 2,760 | 25 | 2 | 2 | 29 |
| Overall | 5,322 | 40 | 10 | 2 | 52 |

The 52-bottle validation is a status-classification / inspection review and is not a physical millimetre accuracy study. It therefore cannot be used to derive MAE, MPE, bias, or percentage-within-±mm-tolerance.

---

## 7. Operational Failure and Disposition Questions

| Question | Current documented behavior / evidence | A01 status |
|---|---|---|
| When required measurements are unavailable? | The runtime defines an INCOMPLETE outcome when required measurements are unavailable. | Documented |
| What creates DEFECTIVE? | Confirmed damage/bump or failed measurement can result in DEFECTIVE status. | Documented |
| What happens physically to a DEFECTIVE bottle? | No physical handling/disposition procedure was supplied. | Not recorded |
| What is the formal production line rate? | Not supplied. | Not recorded |
| What happens on production acceptance failure? | No formal production disposition/escalation procedure supplied. | Not recorded |

---

## 8. Explicit Evidence Gaps

| ID | Gap | Evidence | Consequence |
|---|---|---|---|
| G01 | Physical ground truth | No ground-truth measurements supplied. | Physical accuracy cannot be validated. |
| G02 | Physical dimensions | Bottle dimensions not provided. | No validated dimensional reference. |
| G03 | Camera calibration | Calibration not performed/documented. | No calibrated image geometry. |
| G04 | Pixel-to-mm | Conversion not implemented. | No physical mm output. |
| G05 | Accuracy tolerance | ±mm requirement not established. | No physical pass/fail criterion. |
| G06 | Tolerance yield | % within tolerance not established. | Acceptance rate cannot be calculated. |
| G07 | Repeatability | Formal study not performed/documented. | No repeatability statistic. |
| G08 | Line rate | Formal production rate not recorded. | Throughput acceptance cannot be assessed. |
| G09 | Defective disposition | Physical handling not recorded. | End-to-end rejection process is not evidenced. |

---

## 9. Required Clarifications / Future Re-entry

These items are recorded as re-entry requirements rather than completed project evidence. They should only be marked complete after actual measurement, stakeholder confirmation, or controlled validation produces the required evidence.

| Priority | Action | Required evidence before closure |
|---|---|---|
| High | Obtain/define physical ground truth for the measurement feature and axis. | Traceable instrument, operator, measurement protocol, and recorded reference values. |
| High | Obtain actual physical bottle dimensions for 100/300/500 ml variants. | Measured dimensions and measurement conditions. |
| High | Perform camera calibration if physical measurement is required. | Calibration images, calibration parameters, and documented validation. |
| High | Derive and validate pixel-to-mm conversion. | Known reference, pixels-per-mm calculation, and physical validation results. |
| High | Approve ±mm measurement requirement and percentage-within-tolerance target. | Team Lead/client approval recorded in decision record. |
| Medium | Conduct formal repeatability/reproducibility study. | Repeated measurements, operator/condition information, and calculated statistics. |
| Medium | Record formal production line rate and throughput target. | Confirmed line-rate requirement and acceptance target. |
| Medium | Document physical DEFECTIVE bottle disposition. | Production handling/rejection/escalation procedure. |

---

## 10. Traceability to Source Evidence

| Source artifact | Evidence used in A01 |
|---|---|
| Frosch_Bottle_Inspection_Consolidated_Technical_Report.docx | Project scope, image-space criteria, acquisition/runtime configuration, validation counts, environment, and explicit calibration/physical-measurement gaps. |
| MEASUREMENT_REPORT.md | Measurement workflow requirements and confirmation that pixel-to-mm, physical reference, physical validation, MAE/MPE were not completed. |
| Project runtime / inference records | Runtime behavior, status logic, and observed implementation configuration. |
| Team Lead confirmation | 100/300/500 ml project inputs; calibration not performed; no ground-truth measurements provided. |

---

## 11. A01 Conclusion

> **A01 conclusion: The current project has a defined image-space inspection scope and documented runtime criteria, but does not have sufficient physical metrology evidence to claim millimetre-level measurement accuracy.**

The correct engineering and handbook-compliant position is therefore to preserve the implemented image-space criteria as the current measurement definition, while explicitly recording physical ground truth, calibration, pixel-to-mm conversion, ±mm acceptance, repeatability, production line rate, throughput acceptance, and physical defective disposition as unresolved evidence/requirement gaps.

No unsupported physical dimensions, calibration results, accuracy values, tolerance percentages, or repeatability statistics are introduced in this artifact.

---
*Frosch Bottle Inspection Pipeline | A01-FROSCH-QS-v2 | Evidence-based documentation*
