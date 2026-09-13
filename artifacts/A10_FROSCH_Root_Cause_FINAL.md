**FROSCH BOTTLE INSPECTION PIPELINE \| Evidence-based reconstruction \|
September 2026**

| STATUS: NOT TRIGGERED FOR A MEASUREMENT-DISAGREEMENT INVESTIGATION — NO CALIBRATED PHYSICAL DISAGREEMENT DATASET WAS RECORDED. |
|--------------------------------------------------------------------------------------------------------------------------------|

| **Artifact ID**      | A10-FROSCH-RC-v1                               | **Preparation basis** | Existing project records only              |
|----------------------|------------------------------------------------|-----------------------|--------------------------------------------|
| **Parent artifacts** | A08, A09                                       | **Re-execution**      | None                                       |
| **Scope**            | Measurement disagreement / root-cause analysis | **Review state**      | Pending project lead/reviewer confirmation |
| **Evidence policy**  | No unsupported causal findings                 | **Decision basis**    | Retained Frosch records                    |

# 1. Executive Summary

The retained Frosch evidence does not contain a calibrated physical
measurement study, a physical ground-truth comparison, or a recorded
disagreement dataset against which a formal measurement root-cause
analysis could be performed. A10 is therefore recorded as NOT TRIGGERED
for measurement disagreement. The appropriate evidence-based conclusion
is a process and evidence boundary, not a fabricated technical root
cause.

| Key conclusion: the project record shows why a physical measurement disagreement could not be investigated, but it does not show that a specific camera, calibration, segmentation, or measurement algorithm defect caused a disagreement. |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 2. A10 Trigger and Applicability

| **A10 check**                    | **Required evidence**                                  | **Frosch retained evidence**                          | **Status**                    |
|----------------------------------|--------------------------------------------------------|-------------------------------------------------------|-------------------------------|
| Measurement disagreement present | Paired system-vs-ground-truth disagreement set         | No physical ruler/calliper measurement set retained   | NOT EVIDENCED                 |
| Traceable ground truth           | Reference instrument + physical measurements           | No physical ground-truth measurement study recorded   | NOT EVIDENCED                 |
| Controlled test conditions       | Same feature/axis, operator, conditions, repeatability | No formal physical repeatability study recorded       | NOT EVIDENCED                 |
| Causal investigation             | Candidate causes ruled out with evidence               | No controlled cause-isolation investigation recorded  | NOT PERFORMED                 |
| Root-cause closure               | Cause, corrective action, verification evidence        | No measurement disagreement root-cause closure exists | NOT COMPLETED / NOT TRIGGERED |

# 3. Evidence Boundary: What the Project Actually Implemented

The Frosch runtime performed image-space inspection rather than
calibrated real-world dimension measurement. Horizontal and vertical
centricity were normalized to bottle dimensions, and orientation was
evaluated from bottle-mask geometry. These criteria are not millimetre
measurements.

| **Recorded implementation** | **Evidence**                                                                                                    | **Root-cause implication**                                  |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Horizontal centricity H     | H = (label_center_x - bottle_center_x) / bottle_width; threshold 0.15                                           | Image-space criterion; not evidence of mm disagreement      |
| Vertical centricity V       | Normalized label/bottle center displacement; capacity-specific expected values and allowed deviation documented | Image-space criterion; no physical comparator retained      |
| Physical dimensions         | width_mm / height_mm not implemented; MAE/MPE N/A                                                               | No measurement error can be decomposed into physical causes |
| Calibration                 | No intrinsic calibration / undistortion stage recorded                                                          | Lens/geometry error cannot be isolated from evidence        |
| Reference object            | No calibrated physical reference object used                                                                    | No pixels-per-mm factor was derived or validated            |

# 4. Candidate Cause Register

Because there is no measured disagreement dataset, the following items
are recorded as known evidence gaps or possible prerequisites that were
absent. They are not asserted as proven technical causes of an error.

| **Candidate / prerequisite**         | **Evidence status**                                                 | **What can be concluded**                                            | **Closure evidence required**                                         |
|--------------------------------------|---------------------------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------|
| Camera calibration / lens distortion | Absent in retained record                                           | Cannot be evaluated or ruled out for physical measurement            | Calibration parameters, reprojection error, undistortion verification |
| Pixels-per-mm scaling                | Not implemented                                                     | Physical dimension disagreement cannot be quantified                 | Reference object and documented scale derivation                      |
| Physical ground truth                | Not collected / not retained                                        | No paired truth-vs-system comparison exists                          | Traceable ruler/caliper dataset                                       |
| Physical object dimensions           | Not retained                                                        | Cannot compute absolute dimensional error                            | Authoritative dimensions for each validated instance                  |
| Measurement repeatability            | No formal study recorded                                            | Cannot assess variation under repeated measurements                  | Defined repeated-measurement protocol and results                     |
| Segmentation/model error             | Model-level error patterns exist, but no physical disagreement link | Cannot prove these errors caused a physical measurement disagreement | Paired error examples with causal isolation and verification          |

# 5. Causes Ruled Out

| No candidate technical cause was formally ruled out, because no controlled measurement-disagreement investigation was performed in the retained project evidence. |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|

This distinction is important for auditability. The absence of
calibration, reference-object measurements, and a physical ground-truth
set explains why A09 could not be completed, but it does not establish
that any one of those missing elements caused a numerical measurement
failure. A10 therefore avoids assigning causality where the evidence
does not support it.

# 6. Observed Model Error Patterns — No Causal Attribution

The retained detector evaluation does show model-level confusion
patterns. These are included only as observed error patterns and are not
converted into root causes for physical measurement disagreement.

| **Ground-truth class** | **Selected observed confusion / miss pattern**                   | **Evidence interpretation**                         | **Root-cause status** |
|------------------------|------------------------------------------------------------------|-----------------------------------------------------|-----------------------|
| bump                   | 29 missed; 10 predicted as damage; additional smaller confusions | Bump detection is weaker than bottle/capacity/label | Cause not established |
| damage                 | 2 missed; 2 predicted as capacity                                | Some model-level confusion exists                   | Cause not established |
| scratch                | 22 missed; 6 predicted as damage                                 | Scratch is the weakest retained detector class      | Cause not established |
| label                  | 10 predicted as bump; 15 as capacity; 4 as damage                | Cross-class confusion is present                    | Cause not established |

These values come from the model-level confusion matrix and must not be
interpreted as a bottle-level GOOD/DEFECTIVE/INCOMPLETE confusion
matrix.

# 7. Evidence-Based Root-Cause Statement

Primary process/evidence cause: the retained project record does not
contain the prerequisite calibration, physical reference, traceable
ground-truth measurement set, physical dimensions, or repeatability
study required to establish and investigate physical measurement
disagreement.

Technical root cause: NOT DETERMINED. The available evidence is
insufficient to distinguish among camera distortion, scale derivation,
segmentation geometry, model prediction, setup variation, or other
technical causes.

Corrective action for the record: preserve A09 as not completed, do not
claim millimetre accuracy, and re-enter the validation workflow before
making a physical measurement claim.

# 8. Required Re-entry to Complete A10/A09

| **Priority** | **Action**                                                                                                                            | **Closure evidence**                                                                         |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| HIGH         | Perform intrinsic camera calibration and document undistortion.                                                                       | Calibration parameters, reprojection error, and verified undistorted output.                 |
| HIGH         | Use a calibrated physical reference object and derive pixels-per-mm.                                                                  | Reference-object record, dimensions, scale derivation, and validation.                       |
| HIGH         | Collect a traceable physical ground-truth set, including at least 30 consecutive measurements as required by the validation workflow. | Instrument/operator/condition record and measurement table.                                  |
| HIGH         | Define the physical feature/axis and approved ±mm requirement.                                                                        | Controlled measurement definition and acceptance threshold.                                  |
| HIGH         | Run repeatability/reproducibility analysis and compute bias/spread plus MAE/MPE.                                                      | Signed or versioned result record with calculations.                                         |
| HIGH         | Investigate any disagreement using paired examples and controlled cause isolation.                                                    | Candidate-cause matrix, ruled-out evidence, selected cause, corrective action, verification. |
| MEDIUM       | Link the completed measurement investigation to A08 evaluation and subsequent release/re-entry records.                               | Traceable A08/A09/A10 chain.                                                                 |

# 9. Conclusion

A10 is recorded as NOT TRIGGERED for a formal measurement-disagreement
root-cause investigation. The retained Frosch evidence supports a clear
documentation/process boundary: the implementation used normalized
image-space inspection and did not retain the physical measurement
infrastructure needed to establish a numerical disagreement. No specific
technical root cause can therefore be claimed or ruled out. This record
deliberately preserves that distinction and identifies the evidence
required for future re-entry.

# 10. Traceability

**A08-FROSCH-EV-v1 —** Evaluation boundary and model-level performance
context.

**A09-FROSCH-MV-v1 —** Confirms physical measurement validation was not
performed.

**MEASUREMENT_REPORT.md —** Primary source for calibration,
reference-object, mm, MAE/MPE gaps.
