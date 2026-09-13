**FROSCH BOTTLE INSPECTION PIPELINE** \| Evidence-based reconstruction
\| September 2026

| **Artifact ID**  | **A09-FROSCH-MV-v1**        | **Preparation basis** | **Existing project records only**          |
|------------------|-----------------------------|-----------------------|--------------------------------------------|
| Parent artifacts | A02, A06, A07, A08          | Re-execution          | None                                       |
| Evidence policy  | No unsupported values added | Review state          | Pending project lead/reviewer confirmation |
| Scope            | Physical metric validation  | Project status        | Not completed in recorded implementation   |

**STATUS: NOT COMPLETED — PHYSICAL MEASUREMENT VALIDATION WAS NOT
PERFORMED**

**Executive statement.** The retained Frosch project evidence does not
contain a calibrated physical measurement study. The runtime performed
image-space inspection using normalized H/V geometry; no physical
ruler/calliper ground-truth set, calibrated reference object,
pixel-to-mm factor, MAE/MPE study, or formal
repeatability/reproducibility study was recorded. The 52-bottle runtime
validation is explicitly a status-classification validation and must not
be repurposed as metrology evidence.

# 1. Handbook Validation Requirement Mapping

| **Requirement**                 | **Required evidence**                               | **Frosch retained evidence**                                 | **Status**      |
|---------------------------------|-----------------------------------------------------|--------------------------------------------------------------|-----------------|
| Ground-truth instrument         | Traceable physical measurement instrument           | No physical ruler/calliper measurement set retained          | NOT EVIDENCED   |
| Measurement operator            | Identified operator / reproducibility record        | Operator not recorded in supplied measurement evidence       | NOT EVIDENCED   |
| Calibrated camera               | Intrinsic calibration parameters and undistortion   | No intrinsic calibration; no undistortion stage              | NOT COMPLETED   |
| Reference object                | Known physical reference for pixels-per-mm          | No calibrated physical reference object used                 | NOT COMPLETED   |
| Physical measurement set        | At least 30 consecutive measurements for validation | No physical measurement sequence recorded                    | NOT COMPLETED   |
| Same feature / axis             | Controlled feature and measurement axis definition  | No physical mm feature/axis was validated                    | NOT COMPLETED   |
| Tolerance                       | Approved ±mm accuracy requirement and yield target  | No ±mm requirement established                               | NOT ESTABLISHED |
| Accuracy statistics             | MAE, MPE, bias/spread                               | Not computable without ground truth                          | N/A             |
| Repeatability / reproducibility | Repeated measurements under defined conditions      | No formal repeatability study documented                     | NOT EVIDENCED   |
| Correction offsets              | Documented correction method, if any                | No metric correction stage exists in recorded implementation | NOT APPLICABLE  |

# 2. Measurement Method Actually Implemented

**Measurement domain:** normalized image-space geometry, not physical
millimetres.

| **Measurement**                 | **Recorded implementation**                                           | **Interpretation**              |
|---------------------------------|-----------------------------------------------------------------------|---------------------------------|
| Orientation                     | Calculated from bottle-mask pixels; maximum accepted angle 45°        | Image-space geometric criterion |
| Horizontal centricity           | H = (label_center_x - bottle_center_x) / bottle_width; threshold 0.15 | Normalized ratio; not mm        |
| Vertical centricity             | V = (label_center_y - bottle_center_y) / bottle_height                | Normalized ratio; not mm        |
| Capacity-specific V expectation | 100 ml: 0.12; 300 ml: 0.07; 500 ml: 0.01; allowed deviation 0.05      | Normalized image-space target   |
| Physical width / height         | Not implemented                                                       | width_mm and height_mm = N/A    |

**Source basis.** The Frosch Measurement Report states that
pixel-to-millimetre measurement was not implemented and that the runtime
performs image-space inspection. It further records that no calibrated
reference object or physical ruler/calliper dataset was used, and
consequently width_mm, height_mm, MAE, and MPE are N/A.

# 3. Calibration and Physical Ground-Truth Boundary

| **Control**                  | **Recorded state**       | **Validation consequence**                                 |
|------------------------------|--------------------------|------------------------------------------------------------|
| Intrinsic camera calibration | Not performed            | No traceable lens/camera geometry for physical measurement |
| Undistortion                 | No cv2.undistort() stage | Measurement remains based on raw/image-space geometry      |
| Pixels-per-mm                | Not derived              | Pixel dimensions cannot be defensibly converted to mm      |
| Physical bottle dimensions   | Not provided / retained  | No known geometric reference available                     |
| Reference object             | Not used                 | No scale anchor for metric conversion                      |
| Ground-truth instrument      | Not provided / recorded  | No traceable comparator for error statistics               |

# 4. Evidence That Must Not Be Misclassified as Metrology Validation

**52-bottle runtime validation.** The later consolidated record contains
5,322 frames across 100 ml, 300 ml and 500 ml runs, producing 40 GOOD,
10 DEFECTIVE and 2 INCOMPLETE outcomes. These are runtime inspection
counts. The project documentation explicitly says this validation is not
a millimetre-measurement accuracy study.

| **Capacity** | **Frames** | **GOOD** | **DEFECTIVE** | **INCOMPLETE** |
|--------------|------------|----------|---------------|----------------|
| 100 ml       | 1,292      | 9        | 4             | 0              |
| 300 ml       | 1,270      | 6        | 4             | 0              |
| 500 ml       | 2,760      | 25       | 2             | 2              |
| Overall      | 5,322      | 40       | 10            | 2              |

**Important boundary.** These status outcomes cannot be converted into
MAE, MPE, bias, repeatability, percentage within an agreed ±mm
tolerance, or physical acceptance accuracy because no physical
ground-truth comparator and no approved physical tolerance are retained.

# 5. Measurement Validation Results

| **Metric / check**                    | **Recorded value** | **Status / reason**                                     |
|---------------------------------------|--------------------|---------------------------------------------------------|
| Width (mm)                            | N/A                | Pixel-to-mm conversion not implemented                  |
| Height (mm)                           | N/A                | Pixel-to-mm conversion not implemented                  |
| MAE                                   | N/A                | No physical ground-truth measurement set                |
| MPE                                   | N/A                | No physical ground-truth measurement set                |
| Bias                                  | N/A                | No physical comparator values                           |
| Spread / standard deviation           | N/A                | No repeated physical measurements                       |
| % within ±mm tolerance                | N/A                | No physical tolerance established                       |
| Repeatability                         | N/A                | No formal repeatability study documented                |
| Reproducibility                       | N/A                | No multi-operator / repeated-condition study documented |
| 30-consecutive-measurement validation | Not performed      | Required evidence absent from project record            |

# 6. Required Re-entry to Close A09

The following are future validation requirements, not historical project
evidence:

| **Step** | **Required action before A09 closure**                                                              |
|----------|-----------------------------------------------------------------------------------------------------|
| 1        | Define the exact physical feature and measurement axis.                                             |
| 2        | Obtain traceable ground-truth measurements, including instrument and operator identity.             |
| 3        | Provide known physical reference geometry / bottle dimensions where applicable.                     |
| 4        | Complete intrinsic camera calibration and document calibration parameters and reprojection quality. |
| 5        | Undistort all images used for measurement.                                                          |
| 6        | Derive and document pixels-per-mm using a calibrated physical reference.                            |
| 7        | Collect the required consecutive physical measurement set and corresponding system outputs.         |
| 8        | Calculate MAE, MPE, bias/spread and percentage within the approved ±mm tolerance.                   |
| 9        | Conduct formal repeatability / reproducibility checks under defined conditions.                     |
| 10       | Obtain Team Lead/stakeholder approval of the physical accuracy criterion and closure decision.      |

# 7. Traceability to Parent Artifacts

| **Parent**                      | **Relationship to A09**                                                                                                                 | **Status**                   |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| A02 — Measurable Objective      | Defines that physical ±mm accuracy and tolerance-yield evidence were not established; keeps physical measurement as an open requirement | Referenced                   |
| A03 — Dataset Inventory         | Provides dataset/provenance context and records missing physical dimensions / collection metadata                                       | Referenced                   |
| A04 / A05 — Annotation evidence | Provides annotation semantics and QA limitations relevant to any future physical validation                                             | Referenced                   |
| A06 — Dataset Version           | Provides dataset-version and test-set traceability context                                                                              | Referenced                   |
| A07 — Training Run              | Provides model/checkpoint/training provenance and calibration boundary                                                                  | Referenced                   |
| A08 — Evaluation                | Confirms no physical accuracy, repeatability, bias/spread or mm pass/fail claims are supported                                          | Direct dependency            |
| MEASUREMENT_REPORT.md           | Primary source for the absence of pixel-to-mm implementation and physical measurement study                                             | Primary source               |
| Team Lead confirmation          | Confirms calibration was not performed and ground-truth measurements were not provided                                                  | Primary project confirmation |

# 8. Conclusion

**A09 is NOT COMPLETED for physical metrology.** The recorded Frosch
implementation supports normalized image-space inspection criteria, but
there is no documented physical measurement validation against traceable
ground truth. The retained evidence therefore does not support claims of
millimetre accuracy, MAE/MPE, bias/spread, repeatability,
reproducibility, or percentage within an approved physical tolerance.
This artifact preserves that limitation rather than backfilling missing
measurements.

SOURCE BASIS

Frosch_Bottle_Inspection_Consolidated_Technical_Report.docx;
MEASUREMENT_REPORT.md; A02_FROSCH_Measurable_Objective_v1;
A08_FROSCH_Evaluation_v1; A01_FROSCH_Question_Set_v2; existing Frosch
project records. No project re-execution performed.
