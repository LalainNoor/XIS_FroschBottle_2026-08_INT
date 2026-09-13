**FROSCH BOTTLE INSPECTION PIPELINE**

**A05 — Annotation QA**

*Annotation quality evidence, review coverage, disagreement, and
resolution status*

| **Document Control** | **Value**                                                                      |
|----------------------|--------------------------------------------------------------------------------|
| Document ID          | A05-FROSCH-AQA-v1                                                              |
| Artifact             | A05 — Annotation QA                                                            |
| Project              | Frosch Bottle Inspection Pipeline                                              |
| Status               | RETROSPECTIVE EVIDENCE REVIEW — formal annotation QA not evidenced             |
| Parent artifact      | A04-FROSCH-AS-v2                                                               |
| Dataset              | Frosch bottle dataset 5 v6 / Roboflow COCO Segmentation export                 |
| Export date          | 23 June 2026                                                                   |
| Prepared             | 07 September 2026                                                              |
| Evidence rule        | Missing evidence means the check did not happen; no historical QA is inferred. |

| **IMPORTANT: This artifact does not claim that the handbook-required 10% annotation review was performed. The supplied project evidence contains the exported annotation inventory, but no retained historical QA sample, inter-annotator review record, adjudication log, or quantitative edge-disagreement study.** |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

**1. Purpose and Scope**

A05 records the available evidence concerning annotation quality for the
Frosch Bottle Inspection Pipeline. It is intended to establish whether
the annotation-QA gate required by the XIS AI Departmental Handbook can
be considered satisfied, and to identify the exact evidence needed for
re-entry where it cannot.

The review is limited to the supplied project records. The dataset is a
Roboflow COCO Segmentation export, with 739 training images / 2,370
annotations and 307 validation images / 996 annotations; no test split
is present. The dataset card also records the class distribution and
confirms that no additional annotation-tool history is retained.

**2. Applicable Handbook Requirements**

| **Requirement** | **Handbook expectation**                                                            | **Frosch evidence status**                                                                             |
|-----------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Review coverage | Review 10% of labels/annotations per batch.                                         | NOT EVIDENCED                                                                                          |
| Schema          | Review against the approved A04 schema and its edge definitions.                    | PARTIALLY AVAILABLE — A04 baseline exists, but original annotation manual/edge rules are not retained. |
| Edge agreement  | Quantify disagreement between annotators in millimetres against EDGE_AGREEMENT_MAX. | NOT POSSIBLE FROM RETAINED EVIDENCE — no calibrated physical scale / physical ground truth.            |
| Trend           | Record disagreement trend and identify worsening classes/batches.                   | NOT EVIDENCED                                                                                          |
| Resolution      | Record disagreements, adjudications, and schema decisions.                          | NOT EVIDENCED                                                                                          |
| Release gate    | A05 must establish annotation consistency before A06 dataset versioning.            | NOT SATISFIED by retained historical evidence.                                                         |

**3. Dataset Annotation Baseline**

| **Class**                    | **Train** | **Validation** | **Total** | **Share** |
|------------------------------|-----------|----------------|-----------|-----------|
| Frosch-bottle-UTNY-aUbJ-XBXs | 0         | 0              | 0         | 0.00%     |
| bottle                       | 731       | 304            | 1,035     | 30.75%    |
| bump                         | 145       | 64             | 209       | 6.21%     |
| capacity                     | 682       | 273            | 955       | 28.37%    |
| damage                       | 48        | 24             | 72        | 2.14%     |
| label                        | 727       | 303            | 1,030     | 30.60%    |
| scratch                      | 37        | 28             | 65        | 1.93%     |

The first category is present in the COCO metadata but has zero
annotations. The class distribution is descriptive dataset evidence; it
is not evidence of annotation correctness or inter-annotator agreement.

**4. Formal QA Coverage Assessment**

| **QA item**              | **Required / expected**              | **Recorded evidence**                                                 | **Result**              |
|--------------------------|--------------------------------------|-----------------------------------------------------------------------|-------------------------|
| Historical review sample | At least 10% of labels/annotations   | No review sample or reviewer record retained.                         | FAIL / MISSING EVIDENCE |
| Reviewer identity        | Named reviewer(s) and batch/date     | Not documented.                                                       | MISSING                 |
| Schema version           | Version reviewed against             | A04 reconstructed baseline v2 exists; original schema history absent. | PARTIAL                 |
| Edge definition          | Authoritative edge convention        | Not retained.                                                         | MISSING                 |
| Inter-annotator review   | Independent second annotation/review | Not documented.                                                       | MISSING                 |
| Edge disagreement (mm)   | Mean absolute edge difference in mm  | No calibrated mm reference and no physical GT.                        | NOT COMPUTABLE          |
| EDGE_AGREEMENT_MAX       | Project-specific limit               | Not established in Frosch objective/evidence.                         | NOT ESTABLISHED         |
| Adjudication             | Resolution of disagreements          | No adjudication record retained.                                      | MISSING                 |
| Trend                    | Per-batch/class disagreement trend   | No historical QA series.                                              | MISSING                 |
| Schema change action     | Changes linked to QA findings        | No historical QA-driven schema change log.                            | MISSING                 |

| **Handbook rule applied: missing evidence means the check did not happen. Therefore the absence of a retained 10% review record is recorded as a missing QA gate, not as an assumed pass.** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

**5. Physical Edge-Agreement Limitation**

The handbook defines annotation disagreement as a mean absolute edge
difference in millimetres. For this Frosch project, camera calibration
was not performed, pixel-to-mm conversion was not implemented, physical
bottle dimensions were not recorded, and no physical ground-truth
measurement set was supplied. Consequently, a defensible millimetre
edge-disagreement value cannot be reconstructed from the retained
project evidence.

| **Measurement evidence needed**     | **Frosch status**            | **Effect on A05**                                            |
|-------------------------------------|------------------------------|--------------------------------------------------------------|
| Calibrated pixel-to-mm relationship | Not performed / not retained | Cannot convert annotation edge differences to mm.            |
| Physical bottle dimensions          | Not provided                 | No physical geometry reference.                              |
| Ground-truth measurement set        | Not provided                 | No physical measurement anchor.                              |
| EDGE_AGREEMENT_MAX                  | Not established              | No project acceptance threshold for annotation disagreement. |

**6. Structural / Descriptive Checks Available**

The supplied dataset evidence supports structural and descriptive
observations, but these must not be presented as the handbook's formal
semantic annotation QA. The available evidence confirms the COCO
Segmentation export identity, annotation totals, class inventory, and
preprocessing/export context.

| **Available check**     | **Evidence**                                                                  | **Interpretation**                                             |
|-------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------|
| Dataset identity        | Frosch bottle 5 v6; COCO Segmentation; Roboflow export; 23 June 2026.         | Source/export identity is documented.                          |
| Annotation totals       | 3,366 annotations across train/validation; 0 test annotations.                | Dataset inventory is traceable.                                |
| Class inventory         | Six active detection classes plus one zero-annotation COCO metadata category. | Class composition is documented.                               |
| Preprocessing           | Auto-orientation/EXIF handling; resize to 432×432 using Stretch.              | Export preprocessing is documented.                            |
| Annotation-tool history | No additional history beyond export retained.                                 | Original annotation decisions cannot be reconstructed exactly. |

**7. Weak-Class Observation — Not a Causal QA Finding**

A04 records materially weaker detection performance for bump, damage,
and especially scratch than for bottle, label, and capacity. This is
relevant to annotation QA because inconsistent class definitions can
contribute to apparent model weakness. However, the retained project
evidence contains no annotation-QA study, so these model-performance
differences must not be attributed to annotation quality.

| **Class**                 | **Recorded evaluation context**              | **A05 interpretation**                                      |
|---------------------------|----------------------------------------------|-------------------------------------------------------------|
| bottle / label / capacity | Strong detection performance recorded.       | No annotation-quality conclusion.                           |
| bump                      | Lower performance than core classes.         | Candidate for later QA attention; not proven label-driven.  |
| damage                    | Lower precision with relatively high recall. | Potential ambiguity; not proven label-driven.               |
| scratch                   | Weakest recorded detection performance.      | High-priority QA review candidate; not proven label-driven. |

**8. Required Re-entry Actions**

| **Priority** | **Action**                                                                                                                      | **Closure evidence**                                     |
|--------------|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| HIGH         | Recover or create an approved annotation specification for each active class, including exact edge/occlusion/ambiguity rules.   | Versioned, approved A04 schema/manual.                   |
| HIGH         | Perform the handbook-required annotation review sample (10% minimum) against the approved schema.                               | Reviewer, batch, sample size, date, reviewed IDs.        |
| HIGH         | Quantify edge disagreement in mm where physical calibration is available and establish the project-specific EDGE_AGREEMENT_MAX. | Per-edge / aggregate disagreement report with threshold. |
| HIGH         | Record disagreements and adjudications; document any schema changes caused by findings.                                         | Adjudication log and schema change record.               |
| MEDIUM       | Create class/batch trend reporting and identify recurring disagreement patterns.                                                | Trend table/plot with reviewer decisions.                |
| MEDIUM       | Link the final approved schema version to A06 and subsequent A07 records.                                                       | Traceability chain A04 → A05 → A06 → A07.                |

**9. A05 Closure Status**

| **A05 STATUS: NOT COMPLETE AS A HISTORICAL QA GATE. The retained evidence establishes the dataset annotation inventory and documents the absence of formal QA evidence, but it does not establish the required 10% review, millimetre edge agreement, inter-annotator consistency, trend, or adjudication results.** |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

This status is intentional. It preserves the distinction between what is
known from the dataset export and what would need to be newly performed
to satisfy the handbook. A retrospective record may document the gap,
but it must not rewrite the project history as though the missing review
occurred.

**10. Traceability**

| **Related artifact** | **Relationship**                                                                                           |
|----------------------|------------------------------------------------------------------------------------------------------------|
| A01-FROSCH-QS-v2     | Defines project measurement questions and records missing ground-truth/calibration evidence.               |
| A02-FROSCH-MO-v1     | Defines the current image-space objective and confirms that physical ±mm accuracy is not established.      |
| A03-FROSCH-DI-v2     | Provides dataset identity, split counts, class distribution, preprocessing, and lineage gaps.              |
| A04-FROSCH-AS-v2     | Provides the reconstructed annotation-schema baseline and explicitly carries missing QA evidence into A05. |
| Future A06           | Must reference the final accepted annotation-QA status before issuing a formal dataset version.            |

**11. Evidence References**

Primary source records used for this artifact:

- Frosch bottle dataset 5 v6 — Roboflow COCO Segmentation Dataset Card
  (export 23 June 2026).

- A04-FROSCH-AS-v2 — Annotation Schema.

- XIS AI Departmental Handbook v1.0 — A05 Annotation QA requirements and
  A05 artifact register.

**12. Conclusion**

The Frosch annotation dataset is sufficiently characterized to state its
exported annotation composition and the classes used by the inspection
workflow, but the retained evidence does not contain the formal
annotation-QA record required by the handbook. In particular, no 10%
review sample, inter-annotator disagreement record, millimetre edge
agreement result, project-specific edge-agreement threshold, trend
analysis, or adjudication history is available.

Accordingly, A05 is recorded as a missing historical QA gate with
explicit re-entry actions. No annotation-quality pass, quantitative
edge-agreement value, or historical reviewer decision has been invented.
