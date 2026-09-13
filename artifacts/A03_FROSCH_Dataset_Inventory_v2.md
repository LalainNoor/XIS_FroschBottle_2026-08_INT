# FROSCH BOTTLE INSPECTION PIPELINE
## A03 — Dataset Inventory
*Dataset lineage, composition, acquisition context, coverage, and evidence gaps*

| Document Control | Value |
|---|---|
| Document ID | A03-FROSCH-DI-v2 |
| Artifact | A03 — Dataset Inventory |
| Project | Frosch Bottle Inspection Pipeline |
| Status | Evidence-based dataset inventory — revised |
| Parent artifacts | A01-FROSCH-QS-v2; A02-FROSCH-MO-v1 |
| Primary dataset | Frosch bottle dataset v6 / Roboflow COCO Segmentation export |
| Version | v2.0 |

> **Evidence rule: dataset facts are recorded only where supported by the supplied project documentation. Missing acquisition metadata and physical dimensions are explicitly identified as gaps.**

---

## 1. Purpose and Scope

This artifact inventories the datasets used by the Frosch Bottle Inspection Pipeline and records their lineage, format, composition, split information, acquisition context, known variants, and limitations. The inventory is intended to support reproducibility, training/evaluation traceability, and the downstream handbook artifacts.

---

## 2. Dataset Identity and Lineage

| Field | Recorded project information |
|---|---|
| Dataset | Frosch bottle dataset 5 v6 |
| Annotation/export format | Roboflow COCO Segmentation |
| Export date | 23 June 2026 |
| Image resolution in exported dataset | 432 × 432 |
| Dataset license | CC BY 4.0 |
| Dataset source | Images provided by the project Team Lead / project team |
| Original collection protocol | Not retained in the supplied project evidence |
| Original physical bottle dimensions | Not retained / not provided |
| Processing / export context | Roboflow export used for the project training/evaluation workflow |
| Dataset preprocessing / export | Auto-orientation / EXIF handling and resize to 432 × 432 using Stretch. |
| Custom augmentation policy | No custom augmentation policy is claimed for the final training configuration. |
| Processing script / reproducibility artifact | No specific processing script is identified or retained in the supplied dataset evidence. |

---

## 3. Dataset Composition

| Split | Images | Annotations | Status |
|---|---|---|---|
| Train | 739 | 2,370 | Available |
| Validation | 307 | 996 | Available |
| Test | 0 | 0 | No test split recorded in this exported dataset |
| Total | 1,046 | 3,366 | Recorded total |

**Recorded Split Proportions**

| Split | Images | Share of images |
|---|---|---|
| Train | 739 | 70.65% |
| Validation | 307 | 29.35% |
| Test | 0 | 0.00% |

**Recorded Annotation Class Distribution**

| Class | Train | Validation | Total | Share of annotations |
|---|---|---|---|---|
| Frosch-bottle-UTNY-aUbJ-XBXs | 0 | 0 | 0 | 0.00% |
| bottle | 731 | 304 | 1,035 | 30.75% |
| bump | 145 | 64 | 209 | 6.21% |
| capacity | 682 | 273 | 955 | 28.37% |
| damage | 48 | 24 | 72 | 2.14% |
| label | 727 | 303 | 1,030 | 30.60% |
| scratch | 37 | 28 | 65 | 1.93% |

The absence of a recorded test split is an important limitation for independent held-out evaluation. It must not be silently treated as equivalent to a sealed test set.

---

## 4. Bottle Variants and Coverage

| Variant / factor | Coverage in project | Inventory status |
|---|---|---|
| 100 ml bottle | Included in project scope and validation | CONFIRMED |
| 300 ml bottle | Included in project scope and validation | CONFIRMED |
| 500 ml bottle | Included in project scope and validation | CONFIRMED |
| Bottle colour / finish variants | No complete structured inventory supplied | NOT ESTABLISHED |
| Physical dimensional variants | Physical dimensions not provided | NOT AVAILABLE |
| Camera variants | No structured multi-camera inventory supplied | NOT ESTABLISHED |
| Lighting variants | No structured lighting-condition inventory supplied | NOT ESTABLISHED |
| Shift / operator variants | No structured shift/operator dataset metadata supplied | NOT ESTABLISHED |

---

## 5. Acquisition and Runtime Context

| Parameter | Recorded information | Dataset implication |
|---|---|---|
| Camera acquisition framework | Vimba X / project camera acquisition | Camera source exists, but full original acquisition protocol is not retained. |
| Recorded runtime image configuration | 2048 × 2448 | Runtime acquisition differs from exported 432 × 432 training resolution. |
| Trigger configuration | 40% | Recorded runtime configuration; not a dataset acceptance criterion. |
| Project bottle scope | 100 / 300 / 500 ml | Core variant coverage is defined. |
| Production line rate | Not recorded | No production-rate stratification is available. |
| Lighting protocol | Not formally documented in supplied evidence | Robustness across lighting conditions cannot be quantified. |
| Shift/operator metadata | Not formally documented | Shift/operator effects cannot be assessed. |

---

## 6. Dataset Label / Annotation Context

The project training workflow used object detection and segmentation data exported in COCO format. The recorded model classes include bottle, bump, capacity, damage, label, and scratch for the detection workflow. The segmentation workflow used bottle-related mask annotations.

| Aspect | Recorded status |
|---|---|
| Detection classes | Bottle, bump, capacity, damage, label, scratch |
| Segmentation | COCO segmentation masks used for the segmentation workflow |
| Annotation source | Roboflow COCO Segmentation export |
| Formal annotation protocol retained? | No complete original collection/annotation protocol is recorded in the supplied project evidence. |
| Formal annotation QA statistics | Not established in the available dataset inventory. |

---

## 7. Data Quality and Coverage Considerations

| Area | Evidence-based finding | Impact |
|---|---|---|
| Class balance | Class counts are not presented here beyond the recorded dataset totals; no complete balance analysis was retained in A03 evidence. | Potential imbalance should be assessed before release. |
| Bottle-size coverage | 100/300/500 ml are represented in project scope and validation. | Core size coverage is present. |
| Physical geometry | No physical dimensions provided. | Physical measurement validation cannot be linked to dataset geometry. |
| Acquisition metadata | Original collection protocol not retained. | Reproducibility of capture conditions is limited. |
| Independent test set | Export records 0 test images / 0 annotations. | Independent held-out model assessment is limited. |
| Lighting/camera/shift diversity | No structured inventory supplied. | Generalization across operational conditions cannot be quantified. |

---

## 8. Validation Dataset Used by the Runtime Review

| Bottle variant | Recorded frame count | GOOD | DEFECTIVE | INCOMPLETE | Reviewed total |
|---|---|---|---|---|---|
| 100 ml | 1,292 | 9 | 4 | 0 | 13 |
| 300 ml | 1,270 | 6 | 4 | 0 | 10 |
| 500 ml | 2,760 | 25 | 2 | 2 | 29 |
| Overall | 5,322 | 40 | 10 | 2 | 52 |

These runtime validation-frame counts are separate from the 1,046-image / 3,366-annotation training dataset inventory. They represent the recorded validation/review material used for the inspection workflow and should not be merged into the training dataset counts.

---

## 9. Data Privacy and Retention

| Question | Status / evidence |
|---|---|
| Personal data in bottle images | No personal-data content is identified in the supplied project documentation. |
| Privacy review | No formal privacy assessment is recorded in the supplied dataset documentation. |
| Retention period | Not specified in the supplied project evidence. |
| Raw source storage location | Original source location not retained in the supplied documentation. |
| Processed dataset storage | Project repository / training workflow contains derived dataset references and artifacts; exact original storage lineage is not fully documented. |

---

## 10. Dataset Reproducibility and Lineage Gaps

| Gap ID | Missing / incomplete evidence | Consequence |
|---|---|---|
| D01 | Original collection protocol not retained. | Exact acquisition process cannot be reproduced from the supplied artifacts alone. |
| D02 | Physical bottle dimensions not provided. | No physical geometry reference for metrology validation. |
| D03 | Ground-truth measurement set not provided. | No physical accuracy calculation can be performed. |
| D04 | Camera calibration not performed. | No validated physical image scale. |
| D05 | Export has no recorded test split. | Independent held-out evaluation is unavailable from this dataset export. |
| D06 | Structured camera/lighting/shift metadata not supplied. | Condition-specific performance cannot be quantified. |
| D07 | Retention/raw-data location not documented. | Long-term dataset governance is incomplete. |
| D08 | Specific processing script is not identified or retained in the supplied dataset evidence. | Exact preprocessing/export steps cannot be reproduced from a script record alone. |

---

## 11. Dataset Inventory Closure Status

> **A03 status: PARTIALLY COMPLETE — the dataset identity, export format, split counts and proportions, annotation-class distribution, resolution, license, preprocessing/export details, core bottle variants, and major lineage limitations are documented. Original collection metadata, physical dimensions, ground truth, structured acquisition-condition metadata, and a dedicated processing script remain unavailable.**

---

## 12. Recommended Re-entry Items

| Priority | Action | Closure evidence |
|---|---|---|
| High | Record dataset version/lineage and preserve the exact training export. | Immutable dataset identifier/archive and manifest. |
| High | Create or identify an independent held-out test set if model release requires it. | Sealed test-set record and split/leakage check. |
| High | Document the original image acquisition protocol where recoverable. | Camera, optics, lighting, placement, trigger, and capture settings. |
| High | Record physical bottle dimensions and ground-truth measurements if physical metrology is required. | Traceable measurement records. |
| Medium | Inventory lighting, camera, operator, and shift conditions. | Structured dataset metadata. |
| Medium | Document retention and raw-data storage. | Dataset governance record. |
| Medium | Preserve the dataset export/preprocessing manifest or processing script used to generate the training dataset. | Versioned preprocessing/export script or manifest with dataset hash/identifier. |

---

## 13. Traceability

| Related artifact | Traceability relationship |
|---|---|
| A01-FROSCH-QS-v2 | Defines the measurement questions and evidence gaps inherited by A03. |
| A02-FROSCH-MO-v1 | Defines the measurable objective and distinguishes image-space criteria from unestablished physical accuracy. |
| Training / technical project documentation | Provides dataset identity, export information, split counts, runtime validation counts, and model context. |
| Future A06 Dataset Version | A03 provides the inventory baseline from which the formal dataset version/lineage record should be established. |

---

## 14. A03 Conclusion

> **The project dataset is sufficiently characterized to document its principal identity, composition, annotation-class distribution, preprocessing/export context, and core bottle-size coverage, but it is not fully characterized as a production-grade dataset because original acquisition metadata, physical dimensions, ground truth, an independent test split, structured operating-condition metadata, and a specific processing script are not available in the supplied evidence.**

No unsupported class counts, acquisition conditions, physical dimensions, calibration values, privacy/retention policies, or processing scripts have been invented. These remain explicit evidence gaps for later re-entry where required.

---
*Frosch Bottle Inspection Pipeline | A03-FROSCH-DI-v2 | Evidence-based documentation*
