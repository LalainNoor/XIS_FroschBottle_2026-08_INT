**FROSCH BOTTLE INSPECTION PIPELINE**

**A04 — Annotation Schema**

Evidence-based annotation classes, representation, and schema
limitations

| **Document Control** | **Value**                                                    |
|----------------------|--------------------------------------------------------------|
| Document ID          | A04-FROSCH-AS-v2                                             |
| Artifact             | A04 — Annotation Schema                                      |
| Project              | Frosch Bottle Inspection Pipeline                            |
| Status               | Evidence-based annotation schema baseline — revised          |
| Parent artifacts     | A01-FROSCH-QS-v2; A02-FROSCH-MO-v1; A03-FROSCH-DI-v1         |
| Dataset              | Frosch bottle dataset v6 / Roboflow COCO Segmentation export |
| Version              | v2.0                                                         |

| **Evidence rule: this artifact describes the annotation classes and representations supported by the project records. Where the original annotation protocol, exact edge definitions, or ambiguity rules were not retained, they are explicitly marked as not documented.** |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 1. Purpose and Scope

A04 defines the annotation schema evidenced by the Frosch Bottle
Inspection Pipeline training data and model workflow. It records the
known detection classes, segmentation representation, dataset format,
and the limits of the retained annotation specification. The purpose is
to provide a traceable schema baseline for later annotation QA, dataset
versioning, and model evaluation.

# 2. Annotation Dataset and Representation

| **Aspect**                       | **Recorded project information**                          | **Status** |
|----------------------------------|-----------------------------------------------------------|------------|
| Dataset                          | Frosch bottle dataset 5 v6                                | Confirmed  |
| Export format                    | Roboflow COCO Segmentation                                | Confirmed  |
| Image resolution                 | 432 × 432 in exported dataset                             | Confirmed  |
| Annotation representations       | Object-detection annotations and segmentation masks       | Confirmed  |
| Detection matching               | IoU matching threshold 0.50 recorded for model evaluation | Confirmed  |
| Original annotation protocol     | Not retained in supplied project evidence                 | Gap        |
| Formal schema version/change log | Not retained in supplied project evidence                 | Gap        |

# 3. Detection Class Schema

The detection workflow records six classes. These are the classes
evidenced in the project model/evaluation documentation.

| **Class** | **Role in inspection pipeline**                                          | **Schema status** |
|-----------|--------------------------------------------------------------------------|-------------------|
| bottle    | Bottle/object detection used as a principal inspection object.           | Documented        |
| label     | Label detection used for label position/inspection logic.                | Documented        |
| capacity  | Capacity-related label/content detection for bottle-size identification. | Documented        |
| bump      | Defect class for bump-related conditions.                                | Documented        |
| damage    | Defect class for damage-related conditions.                              | Documented        |
| scratch   | Defect class for scratch-related conditions.                             | Documented        |

**Verified Annotation Inventory**

The supplied dataset card provides the recorded annotation counts by
class. These counts are included here as schema context only; they do
not establish annotation quality, boundary correctness, or an
authoritative annotation manual.

| **Class**                    | **Train** | **Validation** | **Total** | **Share of annotations** |
|------------------------------|-----------|----------------|-----------|--------------------------|
| Frosch-bottle-UTNY-aUbJ-XBXs | 0         | 0              | 0         | 0.00%                    |
| bottle                       | 731       | 304            | 1,035     | 30.75%                   |
| bump                         | 145       | 64             | 209       | 6.21%                    |
| capacity                     | 682       | 273            | 955       | 28.37%                   |
| damage                       | 48        | 24             | 72        | 2.14%                    |
| label                        | 727       | 303            | 1,030     | 30.60%                   |
| scratch                      | 37        | 28             | 65        | 1.93%                    |

The category Frosch-bottle-UTNY-aUbJ-XBXs is present in the COCO
metadata but has zero annotations; it should therefore not be
interpreted as an active annotated class in the supplied dataset.

# 4. Segmentation Schema

The project also contains a segmentation workflow using the Roboflow
COCO Segmentation export. The retained project documentation identifies
the segmentation model/checkpoint and mask output, but does not preserve
a complete annotation manual describing exact boundary/edge conventions.

| **Aspect**                             | **Recorded status**                                                                                                       |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Segmentation representation            | COCO segmentation masks                                                                                                   |
| Segmentation model workflow            | RF-DETR segmentation workflow                                                                                             |
| Training resolution                    | 432 × 432                                                                                                                 |
| Segmentation threshold used in runtime | 0.30                                                                                                                      |
| Exact mask boundary definition         | Not formally documented in retained project evidence                                                                      |
| Edge inclusion/exclusion rules         | Not formally documented                                                                                                   |
| Burr / flash / chamfer handling        | Not formally documented                                                                                                   |
| Ambiguous-boundary rules               | Not formally documented                                                                                                   |
| Annotation-tool history                | No additional annotation-tool history beyond the Roboflow COCO Segmentation export is retained in the supplied artifacts. |

| **Important: this document does not invent edge rules, burr/flash/chamfer rules, or annotation conventions that were not retained from the original annotation process.** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

# 5. Annotation Semantics Evidenced by the Pipeline

| **Annotation / output** | **How it is used in the implemented pipeline**                               | **Evidence status** |
|-------------------------|------------------------------------------------------------------------------|---------------------|
| Bottle detection        | Provides bottle localization and bottle reference geometry.                  | Implemented         |
| Label detection         | Provides label center/location used in normalized H/V position calculations. | Implemented         |
| Capacity detection      | Supports 100/300/500 ml capacity identification.                             | Implemented         |
| Bump detection          | Contributes to defect inspection.                                            | Implemented         |
| Damage detection        | Contributes to defect inspection.                                            | Implemented         |
| Scratch detection       | Contributes to defect inspection.                                            | Implemented         |
| Bottle segmentation     | Provides mask output in the segmentation workflow.                           | Implemented         |

# 6. What Is Not Defined by the Retained Schema

| **Required schema detail**     | **Status**     | **Documentation consequence**                                      |
|--------------------------------|----------------|--------------------------------------------------------------------|
| Exact object-edge definition   | Not documented | Annotator boundary interpretation cannot be reconstructed exactly. |
| Partial/occluded object policy | Not documented | No retained rule for incomplete visibility.                        |
| Burr/flash/chamfer treatment   | Not documented | No retained edge exception policy.                                 |
| Minimum object size            | Not documented | No formal inclusion threshold.                                     |
| Overlapping instances policy   | Not documented | No formal instance separation rule.                                |
| Ambiguous label/capacity cases | Not documented | No formal adjudication rule.                                       |
| Defect severity grading        | Not documented | Classes indicate defect type, not a retained severity scale.       |
| Annotation change history      | Not documented | Schema evolution cannot be fully traced from retained artifacts.   |

# 7. Class-Level Considerations

The recorded evaluation results show materially different performance
across classes. This is relevant to annotation QA because class
definitions and annotation consistency should be reviewed before
interpreting weak-class performance as purely a model limitation.

| **Class** | **Recorded evaluation context**                         | **A04 interpretation**                                                                                        |
|-----------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| bottle    | Strong detection performance recorded.                  | Core object class appears consistently represented in the evaluation evidence.                                |
| label     | Strong detection performance recorded.                  | Core positional reference class.                                                                              |
| capacity  | Strong detection performance recorded.                  | Supports size identification.                                                                                 |
| bump      | Lower detection performance than bottle/capacity/label. | Annotation consistency and class variability should be reviewed.                                              |
| damage    | Lower precision with relatively high recall.            | Potential ambiguity / class variability should be considered in later QA.                                     |
| scratch   | Weakest recorded detection performance.                 | Requires particular annotation/data-quality review before attributing performance solely to model capability. |

These observations do not establish that annotation quality caused the
performance differences; the project evidence does not contain the
annotation QA study required to make that causal conclusion.

# 8. Annotation QA Status

| **A04/A05-relevant check**             | **Current status**                                                                     |
|----------------------------------------|----------------------------------------------------------------------------------------|
| Formal annotation QA sample            | Not documented in retained project evidence.                                           |
| Minimum 10% review                     | Not evidenced; no historical review sample should be inferred from the dataset counts. |
| Quantitative edge disagreement in mm   | Not possible; physical calibration/ground truth were not established.                  |
| Inter-annotator disagreement           | Not documented.                                                                        |
| Adjudication record                    | Not documented.                                                                        |
| Class-balance trend from annotation QA | Not documented.                                                                        |
| Annotation version/change log          | Not documented.                                                                        |

| **A04 defines the schema baseline; the absence of formal annotation-QA evidence is carried forward to A05 rather than being treated as completed QA.** |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|

# 9. Annotation and Dataset Governance Gaps

| **Gap ID** | **Missing evidence**                       | **Impact**                                                          |
|------------|--------------------------------------------|---------------------------------------------------------------------|
| AS01       | Original annotation manual not retained.   | Exact annotation semantics cannot be independently reproduced.      |
| AS02       | Schema version/change log not retained.    | Changes to classes or boundaries cannot be fully traced.            |
| AS03       | Ambiguity/adjudication rules not retained. | Borderline cases cannot be evaluated against an authoritative rule. |
| AS04       | Formal annotation QA not evidenced.        | Annotation consistency cannot be quantitatively certified.          |
| AS05       | Physical edge accuracy cannot be assessed. | No calibrated mm reference or physical GT exists.                   |

# 10. Recommended Re-entry Actions

| **Priority** | **Action**                                                             | **Closure evidence**                                                                 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| High         | Recover or create an approved annotation specification for each class. | Signed/approved annotation manual with version identifier.                           |
| High         | Define exact object and mask boundary rules.                           | Explicit edge, occlusion, ambiguity, and defect rules.                               |
| High         | Perform formal annotation QA.                                          | At least the handbook-required review sample plus quantitative disagreement results. |
| High         | Record schema version and change history.                              | Versioned schema/change log linked to dataset version.                               |
| Medium       | Document class-specific ambiguity examples.                            | Representative examples with adjudicated decisions.                                  |
| Medium       | Link schema version to dataset/training run.                           | Traceable dataset → schema → training run chain.                                     |

# 11. Traceability

| **Related artifact**  | **Relationship**                                                                  |
|-----------------------|-----------------------------------------------------------------------------------|
| A01-FROSCH-QS-v2      | Defines project scope and measurement questions relevant to annotation semantics. |
| A02-FROSCH-MO-v1      | Defines current image-space objective and inspection boundaries.                  |
| A03-FROSCH-DI-v1      | Defines dataset identity, format, counts, and lineage limitations.                |
| A05 — Annotation QA   | Next artifact; will record whether annotation consistency/QA evidence exists.     |
| A06 — Dataset Version | Should reference the final schema version once established.                       |

# 12. A04 Conclusion

| **A04 status: PARTIALLY COMPLETE — the project detection classes, segmentation representation, verified class inventory, and pipeline use are documented. The original annotation manual, exact edge conventions, ambiguity rules, schema version history, and formal annotation-QA evidence were not retained.** |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

The correct evidence-based conclusion is therefore to treat the
documented class list, verified annotation counts, and representations
as the reconstructed schema baseline while carrying the undocumented
annotation semantics and QA requirements forward to A05 and future
dataset re-entry. The recorded class distribution is descriptive
evidence only and does not certify annotation quality. No new annotation
rules are asserted as historical project facts.
