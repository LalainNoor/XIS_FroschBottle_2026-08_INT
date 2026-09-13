**FROSCH BOTTLE INSPECTION PIPELINE**

**A06 — Dataset Version**

*Dataset lineage, split definition, leakage controls, and test-set
status*

| **Document Control** | **Value**                                                                                |
|----------------------|------------------------------------------------------------------------------------------|
| Document ID          | A06-FROSCH-DSV-v1                                                                        |
| Artifact             | A06 — Dataset Version                                                                    |
| Project              | Frosch Bottle Inspection Pipeline                                                        |
| Status               | RECONSTRUCTED DATASET BASELINE — A06 release gate not satisfied                          |
| Parent artifacts     | A03-FROSCH-DI-v2; A05-FROSCH-AQA-v1                                                      |
| Source dataset       | Frosch bottle 5 v6 / Roboflow COCO Segmentation export                                   |
| Dataset export date  | 23 June 2026                                                                             |
| Prepared             | 07 September 2026                                                                        |
| Evidence rule        | No dataset version, split key, leak check, or seal is inferred where evidence is absent. |

| **A06 records the state of the supplied Frosch dataset and determines whether it can qualify as a controlled handbook dataset version. The supplied export has train/validation data but no test split, no retained split key, no documented leakage check, and no sealed held-out test-set record. Therefore this artifact does not claim a completed A06 release.** |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

**1. Purpose and Scope**

A06 establishes the controlled dataset-version record required before
training and evaluation can rely on a traceable, leak-checked split. The
handbook requires the raw-data reference, schema version, split key and
proportions, leak check, seal date, and parent artifacts to be recorded.

For Frosch, the available evidence is sufficient to reconstruct the
source dataset identity and its current exported split, but it does not
establish the controls needed to issue a final A06 dataset version.

**2. Source Dataset Identity**

| **Field**                    | **Recorded evidence**             | **Status** |
|------------------------------|-----------------------------------|------------|
| Dataset                      | Frosch bottle 5                   | Confirmed  |
| Dataset version              | v6                                | Confirmed  |
| Format                       | COCO Segmentation                 | Confirmed  |
| Source                       | Roboflow export                   | Confirmed  |
| Export date                  | 23 June 2026                      | Confirmed  |
| Image size                   | 432 × 432                         | Confirmed  |
| License                      | CC BY 4.0                         | Confirmed  |
| Source images                | Provided by project Team Lead     | Confirmed  |
| Original collection protocol | Not retained in supplied evidence | Gap        |
| Physical bottle dimensions   | Not known/recorded                | Gap        |

**3. Recorded Source Split**

| **Split**  | **Images** | **Annotations** | **Share of images** | **Handbook A06 status** |
|------------|------------|-----------------|---------------------|-------------------------|
| Train      | 739        | 2,370           | 70.65%              | Recorded                |
| Validation | 307        | 996             | 29.35%              | Recorded                |
| Test       | 0          | 0               | 0.00%               | No test split present   |
| Total      | 1,046      | 3,366           | 100%                | Recorded total          |

The supplied dataset therefore uses an approximately 70.65% / 29.35% /
0% image split rather than the handbook's default 80/15/5 proportions.
The dataset card explicitly records that no separate test split is
present.

**4. Handbook A06 Control Assessment**

| **A06 requirement**   | **Required control**                                         | **Frosch evidence**                                               | **Result** |
|-----------------------|--------------------------------------------------------------|-------------------------------------------------------------------|------------|
| Raw data reference    | Exact source/reference recorded                              | Dataset identity and Roboflow export provenance are recorded.     | PARTIAL    |
| Schema version        | Link to A04 schema version                                   | A04-FROSCH-AS-v2 exists as a reconstructed schema baseline.       | PARTIAL    |
| Split key             | Part/batch/shift/camera/etc.; never frame                    | No split key is retained in the supplied Frosch evidence.         | MISSING    |
| Split proportions     | Actual proportions recorded with reason for deviation        | 70.65 / 29.35 / 0 recorded; reason not established.               | PARTIAL    |
| Leak check            | Verify no related part/group crosses splits                  | No Frosch leak-check evidence is retained.                        | MISSING    |
| Sealed test set       | Independent held-out test set sealed before final evaluation | 0 test images / 0 annotations; no seal record.                    | MISSING    |
| Manifest / identifier | Reproducible version reference                               | No immutable A06 dataset manifest/hash is retained.               | MISSING    |
| Parent chain          | A03, A05 named                                               | A03 and A05 are named as parents; A05 is not a completed QA gate. | PARTIAL    |

**5. Split-Key and Leakage Status**

| **NO RETAINED SPLIT KEY: The project evidence does not establish whether the original 739/307 split was made by part, batch, bottle, capture session, camera, frame, or another grouping. The handbook specifically prohibits splitting by frame because correlated frames of the same part can leak across train and test.** |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Because the original grouping rule is not documented, a historical
leakage-free claim cannot be reconstructed from the dataset counts
alone. The absence of a test split also prevents the supplied export
from serving as a sealed independent evaluation dataset under A06.

| **Control**                  | **Current Frosch status**   | **Required evidence for closure**                                   |
|------------------------------|-----------------------------|---------------------------------------------------------------------|
| Split key                    | NOT ESTABLISHED             | Named grouping key and assignment rule.                             |
| Group assignment             | NOT ESTABLISHED             | Manifest showing each group belongs to exactly one split.           |
| Part-level leakage           | NOT CHECKED / NOT EVIDENCED | Automated comparison of part identifiers across splits.             |
| Frame-level leakage          | NOT CHECKED / NOT EVIDENCED | Verification that related frames are not distributed across splits. |
| Validation/test independence | NOT SATISFIED               | Dedicated test split created and sealed before evaluation.          |

**6. Dataset Integrity and Provenance Controls**

The available dataset evidence establishes useful provenance but not a
complete immutable dataset version. The Roboflow export identity,
image/annotation totals, class distribution, and preprocessing context
are documented in A03. The source dataset itself has not been modified
by this A06 reconstruction because no separate Frosch dataset-version
construction process is retained.

| **Control / evidence**       | **Current status**                                                                |
|------------------------------|-----------------------------------------------------------------------------------|
| Source dataset identity      | Documented: Frosch bottle 5 v6.                                                   |
| Export provenance            | Documented: Roboflow COCO Segmentation export, 23 June 2026.                      |
| Schema linkage               | Reconstructed A04-FROSCH-AS-v2 exists; historical schema version not established. |
| Dataset manifest / hash      | Not retained.                                                                     |
| Immutable archive identifier | Not retained.                                                                     |
| Processing script / manifest | No specific processing script is identified in supplied evidence.                 |
| Raw-data location            | Original source location not retained.                                            |
| A05 annotation QA gate       | Not complete; formal 10% review evidence not retained.                            |

**7. Test Set and Evaluation Readiness**

| **A06 TEST-SET STATUS: NOT AVAILABLE. The source export contains 0 test images and 0 test annotations. It therefore cannot provide the handbook-required sealed test set for independent final reporting.** |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

The validation split may be used for development decisions only to the
extent supported by the project workflow, but it must not be relabeled
as a sealed test set. A new held-out test set must be created and sealed
before a final A06 dataset version can support handbook-compliant final
evaluation.

**8. Relationship to A05 Annotation QA**

A05 currently records that the handbook-required annotation-QA evidence
is missing. In particular, the retained evidence does not establish the
10% review sample, inter-annotator disagreement, millimetre edge
agreement, adjudication, or trend analysis. A06 therefore carries A05's
unresolved status forward rather than treating the annotation gate as
passed.

| **Dependency** | **A06 implication**                                                                                |
|----------------|----------------------------------------------------------------------------------------------------|
| A04 schema     | Provides the reconstructed schema baseline but not the original annotation manual/history.         |
| A05 QA         | Annotation-QA gate remains unresolved; no completed historical pass is claimed.                    |
| A03 inventory  | Provides the source dataset identity, counts, class distribution, preprocessing, and lineage gaps. |

**9. Required Re-entry Actions**

| **Priority** | **Action**                                                                                            | **Closure evidence**                                |
|--------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| HIGH         | Define and record a defensible split key based on project grouping information; never split by frame. | Documented split key and assignment manifest.       |
| HIGH         | Create a dedicated held-out test set and seal it before further model/threshold decisions.            | Sealed test-set record and immutable manifest/hash. |
| HIGH         | Run leakage checks across the chosen grouping key and related frames/parts.                           | Automated leak-check report with PASS result.       |
| HIGH         | Link the dataset version to the final accepted A04/A05 state.                                         | Traceable A04 → A05 → A06 chain.                    |
| HIGH         | Record the actual final proportions and explain any deviation from 80/15/5.                           | A06 split rationale and counts.                     |
| MEDIUM       | Preserve an immutable dataset manifest or hash and exact export reference.                            | Versioned manifest/archive identifier.              |

**10. A06 Closure Status**

| **A06 STATUS: PARTIALLY COMPLETE / NOT ISSUED AS A CONTROLLED DATASET VERSION. The source dataset identity and existing train/validation counts are known, but the split key, leakage verification, independent test set, test-set seal, and immutable dataset-version reference are not established in the retained Frosch evidence.** |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

**11. Traceability**

| **Related artifact** | **Relationship**                                                                                                    |
|----------------------|---------------------------------------------------------------------------------------------------------------------|
| A01-FROSCH-QS-v2     | Defines the project questions and evidence gaps that affect dataset suitability.                                    |
| A02-FROSCH-MO-v1     | Defines the measurable objective and distinguishes image-space criteria from physical accuracy.                     |
| A03-FROSCH-DI-v2     | Source inventory: dataset identity, counts, split proportions, class distribution, preprocessing, and lineage gaps. |
| A04-FROSCH-AS-v2     | Reconstructed annotation-schema baseline.                                                                           |
| A05-FROSCH-AQA-v1    | Annotation-QA record documenting the missing historical QA gate.                                                    |
| Future A07           | Training runs should reference a completed/issued A06 version rather than the uncontrolled export.                  |

**12. Evidence References**

- Frosch bottle dataset 5 v6 — Roboflow COCO Segmentation Dataset Card.

- A03-FROSCH-DI-v2 — Dataset Inventory.

- A04-FROSCH-AS-v2 — Annotation Schema.

- A05-FROSCH-AQA-v1 — Annotation QA.

- XIS AI Departmental Handbook v1.0 — A06 dataset versioning, splitting,
  leakage, and test-set requirements.

**13. Conclusion**

The supplied Frosch dataset can be described as a Roboflow COCO
Segmentation export of Frosch bottle 5 v6 with 1,046 images and 3,366
annotations, split into 739 training images and 307 validation images,
with no test split. However, the retained project evidence does not
establish the split key, leakage-free assignment, a sealed independent
test set, or an immutable dataset-version manifest.

Accordingly, this A06 artifact is a reconstruction of the current
dataset state and an explicit record of the controls that remain
outstanding. It does not claim that a handbook-compliant A06 dataset
version was historically issued.
