**FROSCH BOTTLE INSPECTION PIPELINE**

**A07 — Training Run**

*Training configuration, model artifacts, reproducibility and
traceability record*

| **Artifact ID**          | A07-FROSCH-TR-v1                                                                                                |
|--------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Project**              | Frosch Bottle Inspection Pipeline                                                                               |
| **Artifact**             | A07 — Training Run                                                                                              |
| **Evidence basis**       | Existing project documentation and retained training/runtime artifacts                                          |
| **Preparation mode**     | Reconstructed from existing records; no new training run executed for this artifact                             |
| **Status**               | PARTIALLY COMPLETE — training configuration/artifacts documented; handbook traceability gates remain incomplete |
| **Primary model family** | RF-DETR                                                                                                         |

# 1. Purpose and Scope

A07 records the training configurations and model artifacts that are
evidenced in the existing Frosch project documentation. It covers both
RF-DETR models used by the live inspection pipeline: an object-detection
model and a separate instance-segmentation model. The artifact is
intentionally limited to documented evidence; fields that require
evidence not retained in the project record are explicitly marked as not
evidenced rather than reconstructed by assumption.

**Traceability principle:** The project record must distinguish
confirmed training facts from missing reproducibility evidence. No
unsupported run identifiers, hashes, seeds, calibration values, or
final-checkpoint claims are introduced.

# 2. Training Run Summary

| **Item**                                   | **Detection**                                                                                              | **Segmentation**                                                 | **Overall status**  |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|---------------------|
| Architecture                               | RF-DETR Medium                                                                                             | RF-DETR Medium instance segmentation                             | CONFIRMED           |
| Primary role                               | Bottle/class/defect detection and class association                                                        | Bottle mask generation for geometry and defect validation        | CONFIRMED           |
| Training configuration                     | Detailed training hyperparameters not retained in the Frosch training report                               | 50 epochs; batch 4; grad accumulation 4; resolution 432; LR 1e-4 | PARTIALLY AVAILABLE |
| Retained checkpoint                        | runs/frosch_medium/checkpoint_best_regular.pth                                                             | runs/frosch_seg_medium/checkpoint_best_total.pth                 | CONFIRMED           |
| TensorRT deployment artifact               | output/rfdetr-medium.trt                                                                                   | output/rfdetr-seg-medium.trt                                     | CONFIRMED           |
| Training metrics retained in source report | Final metric table for the current live detection checkpoint not retained in the dedicated training report | Aggregate segmentation metrics/loss curves not retained          | PARTIAL             |

# 3. Detection Model Training Record

The project uses RF-DETR Medium for object detection. The retained
checkpoint is explicitly identified in the training report, and the
consolidated project report records its TensorRT deployment artifact and
live confidence thresholds.

| **Field**                             | **Recorded evidence**                                                               | **Status**    |
|---------------------------------------|-------------------------------------------------------------------------------------|---------------|
| Model                                 | RF-DETR Medium (\`RFDETRMedium\`)                                                   | CONFIRMED     |
| Checkpoint                            | \`runs/frosch_medium/checkpoint_best_regular.pth\`                                  | CONFIRMED     |
| TensorRT engine                       | \`output/rfdetr-medium.trt\`                                                        | CONFIRMED     |
| Live classes consumed                 | bottle, capacity, label, damage, bump                                               | CONFIRMED     |
| Runtime checkpoint class count        | Runtime logs reported 7 classes; current inference code consumes five named classes | CONFIRMED     |
| Live detection threshold              | 0.40 in the dedicated training report                                               | CONFIRMED     |
| Per-class deployment thresholds       | bottle 0.70; label 0.35; capacity 0.35; bump 0.50; damage 0.30; scratch 0.30        | CONFIRMED     |
| Training epochs                       | Not retained in the dedicated Frosch training report                                | NOT EVIDENCED |
| Batch size / accumulation             | Not retained in the dedicated Frosch training report                                | NOT EVIDENCED |
| Learning rate / optimizer / scheduler | Not retained in the dedicated Frosch training report                                | NOT EVIDENCED |
| Training seed                         | Not retained in the available Frosch training report                                | NOT EVIDENCED |

# 4. Segmentation Model Training Record

The segmentation model is a separate RF-DETR Medium
instance-segmentation model. The available training report documents its
training script, configuration, output directory, and inference
checkpoint. It starts from its own COCO-pretrained segmentation weights
rather than loading the detection checkpoint.

| **Parameter**             | **Recorded value**                                                                                                                                             | **Status** |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| Architecture              | RF-DETR Medium segmentation                                                                                                                                    | CONFIRMED  |
| Pretrained initialization | COCO-pretrained segmentation weights; detection checkpoint not used                                                                                            | CONFIRMED  |
| Epochs                    | 50                                                                                                                                                             | CONFIRMED  |
| Batch size                | 4                                                                                                                                                              | CONFIRMED  |
| Gradient accumulation     | 4                                                                                                                                                              | CONFIRMED  |
| Training resolution       | 432                                                                                                                                                            | CONFIRMED  |
| Learning rate             | 1e-4                                                                                                                                                           | CONFIRMED  |
| Dataset root              | \`.\` by default in the documented training script                                                                                                             | CONFIRMED  |
| Output directory          | \`runs/frosch_seg_medium\`                                                                                                                                     | CONFIRMED  |
| Inference checkpoint      | \`runs/frosch_seg_medium/checkpoint_best_total.pth\`                                                                                                           | CONFIRMED  |
| Segmentation threshold    | 0.30                                                                                                                                                           | CONFIRMED  |
| Training command          | \`python train_frosch_segmentation.py --dataset . --output runs/frosch_seg_medium --epochs 50 --batch-size 4 --grad-accum-steps 4 --resolution 432 --lr 1e-4\` | CONFIRMED  |

# 5. Dataset and Training Lineage

| **Lineage field**                           | **Documented value**                                                                                         | **Status**                      |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------|
| Dataset identity                            | Frosch bottle 5, version v6; Roboflow COCO Segmentation export dated 23 June 2026                            | CONFIRMED                       |
| Dataset version referenced by existing A06  | A06-FROSCH-DSV-v1 (reconstructed dataset baseline)                                                           | CONFIRMED AS ARTIFACT REFERENCE |
| Training image resolution in source dataset | 432 × 432                                                                                                    | CONFIRMED                       |
| Detection model input                       | 576 × 576 (documented deployed engine input)                                                                 | CONFIRMED                       |
| Segmentation model input                    | 432 × 432                                                                                                    | CONFIRMED                       |
| Test-set state                              | No separate test split in supplied Frosch export; 0 images / 0 annotations                                   | CONFIRMED                       |
| Dataset leakage / split key                 | Original split key and formal pre-training leak-check evidence are not retained in the Frosch project record | NOT EVIDENCED                   |

# 6. Training / Runtime Environment

| **Component**    | **Recorded environment**       | **Status** |
|------------------|--------------------------------|------------|
| Operating system | Ubuntu 22.04.5 LTS             | CONFIRMED  |
| Python           | 3.10.20                        | CONFIRMED  |
| GPU              | NVIDIA GeForce RTX 5080, 16 GB | CONFIRMED  |
| NVIDIA driver    | 595.84                         | CONFIRMED  |
| CUDA             | 12.9.41                        | CONFIRMED  |
| PyTorch          | 2.13.0+cu130                   | CONFIRMED  |
| TensorRT         | 10.14.1.48.post1               | CONFIRMED  |
| RF-DETR          | 1.9.1                          | CONFIRMED  |
| EasyOCR          | 1.7.2                          | CONFIRMED  |
| OpenCV           | 4.10.0                         | CONFIRMED  |
| Harvester        | 1.4.3                          | CONFIRMED  |

# 7. Reproducibility and Handbook Gate Assessment

The handbook requires a training record that can be traced to the exact
dataset, configuration, execution environment, seed, dependency lock,
checkpoint, and calibration state. The following assessment reflects
only what is retained in the Frosch documentation.

| **A07 requirement**                               | **Frosch evidence**                                                                                                                             | **Status**                        | **Action / interpretation**                                                                |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|--------------------------------------------------------------------------------------------|
| Run identifier / experiment ID                    | No dedicated immutable run ID retained in the Frosch training report                                                                            | NOT EVIDENCED                     | Do not invent an ID                                                                        |
| Git commit                                        | No commit hash recorded in the available Frosch training report                                                                                 | NOT EVIDENCED                     | Repository history may exist, but exact training-to-commit mapping is not established here |
| Dataset version                                   | A06-FROSCH-DSV-v1 is the reconstructed parent artifact reference                                                                                | PARTIAL                           | Parent trace exists; formal sealed test-set state is incomplete                            |
| Configuration path                                | Segmentation training script/configuration is documented; detection hyperparameter file is not retained in the dedicated Frosch training report | PARTIAL                           | Keep documented paths only                                                                 |
| Seed                                              | No project-specific seed recorded in the Frosch training report                                                                                 | NOT EVIDENCED                     | Handbook default seed is not asserted as historical fact                                   |
| Dependency lock / hash                            | No lock hash retained in the Frosch training report                                                                                             | NOT EVIDENCED                     | Do not fabricate hash                                                                      |
| Hardware                                          | RTX 5080 16 GB and software environment recorded                                                                                                | CONFIRMED                         |                                                                                            |
| Key hyperparameters                               | Segmentation: 50 epochs, batch 4, grad accumulation 4, resolution 432, LR 1e-4                                                                  | PARTIAL                           | Detection hyperparameters are not retained in available Frosch record                      |
| Calibration values                                | Camera calibration and pixel-to-mm workflow were not implemented                                                                                | NOT APPLICABLE TO IMPLEMENTED RUN | Runtime remained image-space                                                               |
| Checkpoint                                        | Detection and segmentation checkpoint paths retained                                                                                            | CONFIRMED                         |                                                                                            |
| Checkpoint provenance / final selection rationale | Checkpoint paths are documented; formal best-checkpoint selection event is not separately retained                                              | PARTIAL                           | Do not claim a formal selection procedure                                                  |
| Training metrics                                  | Detection metric table exists elsewhere in consolidated report; segmentation aggregate metrics and epoch loss history are not retained          | PARTIAL                           | Evaluation evidence belongs primarily in A08/A17                                           |
| Reproducibility test / rerun                      | No rerun evidence is retained in the available documentation                                                                                    | NOT EVIDENCED                     | Do not claim successful reproduction                                                       |
| License record for pretrained weights             | Segmentation report states COCO-pretrained initialization, but a separate license record for the pretrained checkpoint is not retained          | NOT EVIDENCED                     | Retain as a documentation gap                                                              |

# 8. Calibration and Measurement Boundary

Camera calibration was not implemented in the Frosch pipeline. The
project documentation explicitly records that there is no intrinsic
calibration or undistortion stage and that pixel-to-millimetre
measurement was not implemented. Consequently, A07 does not attach
calibration parameters to the training run and does not imply that the
trained models support calibrated physical measurement.

# 9. Retained Model and Deployment Artifacts

| **Artifact**                 | **Path / reference**                                                | **Status** |
|------------------------------|---------------------------------------------------------------------|------------|
| Detection checkpoint         | \`runs/frosch_medium/checkpoint_best_regular.pth\`                  | CONFIRMED  |
| Detection TensorRT engine    | \`output/rfdetr-medium.trt\`                                        | CONFIRMED  |
| Segmentation checkpoint      | \`runs/frosch_seg_medium/checkpoint_best_total.pth\`                | CONFIRMED  |
| Segmentation TensorRT engine | \`output/rfdetr-seg-medium.trt\`                                    | CONFIRMED  |
| Training script              | \`train_frosch_segmentation.py\` (documented command/configuration) | CONFIRMED  |
| Primary runtime              | \`live_inference.py\`                                               | CONFIRMED  |

# 10. Known Training-Record Gaps and Risks

- No exact training-to-Git-commit mapping is retained in the available
  Frosch training documentation.

- No project-specific training seed is retained; the handbook default
  seed is not presented as historical run evidence.

- No dependency-lock hash is retained in the available Frosch training
  documentation.

- The dedicated training report does not retain epoch-by-epoch loss
  history for the segmentation model.

- Aggregate segmentation mAP/IoU/test metrics are not retained in the
  available project record.

- The supplied Frosch dataset has no separate test split, limiting
  formal held-out evaluation traceability.

- Calibration parameters are absent because camera calibration and
  pixel-to-mm measurement were not implemented.

- No successful reproduction/rerun record is retained in the available
  project documentation.

# 11. A07 Conclusion

**A07 status: PARTIALLY COMPLETE — DOCUMENTED TRAINING RECORD WITH
TRACEABILITY GAPS**

The available Frosch record is sufficient to document the two RF-DETR
model artifacts, the retained checkpoints, the segmentation training
configuration, the deployment engines, and the verified
software/hardware environment. It is not sufficient to claim complete
handbook-grade reproducibility because the exact run identifier,
training-to-commit mapping, project-specific seed, dependency-lock hash,
complete detection training configuration, and rerun evidence are not
retained. These gaps are recorded explicitly and should remain visible
in downstream A08/A13/A17 traceability.

# 12. Source Basis

- Frosch_Bottle_Inspection_Consolidated_Technical_Report.docx — Sections
  5–7 and Appendix A.

- FROSCH_TRAINING_REPORT.md — model roles, checkpoint paths,
  segmentation training configuration, runtime threshold, and
  retained-metric limitations.

- Frosch dataset documentation / DATASET_CARD — dataset identity and
  split state.

- A06-FROSCH-DSV-v1 — parent dataset-version artifact and its documented
  release-gate limitations.

**Document control note:** This artifact was reconstructed from existing
project records in accordance with the Team Lead instruction not to
rerun the project solely for documentation. Missing evidence is
intentionally not backfilled with assumptions.
