**FROSCH BOTTLE INSPECTION PIPELINE** | Evidence-based reconstruction | September 2026

**A17 — Model Performance Report**

*Detection/segmentation quality and inference speed, separated from the end-to-end evaluation in A08*

| **Artifact ID** | A17-FROSCH-MPR-v1 | **Preparation basis** | Existing project records only |
|---|---|---|---|
| **Parent artifacts** | A06, A07 | **Re-execution** | Not performed |
| **Scope** | Model-level quality metrics and measured inference performance | **Review state** | Pending project lead confirmation |
| **Evidence policy** | Only metrics actually retained are reported; unretained metrics are marked as such rather than estimated | **Status** | PARTIALLY COMPLETE |

| **A17 answers how the model performed and how fast, per Handbook §7.3.4. It does not answer whether the measurement met its criterion — that is A08/A09. Several required fields (aggregate segmentation metrics, full latency percentile distribution, per-stage timing) are not retained and are marked NOT EVIDENCED.** |
|---|

# 1. Purpose and Scope

Handbook §5.2/§7.3.4 requires a model performance report distinct from the acceptance-criterion evaluation: detection or segmentation metrics per class, the decision threshold and how it was chosen, inference speed on target hardware, per-stage timing, and comparison against the current production model. This artifact assembles what the retained Frosch records support against that structure.

# 2. Checkpoint and Evaluation Basis

| **Field** | **Recorded value** | **Status** |
|---|---|---|
| Checkpoint | Not individually cited beyond A07's reconstructed training-run reference | Partial |
| Task | Detection (bottle/label/capacity/bump/damage/scratch) + segmentation (bottle mask) | Confirmed |
| Evaluated on | Validation split (307 images / 996 annotations); no separate sealed test set exists (A06 §7) | Confirmed as a gap |

# 3. Detection Quality Metrics (IoU = 0.50 matching)

| **Class** | **Precision** | **Recall** | **F1** | **AP50** | **n_gt** |
|---|---|---|---|---|---|
| bottle | 0.9869 | 0.9934 | 0.9902 | 0.9091 | 304 |
| bump | 0.4487 | 0.5469 | 0.4930 | 0.4130 | 64 |
| capacity | 0.9189 | 0.9963 | 0.9561 | 0.9072 | 273 |
| damage | 0.3607 | 0.9167 | 0.5176 | 0.8455 | 24 |
| label | 0.9869 | 0.9934 | 0.9901 | 0.9088 | 303 |
| scratch | 0.3333 | 0.2143 | 0.2609 | 0.2061 | 28 |

**Confusion matrix (retained):**

| **GT \ Pred** | **bottle** | **bump** | **capacity** | **damage** | **label** | **scratch** | **missed** |
|---|---|---|---|---|---|---|---|
| bottle | 302 | 0 | 0 | 0 | 0 | 0 | 2 |
| bump | 3 | 68 | 4 | 10 | 3 | 2 | 29 |
| capacity | 0 | 0 | 273 | 0 | 0 | 1 | 1 |
| damage | 0 | 0 | 2 | 41 | 0 | 0 | 2 |
| label | 0 | 10 | 15 | 4 | 301 | 1 | 2 |
| scratch | 1 | 0 | 0 | 6 | 1 | 14 | 22 |

# 4. Segmentation Quality Metrics

| **Metric** | **Status** |
|---|---|
| mean IoU | NOT EVIDENCED — not retained in the available project documentation |
| Dice | NOT EVIDENCED |
| Boundary/edge accuracy (px) | NOT EVIDENCED |

Training configuration and checkpoint for the segmentation workflow are retained (A07); aggregate quality metrics for it are not.

# 5. Decision Threshold

| **Value** | **Chosen on** | **Status** |
|---|---|---|
| Detection matching threshold: IoU = 0.50 | Not documented whether validation or test set | Recorded value, chosen-on basis not retained |
| Segmentation runtime threshold: 0.30 | Not documented | Recorded value, chosen-on basis not retained |

No record confirms these thresholds were chosen on a validation set rather than the evaluation set itself; per Handbook §17 rule, a threshold chosen on the test set would be invalid. Since no separate sealed test set exists at all (A06 §7), this question cannot currently be resolved either way.

# 6. Inference Performance

| **Condition** | **Observed** | **Status** |
|---|---|---|
| Idle (no bottle in frame) | 83–96 FPS | Range only |
| Active (full pipeline: dual TensorRT inference, mask geometry, centricity, defect validation, tracking, image saving) | 22–35 FPS | Range only |
| Earlier synchronous-OCR version | ~4–10 FPS | Historical comparison, not current performance |
| Hardware | Not individually re-stated here; see A07/A11 for the tested environment | Partial |
| Precision (FP32/FP16/INT8) | NOT DOCUMENTED | Gap |
| Batch size | NOT DOCUMENTED | Gap |
| Latency mean / p50 / p95 / p99 / worst-case | Only p95 (30.505 ms) and worst-case (101.131 ms) are retained, from the A11 soak run | Partial — full distribution not retained |
| Per-stage timing (preprocess / inference / postprocess) | NOT EVIDENCED | Gap |

# 7. Comparison Against Current Production Model

| **Metric** | **Production** | **This model** | **Change** |
|---|---|---|---|
| — | NOT DOCUMENTED | — | No production baseline is recorded anywhere in the retained project evidence to compare against. |

# 8. Per-Group Results

| **Group** | **F1 (or IoU)** | **FPS** | **Pass/Fail** |
|---|---|---|---|
| — | Detection metrics above are class-level, not group-level (variant/size/camera/shift). No per-group breakdown by capacity is retained at the model-quality level; runtime status counts by capacity are reported in A08 §6 instead, which is a different kind of result. | | NOT EVIDENCED at this level |

# 9. Verdict

| **Question** | **Answer** |
|---|---|
| Fit for release candidate? | NOT DETERMINED — precision/recall/F1/AP50 and the confusion matrix are documented and materially uneven across classes (scratch and damage notably weaker); segmentation aggregate metrics, full latency distribution, per-stage timing, and a production-model comparison are all missing, so a complete §7.3.4/§17 sign-off cannot be made from this evidence alone. |

# 10. Traceability

| **Related artifact** | **Relationship** |
|---|---|
| A06 — Dataset Version | Basis for "evaluated on" split; carries forward the missing-sealed-test-set gap. |
| A07 — Training Run | Source of checkpoint/config/training provenance. |
| A08 — Evaluation | Uses this report's detection metrics as one evidence input; answers acceptance criteria, which A17 does not. |
| A11 — Soak Test | Source of the p95/worst-case latency figures reused in §6. |
| A13 — Release Manifest | This report's incompleteness is one reason release remains blocked. |

# 11. A17 Status

| **A17 STATUS: PARTIALLY COMPLETE. Detection metrics and the confusion matrix are documented. Segmentation aggregate metrics, decision-threshold provenance, full latency-percentile distribution, per-stage timing, per-group breakdown, and a production-model comparison are not retained and are recorded as gaps rather than estimated.** |
|---|
