**FROSCH BOTTLE INSPECTION PIPELINE** | Evidence-based reconstruction | September 2026

**A15 — Re-entry Record Log**

*Items that should route back under Handbook §7.5, reconstructed from the current artifact chain*

| **Artifact ID** | A15-FROSCH-RE-log-v1 | **Preparation basis** | Existing project records only |
|---|---|---|---|
| **Scope** | Re-entry triggers evidenced across A02–A13 | **Re-execution** | Not performed |
| **Evidence policy** | Only findings already recorded elsewhere are logged as triggers; no new defects invented | **Status** | RECONSTRUCTED |

| **No re-entry record was contemporaneously filed for any of the findings below at the time they were discovered. This log reconstructs what §7.5 requires retrospectively, so the recurring-cause review in §7.5.3 has something to work from.** |
|---|

# 1. Purpose

Handbook §7.5.1 lists ten re-entry triggers; §7.5.3 requires every re-entry recorded as A15 within one working day, with a trigger number, re-entry point, and one-sentence cause. None of the findings already on record elsewhere in this project (A06, A08, A09, A11, A12) were filed as A15 at the time. This log closes that gap retrospectively.

# 2. Re-entry Log

| **ID** | **Item** | **Trigger #** | **Trigger** | **Re-enters at** | **Cause (one sentence)** | **Which re-entry** |
|---|---|---|---|---|---|---|
| RE-0001 | Dataset split (A06) | 2 | 5.2 obligation not performed / evidence missing | 7.2 Implementation | No split key, leak check, or sealed test set was recorded for the 739/307 train/validation split. | 1st |
| RE-0002 | Annotation QA (A05) | 2 | 5.2 obligation not performed / evidence missing | 7.2 Implementation | No 10% annotation review sample, edge-disagreement measurement, or adjudication record was retained. | 1st |
| RE-0003 | Measurement validation (A09) | 2 | 5.2 obligation not performed / evidence missing | 7.2 Implementation | Physical calibration and ground-truth measurement were never implemented, so no accuracy/repeatability study could be performed. | 1st |
| RE-0004 | Soak test (A11) | 3 | 5.2 check ran and returned a failing result | 7.2 Implementation | Worst-case latency (101.131 ms) breached the 100 ms budget and was ungated at the time of the retained run. | 1st |
| RE-0005 | Soak test D4 stability (A11) | 3 | 5.2 check ran and returned a failing result | 7.2 Implementation | Two consecutive full-duration runs on identical code produced opposite D4 (memory-growth) verdicts. | 1st |
| RE-0006 | Fallback audit (A12) | 1 | Blocking 7.3 gate fails (defect in the work) | 7.2 Implementation | `finalize_measurements()` backfills an unconfirmed capacity reading to `--expected-capacity` instead of reporting it absent — the prohibited pattern the audit exists to catch. | 1st |
| RE-0007 | Objective (A02) | 9 | An assumption recorded under §4.2 proved false | 4.3 Objective | No customer [ACCURACY_REQ] was ever supplied; the objective reconstructed here is image-space only, not the physical accuracy objective the handbook assumes exists. | 1st |

# 3. Re-entry Limit Check (§7.5.3)

Each item above is on its first reconstructed re-entry. None has reached the third-re-entry escalation threshold. Should any of RE-0001 through RE-0007 recur after the fix implied by its parent artifact's "Recommended Re-entry Actions" section, the next occurrence should be logged as that item's 2nd re-entry here, not as a new ID.

# 4. Recurring-Cause Pattern (for §7.5.3 monthly review)

| **Pattern** | **Items** | **Candidate amendment area** |
|---|---|---|
| "5.2 obligation not performed" (trigger 2) appears on three separate items (RE-0001, RE-0002, RE-0003) | A05, A06, A09 | Suggests a standard or process gap in how self-verification obligations were scheduled during this project — a §7.5.3 amendment candidate against §5.2/§7.1 scheduling, not a single item's defect. |

# 5. Traceability

| **Related artifact** | **Relationship** |
|---|---|
| A05, A06, A09, A11, A12 | Source findings reconstructed into the log above. |
| A02 | Source of RE-0007. |
| A13 — Release Manifest | Lists the same gaps as release blockers; this log adds the required trigger/routing detail. |

# 6. A15 Status

| **A15 STATUS: RECONSTRUCTED, RETROSPECTIVE. Seven items are logged against their evidenced findings. Because none was filed within one working day of discovery as §7.5.3 requires, this log itself is a documentation gap being closed late, not a contemporaneous record.** |
|---|
