**FROSCH BOTTLE INSPECTION PIPELINE** | Evidence-based reconstruction | September 2026

**A14 — Decision Record Log**

*Decisions evidenced by the retained project records (Handbook §7.1.2, §9)*

| **Artifact ID** | A14-FROSCH-DEC-log-v1 | **Preparation basis** | Existing project records only |
|---|---|---|---|
| **Scope** | All reconstructable decisions across A01–A13 | **Re-execution** | Not performed |
| **Evidence policy** | Only the choice actually evidenced is recorded; alternatives are marked NOT DOCUMENTED where absent, never invented | **Status** | RECONSTRUCTED, PARTIAL |

| **Handbook §9 treats "chose X" without recorded alternatives as an incomplete decision record. Several entries below are marked incomplete for exactly this reason — the alternatives were not retained, not that none existed.** |
|---|

# 1. Purpose

Decision records state what was decided, by whom, when, and what was rejected and why (Handbook §7.1.2 technical-approach field, §9). No standalone decision log was retained for Frosch; this artifact reconstructs the decisions evidenced across the other artifacts into one traceable log.

# 2. Decision Log

| **ID** | **Decision** | **Alternatives rejected / why** | **Decider** | **Reversible?** | **Affects** | **Evidence status** |
|---|---|---|---|---|---|---|
| DEC-0001 | Use TensorRT native inference engine (`NativeTRTEngine`) for production runtime | Not documented | Not documented | Yes | A07, A11 | INCOMPLETE — choice known, rationale not retained |
| DEC-0002 | Use RF-DETR segmentation workflow at 432×432 training resolution | Not documented | Not documented | Yes | A04, A07 | INCOMPLETE |
| DEC-0003 | Treat the 52-bottle runtime run as a status-classification validation, explicitly not a formal model-accuracy or metrology study | Implied rejection of treating it as a calibrated accuracy study | Not individually named | Yes | A08, A09 | Recorded rationale exists (avoids overstating evidence); decider not named |
| DEC-0004 | Deploy in `--input folder` mode only, no camera integration | Camera-mode fetch code exists but is explicitly marked untested — implies camera-mode deployment was considered and deferred | Not documented | Yes | A11 §Recovery | PARTIAL — deferral evidenced, formal rejection reasoning not retained |
| DEC-0005 | Fix `mem_check.py`'s per-frame RSS/latency accumulation (switch to preallocated numpy float32 buffers) before trusting soak results | Continuing with unbounded Python lists, which added ~12MB of harness-induced growth | Not documented | N/A — already applied | A11 | Recorded in A11; decider not named |
| DEC-0006 | Record 60 of 61 A12 fallback-audit findings as false positives after manual triage, rather than accepting the raw scanner FAIL verdict | Accepting the raw automated FAIL without triage | Not documented | N/A — analysis decision | A12 | Rationale per-pattern is recorded; decider not named |
| DEC-0007 | Leave the capacity-backfill fallback (A12 open finding) unresolved rather than fixing it before this audit closed | Fixing it immediately | Not documented | N/A — deferred | A12, A13 | Recorded as OPEN, not accepted-as-design; decider not named |

# 3. Decisions Referenced But Not Reconstructable

| **Decision area** | **Why it cannot be logged here** |
|---|---|
| Original model-architecture choice (why detection + segmentation rather than a single-stage approach) | No A14-equivalent record or planning note retained |
| Choice of the 70.65/29.35/0 train/validation/test split proportions instead of the Handbook D2 default (80/15/5) | No rationale retained (see A06 §4) |
| Choice not to implement camera calibration / pixel-to-mm conversion | No rationale retained; only its absence is documented (A09) |

# 4. Traceability

| **Related artifact** | **Relationship** |
|---|---|
| A07, A08, A11, A12 | Source evidence for the decisions logged above. |
| A13 — Release Manifest | Decisions here explain some of the gaps blocking release. |

# 5. A14 Status

| **A14 STATUS: PARTIAL LOG. Seven decisions are reconstructed from surviving evidence; most lack a named decider or a recorded rejected-alternative, which Handbook §9 treats as an incomplete decision record. Three further decision areas are known to exist but could not be reconstructed at all.** |
|---|
