**FROSCH BOTTLE INSPECTION PIPELINE** | Evidence-based reconstruction | September 2026

**A16 — Exception Record**

*Standards or gates waived, evidenced by the retained project records (Handbook §11)*

| **Artifact ID** | A16-FROSCH-EXC-v1 | **Preparation basis** | Existing project records only |
|---|---|---|---|
| **Scope** | Exceptions to Handbook standards/gates | **Re-execution** | Not performed |
| **Evidence policy** | No exception is recorded as granted unless a grant is actually evidenced | **Status** | NO GRANTED EXCEPTION FOUND |

| **STATUS: NO EXCEPTION RECORD EXISTS IN THE RETAINED EVIDENCE. This artifact exists to state that clearly, and to identify where an exception request would be the correct next step given the open findings in A11–A13, rather than to document a waiver that was never actually granted.** |
|---|

# 1. Purpose

Handbook §11 requires every waived standard, obligation, or gate to be requested before the work, recorded as A16 with a rule, reason, substitute action, risk, approver, and expiry (max 90 days). No such record was retained for Frosch.

# 2. Findings Against Which an Exception Has Not Been Requested

The following gate failures are currently unresolved (per A11–A13) and are release-blocking. None has an associated A16 exception on record — meaning the project has neither fixed them nor formally accepted the risk of releasing with them outstanding.

| **Finding** | **Rule/gate implicated** | **Waivable?** | **Exception on record?** |
|---|---|---|---|
| Capacity backfill (A12 open finding) | §5.1.6 Fallback Prohibition | NOT WAIVABLE per §11 — the fallback prohibition is explicitly excluded from exception | No — correctly so, since it cannot be waived |
| Worst-case latency breach, ungated at time of run (A11) | §7.3.3 Stress and Resilience | Waivable only with CTO/COO approval (stress-test budget breach) | No |
| D4 memory-growth instability across repeat runs (A11) | §5.2 Self-Verification | Waivable by AI lead, to defer only, never to omit | No |
| Missing [ACCURACY_REQ] / unconfirmed objective (A02) | §4.3 objective confirmation | Not a §11 exception category — this is a re-entry (A15 RE-0007), not a waivable standard | No — correctly routed as re-entry instead |

# 3. Why No Exception Record Should Be Fabricated Here

Handbook §11 makes clear that an exception is requested before the work and recorded with an approver and expiry — it is a specific authorization event, not a retrospective label for "this gap exists." Since no such authorization event is evidenced for any of the findings in §2 above, this artifact records their absence rather than inventing an approver or expiry date that was never set.

# 4. Recommended Next Step

For the two items above that are genuinely waivable (latency breach, D4 instability), the correct path per §11 is:

1. The AI lead (for D4, to defer only) or the CTO/COO (for the stress-test budget breach) formally requests the exception before any release proceeds.
2. A new A16 record is created at that time with rule, item(s), justification, substitute action, risk, approver, and an expiry of at most 90 days.
3. This artifact (A16-FROSCH-EXC-v1) is superseded by that record once it exists.

# 5. Traceability

| **Related artifact** | **Relationship** |
|---|---|
| A11 — Soak Test | Source of the two waivable findings discussed in §2. |
| A12 — Fallback Audit | Source of the non-waivable finding. |
| A13 — Release Manifest | Release remains blocked in the absence of either a fix or a genuine A16 grant for each item. |

# 6. A16 Status

| **A16 STATUS: NO EXCEPTION GRANTED. Nothing in this project's retained evidence has been formally waived under §11. Where a waiver would be the appropriate path (latency, D4), it has not been sought; where it would not be appropriate (the fallback prohibition), none has been attempted either.** |
|---|
