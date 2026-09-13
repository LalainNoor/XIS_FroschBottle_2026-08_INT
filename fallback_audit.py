"""
fallback_audit.py — A12 Fallback Audit for FROSCH pipeline (§5.1.6 / §5.2).

Scans live_inference_combined.py and related source files for every
failure path and checks against the §5.1.6 prohibited patterns.

Handbook requirement: "No prohibited pattern; every failure path returns
an explicit absent status."

Prohibited patterns (§5.1.6):
  P1  except/try returning 0, None, or a constant as a measurement
  P2  Catching an exception to keep the pipeline running without reporting failure
  P3  Returning the previous measurement when no contour / detection found
  P4  Taking contours[0] / detections[0] inside a try without length check
  P5  Hardcoded fallback values when config / model is missing
  P6  Clamping a result into the expected range
  P7  Reusing the previous frame on a grab failure

Usage:
    python fallback_audit.py [--sources src1.py src2.py ...] [--output A12_audit.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Prohibited pattern definitions — (id, description, regex)
# ---------------------------------------------------------------------------

PROHIBITED: list[tuple[str, str, str]] = [
    (
        "P1",
        "except/except Exception returning 0 / None / constant as measurement",
        r"except[^:]*:\s*\n(?:[^\n]*\n){0,10}?\s*return\s+(0|None|0\.0|False|\"\"|\[\]|\{\})",
    ),
    (
        "P2",
        "bare except: pass (swallows failure silently)",
        r"except[^:]*:\s*\n\s*pass\b",
    ),
    (
        "P3",
        "or <default> on measurement variable (hides absent result)",
        r"\b(measurement|result|value|score|distance|height|width|count)\s*=\s*\w+\s+or\s+\d",
    ),
    (
        "P4",
        "accessing [0] on detections/contours inside try without prior length check",
        r"(detections|contours|boxes|masks|results)\[0\]",
    ),
    (
        "P5",
        ".get(key, <non-None default>) on config/measurement dict (hides missing config)",
        r"\.get\(\s*['\"][^'\"]+['\"]\s*,\s*(?!None)[^\)]+\)",
    ),
    (
        "P6",
        "np.clip / min(max()) on a measurement result (clamps into range)",
        r"np\.clip\s*\(|min\s*\(\s*max\s*\(",
    ),
    (
        "P7",
        "reusing previous frame / prev_frame on grab failure",
        r"\b(prev_frame|last_frame|previous_frame)\b",
    ),
]


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

class Finding(NamedTuple):
    pattern_id: str
    description: str
    file: str
    line: int
    snippet: str


# ---------------------------------------------------------------------------
# AST-based check: bare except with return inside
# ---------------------------------------------------------------------------

def ast_bare_except_with_return(source: str, filepath: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if handler.type is None:  # bare except:
                for child in ast.walk(handler):
                    if isinstance(child, ast.Return):
                        val = child.value
                        if val is None or (isinstance(val, ast.Constant) and val.value in (0, None, 0.0, "", [], {})):
                            findings.append(Finding(
                                pattern_id="P1_ast",
                                description="Bare except: with return of 0/None/constant",
                                file=filepath,
                                line=handler.lineno,
                                snippet=f"except: ... return {ast.unparse(child.value) if child.value else 'None'}",
                            ))
    return findings


# ---------------------------------------------------------------------------
# Regex scan
# ---------------------------------------------------------------------------

def regex_scan(source: str, filepath: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()

    for pid, desc, pattern in PROHIBITED:
        for match in re.finditer(pattern, source, re.MULTILINE):
            line_no = source[:match.start()].count("\n") + 1
            snippet = lines[line_no - 1].strip() if line_no <= len(lines) else ""
            findings.append(Finding(
                pattern_id=pid,
                description=desc,
                file=filepath,
                line=line_no,
                snippet=snippet,
            ))
    return findings


# ---------------------------------------------------------------------------
# Manual failure-path extraction (AST)
# ---------------------------------------------------------------------------

def extract_failure_paths(source: str, filepath: str) -> list[dict]:
    """
    Walk the AST to enumerate all except handlers and return statements
    that follow an error condition.  Records each as a failure path entry.
    """
    paths: list[dict] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return paths

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                exc_type = ast.unparse(handler.type) if handler.type else "bare except"
                returns = []
                for child in ast.walk(handler):
                    if isinstance(child, ast.Return):
                        returns.append(ast.unparse(child.value) if child.value else "None")
                    if isinstance(child, ast.Raise):
                        returns.append("raise")

                behaviour = returns[0] if returns else "no explicit return (falls through)"
                paths.append({
                    "file": filepath,
                    "line": handler.lineno,
                    "except_type": exc_type,
                    "behaviour": behaviour,
                    "verdict": (
                        "OK — raises or returns absent status"
                        if ("raise" in behaviour or behaviour in ("None", ""))
                        else "REVIEW — check for prohibited substitution"
                    ),
                })
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FROSCH A12 Fallback Audit")
    parser.add_argument(
        "--sources", nargs="+",
        default=["live_inference.py"],
        help="Python source files to audit",
    )
    parser.add_argument("--output", default="A12_fallback_audit.json")
    parser.add_argument("--commit", default="", help="Git commit SHA")
    args = parser.parse_args()

    all_findings: list[Finding] = []
    all_paths: list[dict] = []
    missing_files: list[str] = []

    for src_path in args.sources:
        p = Path(src_path)
        if not p.exists():
            missing_files.append(src_path)
            print(f"[fallback_audit] WARNING: {src_path} not found — skipped", flush=True)
            continue

        source = p.read_text(encoding="utf-8")
        all_findings.extend(regex_scan(source, src_path))
        all_findings.extend(ast_bare_except_with_return(source, src_path))
        all_paths.extend(extract_failure_paths(source, src_path))

    prohibited_found = [f._asdict() for f in all_findings]
    overall_pass = len(prohibited_found) == 0

    result = {
        "artifact_id": "A12",
        "project": "FROSCH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": args.commit,
        "files_audited": args.sources,
        "missing_files": missing_files,
        "failure_paths": all_paths,
        "prohibited_patterns_found": prohibited_found,
        "prohibited_count": len(prohibited_found),
        "overall_pass": overall_pass,
        "verdict": (
            "PASS — no prohibited fallback patterns detected"
            if overall_pass else
            f"FAIL — {len(prohibited_found)} prohibited pattern(s) found; see prohibited_patterns_found"
        ),
    }

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"[fallback_audit] Results written to {out_path}", flush=True)
    print(f"[fallback_audit] Failure paths found: {len(all_paths)}", flush=True)
    print(f"[fallback_audit] Prohibited patterns: {len(prohibited_found)}", flush=True)
    print(f"[fallback_audit] Overall pass: {overall_pass}", flush=True)

    if not overall_pass:
        for f in prohibited_found:
            print(f"  [{f['pattern_id']}] {f['file']}:{f['line']} — {f['snippet']}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
