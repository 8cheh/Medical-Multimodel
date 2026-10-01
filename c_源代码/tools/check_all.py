"""Run every self-check in one command and record the verdict.

The mid-term report and the deck both claim "six automatic checks". A claim like that is
only usable if one command reproduces it, so this is that command: each check runs as its
own subprocess with its own timeout, and the outcome is written to results/check_all.json
so the claim cites a file instead of a person's memory.

    python tools/check_all.py                # everything runnable here
    python tools/check_all.py --list         # just the roster
    python tools/check_all.py --no-data      # skip the checks that need the local data copy
    python tools/check_all.py --only numbers,delivery
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(r"D:\vitaldb_local")
OUT_JSON = ROOT / "results" / "check_all.json"
ROSTER_JSON = ROOT / "results" / "check_all_roster.json"
PY = sys.executable

# name, argv, needs the local data copy, what it proves
CHECKS: list[tuple[str, list[str], bool, str]] = [
    ("numbers", [PY, "tools/check_report_numbers.py"], False,
     "every number in REPORT/SUMMARY/MIDTERM/delivery README is re-derived from results/"),
    ("delivery", [PY, "tools/check_algorithm_delivery.py"], False,
     "evaluation pipeline, waveform extractors, label definition, merge modes"),
    ("final_audit", [PY, "tools/final_audit.py"], False,
     "repository-wide audit of the objective's deliverables"),
    ("p0_eval", [PY, "remote/check_p0_eval.py"], False,
     "self-check: the non-trivial evaluation logic, no dataset needed"),
    ("abp_features", [PY, "remote/check_abp_features.py"], False,
     "self-check: the arterial-waveform extractor on synthetic input"),
    ("extra_merge", [PY, "remote/check_extra_merge.py"], False,
     "self-check: both --extra-features merge modes on tiny frames"),
    ("alarm_policy", [PY, "remote/alarm_policy.py"], False,
     "self-check: the alarm layer (persistence + hysteresis)"),
    ("build_refactor", [PY, "remote/check_build_refactor.py", "--tracks-dir",
                        str(LOCAL / "pilot")], True,
     "the extracted episode definition still reproduces the labels"),
    ("stream", [PY, "remote/stream_monitor.py", "--check", "--caseid", "19"], True,
     "streamed features and scores equal the offline run, case by case"),
]

# The document checks run after the code checks: they consume their artifacts.
DOC_CHECKS: list[tuple[str, list[str], bool, str]] = [
    ("pdf_report", [PY, "tools/verify_pdf.py", "--pdf", "report/REPORT.pdf"], False,
     "the report PDF really carries its pages, figures, numbers and CJK glyphs"),
    ("pdf_midterm", [PY, "tools/verify_pdf.py", "--pdf", "report/MIDTERM.pdf",
                     "--structural-only"], False,
     "the mid-term PDF really carries its pages, figures and glyphs"),
    ("slides", [PY, "tools/make_slides_pdf.py"], False,
     "the deck PDF matches the current .pptx and the deck has no placeholder text"),
    ("latex", [PY, "tools/build_latex_midterm.py"], False,
     "the LaTeX mid-term report compiles and carries its numbers"),
    ("freshness", [PY, "tools/check_freshness.py"], False,
     "no rendered PDF or deck is older than the file it was rendered from"),
    ("citations", [PY, "papers/check_citations.py"], False,
     "every citation resolves and no reference is orphaned"),
]


def run_one(name: str, argv: list[str], timeout: int) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True,
                              check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "timeout", "seconds": round(time.time() - started, 1),
                "tail": f"exceeded {timeout}s"}
    except OSError as exc:
        return {"name": name, "status": "error", "seconds": round(time.time() - started, 1),
                "tail": str(exc)}
    raw = proc.stdout.decode("utf-8", errors="replace")
    tail = ""
    for line in reversed(raw.splitlines()):
        if line.strip():
            tail = line.strip()[:160]
            break
    return {"name": name, "status": "pass" if proc.returncode == 0 else "fail",
            "seconds": round(time.time() - started, 1), "returncode": proc.returncode,
            "tail": tail}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the roster and exit")
    ap.add_argument("--no-data", action="store_true",
                    help="skip the checks that need the local data copy")
    ap.add_argument("--only", default="", help="comma-separated names to run")
    ap.add_argument("--timeout", type=int, default=1800, help="per-check seconds")
    args = ap.parse_args(argv)

    roster = CHECKS + DOC_CHECKS
    if args.list:
        for name, _, needs_data, why in roster:
            print(f"{name:14s} {'(needs data)' if needs_data else '            '} {why}")
        return 0

    # Record the roster *before* running anything. Documents quote "N/N passed", and
    # comparing that against the previous run's verdict would never converge (a failing
    # check keeps the stored verdict below N, which keeps failing the comparison). The
    # roster is stable and independent of how this run ends, so it is the right anchor;
    # whether the checks actually pass is this script's exit code.
    try:
        with open(ROSTER_JSON, "w", encoding="utf-8") as fh:
            json.dump({"total": len(roster), "names": [name for name, _, _, _ in roster]},
                      fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    except OSError as exc:
        print(f"  (cannot record the roster: {exc})")

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    results = []
    for name, cmd, needs_data, why in roster:
        if wanted and name not in wanted:
            continue
        if needs_data and (args.no_data or not LOCAL.is_dir()):
            results.append({"name": name, "status": "skipped", "seconds": 0.0,
                            "tail": "needs the local data copy" if not LOCAL.is_dir()
                                    else "skipped by --no-data", "proves": why})
            print(f"{name:14s} SKIP  {why}")
            continue
        outcome = run_one(name, cmd, args.timeout)
        outcome["proves"] = why
        results.append(outcome)
        mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP",
                "timeout": "TIME", "error": "ERR "}[outcome["status"]]
        print(f"{name:14s} {mark}  {outcome['seconds']:6.1f}s  {outcome['tail']}")

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = [r["name"] for r in results if r["status"] in ("fail", "timeout", "error")]
    skipped = [r["name"] for r in results if r["status"] == "skipped"]
    print(f"\n{passed}/{len(results)} checks passed"
          + (f"; failed: {failed}" if failed else "")
          + (f"; skipped: {skipped}" if skipped else ""))
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump({"checks": results, "passed": passed, "total": len(results),
                       "failed": failed, "skipped": skipped}, fh,
                      ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"recorded {OUT_JSON.relative_to(ROOT)}")
    except OSError as exc:
        print(f"  (cannot record the verdict: {exc})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
