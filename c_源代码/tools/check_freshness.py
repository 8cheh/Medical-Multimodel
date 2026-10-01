"""Fail if a rendered artifact is older than the thing it was rendered from.

Twice in this project a document nearly shipped a stale render: report/MIDTERM.pdf after its
Markdown had been corrected, and the deck's PDF after the generator changed (LibreOffice had
handed the conversion to a still-running instance and written the previous deck). Both were
caught by eye. This catches them by timestamp instead, so they cannot be caught by luck.

    python tools/check_freshness.py
    python tools/check_freshness.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Artifacts that are rewritten by every verification run. A document may legitimately quote
# an older verdict than the live file, so counting them as inputs would make the deck look
# stale right after every green check - a check that can never pass is worse than none.
VERDICTS = {
    "check_all.json", "check_all_roster.json", "check_numbers.json", "check_extra_docs.json",
    "verify_pdf.json", "verify_pdf_MIDTERM.json", "verify_slides.json",
    "verify_latex_midterm.json",
}


def deck_inputs() -> list[Path]:
    """Everything tools/make_slides.js reads, minus the per-run verdict files."""
    inputs = [ROOT / "tools" / "make_slides.js", ROOT / "算法实现" / "inventory.json"]
    for pattern in ("results/*.json", "results/*.log"):
        for path in sorted((ROOT / "results").glob(pattern.split("/")[-1])):
            if path.name not in VERDICTS:
                inputs.append(path)
    return inputs


# (what it is, the rendered file, the things it was rendered from)
JOBS: list[tuple[str, Path, list[Path]]] = [
    ("报告 PDF", ROOT / "report" / "REPORT.pdf",
     [ROOT / "report" / "REPORT.md", *(ROOT / "report" / "figures").glob("fig16*.png")]),
    ("中期报告 PDF（Markdown 版）", ROOT / "report" / "MIDTERM.pdf",
     [ROOT / "report" / "MIDTERM.md", *(ROOT / "report" / "figures").glob("*.png")]),
    ("中期报告 PDF（LaTeX 版）", ROOT / "report" / "latex" / "MIDTERM.pdf",
     [ROOT / "report" / "latex" / "MIDTERM.tex", ROOT / "report" / "latex" / "figs" / "logo.png",
      *(ROOT / "report" / "figures").glob("*.png")]),
    ("汇报幻灯片 pptx", ROOT / "report" / "ppt" / "VitalDB_multimodal_monitoring.pptx",
     deck_inputs()),
    ("汇报幻灯片 PDF", ROOT / "report" / "ppt" / "VitalDB_multimodal_monitoring.pdf",
     [ROOT / "report" / "ppt" / "VitalDB_multimodal_monitoring.pptx",
      ROOT / "tools" / "make_slides_pdf.py"]),
]


def newest(paths: list[Path]) -> tuple[Path | None, float]:
    latest, stamp = None, 0.0
    for path in paths:
        if path.is_file() and path.stat().st_mtime > stamp:
            latest, stamp = path, path.stat().st_mtime
    return latest, stamp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the jobs and exit")
    args = ap.parse_args(argv)
    if args.list:
        for what, target, sources in JOBS:
            print(f"{what:26s} {target.relative_to(ROOT)}  <- {len(sources)} source(s)")
        return 0

    problems = []
    for what, target, sources in JOBS:
        if not target.is_file():
            problems.append(f"{what}: missing {target.relative_to(ROOT)}")
            print(f"{what:26s} MISSING")
            continue
        source, stamp = newest(sources)
        if source is None:
            print(f"{what:26s} no sources to compare")
            continue
        target_stamp = target.stat().st_mtime
        ok = target_stamp >= stamp
        if not ok:
            problems.append(f"{what}: {target.relative_to(ROOT)} is older than "
                            f"{source.relative_to(ROOT)}")
        delta = target_stamp - stamp
        print(f"{what:26s} {'OK  ' if ok else 'STALE'}  target is {delta:+.1f}s vs the "
              f"newest source ({source.name})")

    for item in problems:
        print("  PROBLEM", item)
    if problems:
        print(f"FAIL {len(problems)} rendered artifact(s) are older than their source")
        return 1
    print("OK every rendered artifact is at least as new as what it was rendered from")
    return 0


if __name__ == "__main__":
    sys.exit(main())
