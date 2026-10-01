"""Verify 算法实现/ against the repository: every copy byte-identical, nothing unlisted.

A delivery folder is only evidence if the reader can prove it still matches the code
and artifacts that produced the report. Three checks:

  1. every MANIFEST entry exists, has the recorded size, and hashes to the recorded
     sha256 *and* to the sha256 of its repository source (so drift in either direction
     fails, not just a corrupted copy);
  2. no file sits in the folder that the manifest does not list and no hand-authored
     document is expected (a stale copy from an earlier run is exactly the kind of
     thing that quietly contradicts the report, so it is an error);
  3. the five hand-authored documents exist, and inventory.json still agrees with the
     live repository and local data (so no README quotes a census that has moved).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "算法实现"
sys.path.insert(0, str(ROOT / "tools"))

AUTHORED = {
    "README.md",
    "a_数据来源/README.md",
    "b_算法说明/README.md",
    "c_源代码/README.md",
    "d_运行结果/README.md",
    "inventory.json",
    "MANIFEST.json",
}
REQUIRED = ["README.md", "a_数据来源/README.md", "b_算法说明/README.md",
            "c_源代码/README.md", "d_运行结果/README.md"]
# Linters write caches beside the files they read. They are not deliverables, and
# flagging them would make this check fail for a reason that has nothing to do with
# the folder's contents.
IGNORED_DIRS = {"__pycache__", ".ruff_cache"}
# These artifacts are rewritten by every verification run (durations, byte counts, PDF
# hashes), so a copy in the package can legitimately be older than the live file. What is
# waived is only the "still identical to the source right now" direction; integrity is
# still provable against the sha256 recorded in the manifest at copy time. The folder's
# README says so, so a reader is not misled about what the copies represent.
VOLATILE = {
    "results/check_all.json",
    "results/check_extra_docs.json",
    "results/verify_pdf.json",
    "results/verify_pdf_MIDTERM.json",
    "results/verify_slides.json",
    "results/verify_latex_midterm.json",
}


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def read_text(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def sha256(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            h = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
            return h.hexdigest()
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def main() -> int:
    problems: list[str] = []
    if not (OUT / "MANIFEST.json").is_file():
        raise SystemExit("no MANIFEST.json - run tools/build_algorithm_delivery.py first")
    entries = read_json(OUT / "MANIFEST.json")["entries"]

    # 1. integrity, in both directions
    waived = 0
    for entry in entries:
        dst, src = OUT / entry["dst"], ROOT / entry["src"]
        if not dst.is_file():
            problems.append(f"missing copy: {entry['dst']}")
            continue
        if dst.stat().st_size != entry["bytes"]:
            problems.append(f"size changed: {entry['dst']}")
        if sha256(dst) != entry["sha256"]:
            problems.append(f"copy no longer matches its manifest hash: {entry['dst']}")
        if not src.is_file():
            problems.append(f"source vanished: {entry['src']}")
        elif entry["src"] in VOLATILE:
            waived += 1
        elif sha256(src) != entry["sha256"]:
            problems.append(f"source moved on since the copy was made: {entry['src']}")

    # 2. nothing unlisted (a stale copy is a contradiction waiting to happen)
    listed = {e["dst"] for e in entries}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and not IGNORED_DIRS.intersection(path.parts):
            rel = str(path.relative_to(OUT)).replace("\\", "/")
            if rel not in listed and rel not in AUTHORED:
                problems.append(f"unlisted file in the delivery folder: {rel}")

    # 3. authored documents present, inventory still true
    for rel in REQUIRED:
        path = OUT / rel
        if not path.is_file():
            problems.append(f"missing document: {rel}")
        elif path.stat().st_size < 400:
            problems.append(f"document looks empty: {rel} ({path.stat().st_size} B)")

    inv_path = OUT / "inventory.json"
    if inv_path.is_file():
        import build_algorithm_delivery as builder  # noqa: PLC0415 - same-repo helper
        fresh = builder.build_inventory()
        stored = read_json(inv_path)
        for section in ("repo", "channels", "local_data"):
            if stored.get(section) != fresh.get(section):
                problems.append(f"inventory.{section} is stale - re-run "
                                f"tools/build_algorithm_delivery.py")
    else:
        problems.append("missing inventory.json")

    # The index states the folder's size; if that drifts, a reader counts files that are
    # not there (or misses ones that are).
    top_readme = OUT / "README.md"
    if top_readme.is_file() and f"合计 {len(entries)} 个副本" not in read_text(top_readme):
        problems.append(f"算法实现/README.md does not state 合计 {len(entries)} 个副本")

    n_files = sum(1 for p in OUT.rglob("*") if p.is_file()
                  and not IGNORED_DIRS.intersection(p.parts))
    print(f"manifest entries: {len(entries)}; files checked: {n_files}; "
          f"build-time snapshots (not re-checked against the live source): {waived}")
    for item in problems:
        print("  PROBLEM", item)
    if problems:
        print(f"FAIL {len(problems)} problem(s) in 算法实现/")
        return 1
    print("OK 算法实现/ is byte-identical to the repository, complete and internally "
          "consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
