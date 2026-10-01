"""Build report/latex/MIDTERM.pdf with Tectonic, then verify what actually landed in it.

The LaTeX report reuses the figures and the numbers of report/MIDTERM.md, so it gets the
same treatment as every other artifact here: compile, then prove the PDF carries the pages,
the Chinese text, the artifact-backed numbers, and none of the template's placeholder
fields. The verdict is written to results/verify_latex_midterm.json.

Tectonic is a single self-contained binary (it fetches TeX Live packages on demand), so the
build does not require a system TeX installation:

    python tools/build_latex_midterm.py                 # compile + verify
    python tools/build_latex_midterm.py --tectonic PATH # point at a specific binary
    python tools/build_latex_midterm.py --verify-only   # skip compilation
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX_DIR = ROOT / "report" / "latex"
TEX = TEX_DIR / "MIDTERM.tex"
PDF = TEX_DIR / "MIDTERM.pdf"
OUT_JSON = ROOT / "results" / "verify_latex_midterm.json"

TECTONIC_CANDIDATES = (
    os.environ.get("TECTONIC", ""),
    shutil.which("tectonic") or "",
    r"D:\vitaldb_local\tectonic\tectonic.exe",
    str(Path.home() / "tectonic" / "tectonic.exe"),
)

# Numbers the report quotes from results/; a PDF without them is not this report.
MUST_HAVE = ["12.48%", "0.9022", "0.9536", "0.8423", "99.85%", "1.13%", "0.9723", "3.74%",
             "0.9889", "0.6527", "0.8787", "0.9202", "0.0820", "0.2011", "87.32%", "93.0%",
             "0.9469", "0.9858", "0.9838", "0.7938", "0.8219", "100.00%", "98.03%", "96.87%",
             "92.65%", "0.0432", "0.0025", "0.0983", "0.0079", "0.0024", "0.0069", "94.49%",
             "0.9100", "0.8349", "0.8848", "0.7954", "+0.0514", "+0.0750", "+0.0895",
             "+0.0051", "+0.0009", "+0.0406", "2.973e-08", "44.9%", "41.7", "0.9537",
             "0.9519", "0.9602", "0.9552", "1,553,854", "3,495", "95,402", "37.13", "67.59",
             "66,375", "32.38", "196", "239 + 96"]
MUST_PHRASE = ["中期检查报告", "模态的价值", "外部验证", "通气不足", "高碳酸血症",
               "指导教师意见", "学院审核意见", "支出科目", "配对", "阴性"]
# Fields the template expects the author to fill: if one survives, the cover is unfinished.
TEMPLATE_PLACEHOLDERS = ["你的名字", "你的学号", "你的学院", "你的专业", "你的班级",
                         "你的指导教师", "课程报告题目", "课程名称：", "简约扁平风"]


def find_tectonic(explicit: str = "") -> str:
    for candidate in (explicit, *TECTONIC_CANDIDATES):
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "no tectonic binary found. Install one with:\n"
        "  curl -L -o tectonic.zip https://github.com/tectonic-typesetting/tectonic/"
        "releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip\n"
        "  or point this script at one with --tectonic PATH (a full TeX Live works too:"
        " xelatex MIDTERM.tex twice)")


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def compile_pdf(binary: str) -> None:
    try:
        proc = subprocess.run([binary, TEX.name], cwd=str(TEX_DIR), capture_output=True,
                              check=False, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"cannot run {binary}: {exc}") from exc
    text = proc.stdout.decode("utf-8", errors="replace")
    overfull = [line for line in text.splitlines() if "Overfull" in line]
    print(f"tectonic rc={proc.returncode}; overfull boxes: {len(overfull)}")
    for line in overfull[:5]:
        print("  ", line.strip()[:120])
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-20:])
        raise SystemExit(f"compilation failed:\n{tail}")


def verify() -> dict:
    import fitz
    try:
        doc = fitz.open(PDF)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"cannot open {PDF}: {exc}") from exc
    pages = doc.page_count
    text = "\n".join(str(doc[index].get_text()) for index in range(pages))
    doc.close()

    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    bad = text.count("\ufffd")
    missing = [needle for needle in MUST_HAVE if needle not in text]
    missing_phrase = [needle for needle in MUST_PHRASE if needle not in text]
    leftovers = [needle for needle in TEMPLATE_PLACEHOLDERS if needle in text]

    copies = read_json(ROOT / "算法实现" / "MANIFEST.json")["n_entries"]
    copies_ok = f"{copies} 个文件副本" in text

    print(f"pages {pages} | CJK glyphs {cjk} | replacement chars {bad}")
    print(f"numbers {len(MUST_HAVE) - len(missing)}/{len(MUST_HAVE)} present"
          + (f" - MISSING {missing}" if missing else ""))
    print(f"phrases {len(MUST_PHRASE) - len(missing_phrase)}/{len(MUST_PHRASE)} present"
          + (f" - MISSING {missing_phrase}" if missing_phrase else ""))
    print(f"template placeholders left: {leftovers or 'none'}")
    print(f"delivery-copy count ({copies}) quoted: {copies_ok}")

    verdict = {
        "tex": str(TEX.relative_to(ROOT)).replace("\\", "/"),
        "pdf": str(PDF.relative_to(ROOT)).replace("\\", "/"),
        "pages": pages,
        "bytes": PDF.stat().st_size,
        "cjk_glyphs": cjk,
        "replacement_chars": bad,
        "numbers": {"checked": len(MUST_HAVE), "missing": missing},
        "phrases": {"checked": len(MUST_PHRASE), "missing": missing_phrase},
        "template_placeholders": leftovers,
        "delivery_copies": {"expected": copies, "quoted": copies_ok},
    }
    verdict["passed"] = (not missing and not missing_phrase and not leftovers
                         and copies_ok and bad == 0 and cjk > 2000)
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(verdict, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"recorded {OUT_JSON.relative_to(ROOT)}")
    except OSError as exc:
        print(f"  (cannot record the verdict: {exc})")
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default="", help="path to a tectonic binary")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    if not TEX.is_file():
        raise SystemExit(f"missing {TEX}")
    if not args.verify_only:
        compile_pdf(find_tectonic(args.tectonic))
    verdict = verify()
    if not verdict["passed"]:
        print("FAIL the LaTeX report does not match what it claims")
        return 1
    print("OK the LaTeX report compiles, carries its numbers, and has no leftover "
          "template fields")
    return 0


if __name__ == "__main__":
    sys.exit(main())
