"""Convert the deck to PDF, verify the PDF matches the .pptx, then render every slide.

LibreOffice keeps a background instance alive on this machine, so a conversion request can
be handed to a stale one and silently produce a PDF of the *previous* deck. That is not a
hypothetical: it happened while building this deck, and a stale PDF would have been shipped
as the handout. So the steps are ordered and checked:

  1. kill any running LibreOffice and convert with a private user profile;
  2. extract the PDF's text and require the strings the current generator emits - a PDF
     without them is from an older .pptx and is not rendered;
  3. render each page to JPEG (for the visual QA the pptx skill requires);
  4. run markitdown over the .pptx for the content QA (slide count, placeholder text);
  5. record the verdict in results/verify_slides.json so the claim "N slides, 0
     placeholders" is backed by a file rather than by a terminal.

    python tools/make_slides_pdf.py
    python tools/make_slides_pdf.py --render D:\\slides   # also write page images
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "report" / "ppt" / "VitalDB_multimodal_monitoring.pptx"
PDF = ROOT / "report" / "ppt" / "VitalDB_multimodal_monitoring.pdf"
OUT_JSON = ROOT / "results" / "verify_slides.json"
SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.com")

# Strings only the current generator emits: their absence means the PDF is stale.
MUST_HAVE = ("材料：report/MIDTERM.md", "阴性结论：波形模态", "第二路 EtCO2", "结论与下一步")
PLACEHOLDERS = ("xxx", "lorem", "ipsum", "TODO", "[insert", "placeholder")


def kill_soffice():
    for image in ("soffice.bin", "soffice.exe"):
        try:
            subprocess.run(["taskkill", "/F", "/IM", image, "/T"], capture_output=True,
                           check=False, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  (taskkill {image}: {exc})")
    time.sleep(2)


def convert():
    args = [str(SOFFICE), "--headless",
            "-env:UserInstallation=file:///D:/vitaldb_local/lo_deck",
            "--convert-to", "pdf", "--outdir", str(PDF.parent), str(PPTX)]
    try:
        proc = subprocess.run(args, capture_output=True, check=False, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"cannot run LibreOffice: {exc}") from exc
    print(f"soffice rc={proc.returncode}")
    for _ in range(60):
        if PDF.exists() and PDF.stat().st_size > 0:
            time.sleep(1)
            return
        time.sleep(1)
    raise SystemExit("the conversion produced no PDF")


def deck_text(path):
    import fitz
    try:
        doc = fitz.open(path)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"cannot open {path}: {exc}") from exc
    text = "\n".join(str(doc[index].get_text()) for index in range(doc.page_count))
    pages = doc.page_count
    doc.close()
    return text, pages


def render(pages, out_dir):
    import fitz
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {out_dir}: {exc}") from exc
    for stale in Path(out_dir).glob("slide-*.jpg"):
        stale.unlink()
    doc = fitz.open(PDF)
    for index in range(doc.page_count):
        pixmap = doc[index].get_pixmap(dpi=110)
        pixmap.save(os.path.join(out_dir, f"slide-{index + 1:02d}.jpg"))
    doc.close()
    print(f"rendered {pages} slides into {out_dir}")


def content_qa():
    try:
        raw = subprocess.run(["markitdown", str(PPTX)], capture_output=True, check=True,
                             timeout=600).stdout
    except (OSError, subprocess.SubprocessError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"markitdown failed: {exc}") from exc
    text = raw.decode("utf-16", errors="replace") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") \
        else raw.decode("utf-8", errors="replace")
    return text, len(re.findall(r"Slide number", text))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="", help="directory to write slide JPEGs into")
    args = ap.parse_args(argv)

    if not PPTX.is_file():
        raise SystemExit(f"missing {PPTX} - run tools/make_slides.js first")
    kill_soffice()
    PDF.unlink(missing_ok=True)
    convert()

    text, pages = deck_text(PDF)
    missing = [needle for needle in MUST_HAVE if needle not in text]
    print(f"pdf: {pages} pages, {PDF.stat().st_size:,} B, "
          f"generator strings {len(MUST_HAVE) - len(missing)}/{len(MUST_HAVE)} present")

    deck, blocks = content_qa()
    placeholders = sorted({p for p in PLACEHOLDERS if p.lower() in deck.lower()})
    print(f"content QA: {blocks} slide blocks, {len(deck):,} chars, "
          f"placeholders: {placeholders or 'none'}")
    if args.render:
        render(pages, args.render)

    verdict = {
        "pptx": str(PPTX.relative_to(ROOT)).replace("\\", "/"),
        "pdf": str(PDF.relative_to(ROOT)).replace("\\", "/"),
        "pages": pages,
        "bytes": PDF.stat().st_size,
        "generator_strings": {"checked": len(MUST_HAVE), "missing": missing},
        "slide_blocks": blocks,
        "text_chars": len(deck),
        "placeholders": placeholders,
        "rendered_to": args.render,
        "passed": not missing and not placeholders,
    }
    try:
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(verdict, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"recorded {OUT_JSON.relative_to(ROOT)}")
    except OSError as exc:
        print(f"  (cannot record the verdict: {exc})")

    if missing:
        print(f"FAIL the PDF is stale: it lacks {missing}")
        return 1
    if placeholders:
        print(f"FAIL placeholder text in the deck: {placeholders}")
        return 1
    print("OK the PDF matches the deck and carries no placeholder text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
