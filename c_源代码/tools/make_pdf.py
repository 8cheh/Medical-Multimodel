"""Render report/REPORT.md to a self-contained PDF.

Pipeline: Markdown -> HTML (figures inlined as base64 data URIs) -> Edge headless
print-to-pdf. Edge is used because pandoc/wkhtmltopdf are absent here and the
installed WeasyPrint cannot load its native pango/gobject libraries.

Heading ids are re-derived with the same slug rules the report's own table of
contents uses, so the in-document links keep working inside the PDF.
"""
import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_MD = os.path.join(ROOT, "report", "REPORT.md")
REPORT_HTML = os.path.join(ROOT, "report", "REPORT.html")
REPORT_PDF = os.path.join(ROOT, "report", "REPORT.pdf")

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 15mm 13mm 16mm 13mm; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Microsoft YaHei", "Segoe UI", "SimHei", sans-serif;
  font-size: 10pt; line-height: 1.62; color: #1a202c; margin: 0;
}
h1 { font-size: 19pt; margin: 0 0 4mm; padding-bottom: 2.5mm;
     border-bottom: 2.5px solid #2b6cb0; }
h2 { font-size: 14.5pt; margin: 8mm 0 3mm; padding: 1.6mm 0 1.6mm 2.6mm;
     border-left: 4px solid #2b6cb0; background: #f5f8fc;
     page-break-after: avoid; page-break-before: auto; }
h3 { font-size: 12pt; margin: 6mm 0 2.4mm; color: #2c5282;
     page-break-after: avoid; }
h1, h2, h3, h4 { page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
a { color: #2b6cb0; text-decoration: none; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt;
       background: #f2f4f7; padding: 0.4mm 1mm; border-radius: 2px; }
pre { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 3px;
      padding: 2.6mm 3mm; font-size: 8.5pt; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; }
img { max-width: 100%; height: auto; display: block; margin: 2.5mm auto;
      page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 8.8pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #cbd5e0; padding: 1.5mm 2mm; text-align: left;
         vertical-align: top; }
th { background: #edf2f7; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote { margin: 3mm 0; padding: 2.4mm 3mm; background: #fffaf0;
             border-left: 3.5px solid #dd6b20; page-break-inside: avoid; }
blockquote p { margin: 1mm 0; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 6mm 0; }
figure { page-break-inside: avoid; }
"""


def find_browser():
    for path in EDGE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def slug(text):
    """Must match tools/normalize_report_headings.py so anchors keep resolving."""
    out = []
    for ch in text.strip().lower():
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        elif ch == " ":
            out.append("-")
    return "".join(out)


def cleanup(path):
    """Best-effort removal of the temporary browser profile."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError as exc:
        print(f"  ! profile cleanup failed: {exc}")


def inline_images(html, base_dir):
    """Replace local image srcs with data URIs so the PDF is self-contained."""
    def repl(match):
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = os.path.join(base_dir, src)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError as exc:
            print(f"  ! cannot read image {path}: {exc}")
            return match.group(0)
        mime = "image/png" if src.lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(raw).decode("ascii")
        return f'src="data:{mime};base64,{b64}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def fix_heading_ids(html):
    """Force heading ids to the same slugs the report's links point at."""
    def repl(match):
        level, text = match.group(1), match.group(2)
        plain = re.sub(r"<[^>]+>", "", text).strip()
        return f'<h{level} id="{slug(plain)}">{text}</h{level}>'

    return re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", repl, html, flags=re.S)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=REPORT_MD)
    ap.add_argument("--html", default=REPORT_HTML)
    ap.add_argument("--pdf", default=REPORT_PDF)
    ap.add_argument("--keep-html", action="store_true", default=True)
    args = ap.parse_args(argv)

    try:
        with open(args.md, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"cannot read {args.md}: {exc}")
        return 1

    body = markdown.markdown(
        text, extensions=["extra", "sane_lists", "fenced_code", "attr_list"],
        output_format="html",
    )
    body = fix_heading_ids(body)
    body = inline_images(body, os.path.dirname(os.path.abspath(args.md)))

    doc = (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
           f"<title>VitalDB 术中低血压预测 + VLA GPU 训练报告</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")

    try:
        with open(args.html, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(doc)
    except OSError as exc:
        print(f"cannot write {args.html}: {exc}")
        return 1
    print(f"wrote {args.html} ({os.path.getsize(args.html):,} B)")

    browser = find_browser()
    if browser is None:
        print("no Edge/Chrome found; HTML written but PDF not generated")
        return 1

    profile = tempfile.mkdtemp(prefix="pdfprofile-")
    uri = "file:///" + os.path.abspath(args.html).replace("\\", "/")
    if os.path.exists(args.pdf):
        try:
            os.remove(args.pdf)
        except OSError as exc:
            print(f"cannot remove old pdf: {exc}")

    cmd = [
        browser, "--headless=new", "--disable-gpu", "--no-sandbox",
        f"--user-data-dir={profile}", "--no-pdf-header-footer",
        "--virtual-time-budget=20000", f"--print-to-pdf={os.path.abspath(args.pdf)}",
        uri,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"browser invocation failed: {type(exc).__name__}: {exc}")
        cleanup(profile)
        return 1
    cleanup(profile)

    if not os.path.exists(args.pdf):
        print(f"PDF not produced (rc={proc.returncode})")
        print(proc.stderr[-1500:] if proc.stderr else "(no stderr)")
        return 1

    print(f"wrote {args.pdf} ({os.path.getsize(args.pdf):,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
