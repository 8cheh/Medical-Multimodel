"""Verify report/REPORT.pdf actually contains what the report claims.

Checks page count, embedded figures, CJK text integrity (a font-substitution
failure shows up as tofu boxes or mojibake here), the headline numbers, and that
the table-of-contents links point at real destinations.

Optionally rasterises pages to PNG so the layout can be inspected by eye.
"""
import argparse
import json
import os
import re
import sys

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "report", "REPORT.pdf")

# Numbers the report asserts; each must survive into the rendered PDF.
KEY_NUMBERS = [
    "6157", "125.58", "134,840,025,296", "0.9440", "0.9537",
    "0.9326", "0.9037", "0.947909", "0.947709", "0.951586", "0.950724",
    "3000", "906,712,520", "2,740,247", "117", "12 bpm", "3282", "1,400,689",
    # part four: clinical baseline, longer horizons, optimisation (results/p0_*.json)
    "1392", "99.85%", "1.26", "1.72", "0.9022", "0.7954", "0.8848",
    "+0.0514", "+0.0895", "0.8917", "+0.0069", "158",
    # part six: device registry, multi-endpoint, waveforms-vs-endpoint
    "196", "0.2445", "0.9723", "99.34%", "0.8851", "0.9889", "+0.0000",
    "+0.0002", "0.5179",
    # part six: the respiratory endpoint, where the matching modality did pay
    "0.9202", "0.2011", "0.6527", "0.2954", "+0.0406", "+0.1184", "0.53%",
]
KEY_PHRASES = [
    "图文并茂报告", "PPG", "SmolVLA", "阴性结果", "灵敏度",
    "时间对齐", "多数类基线", "End of training", "工作点",
    # part four headings and the claims they must keep carrying
    "临床基线", "更长前瞻", "算法优化", "长窗", "配对", "HPI 未复现",
    "当前 MAP", "边界修正",
    # part six: the platform claims
    "多设备", "通道注册表", "被监测终点", "心动过缓", "神经肌肉阻滞",
    "报警策略", "监护显示原型", "通气不足", "流式推理",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--render", default="", help="directory to write page PNGs into")
    ap.add_argument("--max-render", type=int, default=4)
    ap.add_argument("--expect-numbers", default="",
                    help="comma-separated numbers that must appear (overrides default)")
    ap.add_argument("--expect-phrases", default="",
                    help="comma-separated phrases that must appear (overrides default)")
    ap.add_argument("--structural-only", action="store_true",
                    help="skip content expectations; check pages/images/CJK/links only")
    ap.add_argument("--min-links", type=int, default=-1,
                    help="required internal links; default is 10 for the report profile, 0 otherwise")
    args = ap.parse_args(argv)

    key_numbers = [s.strip() for s in args.expect_numbers.split(",") if s.strip()]
    key_phrases = [s.strip() for s in args.expect_phrases.split(",") if s.strip()]
    if not key_numbers and not key_phrases and not args.structural_only:
        key_numbers, key_phrases = KEY_NUMBERS, KEY_PHRASES
    if args.structural_only:
        key_numbers, key_phrases = [], []

    # A paper has no clickable figure index, so only the report profile demands links.
    report_profile = (key_numbers == KEY_NUMBERS and key_phrases == KEY_PHRASES)
    min_links = args.min_links if args.min_links >= 0 else (10 if report_profile else 0)

    if not os.path.exists(args.pdf):
        print(f"missing {args.pdf}")
        return 1

    try:
        doc = fitz.open(args.pdf)
    except (OSError, RuntimeError) as exc:
        print(f"cannot open {args.pdf}: {exc}")
        return 1

    failures = []
    print(f"pages      : {doc.page_count}")
    print(f"size       : {os.path.getsize(args.pdf):,} B")

    full_text = []
    images = 0
    for page in doc:
        full_text.append(page.get_text())
        images += len(page.get_images(full=True))
    text = "\n".join(full_text)
    print(f"images     : {images} embedded image placements")
    print(f"text chars : {len(text):,}")

    # CJK must be real glyphs, not replacement characters.
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    bad = text.count("\ufffd")
    print(f"CJK glyphs : {cjk:,}   replacement chars: {bad}")
    if cjk < 2000:
        failures.append(f"too few CJK glyphs ({cjk}) - font substitution likely failed")
    if bad > 0:
        failures.append(f"{bad} replacement characters - encoding problem")

    missing = [k for k in key_numbers if k not in text]
    print(f"key numbers: {len(key_numbers) - len(missing)}/{len(key_numbers)} present")
    if missing:
        failures.append(f"missing numbers: {missing}")

    missing_p = [k for k in key_phrases if k not in text]
    print(f"key phrases: {len(key_phrases) - len(missing_p)}/{len(key_phrases)} present")
    if missing_p:
        failures.append(f"missing phrases: {missing_p}")

    # Internal links must land on a real page. Browsers emit these as LINK_NAMED
    # (kind=4), not LINK_GOTO, so detect them by having a destination page.
    internal = external = broken = 0
    for page in doc:
        for link in page.get_links():
            target = link.get("page", -1)
            if target is not None and target >= 0:
                internal += 1
                if target >= doc.page_count:
                    broken += 1
            elif link.get("uri"):
                external += 1
    print(f"links      : internal={internal} external={external} broken={broken}")
    check_min = min_links
    if internal < check_min:
        failures.append(f"only {internal} internal links, expected >= {check_min}")
    if broken:
        failures.append(f"{broken} internal links point off-document")

    if args.render:
        try:
            os.makedirs(args.render, exist_ok=True)
        except OSError as exc:
            print(f"cannot create {args.render}: {exc}")
        else:
            for i in range(min(args.max_render, doc.page_count)):
                pix = doc[i].get_pixmap(dpi=110)
                out = os.path.join(args.render, f"page{i + 1:02d}.png")
                try:
                    pix.save(out)
                    print(f"  rendered {out}")
                except (OSError, RuntimeError) as exc:
                    print(f"  ! cannot render page {i + 1}: {exc}")

    # Leave the verdict behind as an artifact: the report and the mid-term report both
    # quote "N pages / N figures / N numbers / 0 replacement characters", and a quoted
    # number has to come from a file rather than from a terminal's scrollback.
    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    record = {
        "pdf": os.path.relpath(args.pdf, ROOT).replace("\\", "/"),
        "pages": doc.page_count,
        "bytes": os.path.getsize(args.pdf),
        "images": images,
        "text_chars": len(text),
        "cjk_glyphs": cjk,
        "replacement_chars": bad,
        "key_numbers": {"checked": len(key_numbers), "missing": missing},
        "key_phrases": {"checked": len(key_phrases), "missing": missing_p},
        "links": {"internal": internal, "external": external, "broken": broken},
        "failures": failures,
        "passed": not failures,
    }
    out_json = os.path.join(ROOT, "results",
                            "verify_pdf.json" if stem == "REPORT"
                            else f"verify_pdf_{stem}.json")
    try:
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"recorded   : {os.path.relpath(out_json, ROOT)}")
    except OSError as exc:
        print(f"  (cannot record the verdict: {exc})")

    doc.close()
    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PDF verification PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
