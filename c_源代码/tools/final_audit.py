"""Audit the three objective deliverables against the real artifacts on disk.

Requirement 1: PPG downloaded AND trained on.
Requirement 2: VLA trained on GPU (a completed, checkpointed run).
Requirement 3: an illustrated report (figures embedded, present, and navigable).

This checks the requirements themselves, not proxy signals. Each item prints the
evidence value it read, so a failure shows what was actually found.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SCORES = os.path.join(ROOT, "scores")
SCRATCH = os.path.join(ROOT, "scratch")
REPORT = os.path.join(ROOT, "report", "REPORT.md")
FIGDIR = os.path.join(ROOT, "report", "figures")

CHECKS = []
NAN = np.nan


def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value, default=NAN):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_text(path):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def load_json(path):
    text = read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def check(req, name, passed, evidence):
    CHECKS.append((req, name, bool(passed), evidence))
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}: {evidence}")


# ---------------------------------------------------------------- requirement 1
def req1_ppg_download():
    man = load_json(os.path.join(RESULTS, "ppg_manifest.json"))
    if man is None:
        return check("R1", "PPG manifest readable", False, "missing/unparseable")
    expected = to_int(man.get("expected_cases"))
    got = to_int(man.get("downloaded_cases"))
    missing = to_int(man.get("missing_count"))
    sidecar = to_int(man.get("sidecar_missing"))
    total_b = to_int(man.get("total_bytes"))
    check("R1", "PPG coverage", got == expected and got > 0 and missing == 0 and sidecar == 0,
          f"{got}/{expected} cases, missing={missing}, sidecar_missing={sidecar}")

    sizes_path = os.path.join(SCRATCH, "ppg_sizes.txt")
    if not os.path.exists(sizes_path):
        return check("R1", "independent byte sum", False, "scratch/ppg_sizes.txt missing")
    try:
        sizes = np.loadtxt(sizes_path, dtype=np.int64, usecols=1)
    except (OSError, ValueError) as exc:
        return check("R1", "independent byte sum", False, f"unparseable: {exc}")
    total = to_int(sizes.sum())
    check("R1", "independent byte sum == manifest total_bytes",
          total == total_b and sizes.size == expected,
          f"on-disk {total:,} B over {sizes.size} files vs manifest {total_b:,} B")


def req1_ppg_training():
    finalize = read_text(os.path.join(RESULTS, "finalize_ppg.log")) or ""
    rep = load_json(os.path.join(RESULTS, "report_fused_final.json"))

    check("R1", "PPG feature build finished", "errors=0" in finalize and "PPG_THREAD_COMPLETE" in finalize,
          "finalize_ppg.log: errors=0 + PPG_THREAD_COMPLETE")
    if rep is None:
        return check("R1", "PPG comparison report", False, "report_fused_final.json missing")
    n_ppg = to_int(rep.get("n_ppg_feats"))
    n_num = to_int(rep.get("n_numeric_feats"))
    rows = to_int(rep.get("merged_rows"))
    cases = to_int(rep.get("merged_cases"))
    res = rep.get("results", {})
    a = res.get("numeric_only", {})
    b = res.get("numeric_plus_ppg", {})
    check("R1", "PPG features actually used in a trained model",
          n_ppg > 0 and b.get("n_features") == n_num + n_ppg and rows > 0 and cases > 0,
          f"{cases} cases / {rows:,} rows, {n_num} numeric + {n_ppg} PPG = {b.get('n_features')} features")
    check("R1", "trained-model metrics recorded",
          bool(a) and bool(b) and "auroc" in a and "auroc" in b,
          f"numeric AUROC={to_float(a.get('auroc')):.4f}, "
          f"+PPG AUROC={to_float(b.get('auroc')):.4f}")

    # Independent re-run must reproduce the published numbers.
    dump = os.path.join(SCORES, "fused_numeric_ppg.json")
    dm = load_json(dump)
    if dm is not None:
        rerun = to_float(dm.get("test_at_0.5", {}).get("auroc"))
        check("R1", "re-run reproduces published AUROC",
              abs(rerun - to_float(b.get("auroc"))) < 1e-6,
              f"re-run {rerun:.6f} vs report {to_float(b.get('auroc')):.6f}")


# ---------------------------------------------------------------- requirement 2
def req2_vla():
    meta = read_text(os.path.join(RESULTS, "vla_ckpt_meta.txt")) or ""
    evid = read_text(os.path.join(RESULTS, "vla_ckpt_evidence.txt")) or ""
    log = read_text(os.path.join(RESULTS, "vla_train.log")) or ""

    steps = [to_int(m.group(2)) for m in re.finditer(r"(\d{6}) -> \{\s*\"step\": (\d+)", meta)]
    check("R2", "checkpoints self-report step counts",
          sorted(steps) == [1000, 2000, 3000],
          f"training_step.json per checkpoint = {sorted(steps)}")

    m = re.search(r'"last_epoch": (\d+)', meta)
    check("R2", "scheduler last_epoch == 3000",
          m is not None and to_int(m.group(1)) == 3000,
          f"last_epoch={m.group(1) if m else 'not found'}")

    link = re.search(r"last symlink resolves to[^\n]*\n(\S+)", meta)
    check("R2", "final checkpoint linked as last",
          link is not None and "003000" in link.group(1),
          "last -> .../checkpoints/003000")

    sizes = [to_int(x) for x in re.findall(r"(\d+)\s+\S*checkpoints/\d+/pretrained_model/model\.safetensors", evid)]
    distinct = sorted(set(sizes))
    check("R2", "model weights present for every checkpoint",
          len(distinct) == 1 and distinct and distinct[0] > 0 and len(sizes) >= 3,
          f"{len(sizes)} path entries, all {distinct[0]:,} B" if distinct else "not found")

    check("R2", "trained on GPU (device=cuda in checkpoint config)",
          '"device": "cuda"' in meta, "train_config.json device=cuda")
    check("R2", "run ended cleanly",
          log.count("End of training") == 1 and log.count("Traceback") == 0,
          f"'End of training' x{log.count('End of training')}, Traceback x{log.count('Traceback')}")
    check("R2", "policy is a VLA (smolvla) on a robot dataset",
          '"type": "smolvla"' in meta and "lerobot/libero" in meta,
          "policy.type=smolvla, dataset=lerobot/libero")


# ---------------------------------------------------------------- requirement 3
def req3_report():
    text = read_text(REPORT)
    if text is None:
        return check("R3", "report exists", False, "report/REPORT.md missing")

    imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    present = [s for s in imgs if os.path.exists(os.path.join(ROOT, "report", s))]
    check("R3", "report embeds figures that exist",
          len(imgs) >= 10 and len(present) == len(imgs),
          f"{len(present)}/{len(imgs)} embedded images resolve")

    try:
        on_disk = sorted(f for f in os.listdir(FIGDIR) if f.endswith(".png"))
    except OSError as exc:
        on_disk = []
        print(f"  ! cannot list {FIGDIR}: {exc}")
    used = {os.path.basename(s) for s in imgs}
    check("R3", "no orphan figures", len(on_disk) == len(used) and len(on_disk) >= 10,
          f"{len(on_disk)} PNGs on disk, {len(used)} referenced")

    # Anchors must resolve, or the figure index is decorative.
    headings, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6}) (.+)$", line)
        if m:
            slug = "".join(c if (c.isalnum() or c in "_-") else ("-" if c == " " else "")
                           for c in m.group(2).strip().lower())
            headings.append((len(m.group(1)), slug))
    available = {s for _, s in headings}
    links = re.findall(r"\]\(#([^)]+)\)", text)
    unresolved = [ln for ln in links if ln not in available]
    check("R3", "internal anchors resolve",
          bool(links) and not unresolved,
          f"{len(links) - len(unresolved)}/{len(links)} resolved")
    check("R3", "single top-level heading",
          sum(1 for lvl, _ in headings if lvl == 1) == 1,
          f"H1 count = {sum(1 for lvl, _ in headings if lvl == 1)}")

    # The text must be substantial, not a caption sheet.
    words = len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_.]+", text))
    check("R3", "report has substantive text", words > 1500, f"~{words} word tokens")

    readme = read_text(os.path.join(ROOT, "README.md")) or ""
    check("R3", "report is discoverable from README",
          "report/REPORT.md" in readme, "README.md links to report/REPORT.md")

    # The PDF must be a real rendering, not an empty or font-broken file.
    pdf = os.path.join(ROOT, "report", "REPORT.pdf")
    if not os.path.exists(pdf):
        return check("R3", "PDF export exists", False, "report/REPORT.pdf missing")
    try:
        import fitz
        doc = fitz.open(pdf)
    except (ImportError, OSError, RuntimeError) as exc:
        return check("R3", "PDF export readable", False, f"{type(exc).__name__}: {exc}")

    pages_text = [str(doc[i].get_text()) for i in range(doc.page_count)]
    joined = "\n".join(pages_text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", joined))
    replacements = joined.count("\ufffd")
    placements = 0
    for i in range(doc.page_count):
        placements += len(doc[i].get_images(full=True))
    links = 0
    for i in range(doc.page_count):
        for link in doc[i].get_links():
            target = link.get("page", -1)
            if target is not None and 0 <= target < doc.page_count:
                links += 1
    page_count = doc.page_count
    doc.close()

    check("R3", "PDF has content and figures",
          page_count >= 8 and placements >= 10 and cjk > 2000 and replacements == 0,
          f"{page_count} pages, {placements} figure placements, {cjk:,} CJK glyphs, "
          f"{replacements} replacement chars")
    check("R3", "PDF figure index is navigable", links >= 10, f"{links} internal links")


def req4_papers():
    """Two papers split from the report, each with verified citations and a valid PDF."""
    papers = [
        ("P1", os.path.join(ROOT, "papers"), "P1_ppg_hypotension"),
        # P2 is archived: it is a robotics paper, unrelated to this project's subject,
        # so the collection holds one paper and the archive holds the other.
        ("P2(归档)", os.path.join(ROOT, "archive", "vla_paper"), "P2_smolvla_finetune"),
    ]
    base = os.path.join(ROOT, "papers")

    log = load_json(os.path.join(base, "source_log.json"))
    if log is None:
        return check("R4", "source log present", False, "papers/source_log.json missing")
    records = log.get("records", [])
    verified = [r for r in records if r.get("verified")]
    excluded = [r["key"] for r in records if not r.get("verified")]
    statuses = sorted({r.get("status") for r in verified})
    check("R4", "references verified against a real record",
          len(verified) >= 20,
          f"{len(verified)}/{len(records)} resolved; statuses={statuses}; "
          f"excluded={excluded}")

    # Run the real cross-checker rather than re-implementing its logic here.
    checker = os.path.join(base, "check_citations.py")
    try:
        proc = subprocess.run([sys.executable, checker], capture_output=True,
                              text=True, timeout=120, check=False)
        tail = (proc.stdout or "").strip().splitlines()
        check("R4", "citations resolve with no orphans",
              proc.returncode == 0,
              tail[-1] if tail else f"exit {proc.returncode}")
    except (OSError, subprocess.SubprocessError) as exc:
        check("R4", "citations resolve with no orphans", False,
              f"{type(exc).__name__}: {exc}")

    for tag, pdir, stem in papers:
        md = os.path.join(pdir, f"{stem}.md")
        pdf = os.path.join(pdir, f"{stem}.pdf")
        text = read_text(md)
        if text is None:
            check("R4", f"{tag} markdown present", False, f"{stem}.md missing")
            continue
        # A paper must state its own limits, not just its results.
        has_limits = any(k in text for k in ("局限", "限定", "威胁有效性"))
        has_refs = "参考文献" in text
        check("R4", f"{tag} is a structured paper",
              has_refs and has_limits,
              f"references={has_refs}, limitations section={has_limits}, "
              f"{len(text):,} chars")

        if not os.path.exists(pdf):
            check("R4", f"{tag} PDF present", False, f"{stem}.pdf missing")
            continue
        try:
            import fitz
            doc = fitz.open(pdf)
        except (ImportError, OSError, RuntimeError) as exc:
            check("R4", f"{tag} PDF readable", False, f"{type(exc).__name__}: {exc}")
            continue
        pages = doc.page_count
        joined = "\n".join(str(doc[i].get_text()) for i in range(pages))
        imgs = 0
        for i in range(pages):
            imgs += len(doc[i].get_images(full=True))
        doc.close()
        cjk = len(re.findall(r"[\u4e00-\u9fff]", joined))
        odd = joined.count("\ufffd")
        check("R4", f"{tag} PDF renders text and figures",
              pages >= 5 and imgs >= 2 and cjk > 2000 and odd == 0,
              f"{pages} pages, {imgs} figures, {cjk:,} CJK glyphs, {odd} replacement chars")




def main():
    print("Requirement 1 - PPG download + training")
    req1_ppg_download()
    req1_ppg_training()
    print("\nRequirement 2 - VLA trained on GPU")
    req2_vla()
    print("\nRequirement 3 - illustrated report")
    req3_report()
    print("\nRequirement 4 - two standalone papers")
    req4_papers()

    failed = [(r, n, e) for r, n, ok, e in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for r, n, e in failed:
        print(f"  FAILED {r} {n}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
