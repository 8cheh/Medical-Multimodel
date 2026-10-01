"""Assemble the 算法实现/ delivery folder: data sources, algorithm notes, source code, results.

The report cites artifacts, so a delivery folder has to ship them. This copies the
pipeline that produced every reported number into one tree, records a sha256 for each
copy, and writes 算法实现/inventory.json - the census the folder's READMEs quote, so no
README figure is a memory. `tools/check_algorithm_delivery.py` re-verifies all of it.

The four `*/README.md` files are authored by hand (prose); this script never touches
them, so re-running it cannot clobber the writing.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "算法实现"
LOCAL = Path(r"D:\vitaldb_local")

# --- what gets copied, grouped by the subfolder it lands in -------------------
DATA = [  # a. 数据来源: the evidence for where every number comes from
    "results/channel_registry.json",
    "results/func_download.json",
    "results/ppg_local_store.md",
    "results/abp_local_store.md",
    "results/ppg_manifest.json",
    "results/disk_volumes.txt",
]
ALGO = [  # b. 算法说明: the one figure that draws the pipeline end to end
    "report/figures/fig16_architecture.png",
]
RESULT_JSON = [
    "results/p0_report.json",
    "results/p0_report_h600.json",
    "results/p0_report_h900.json",
    "results/p0_report_h900opt.json",
    "results/p0_report_hypoxemia.json",
    "results/p0_report_bradycardia.json",
    "results/p0_report_hypoventilation.json",
    "results/p0_report_hypercapnia.json",
    "results/p0_report_multimodal_h300.json",
    "results/p0_report_multimodal_h900.json",
    "results/p0_report_hypoxemia_mm.json",
    "results/p0_report_ppg_h300.json",
    "results/p0_report_ppg_h600.json",
    "results/p0_report_ppg_h900.json",
    "results/alarm_pareto.json",
    "results/calibration_hypotension.json",
    "results/calibration_hypoxemia.json",
    "results/operating_points.json",
    "results/quality_gate.json",
    "results/p0_significance.json",
    "results/report_main.json",
    "results/report_fused.json",
    "results/report_fused_final.json",
    "results/check_numbers.json",
    "results/check_extra_docs.json",
    "results/check_all.json",
    "results/verify_pdf.json",
    "results/verify_pdf_MIDTERM.json",
    "results/verify_slides.json",
]
RESULT_LOG = [
    "results/p0_dataset.log",
    "results/p0_h600.log",
    "results/p0_h900.log",
    "results/p0_h900opt.log",
    "results/p0_multimodal_h300.log",
    "results/p0_lead_h300.log",
    "results/p0_hypoxemia.log",
    "results/p0_hypoxemia_mm.log",
    "results/p0_bradycardia.log",
    "results/stream_check.log",
    "results/verify_seeds.log",
    "results/verify_onset.log",
    "results/verify_ppg_align.log",
    "results/run_main.log",
]


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
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


def tree_stat(root: Path, pattern: str) -> dict:
    """File count and total bytes for one glob, without loading anything."""
    n = 0
    total = 0
    if root.is_dir():
        for path in root.glob(pattern):
            if path.is_file():
                n += 1
                total += path.stat().st_size
    return {"files": n, "bytes": total, "gb": round(total / 1e9, 2)}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    except OSError as exc:
        raise SystemExit(f"cannot write {path}: {exc}") from exc


def prune_caches(root: Path) -> int:
    """Linters drop caches next to the files they read; they are not deliverables."""
    pruned = 0
    for name in ("__pycache__", ".ruff_cache"):
        for path in root.rglob(name):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    pruned += 1
            except OSError:  # a cache we cannot remove is not a delivery failure
                continue
    return pruned


def registry_census() -> dict:
    reg = read_json(ROOT / "results" / "channel_registry.json")
    devices = reg.get("devices") or {}
    functions = reg.get("functions") or {}
    local = reg.get("locally_available") or {}
    return {"indexed": reg.get("channels_total", 0), "local": len(local),
            "devices": len(devices), "device_names": sorted(devices),
            "function_groups": len(functions), "local_names": sorted(local)[:60]}


def build_inventory() -> dict:
    py = [p for p in sorted(ROOT.rglob("*.py"))
          if not {"archive", "算法实现", ".venv"}.intersection(p.parts)]
    inv = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": {"python_files": len(py),
                 "python_lines": sum(count_lines(p) for p in py),
                 "results_artifacts": len(list((ROOT / "results").glob("*"))),
                 "report_figures": len(list((ROOT / "report" / "figures").glob("*.png")))},
        "channels": registry_census(),
        "local_copy_root": str(LOCAL),
        "local_data": {
            "numeric_csv": tree_stat(LOCAL / "numeric", "*.csv"),
            "abp_npy": tree_stat(LOCAL / "abp_npy", "*.npy"),
            "abp_features": tree_stat(LOCAL / "abp_features_full", "*.parquet"),
            "ppg_features": tree_stat(LOCAL / "ppg_features_full", "*.parquet"),
            "windows_parquet": tree_stat(LOCAL / "processed", "windows.parquet"),
        },
    }
    dl = read_json(ROOT / "results" / "func_download.json")
    inv["download"] = {k: dl[k] for k in
                       ("channels_requested", "files_on_disk", "bytes_on_disk", "gb_on_disk",
                        "downloader_ok", "downloader_fail", "downloader_mb")
                       if k in dl}
    inv["report_number_check"] = read_json(ROOT / "results" / "check_numbers.json")
    return inv


def main() -> int:
    plan: list[tuple[Path, Path]] = []
    for rel in DATA:
        plan.append((ROOT / rel, OUT / "a_数据来源" / Path(rel).name))
    for rel in ALGO:
        plan.append((ROOT / rel, OUT / "b_算法说明" / Path(rel).name))
    for path in sorted((ROOT / "remote").glob("*.py")):
        plan.append((path, OUT / "c_源代码" / "remote" / path.name))
    for path in sorted((ROOT / "tools").glob("*.py")):
        plan.append((path, OUT / "c_源代码" / "tools" / path.name))
    plan.append((ROOT / "requirements.txt", OUT / "c_源代码" / "requirements.txt"))
    # The deck's generator belongs with the code it cites: every number on a slide is read
    # from results/*.json by this script, so shipping it keeps the slides reproducible too.
    plan.append((ROOT / "tools" / "make_slides.js", OUT / "c_源代码" / "tools" / "make_slides.js"))
    for rel in RESULT_JSON:
        plan.append((ROOT / rel, OUT / "d_运行结果" / Path(rel).name))
    for rel in RESULT_LOG:
        plan.append((ROOT / rel, OUT / "d_运行结果" / "logs" / Path(rel).name))
    for path in sorted((ROOT / "report" / "figures").glob("*.png")):
        plan.append((path, OUT / "d_运行结果" / "figures" / path.name))

    entries = []
    missing = []
    for src, dst in plan:
        if not src.is_file():
            missing.append(str(src.relative_to(ROOT)))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = sha256(dst)
        if digest != sha256(src):
            raise SystemExit(f"copy is not byte-identical: {src}")
        entries.append({"src": str(src.relative_to(ROOT)).replace("\\", "/"),
                        "dst": str(dst.relative_to(OUT)).replace("\\", "/"),
                        "bytes": dst.stat().st_size, "sha256": digest})

    inventory = build_inventory()
    pruned = prune_caches(OUT)
    write_json(OUT / "inventory.json", inventory)
    write_json(OUT / "MANIFEST.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(ROOT),
        "note": "every entry is byte-identical to its source; the */README.md files are "
                "authored by hand and deliberately absent here",
        "n_entries": len(entries),
        "entries": entries,
    })

    print(f"copied {len(entries)} files into {OUT}"
          + (f" (pruned {pruned} linter cache dir(s))" if pruned else ""))
    if missing:
        print(f"  !! {len(missing)} planned sources did not exist:")
        for item in missing:
            print(f"     {item}")
    print(json.dumps({k: inventory[k] for k in
                      ("repo", "channels", "local_data", "download")},
                     ensure_ascii=False, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
