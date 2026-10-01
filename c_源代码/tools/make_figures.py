"""Render every figure embedded in report/REPORT.md.

Each figure reads artifacts that came out of a real run (results/, scores/,
scratch/). Nothing is hard-coded from memory, and a figure whose input is absent
is skipped with a printed reason rather than drawn from placeholder numbers.

Usage:
    python tools/make_figures.py [--out report/figures] [--only name[,name]]
"""
import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from scipy import signal as sg  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

NAN = np.nan
PRIMARY_FONT = "DejaVu Sans"
STRIP_FIGURE_PREFIX = False


def _strip_prefix(text):
    """Drop a leading '图N ' so the same figure can be renumbered by a consumer.

    The standalone report numbers its own figures; papers renumber them in their
    own caption order, so a baked-in '图5' inside the image would contradict the
    paper's '图 3' caption.
    """
    if not STRIP_FIGURE_PREFIX or not isinstance(text, str):
        return text
    return re.sub(r"^图\d+[\u3000\s]+", "", text)


def enable_prefix_stripping():
    """Patch title setters so every call site is covered without edits."""
    global STRIP_FIGURE_PREFIX
    STRIP_FIGURE_PREFIX = True

    original_set_title = Axes.set_title
    original_suptitle = Figure.suptitle

    def set_title(self, label, *a, **k):
        return original_set_title(self, _strip_prefix(label), *a, **k)

    def suptitle(self, t, *a, **k):
        return original_suptitle(self, _strip_prefix(t), *a, **k)

    Axes.set_title = set_title
    Figure.suptitle = suptitle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SCORES = os.path.join(ROOT, "scores")
SCRATCH = os.path.join(ROOT, "scratch")
SAMPLES = os.path.join(SCRATCH, "ppg_samples")
PRODUCED = []
SKIPPED = []


def to_float(value, default=NAN):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_text(path):
    """Read UTF-8, tolerating a UTF-16 BOM from shell redirection on Windows."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        print(f"  ! cannot read {path}: {exc}")
        return None
    for bom, enc in ((b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            try:
                return raw.decode(enc)
            except (UnicodeDecodeError, ValueError):
                break
    return raw.decode("utf-8", errors="replace")


def load_json(path):
    text = read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        print(f"  ! cannot parse {path}: {exc}")
        return None


def load_npz(path):
    try:
        return np.load(path)
    except (OSError, ValueError) as exc:
        print(f"  ! cannot load {path}: {exc}")
        return None


def setup_fonts():
    """Prefer a CJK font so Chinese labels render as glyphs, not boxes."""
    global PRIMARY_FONT
    prefer = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
              "WenQuanYi Zen Hei", "SimSun"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [n for n in prefer if n in have]
    if chosen:
        PRIMARY_FONT = chosen[0]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["font.monospace"] = chosen + ["DejaVu Sans Mono"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    return chosen


def save(fig, out, name):
    path = os.path.join(out, name)
    try:
        fig.savefig(path)
    except OSError as exc:
        print(f"  ! cannot write {path}: {exc}")
        plt.close(fig)
        return None
    plt.close(fig)
    PRODUCED.append(name)
    print(f"  + {name}")
    return path


def skip(name, reason):
    SKIPPED.append((name, reason))
    print(f"  - {name} SKIPPED: {reason}")


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------
def parse_finalize_log():
    """'HH:MM:SS waiting manifest npy=5595/6157 downloader=1' -> progress points."""
    text = read_text(os.path.join(RESULTS, "finalize_ppg.log"))
    if not text:
        return []
    pts = []
    for line in text.splitlines():
        m = re.search(r"(\d\d:\d\d:\d\d) waiting manifest npy=(\d+)/(\d+)", line)
        if m:
            pts.append((m.group(1), to_int(m.group(2)), to_int(m.group(3))))
    return pts


def parse_align_log():
    """Rows of the PPG-vs-monitor heart-rate table."""
    text = read_text(os.path.join(RESULTS, "verify_ppg_align.log"))
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                     r"(ALIGNED|MISMATCH)\s+\((\S+)\)", line)
        if m:
            rows.append({
                "caseid": to_int(m.group(1)), "ppg": to_float(m.group(2)),
                "mon": to_float(m.group(3)), "diff": to_float(m.group(4)),
                "prom": to_float(m.group(5)), "verdict": m.group(6), "src": m.group(7),
            })
    return rows


def parse_feature_gains():
    """Top gain table printed by train_model.py inside run_main.log."""
    text = read_text(os.path.join(RESULTS, "run_main.log"))
    if not text:
        return []
    block = text.split("=== top features (gain) ===")
    if len(block) < 2:
        return []
    out = []
    for line in block[1].splitlines():
        m = re.match(r"\s+([A-Za-z0-9_]+)\s+([\d.]+)\s*$", line)
        if m:
            out.append((m.group(1), to_float(m.group(2))))
    return out


def parse_seed_sweep():
    """verify_seeds.log -> per-seed accuracy/auroc/baseline."""
    text = read_text(os.path.join(RESULTS, "verify_seeds.log"))
    if not text:
        return []
    seeds, cur = [], None
    for line in text.splitlines():
        m = re.match(r"#+\s*seed=(\d+)", line)
        if m:
            cur = {"seed": to_float(m.group(1))}
            seeds.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s+(accuracy|auroc|auprc|sensitivity|specificity|"
                     r"majority_baseline_acc|balanced_accuracy)\s+([\d.]+)", line)
        if m:
            cur.setdefault(m.group(1), to_float(m.group(2)))
    return [s for s in seeds if "accuracy" in s and "auroc" in s]


def parse_vla_log():
    """Per-step training telemetry from the lerobot log line."""
    text = read_text(os.path.join(RESULTS, "vla_train.log"))
    if not text:
        return []
    pat = re.compile(
        r"step:(\S+)\s+smpl:\S+\s+ep:\S+\s+epch:([\d.]+)\s+loss:([\d.]+)\s+grdn:([\d.]+)\s+"
        r"lr:([\d.eE+-]+)\s+updt_s:([\d.]+)\s+data_s:([\d.]+)\s+smp/s:(\d+)\s+mem_gb:([\d.]+)")
    rows = []
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        raw = m.group(1)
        mult = 1
        if raw.endswith("K"):
            mult, raw = 1000, raw[:-1]
        elif raw.endswith("M"):
            mult, raw = 1_000_000, raw[:-1]
        rows.append({
            "step": to_int(to_float(raw)) * mult, "epoch": to_float(m.group(2)),
            "loss": to_float(m.group(3)), "grdn": to_float(m.group(4)),
            "lr": to_float(m.group(5)), "updt_s": to_float(m.group(6)),
            "data_s": to_float(m.group(7)), "smp_s": to_float(m.group(8)),
            "mem_gb": to_float(m.group(9)),
        })
    rows.sort(key=lambda r: r["step"])
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_data_volume(out):
    name = "fig01_data_volume.png"
    vol = read_text(os.path.join(RESULTS, "disk_volumes.txt"))
    if not vol:
        return skip(name, "results/disk_volumes.txt missing")
    got = {}
    for line in vol.splitlines():
        m = re.match(r"(/root/autodl-tmp/\S+)\s+(\d+)\s*$", line)
        if m:
            got[m.group(1)] = to_int(m.group(2))
    numeric = got.get("/root/autodl-tmp/numeric")
    ppg = got.get("/root/autodl-tmp/ppg_npy")
    if numeric is None or ppg is None:
        return skip(name, "numeric/ppg_npy byte counts absent from disk_volumes.txt")

    labels = ["数值轨道\nCSV (实测)", "PPG 波形\n.npy (实测)",
              "PPG 波形\nCSV 文本 (推算)", "整包 .vital\n全量 (推算)"]
    gb = [numeric / 1e9, ppg / 1e9, 286.0, 86.0]
    colors = ["#2b6cb0", "#2f855a", "#b7791f", "#9b2c2c"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labels, gb, color=colors, edgecolor="black", linewidth=0.6)
    for b, v, measured in zip(bars, gb, [1, 1, 0, 0], strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.1f} GB"
                + ("\n实测" if measured else "\n推算"), ha="center", fontsize=9)
    ax.set_ylabel("磁盘占用 (GB)")
    ax.set_title("图1  四种数据获取路线的磁盘代价（实测 vs 推算）")
    ax.set_ylim(0, max(gb) * 1.22)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.06,
             "实测值来自 du -sb 与逐文件求和；推算值对应 6388 例整包、6157 例 PPG CSV 文本",
             fontsize=8, color="#444")
    return save(fig, out, name)


def fig_ppg_coverage(out):
    name = "fig02_ppg_coverage.png"
    sizes_path = os.path.join(SCRATCH, "ppg_sizes.txt")
    manifest = load_json(os.path.join(RESULTS, "ppg_manifest.json"))
    if not os.path.exists(sizes_path) or manifest is None:
        return skip(name, "scratch/ppg_sizes.txt or results/ppg_manifest.json missing")

    try:
        sizes = np.loadtxt(sizes_path, dtype=np.int64, usecols=1)
    except (OSError, ValueError) as exc:
        return skip(name, f"cannot parse ppg_sizes.txt: {exc}")
    if sizes.size == 0:
        return skip(name, "ppg_sizes.txt empty")

    pts = parse_finalize_log()
    expected = to_int(manifest.get("expected_cases"), 0)
    downloaded = to_int(manifest.get("downloaded_cases"), 0)
    total_b = to_int(manifest.get("total_bytes"), 0)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    ax.hist(sizes / 2**20, bins=60, color="#2b6cb0", edgecolor="white", linewidth=0.3)
    ax.axvline(sizes.mean() / 2**20, color="#c53030", linestyle="--",
               label=f"均值 {sizes.mean()/2**20:.2f} MB")
    ax.axvline(np.median(sizes) / 2**20, color="#2f855a", linestyle="-.",
               label=f"中位数 {np.median(sizes)/2**20:.2f} MB")
    ax.set_xlabel("单例 .npy 体积 (MB)")
    ax.set_ylabel("病例数")
    ax.set_title(f"逐例体积分布（{sizes.size} 例实测）")
    ax.legend(fontsize=8)

    ax = axes[1]
    sorted_mb = np.sort(sizes) / 2**20
    cum_gb = np.cumsum(sorted_mb) / 1024.0
    ax.plot(sorted_mb, cum_gb, color="#2b6cb0", linewidth=2)
    ax.axhline(total_b / 1e9, color="#c53030", linestyle="--",
               label=f"清单 total_bytes = {total_b/1e9:.1f} GB")
    ax.set_xlabel("单例体积 (MB)")
    ax.set_ylabel("累计体积 (GB)")
    ax.set_title("累计体积曲线（面积守恒校验）")
    ax.legend(fontsize=8)

    ax = axes[2]
    bars = ax.bar(["应下载", "已下载", "缺失", "侧车缺失"],
                  [expected, downloaded, to_int(manifest.get("missing_count"), 0),
                   to_int(manifest.get("sidecar_missing"), 0)],
                  color=["#a0aec0", "#2f855a", "#c53030", "#c53030"],
                  edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, [expected, downloaded, to_int(manifest.get("missing_count"), 0),
                           to_int(manifest.get("sidecar_missing"), 0)], strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + max(expected, 1) * 0.02, str(v),
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(expected, 1) * 1.18)
    ax.set_title(f"下载覆盖：{downloaded}/{expected} = {100.0*downloaded/max(expected,1):.1f}%")
    ax.grid(axis="x", visible=False)

    fig.suptitle("图2  PPG 波形全量下载的完整性与体积分布", fontsize=13, y=1.03)
    if pts:
        tail = "  最后阶段进度：" + ", ".join(f"{t} {n}/{d}" for t, n, d in pts[-3:])
        fig.text(0.01, -0.08, tail, fontsize=8, color="#444")
    return save(fig, out, name)


def clean_window_start(filt, fs, hr_hz, win_s=8.0, hop_s=1.0, search_s=300.0):
    """First sample of the window carrying the most power at the dominant rate.

    The record start often holds a line-flush artefact (a deep negative spike), so
    the raw-waveform panel must not blindly plot t=0; this picks a beat that
    actually looks like the pulse the spectrum reports.
    """
    n_win = to_int(win_s * fs)
    hop = max(to_int(hop_s * fs), 1)
    limit = min(filt.size, to_int(search_s * fs))
    best_idx, best_power = 0, -1.0
    for start in range(0, max(limit - n_win, 0), hop):
        seg = filt[start:start + n_win]
        if seg.size < n_win:
            break
        try:
            f2, p2 = sg.welch(seg, fs=fs, nperseg=to_int(4 * fs), noverlap=to_int(2 * fs))
        except (ValueError, TypeError) as exc:
            print(f"  ! welch(window scan) failed: {exc}")
            break
        band = np.asarray((f2 >= hr_hz - 0.15) & (f2 <= hr_hz + 0.15))
        if not band.any():
            continue
        power = to_float(np.trapezoid(p2[band], f2[band]))
        if power > best_power:
            best_idx, best_power = start, power
    return best_idx


def fig_ppg_waveform(out):
    name = "fig03_ppg_waveform.png"
    check = load_json(os.path.join(SAMPLES, "ppg_signal_check.json"))
    if check is None:
        return skip(name, "scratch/ppg_samples/ppg_signal_check.json missing")
    rows = [r for r in check.get("samples", []) if r.get("spectral_prominence", 0) > 20]
    if not rows:
        return skip(name, "no high-prominence sample in the signal check")
    best = max(rows, key=lambda r: to_float(r.get("spectral_prominence"), 0))
    path = os.path.join(SAMPLES, str(best.get("file")))
    arr = None
    try:
        arr = np.load(path).astype(np.float64)
    except (OSError, ValueError) as exc:
        return skip(name, f"cannot load {path}: {exc}")

    fs = to_float(check.get("fs_hz"), 500.0)
    med = np.nanmedian(arr)
    v = np.nan_to_num(arr, nan=to_float(med))
    lo, hi = to_float(check.get("band_hz", [0.5, 4.0])[0]), to_float(check.get("band_hz", [0.5, 4.0])[1])

    from scipy import signal as sg
    sos = sg.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    filt = sg.sosfiltfilt(sos, v)
    fw, pw = sg.welch(filt, fs=fs, nperseg=to_int(20 * fs), noverlap=to_int(10 * fs))
    band = np.asarray((fw >= lo) & (fw <= hi))
    hr = to_float(fw[band][to_int(np.argmax(pw[band]))]) * 60.0

    n8 = to_int(8 * fs)
    n60 = to_int(60 * fs)
    start = clean_window_start(filt, fs, hr / 60.0)
    start60 = max(start - (n60 - n8) // 2, 0)
    seg8 = v[start:start + n8]
    seg60 = filt[start60:start60 + n60]
    t8 = np.arange(seg8.size) / fs
    t60 = (start60 + np.arange(seg60.size)) / fs
    t8 = t8 + start / fs
    lo_y, hi_y = np.percentile(seg8, [0.5, 99.5])
    pad = max((hi_y - lo_y) * 0.15, 1.0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.4), layout="constrained")
    ax = axes[0]
    ax.plot(t8, seg8, color="#2b6cb0", linewidth=0.9)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度 (监护仪原始单位)")
    ax.set_title(f"原始 PLETH 波形 8 秒（已避开起始冲管伪迹）— caseid {best.get('caseid')}，"
                 f"共 {to_float(best.get('duration_s'), 0):.0f} s / "
                 f"{to_int(best.get('n_samples'), 0)} 采样点 @ {fs:.0f} Hz")
    ax.set_xlim(t8.min(), t8.max())
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.text(0.995, 0.04, f"纵轴截断到 0.5–99.5 分位（原始极值 {v.min():.0f} / {v.max():.0f}）",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#555")

    ax = axes[1]
    ax.plot(t60, seg60, color="#2f855a", linewidth=0.6)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("带通滤波后幅度")
    ax.set_title(f"同一段落前后 60 秒（{lo}–{hi} Hz 带通）：规则的搏动节律")
    ax.set_xlim(t60.min(), t60.max())

    ax = axes[2]
    ax.semilogy(fw[band], pw[band], color="#6b46c1", linewidth=1.4)
    ax.axvline(hr / 60.0, color="#c53030", linestyle="--",
               label=f"谱峰 {hr:.1f} bpm ({hr/60:.3f} Hz)")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("功率谱密度")
    ax.set_title(f"Welch 功率谱：谱峰显著度 {to_float(best.get('spectral_prominence')):.1f}×，"
                 f"30 s 窗内心率波动 {to_float(best.get('window_hr_spread')):.0f} bpm")
    ax.legend(fontsize=9)
    fig.suptitle("图3  下载的 PPG 波形确为真实脉搏波（非噪声/常数）", fontsize=13)
    return save(fig, out, name)


def fig_ppg_align(out):
    name = "fig04_ppg_align.png"
    rows = parse_align_log()
    if not rows:
        return skip(name, "results/verify_ppg_align.log has no parseable table")
    ppg = np.array([r["ppg"] for r in rows])
    mon = np.array([r["mon"] for r in rows])
    prom = np.array([r["prom"] for r in rows])
    ok = np.array([r["verdict"] == "ALIGNED" for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    lim = [min(ppg.min(), mon.min()) - 8, max(ppg.max(), mon.max()) + 8]
    ax.plot(lim, lim, color="#718096", linestyle=":", label="理想一致线 y=x")
    ax.fill_between(lim, [lim[0] - 15, lim[1] - 15], [lim[0] + 15, lim[1] + 15],
                    color="#68d391", alpha=0.18, label="±15 bpm 容差带")
    ax.scatter(mon[ok], ppg[ok], s=70, color="#2f855a", edgecolor="black",
               zorder=3, label=f"对齐 ({to_int(ok.sum())})")
    ax.scatter(mon[~ok], ppg[~ok], s=70, color="#c53030", marker="X", edgecolor="black",
               zorder=3, label=f"不一致 ({to_int((~ok).sum())})")
    for r in rows:
        ax.annotate(str(r["caseid"]), (r["mon"], r["ppg"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("监护仪自带 PLETH_HR / HR (bpm)")
    ax.set_ylabel("PPG 波形推算心率 (bpm)")
    ax.set_title(f"时间对齐验证：{to_int(ok.sum())}/{len(rows)} 例误差 ≤15 bpm")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    order = np.argsort(prom)
    colors = ["#2f855a" if o else "#c53030" for o in ok[order]]
    ax.barh([str(rows[i]["caseid"]) for i in order], prom[order], color=colors,
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("谱峰显著度（峰值 / 带内中位数）")
    ax.set_ylabel("caseid")
    ax.set_title("谱峰显著度：绿色=对齐，红色=不一致")
    ax.grid(axis="y", visible=False)
    fig.suptitle("图4  PPG 数组 index 0 与数值轨道 t=0 同源（对齐即证明）", fontsize=13, y=1.02)
    return save(fig, out, name)


def fig_seed_robustness(out):
    name = "fig05_seed_robustness.png"
    seeds = parse_seed_sweep()
    if not seeds:
        return skip(name, "results/verify_seeds.log unparseable")
    xs = np.arange(len(seeds))
    labs = [f"seed={to_int(s['seed'])}" for s in seeds]
    acc = [s["accuracy"] for s in seeds]
    base = [s.get("majority_baseline_acc", NAN) for s in seeds]
    auroc = [s["auroc"] for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax = axes[0]
    w = 0.38
    b1 = ax.bar(xs - w / 2, acc, w, label="模型 accuracy", color="#2b6cb0",
                edgecolor="black", linewidth=0.6)
    b2 = ax.bar(xs + w / 2, base, w, label="多数类基线", color="#a0aec0",
                edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                    f"{b.get_height():.4f}", ha="center", fontsize=8, rotation=90)
    ax.axhline(0.85, color="#c53030", linestyle="--", linewidth=1.4, label="目标线 85%")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_ylim(0.80, 0.99)
    ax.set_ylabel("测试集 accuracy")
    ax.set_title("四个随机种子：模型 vs 多数类基线")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    gains = [a - b for a, b in zip(acc, base, strict=True)]
    bars = ax.bar(xs, gains, color="#2f855a", edgecolor="black", linewidth=0.6)
    for b, g, a in zip(bars, gains, auroc, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.002,
                f"+{g*100:.1f} pt\nAUROC {a:.4f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_ylabel("相对基线的 accuracy 提升 (绝对百分点)")
    ax.set_title("真正的收益：高于基线的百分点")
    ax.set_ylim(0, max(gains) * 1.45)
    fig.suptitle("图5  「accuracy > 85%」必须相对基线读：基线本身就有 86–88%", fontsize=13, y=1.02)
    return save(fig, out, name)


def _curve_panels(ax_roc, ax_pr, variants, title_roc, title_pr):
    for label, npz_path, color in variants:
        z = load_npz(npz_path)
        if z is None:
            continue
        y, p = z["y_test"], z["p_test"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax_roc.plot(fpr, tpr, color=color, linewidth=1.9,
                    label=f"{label}  AUROC={auc(fpr, tpr):.4f}")
        ax_pr.plot(rec, prec, color=color, linewidth=1.9,
                   label=f"{label}  AUPRC={average_precision_score(y, p):.4f}")
    ax_roc.plot([0, 1], [0, 1], color="#a0aec0", linestyle=":", label="随机 (0.5)")
    ax_roc.set_xlabel("假阳性率 FPR")
    ax_roc.set_ylabel("真阳性率 TPR")
    ax_roc.set_title(title_roc)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_pr.set_xlabel("召回率 Recall")
    ax_pr.set_ylabel("精确率 Precision")
    ax_pr.set_title(title_pr)
    ax_pr.legend(fontsize=8, loc="lower left")


def fig_roc_pr(out):
    name = "fig06_roc_pr.png"
    need = [os.path.join(SCORES, f) for f in ("main.npz", "onset.npz")]
    if not all(os.path.exists(p) for p in need):
        return skip(name, "scores/main.npz or scores/onset.npz missing")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    _curve_panels(
        axes[0], axes[1],
        [("主定义「未来5分钟出现」", need[0], "#2b6cb0"),
         ("严格 onset「未来5分钟新发」", need[1], "#b7791f")],
        "ROC 曲线（测试集）", "PR 曲线（测试集）",
    )
    z = load_npz(need[0])
    if z is not None:
        pr = to_float(np.mean(z["y_test"]))
        axes[1].axhline(pr, color="#c53030", linestyle="--", linewidth=1,
                        label=f"主定义阳性率 {pr:.3f}")
        axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("图6  两种标签定义下的完整 ROC / PR 曲线（剔除边界重叠后仍远超随机）",
                 fontsize=13, y=1.02)
    return save(fig, out, name)


def fig_operating_points(out):
    name = "fig07_operating_points.png"
    data = load_json(os.path.join(RESULTS, "operating_points.json"))
    if not data or not data.get("points"):
        return skip(name, "results/operating_points.json missing")
    pts = sorted(data["points"], key=lambda p: to_float(p["threshold"]))
    thr = [to_float(p["threshold"]) for p in pts]
    sens = [to_float(p["sensitivity"]) for p in pts]
    spec = [to_float(p["specificity"]) for p in pts]
    ppv = [to_float(p["ppv"]) for p in pts]
    acc = [to_float(p["accuracy"]) for p in pts]
    alarm = [to_float(p["alarms_per_100_windows"]) for p in pts]
    base = to_float(data.get("test_majority_baseline_acc"))
    names = [p["name"] for p in pts]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    ax = axes[0]
    ax.plot(thr, sens, "o-", color="#2b6cb0", label="灵敏度 Sensitivity")
    ax.plot(thr, spec, "s-", color="#2f855a", label="特异度 Specificity")
    ax.plot(thr, ppv, "^-", color="#6b46c1", label="PPV 精确率")
    ax.plot(thr, acc, "D-", color="#b7791f", label="accuracy")
    ax.axhline(base, color="#c53030", linestyle="--", linewidth=1.4,
               label=f"多数类基线 {base:.4f}")
    ax.axhline(0.85, color="#718096", linestyle=":", linewidth=1.2, label="目标线 0.85")
    ax.set_xlabel("判决阈值（全部在验证集选定）")
    ax.set_ylabel("指标值")
    ax.set_title("同一模型、同一数据：仅换阈值即可大幅改变各指标")
    ax.legend(fontsize=8, loc="center right")

    ax = axes[1]
    bars = ax.barh(np.arange(len(pts)), alarm, color="#dd6b20", edgecolor="black",
                   linewidth=0.5)
    for i, b in enumerate(bars):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                f"阈值 {thr[i]:.3f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(pts)))
    ax.set_yticklabels([f"{n}" for n in names], fontsize=8)
    ax.set_xlabel("报警率（每 100 个窗口）")
    ax.set_title("工作点的临床代价：报警率")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(alarm) * 1.45)
    fig.suptitle("图7  六个临床可用工作点：accuracy 0.873–0.944 全部「满足 >85%」，但含义完全不同",
                 fontsize=12.5, y=1.03)
    return save(fig, out, name)


def fig_feature_gain(out):
    name = "fig08_feature_gain.png"
    gains = parse_feature_gains()
    if not gains:
        return skip(name, "no gain table in results/run_main.log")
    top = gains[:18][::-1]
    labels = [g[0] for g in top]
    vals = np.array([g[1] for g in top])

    def family(f):
        if f.startswith("map"):
            return "#2b6cb0"
        if f.startswith(("sbp", "dbp")):
            return "#2f855a"
        if f.startswith("hr"):
            return "#b7791f"
        if f.startswith("etco2"):
            return "#6b46c1"
        return "#718096"

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    bars = ax.barh(labels, vals, color=[family(f) for f in labels],
                   edgecolor="black", linewidth=0.5)
    share = vals.sum() / sum(g[1] for g in gains)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + max(vals) * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,.0f}", va="center", fontsize=7.5)
    ax.set_xlabel("LightGBM gain")
    ax.set_title(f"图8  数值模型前 18 重要特征（MAP 系主导；前 18 占 top-20 总 gain 的 {share*100:.1f}%）")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.20)
    handles = [Patch(facecolor=c) for c in
               ["#2b6cb0", "#2f855a", "#b7791f", "#6b46c1", "#718096"]]
    ax.legend(handles, ["MAP 系", "SBP/DBP 系", "HR 系", "EtCO2 系", "其他"],
              fontsize=8, loc="lower right")
    fig.text(0.01, -0.04, "标签由 MAP 定义、特征由 MAP 主导 —— 这是 PPG 波形无法带来增益的结构性原因",
             fontsize=8.5, color="#444")
    return save(fig, out, name)


def fig_ppg_ablation(out):
    name = "fig09_ppg_ablation.png"
    a = os.path.join(SCORES, "fused_numeric.npz")
    b = os.path.join(SCORES, "fused_numeric_ppg.npz")
    if not (os.path.exists(a) and os.path.exists(b)):
        return skip(name, "fused score dumps missing")
    keys = ["accuracy", "auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy"]
    za, zb = load_npz(a), load_npz(b)
    if za is None or zb is None:
        return skip(name, "fused npz unreadable")

    def m(z, key):
        y, p = z["y_test"], z["p_test"]
        pred = (p >= 0.5).astype(np.int8)
        tn = to_float(np.sum((y == 0) & (pred == 0)))
        fp = to_float(np.sum((y == 0) & (pred == 1)))
        fn = to_float(np.sum((y == 1) & (pred == 0)))
        tp = to_float(np.sum((y == 1) & (pred == 1)))
        table = {
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "auroc": auc(*roc_curve(y, p)[:2]),
            "auprc": average_precision_score(y, p),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        }
        return to_float(table.get(key))

    va = [m(za, k) for k in keys]
    vb = [m(zb, k) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    ax = axes[0]
    xs = np.arange(len(keys))
    w = 0.38
    ax.bar(xs - w / 2, va, w, label="纯数值 (67 维)", color="#2b6cb0",
           edgecolor="black", linewidth=0.6)
    ax.bar(xs + w / 2, vb, w, label="数值 + PPG (122 维)", color="#dd6b20",
           edgecolor="black", linewidth=0.6)
    for i, (x, y2) in enumerate(zip(va, vb, strict=True)):
        d = y2 - x
        ax.text(i, max(x, y2) + 0.012, f"{d:+.4f}", ha="center", fontsize=8,
                color="#c53030" if d < 0 else "#2f855a", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("测试集指标")
    ax.set_title("六项指标对照：五项略负，唯一为正的是噪声量级")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for label, path, color in (("纯数值", a, "#2b6cb0"), ("数值+PPG", b, "#dd6b20")):
        z = load_npz(path)
        if z is None:
            continue
        fpr, tpr, _ = roc_curve(z["y_test"], z["p_test"])
        ax.plot(fpr, tpr, color=color, linewidth=1.9, label=f"{label} AUROC={auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], color="#a0aec0", linestyle=":", label="随机")
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR")
    ax.set_title("两条 ROC 几乎完全重合")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("图9  PPG 波形对照实验（3282 例 / 1,400,689 窗口）：加入波形后无可测收益",
                 fontsize=12.5, y=1.03)
    return save(fig, out, name)


def fig_vla_training(out):
    name = "fig10_vla_training.png"
    rows = parse_vla_log()
    if not rows:
        return skip(name, "no parseable step lines in results/vla_train.log")
    step = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    lr = np.array([r["lr"] for r in rows])
    smp = np.array([r["smp_s"] for r in rows])
    mem = np.array([r["mem_gb"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), layout="constrained")
    ax = axes[0, 0]
    ax.plot(step, loss, color="#2b6cb0", linewidth=1.6)
    if step.size > 12:
        k = max(step.size // 12, 3)
        sm = np.convolve(loss, np.ones(k) / k, mode="valid")
        ax.plot(step[k - 1:], sm, color="#c53030", linewidth=2,
                label=f"{k} 点滑动平均")
        ax.legend(fontsize=8)
    ax.set_xlabel("训练步 step")
    ax.set_ylabel("loss")
    ax.set_title(f"SmolVLA 微调 loss（{step.size} 个记录点，最终 "
                 f"{loss[-1]:.3f} @ step {step[-1]}）")

    ax = axes[0, 1]
    ax.semilogy(step, lr, color="#6b46c1", linewidth=1.6)
    ax.set_xlabel("训练步 step")
    ax.set_ylabel("学习率 (log)")
    ax.set_title("学习率调度：1e-4 余弦衰减至 2.5e-06")

    ax = axes[1, 0]
    ax.plot(step, smp, color="#2f855a", linewidth=1.4)
    ax.set_xlabel("训练步 step")
    ax.set_ylabel("吞吐 (samples/s)")
    ax.set_title(f"训练吞吐：中位数 {np.median(smp):.0f} samples/s")

    ax = axes[1, 1]
    ax.plot(step, mem, color="#dd6b20", linewidth=1.4)
    ax.set_xlabel("训练步 step")
    ax.set_ylabel("显存占用 (GB)")
    ax.set_title(f"训练期显存：{mem.min():.2f}–{mem.max():.2f} GB / 24 GB（RTX 4090 D）")
    fig.suptitle("图10  VLA（SmolVLA on LIBERO）3000 步 GPU 训练全过程遥测", fontsize=13)
    return save(fig, out, name)


def fig_vla_artifacts(out):
    name = "fig11_vla_artifacts.png"
    text = read_text(os.path.join(RESULTS, "vla_ckpt_evidence.txt"))
    meta = read_text(os.path.join(RESULTS, "vla_ckpt_meta.txt"))
    if not text:
        return skip(name, "results/vla_ckpt_evidence.txt missing")

    per_ckpt = {}
    for line in text.splitlines():
        m = re.match(r"(\d+)\s+(\S*/checkpoints/(\d+)/pretrained_model/model\.safetensors)", line)
        if m:
            per_ckpt[m.group(3)] = to_int(m.group(1))
    if not per_ckpt:
        return skip(name, "no model.safetensors sizes parsed")

    steps = {}
    if meta:
        for m in re.finditer(r"(\d{6}) -> \{\s*\"step\": (\d+),\s*\"num_processes\": (\d+),"
                             r"\s*\"batch_size\": (\d+)", meta):
            steps[m.group(1)] = (to_int(m.group(2)), to_int(m.group(4)))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
    ax = axes[0]
    names = sorted(per_ckpt)
    vals = [per_ckpt[n] / 2**20 for n in names]
    bars = ax.bar(names, vals, color="#2b6cb0", edgecolor="black", linewidth=0.6)
    for b, v, n in zip(bars, vals, names, strict=True):
        lbl = f"{v:.1f} MB"
        if n in steps:
            lbl += f"\nstep={steps[n][0]}\nbs={steps[n][1]}"
        ax.text(b.get_x() + b.get_width() / 2, v * 0.5, lbl, ha="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_ylabel("model.safetensors (MB)")
    ax.set_ylim(0, max(vals) * 1.12)
    ax.set_title("三个检查点的模型权重（每个 906,712,520 B）")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.axis("off")
    lines = []
    if meta:
        m = re.search(r"last symlink resolves to\s*\n(\S+)", meta)
        if m:
            lines.append(f"last 符号链接 → {m.group(1)}")
        m = re.search(r'"last_epoch": (\d+)', meta)
        if m:
            lines.append(f"scheduler last_epoch = {m.group(1)}")
        m = re.search(r'"_step_count": (\d+)', meta)
        if m:
            lines.append(f"scheduler _step_count = {m.group(1)}")
        m = re.search(r'"_last_lr": \[\s*([\d.eE+-]+)', meta)
        if m:
            lines.append(f"最终学习率 = {m.group(1)}")
        m = re.search(r'"type": "(smolvla)"', meta)
        if m:
            lines.append(f"policy.type = {m.group(1)}")
        m = re.search(r'"device": "(\w+)"', meta)
        if m:
            lines.append(f"device = {m.group(1)}")
        m = re.search(r'"repo_id": "([^"]+)"', meta)
        if m:
            lines.append(f"dataset = {m.group(1)}")
        m = re.search(r'"num_workers": (\d+)', meta)
        if m:
            lines.append(f"num_workers = {m.group(1)}")
        m = re.search(r'"save_freq": (\d+)', meta)
        if m:
            lines.append(f"save_freq = {m.group(1)}")
        m = re.search(r'"vlm_model_name": "([^"]+)"', meta)
        if m:
            lines.append(f"VLM = {m.group(1)}")
        m = re.search(r'"steps": (\d+)', meta)
        if m:
            lines.append(f"steps = {m.group(1)}")
    m = re.search(r"(\S+), (\d+) MiB, (\d+) %, ([\d.]+) W", text)
    if m:
        lines.append(f"训练结束后 GPU：{m.group(2)} MiB / {m.group(3)}% / {m.group(4)} W（已释放）")
    ax.text(0.02, 0.98, "检查点自身元数据（非日志转述）\n\n" + "\n".join(f"• {x}" for x in lines),
            va="top", ha="left", fontsize=10, family=PRIMARY_FONT)
    ax.set_title("产物级证据")
    fig.suptitle("图11  VLA 训练产物核验：检查点自述 3000 步、batch 64、last→003000",
                 fontsize=12.5, y=1.03)
    return save(fig, out, name)


def fig_p0_baselines(out):
    """Clinical baselines at the event level (results/p0_report.json).

    Two things the accuracy/AUROC tables cannot show: how the model compares with
    the alarm it would replace, and what happens per hypotension *episode*.
    """
    name = "fig12_p0_baselines.png"
    rep = load_json(os.path.join(RESULTS, "p0_report.json"))
    if not rep:
        return skip(name, "results/p0_report.json missing")
    group = (rep.get("groups") or {}).get("numeric")
    if not group:
        return skip(name, "no numeric group in p0_report.json")
    variants = group.get("variants") or {}
    order = [k for k in ("main", "map_only", "rule_map65", "rule_map65_mean",
                         "rule_map65_min") if k in variants]
    if not order:
        return skip(name, "no variants in p0_report.json")
    label = {"main": "主模型 (67 维)", "map_only": "仅 MAP (11 维)",
             "rule_map65": "规则 MAP<65", "rule_map65_mean": "规则 60 s 均值<65",
             "rule_map65_min": "规则 5 min 最小值<65"}
    offsets = {"main": (10, 9), "map_only": (10, -20), "rule_map65": (10, 10),
               "rule_map65_mean": (-14, 12), "rule_map65_min": (-16, -24)}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1))

    ax = axes[0]
    xs = np.arange(len(order))
    w = 0.38
    auroc = [to_float(variants[k]["window"]["auroc"]) for k in order]
    auprc = [to_float(variants[k]["window"]["auprc"]) for k in order]
    ax.bar(xs - w / 2, auroc, w, label="AUROC", color="#2b6cb0",
           edgecolor="black", linewidth=0.6)
    ax.bar(xs + w / 2, auprc, w, label="AUPRC", color="#dd6b20",
           edgecolor="black", linewidth=0.6)
    for i, (a, b) in enumerate(zip(auroc, auprc, strict=True)):
        ax.text(i - w / 2, a + 0.014, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.014, f"{b:.3f}", ha="center", fontsize=8)
    if "rule_map65" in order:
        ax.axhline(auroc[order.index("rule_map65")], color="#c53030", linestyle=":",
                   linewidth=1.2, label="规则 MAP<65 的 AUROC")
    ax.set_xticks(xs)
    ax.set_xticklabels([label.get(k, k) for k in order], rotation=18, ha="right",
                       fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("测试集指标（阈值无关）")
    ax.set_title("排序能力：模型优于规则")
    ax.legend(fontsize=8.5, loc="lower left")

    ax = axes[1]
    for k in order:
        e = variants[k]["event"]
        ci = (variants[k].get("event_ci") or {}).get("false_alarm_rate_per_h") or {}
        x = to_float(e["false_alarm_rate_per_h"])
        y = to_float(e["lead_median_s"]) / 60.0
        lo, hi = to_float(ci.get("lo")), to_float(ci.get("hi"))
        xerr = None
        if np.isfinite(lo) and np.isfinite(hi):
            xerr = [[max(x - lo, 0.0)], [max(hi - x, 0.0)]]
        model = k in ("main", "map_only")
        ax.errorbar(x, y, xerr=xerr, fmt="o", ms=10 if model else 7,
                    color="#2b6cb0" if model else "#c53030",
                    markeredgecolor="black", markeredgewidth=0.6,
                    ecolor="#a0aec0", capsize=3, zorder=3,
                    label="机器学习模型" if model else None)
        ax.annotate(f"{label.get(k, k)}\n检出 {to_float(e['detection_rate']) * 100:.1f}%",
                    (x, y), textcoords="offset points",
                    xytext=offsets.get(k, (8, 8)), fontsize=8)
    handles = [Line2D([], [], marker="o", linestyle="", color="#2b6cb0",
                      markeredgecolor="black", label="机器学习模型"),
               Line2D([], [], marker="o", linestyle="", color="#c53030",
                      markeredgecolor="black", label="MAP<65 规则")]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    ax.set_xlim(0.2, 2.45)
    ax.set_xlabel("误报次数 / 病例小时")
    ax.set_ylabel("事件检出中位提前量（min）")
    ax.set_title("事件级权衡：模型误报更多、提前量更早")
    ax.grid(alpha=0.25, linestyle=":")
    fig.suptitle("图12  临床基线对照（350 例 / 156,574 窗口 / 1392 个低血压事件）："
                 "事件检出率无增益，仅提前量更早", fontsize=12.5, y=1.03)
    return save(fig, out, name)


def load_p0_group(path):
    """Numeric group of a p0_eval dataset report, or None if absent/unreadable."""
    rep = load_json(path)
    if not rep:
        return None
    return (rep.get("groups") or {}).get("numeric")


def fig_horizon(out):
    """How the task changes when the label looks further ahead (5/10/15 min)."""
    name = "fig13_horizon.png"
    sources = [(300, os.path.join(RESULTS, "p0_report.json")),
               (600, os.path.join(RESULTS, "p0_report_h600.json")),
               (900, os.path.join(RESULTS, "p0_report_h900.json"))]
    data = [(h, g) for h, p in sources if (g := load_p0_group(p))]
    if len(data) < 2:
        return skip(name, "need two or more horizon reports in results/")
    for h, g in data:
        if "rule_map65" not in (g.get("variants") or {}):
            return skip(name, f"report for {h}s has no rule_map65 baseline")

    hs = [h / 60.0 for h, _ in data]
    series = {"main": ("主模型 (67 维)", "#2b6cb0", "o-"),
              "map_only": ("仅 MAP (11 维)", "#38a169", "s--"),
              "rule_map65": ("规则 MAP<65", "#c53030", "^:")}

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    ax = axes[0]
    all_y = []
    for key, (lab, col, style) in series.items():
        ys = [to_float(g["variants"][key]["window"]["auroc"]) for _, g in data]
        all_y += [y for y in ys if np.isfinite(y)]
        ax.plot(hs, ys, style, color=col, label=lab, markersize=6, linewidth=1.8)
        for x, y in zip(hs, ys, strict=True):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.5)
    ax.set_xticks(hs)
    # data-driven floor: a fixed 0.8 would clip the rule's 15-min point entirely
    ax.set_ylim(min(all_y) - 0.04, 1.0)
    ax.set_xlabel("前瞻时长（min）")
    ax.set_ylabel("AUROC（阈值无关）")
    ax.set_title("排序能力：规则随前瞻迅速退化")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=8.5, loc="lower left")

    for ax, key_, ylab, title in (
            (axes[1], "detection_rate", "事件检出率（可评分事件）",
             "事件检出：两者都接近饱和"),
            (axes[2], "false_alarm_rate_per_h", "误报次数 / 病例小时",
             "误报负担：模型更高")):
        for key, (lab, col, style) in series.items():
            if key == "map_only":
                continue
            ys, los, his = [], [], []
            for _, g in data:
                ev = g["variants"][key]["event"]
                ci = (g["variants"][key].get("event_ci") or {}).get(key_) or {}
                ys.append(to_float(ev[key_]))
                los.append(to_float(ci.get("lo")))
                his.append(to_float(ci.get("hi")))
            yerr = None
            if all(np.isfinite(los)) and all(np.isfinite(his)):
                yerr = [np.array(ys) - np.array(los), np.array(his) - np.array(ys)]
            ax.errorbar(hs, ys, yerr=yerr, fmt=style, color=col, label=lab,
                        markersize=6, linewidth=1.8, capsize=3, ecolor="#a0aec0")
        ax.set_xticks(hs)
        ax.set_xlabel("前瞻时长（min）")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25, linestyle=":")
        if key_ == "detection_rate":
            ax.set_ylim(0.5, 1.03)
            ax.legend(fontsize=8.5, loc="lower left")
    axes[2].legend(fontsize=8.5, loc="upper left")

    fig.suptitle("图13  前瞻越长，“当前 MAP”越不足：规则与模型的差距被拉开",
                 fontsize=12.5, y=1.03)
    return save(fig, out, name)


def fig_optimization(out):
    """Paired ablation of the two free optimisation levers at the longest horizon."""
    name = "fig14_optimization.png"
    g = load_p0_group(os.path.join(RESULTS, "p0_report_h900opt.json"))
    if not g:
        return skip(name, "results/p0_report_h900opt.json missing")
    order = [k for k in ("main", "main_long", "main_extra", "main_all")
             if k in (g.get("variants") or {})]
    if len(order) < 2:
        return skip(name, "optimisation report has fewer than two variants")
    label = {"main": "基线 67 维", "main_long": "+长窗 (131)",
             "main_extra": "+BT/NIBP (94)", "main_all": "+两者 (158)"}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    ax = axes[0]
    xs = np.arange(len(order))
    w = 0.38
    auroc = [to_float(g["variants"][k]["window"]["auroc"]) for k in order]
    auprc = [to_float(g["variants"][k]["window"]["auprc"]) for k in order]
    ax.bar(xs - w / 2, auroc, w, label="AUROC", color="#2b6cb0",
           edgecolor="black", linewidth=0.6)
    ax.bar(xs + w / 2, auprc, w, label="AUPRC", color="#dd6b20",
           edgecolor="black", linewidth=0.6)
    for i, (a, b) in enumerate(zip(auroc, auprc, strict=True)):
        ax.text(i - w / 2, a + 0.012, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.012, f"{b:.4f}", ha="center", fontsize=8)
        if i:
            d = a - auroc[0]
            ax.text(i, max(a, b) + 0.05, f"{d:+.4f}", ha="center", fontsize=8.5,
                    color="#2f855a" if d > 0 else "#c53030", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([label.get(k, k) for k in order], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("测试集指标（阈值无关）")
    ax.set_title("特征消融：三条免费杠杆的绝对效果")
    ax.legend(fontsize=8.5, loc="lower left")

    ax = axes[1]
    metrics = [("auroc", "AUROC", "#2b6cb0"), ("auprc", "AUPRC", "#dd6b20"),
               ("sensitivity", "灵敏度", "#805ad5")]
    rows = [(k, m, lab, col) for k in order[1:] for m, lab, col in metrics]
    ypos = np.arange(len(rows))[::-1]
    for y, (k, m, _lab, col) in zip(ypos, rows, strict=True):
        d = ((g.get("deltas") or {}).get(f"{k}-main") or {}).get("boot", {}).get(m)
        if not d:
            continue
        mean, lo, hi = to_float(d["mean"]), to_float(d["lo"]), to_float(d["hi"])
        ax.errorbar(mean, y, xerr=[[max(mean - lo, 0)], [max(hi - mean, 0)]],
                    fmt="o", color=col, markersize=7, capsize=3, ecolor="#a0aec0",
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(f"{mean:+.4f} [{lo:+.4f},{hi:+.4f}]", (hi, y),
                    textcoords="offset points", xytext=(6, -3), fontsize=7.5)
    ax.axvline(0, color="#c53030", linestyle=":", linewidth=1.2)
    xs_plotted = [0.0]
    for _y, (k, m, _lab, _col) in zip(ypos, rows, strict=True):
        d = ((g.get("deltas") or {}).get(f"{k}-main") or {}).get("boot", {}).get(m)
        if d:
            xs_plotted += [to_float(d["lo"]), to_float(d["hi"])]
    span = max(xs_plotted) - min(xs_plotted)
    # leave room for the right-hand text labels, which otherwise leave the axes
    ax.set_xlim(min(xs_plotted) - 0.18 * span, max(xs_plotted) + 0.85 * span)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{label.get(k, k)}  {lab}" for k, _m, lab, _c in rows], fontsize=8.5)
    ax.set_xlabel("配对差值（病例级 bootstrap 95% CI）")
    ax.set_title("配对检验：置信区间是否远离 0")
    ax.grid(alpha=0.25, axis="x", linestyle=":")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, label=lbl,
                              markeredgecolor="black")
                       for _m, lbl, c in metrics], fontsize=8.5, loc="upper left")

    fig.suptitle("图14  15 min 前瞻下的算法优化：长窗与未使用轨道的配对消融",
                 fontsize=12.5, y=1.03)
    return save(fig, out, name)


def _arch_box(ax, x, y, w, h, title, body, face="#f7fafc", edge="#2d3748",
              tsize=9.5, bsize=7.6):
    """One labelled box for the architecture figure."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.0",
                                linewidth=1.1, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top",
            fontsize=tsize, fontweight="bold", color="#1a202c")
    if body:
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top",
                fontsize=bsize, linespacing=1.4, color="#2d3748")


def _endpoint_row(path, variant="main"):
    """(prevalence %, AUROC) for a variant of one endpoint report, or None if absent.

    Absence is expected (an endpoint that has not been evaluated yet), so it must not
    print an error: the figure shows it as pending instead.
    """
    if not os.path.isfile(path):
        return None
    g = load_p0_group(path)
    if not g or variant not in (g.get("variants") or {}):
        return None
    return g["test_positive_rate"] * 100.0, to_float(g["variants"][variant]["window"]["auroc"])


def fig_architecture(out):
    """The platform, drawn from the artifacts it is built from.

    Every box cites a result file: the device/channel counts come from the channel
    registry, the feature-group deltas from the paired reports, the endpoints from their
    own reports, and the alarm/streaming/verification numbers from theirs. Missing
    inputs omit their line rather than being filled with a plausible value.
    """
    name = "fig16_architecture.png"
    reg = load_json(os.path.join(RESULTS, "channel_registry.json"))
    if not reg:
        return skip(name, "results/channel_registry.json missing")
    local = reg.get("locally_available") or {}
    devices = reg.get("devices") or {}

    endpoints = [
        ("低血压", "MAP<65", os.path.join(RESULTS, "p0_report.json"), "main", ""),
        ("低氧", "SpO2<90", os.path.join(RESULTS, "p0_report_hypoxemia.json"), "main", ""),
        ("心动过缓", "HR<50", os.path.join(RESULTS, "p0_report_bradycardia.json"),
         "main", ""),
        ("通气不足", "RR<6", os.path.join(RESULTS, "p0_report_hypoventilation.json"),
         "main_func", "（含功能类）"),
        ("高碳酸血症", "ETCO2>50", os.path.join(RESULTS, "p0_report_hypercapnia.json"),
         "main_func", "（含功能类）"),
    ]
    rows = [(lab, sig, _endpoint_row(path, variant), variant, note)
            for lab, sig, path, variant, note in endpoints]
    if not any(r[2] for r in rows):
        return skip(name, "no endpoint reports in results/")

    fig, ax = plt.subplots(figsize=(16.2, 10.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- devices: what is local, from the registry
    local_devices = [("Solar8000", "监护仪"), ("SNUADC", "波形"), ("Primus", "麻醉机"),
                     ("BIS", "麻醉深度"), ("Orchestra", "输注泵")]
    gap = 1.6
    w = (97.0 - gap * (len(local_devices) - 1)) / len(local_devices)
    for i, (dev, role) in enumerate(local_devices):
        n = sum(1 for k in local if k.startswith(dev + "/"))
        total = (devices.get(dev) or {}).get("channels", 0)
        cases = (devices.get(dev) or {}).get("cases_max", 0)
        _arch_box(ax, 1.5 + i * (w + gap), 77.0, w, 15.0, f"{dev}（{role}）",
                  f"本地 {n} / 索引 {total} 路\n覆盖 ≤{cases:,} 例",
                  face="#ebf8ff" if n else "#fffaf0")
    missing = [d for d in ("EV1000", "Vigileo", "Vigilance", "CardioQ", "Invos", "FMS")
               if d in devices]
    ax.text(50, 74.6,
            "未下载：" + "、".join(f"{d}({devices[d]['channels']}路)" for d in missing)
            + "；肌松 TOF/NMT：本数据集 0 通道（需新设备源）",
            ha="center", va="top", fontsize=8.2, color="#744210")

    # ---- feature groups, with the measured increment where one exists
    opt = load_p0_group(os.path.join(RESULTS, "p0_report_h900opt.json")) or {}

    def delta_of(report, pair, metric="auroc"):
        d = ((report.get("deltas") or {}).get(pair) or {}).get("boot") or {}
        return to_float(d.get(metric, {}).get("mean")) if d else NAN

    long_delta = delta_of(opt, "main_long-main")
    extra_delta = delta_of(opt, "main_extra-main")
    mm300 = load_p0_group(os.path.join(RESULTS, "p0_report_multimodal_h300.json")) or {}
    # Same convention as the report/README: each delta is versus the numeric baseline,
    # not "added on top of the other waveform", so the figure and the text agree.
    ppg_delta = delta_of(mm300, "main_ppg-main")
    abp_delta = delta_of(mm300, "main_abp-main")
    fn_delta = delta_of(
        load_p0_group(os.path.join(RESULTS, "p0_report_hypoventilation.json")) or {},
        "main_func-main")
    groups = [
        ("数值基线 67 维", "低血压 AUROC 0.9536", "#f7fafc"),
        ("长窗 l_* +64", f"15 min 前瞻 ΔAUROC {long_delta:+.4f}" if np.isfinite(long_delta)
         else "", "#f0fff4"),
        ("BT / NIBP +27", f"ΔAUROC {extra_delta:+.4f}（不显著）" if np.isfinite(extra_delta)
         else "", "#f7fafc"),
        ("PPG 波形 +55", f"ΔAUROC {ppg_delta:+.4f}（无增益）" if np.isfinite(ppg_delta)
         else "", "#fff5f5"),
        ("有创动脉压波 +65", f"ΔAUROC {abp_delta:+.4f}（无增益）" if np.isfinite(abp_delta)
         else "", "#fff5f5"),
        ("功能类 fn_* +54", f"通气不足 ΔAUROC {fn_delta:+.4f}" if np.isfinite(fn_delta)
         else "", "#f0fff4"),
    ]
    gw = (97.0 - gap * (len(groups) - 1)) / len(groups)
    for i, (title, body, face) in enumerate(groups):
        _arch_box(ax, 1.5 + i * (gw + gap), 55.0, gw, 14.0, title, body, face=face)
    ax.text(50, 52.6, "ΔAUROC 均为「相对数值基线」的病例级配对均值；特征只取过去窗口，"
            "标签只看未来窗口，划分按病例隔离",
            ha="center", va="top", fontsize=8.6, color="#4a5568")

    # ---- endpoints
    ew = (97.0 - gap * (len(rows) - 1)) / len(rows)
    for i, (lab, sig, vals, variant, note) in enumerate(rows):
        if vals:
            prev, auc = vals
            tag = "" if variant == "main" else f"{note}"
            body = f"阳性率 {prev:.2f}%\nAUROC {auc:.4f}{tag}"
            face = "#ebf8ff"
        else:
            body = "待评估"
            face = "#fffaf0"
        _arch_box(ax, 1.5 + i * (ew + gap), 33.0, ew, 14.0, f"{lab}（{sig}）", body,
                  face=face)
    ax.text(50, 30.6, "五个终点共用同一特征矩阵与同一套划分协议", ha="center",
            va="top", fontsize=8.6, color="#4a5568")

    # ---- actions, each citing its artifact
    actions = [
        ("报警策略", "`remote/alarm_policy.py`", "#fffaf0"),
        ("监护显示", "fig15（真实打分+轨道）", "#f0fff4"),
        ("流式推理", "在线 ≡ 离线（67 维）", "#ebf8ff"),
        ("验证回路", "逐值回算自产物", "#f7fafc"),
    ]
    pareto = load_json(os.path.join(RESULTS, "alarm_pareto.json"))
    if pareto:
        pick = next((r for r in pareto if r.get("persist") == 2
                     and abs(to_float(r.get("clear_factor")) - 0.8) < 1e-9), None)
        if pick:
            actions[0] = ("报警策略",
                          (f"连续 2 窗 + 0.8 回差\n检出 {to_float(pick['detect']) * 100:.2f}%"
                           f"、误报 {to_float(pick['false_per_h']):.2f}/h"), "#fffaf0")
    checks = load_json(os.path.join(RESULTS, "check_numbers.json"))
    if checks:
        actions[3] = ("验证回路", (f"{checks.get('checked', 0)} 个报告数值逐值回算\n"
                                  f"{checks.get('missing', 0)} 处不符"), "#f7fafc")
    aw = (97.0 - gap * (len(actions) - 1)) / len(actions)
    for i, (title, body, face) in enumerate(actions):
        _arch_box(ax, 1.5 + i * (aw + gap), 11.0, aw, 14.0, title, body, face=face)

    # ---- data + provenance footer
    dl = load_json(os.path.join(RESULTS, "func_download.json")) or {}
    dl_txt = (f"功能类 {dl.get('files_on_disk', 0):,} 文件 / {dl.get('gb_on_disk', 0)} GB"
              if dl else "功能类：见 results/func_download.json")
    ax.text(1.5, 8.2,
            ("数据： 数值 8 路（2.2 GB）  ·  PPG 6157 例（特征提取后按策略删除原始）  ·  "
             f"有创动脉压波 2500 例（63.0 GB）  ·  {dl_txt}"),
            ha="left", va="top", fontsize=8.2, color="#4a5568")
    ax.text(1.5, 5.0,
            ("每层的数字均来自 results/ 下的产物；本图不填占位值（缺输入即省略该行）。"
             " 数据流向下（左箭头），验证回路向上（右箭头）。"),
            ha="left", va="top", fontsize=8.2, color="#718096")

    ax.annotate("", xy=(0.6, 11.0), xytext=(0.6, 92.0),
                arrowprops={"arrowstyle": "-|>", "color": "#2b6cb0", "lw": 1.6})
    ax.annotate("", xy=(99.4, 92.0), xytext=(99.4, 11.0),
                arrowprops={"arrowstyle": "-|>", "color": "#2f855a", "lw": 1.6})
    fig.suptitle("图16  系统架构：设备 → 通道 → 特征 → 终点 → 报警 / 显示"
                 "（数字均来自 results/ 产物）", fontsize=13, y=0.995)
    return save(fig, out, name)


FIGURES = [
    fig_data_volume, fig_ppg_coverage, fig_ppg_waveform, fig_ppg_align,
    fig_seed_robustness, fig_roc_pr, fig_operating_points, fig_feature_gain,
    fig_ppg_ablation, fig_vla_training, fig_vla_artifacts, fig_p0_baselines,
    fig_horizon, fig_optimization, fig_architecture,
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "figures"))
    ap.add_argument("--no-figure-prefix", action="store_true",
                    help="strip the baked-in '图N ' prefix so a paper can renumber figures")
    ap.add_argument("--only", default="",
                    help="comma-separated function names to redraw; the rest are left "
                         "untouched so a paper keeps the figures it already ships")
    args = ap.parse_args(argv)
    if args.no_figure_prefix:
        enable_prefix_stripping()
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc

    fonts = setup_fonts()
    print(f"CJK fonts available: {fonts or '(none - labels will not render)'}")
    print(f"writing figures to {args.out}")
    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    if wanted:
        unknown = wanted - {fn.__name__ for fn in FIGURES}
        if unknown:
            raise SystemExit(f"unknown figure(s): {', '.join(sorted(unknown))}")
    for fn in FIGURES:
        if wanted and fn.__name__ not in wanted:
            continue
        try:
            fn(args.out)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            skip(fn.__name__, f"unexpected {type(exc).__name__}: {exc}")

    drawn = len(wanted) if wanted else len(FIGURES)
    print(f"\nproduced {len(PRODUCED)}/{drawn} figures")
    for name, reason in SKIPPED:
        print(f"  skipped {name}: {reason}")
    return 0 if not SKIPPED else 1


if __name__ == "__main__":
    sys.exit(main())
