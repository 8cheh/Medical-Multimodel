"""Monitoring display: one case, the monitored channels, the risk curve, the alarms.

This is the "display" side of the monitor - what a clinician would actually look at.
Drawn from local artifacts only: the case's own numeric tracks, the per-window risk
scores dumped by p0_eval (--dump-scores), and the alarm policy in alarm_policy.py, so
the panel shows what the pipeline produces rather than an illustration of it.

    python make_monitor_display.py --scores <dir with scores_*.parquet> --out <png>
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import alarm_policy
import build_dataset as bd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What to show for each monitored risk: the channel that defines it, plus the other
# channels a clinician would want beside it.
ENDPOINT_VIEW = {
    "hypotension": {"primary": "map", "cut": 65.0, "unit": "mmHg",
                    "others": ("hr", "spo2", "etco2")},
    "hypoxemia": {"primary": "spo2", "cut": 90.0, "unit": "%",
                  "others": ("map", "hr", "etco2")},
    "bradycardia": {"primary": "hr", "cut": 50.0, "unit": "bpm",
                    "others": ("map", "spo2", "etco2")},
}
LABEL = {"map": "MAP (mmHg)", "spo2": "SpO2 (%)", "hr": "HR (bpm)",
         "etco2": "EtCO2 (mmHg)"}


def as_int(value, default=0):
    """int() that never raises (counts come from pandas/argparse values)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def setup_cjk_fonts():
    """Pick an installed CJK family so the panel labels are not tofu boxes."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return ""


def list_parquet(directory):
    try:
        names = sorted(os.listdir(directory))
    except OSError as exc:
        raise SystemExit(f"cannot list {directory}: {exc}") from exc
    return [os.path.join(directory, n) for n in names if n.endswith(".parquet")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=r"D:\vitaldb_local\scores_hypmm")
    ap.add_argument("--caseid", type=int, default=0, help="0 = pick automatically")
    ap.add_argument("--variant", default="main_multimodal")
    ap.add_argument("--endpoint", default="hypoxemia")
    ap.add_argument("--tracks-dir", default=r"D:\vitaldb_local\numeric")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--clear-factor", type=float, default=0.8)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    paths = list_parquet(args.scores)
    if not paths:
        raise SystemExit(f"no score dumps in {args.scores}")
    scores = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    col = f"p_{args.variant}"
    if col not in scores.columns:
        raise SystemExit(f"{col} not in the dumps ({list(scores.columns)})")

    view = ENDPOINT_VIEW[args.endpoint]
    per_case = scores.groupby("caseid")[col].max()
    if args.caseid:
        caseid = args.caseid
    else:
        # prefer a case with real episodes and a firing alarm, so the panel is not
        # an empty illustration
        cands = scores.groupby("caseid").label.sum().sort_values(ascending=False)
        caseid = as_int(next((c for c in cands.index
                              if per_case.get(c, 0) > args.threshold), cands.index[0]))
    sub = scores[scores.caseid == caseid].sort_values(by=["t"])
    grid, sig, _dur = bd.load_case_signals(caseid, args.tracks_dir)

    t_min = grid / 60.0
    limit = as_int(min(args.max_hours * 3600, grid[-1]))
    keep = grid <= limit
    flags = bd.endpoint_flags(sig, args.endpoint)
    onsets = np.flatnonzero(flags[1:] & ~flags[:-1]) + 1 if flags.size > 1 else np.array([])
    runs = alarm_policy.alarm_runs(sub["t"].to_numpy(), sub[col].to_numpy(),
                                   args.threshold, args.threshold * args.clear_factor,
                                   args.persist)

    font = setup_cjk_fonts()
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 8.6), sharex=True)
    ax = axes[0]
    key = view["primary"]
    ax.plot(t_min[keep], sig[key][keep], color="#2b6cb0", linewidth=1.0,
            label=LABEL[key])
    ax.axhline(view["cut"], color="#c53030", linestyle="--", linewidth=1.2,
               label=f"阈值 {view['cut']:g} {view['unit']}")
    for onset in onsets:
        if onset <= limit:
            ax.axvspan(onset / 60.0, (onset + 60) / 60.0, color="#c53030", alpha=0.14)
    ax.set_ylabel(LABEL[key])
    ax.set_title(f"case {caseid} — 被监测通道与事件（{args.endpoint}）", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, linestyle=":")

    ax = axes[1]
    for name, color in zip(view["others"], ("#805ad5", "#2f855a", "#dd6b20"),
                           strict=False):
        v = sig[name][keep]
        fin = np.isfinite(v)
        if not fin.any():
            continue
        ax.plot(t_min[keep][fin], v[fin], color=color, linewidth=0.9, label=LABEL[name])
    ax.set_ylabel("其余通道")
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    ax.grid(alpha=0.25, linestyle=":")

    ax = axes[2]
    ax.plot(sub["t"] / 60.0, sub[col], color="#1a202c", linewidth=1.4,
            label=f"风险分数（{args.variant}）")
    ax.axhline(args.threshold, color="#c53030", linestyle="--", linewidth=1.2,
               label=f"报警阈值 {args.threshold:g}")
    for raise_t, clear_t in runs:
        end = alarm_policy.as_float(clear_t, 0.0) if np.isfinite(clear_t) \
            else alarm_policy.as_float(sub["t"].max(), 0.0)
        ax.axvspan(raise_t / 60.0, end / 60.0, color="#dd6b20", alpha=0.18)
    for raise_t, _clear in runs:
        ax.annotate("报警", (raise_t / 60.0, 1.02), fontsize=7.5, color="#c05621",
                    ha="center")
    hysteresis = as_int((1.0 - args.clear_factor) * 100)
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel("时间 (min)")
    ax.set_ylabel("报警风险")
    ax.set_title(f"报警策略：连续 {args.persist} 窗确认 + {hysteresis}% 回差"
                 f" → {len(runs)} 次报警", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, linestyle=":")
    for a in axes:
        a.set_xlim(0, limit / 60.0)

    fig.tight_layout()
    out = args.out or os.path.join(ROOT, "report", "figures", "fig15_monitor_display.png")
    try:
        fig.savefig(out)
    except OSError as exc:
        raise SystemExit(f"cannot write {out}: {exc}") from exc
    plt.close(fig)
    print(f"fonts={font or '(none)'} case {caseid}: {len(runs)} alarms, "
          f"{as_int(flags.sum())} s in episode, {len(onsets)} onsets -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
