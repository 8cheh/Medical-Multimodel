"""Does signal-quality gating cut false alarms, and what does it cost in detection?

The pipeline has no runtime quality check: range clipping happens when the dataset is
built, and the monitoring display already showed real artifacts (HR spikes to 200+,
MAP spikes). This derives a per-window quality flag from features the dataset already
carries, gates the risk score when quality is poor, and measures the trade against the
ungated detector - alarms, false alarms and episode detection, all on the real test set.

Quality is deliberately simple and auditable:
  * coverage of the channels the model leans on (map/sbp/dbp/hr/spo2) >= 0.5, and
  * not a flat line (a zero standard deviation means a disconnected/constant sensor).

    python quality_gate.py --scores <dir> --windows <windows.parquet>
"""
import argparse
import json
import os
import sys

import alarm_pareto
import alarm_policy
import build_dataset as bd
import numpy as np
import pandas as pd

COVERAGE_CHANNELS = ("map", "sbp", "dbp", "hr", "spo2")
MIN_COVERAGE = 0.5


def quality_mask(features, index):
    """Boolean 'usable' per row of `index` (caseid, t), from the dataset's own columns."""
    cov_cols = [f"{c}_cov" for c in COVERAGE_CHANNELS if f"{c}_cov" in features.columns]
    std_cols = [f"{c}_std" for c in COVERAGE_CHANNELS if f"{c}_std" in features.columns]
    if not cov_cols:
        raise SystemExit("no *_cov columns in the feature table; cannot gate on quality")
    cov = np.asarray(features[cov_cols], dtype="float64")
    usable = (cov >= MIN_COVERAGE).all(axis=1)
    if std_cols:
        std = np.asarray(features[std_cols], dtype="float64")
        flat = np.isnan(std) | (std <= 0.0)          # constant channel == no signal
        usable &= ~flat.all(axis=1)                  # only if ALL are flat
    return usable


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=r"D:\vitaldb_local\scores_h300")
    ap.add_argument("--windows", default=r"D:\vitaldb_local\processed\windows.parquet")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--endpoint", default="hypotension")
    ap.add_argument("--tracks-dir", default=r"D:\vitaldb_local\numeric")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--clear-factor", type=float, default=0.8)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cases = alarm_pareto.load_cases(args.scores, args.variant, args.tracks_dir,
                                    args.endpoint)
    if not cases:
        raise SystemExit("no cases with both dumped scores and readable tracks")
    feats = pd.read_parquet(args.windows,
                            columns=["caseid", "t"]
                            + [f"{c}_cov" for c in COVERAGE_CHANNELS]
                            + [f"{c}_std" for c in COVERAGE_CHANNELS])
    keyed = feats.set_index(["caseid", "t"])

    used = 0
    total = 0
    for c in cases:
        idx = pd.MultiIndex.from_arrays([np.full(c["t"].size, c["caseid"]), c["t"]])
        try:
            sub = keyed.loc[idx]
        except KeyError:
            continue
        usable = quality_mask(sub, idx)
        total += usable.size
        used += bd._i(usable.sum())
        gated = np.where(usable, c["p"][args.variant], 0.0)  # poor quality -> no alarm
        c["p"]["gated"] = gated

    print(f"variant={args.variant} endpoint={args.endpoint} cases={len(cases)} "
          f"windows={total} usable={used} ({used / max(total, 1) * 100:.2f}%)")

    rows = {}
    for name, variant in (("ungated", args.variant), ("gated", "gated")):
        burden = alarm_policy.alarm_burden(
            cases, variant, args.threshold, pre=bd.HORIZON_S,
            persist_windows=args.persist, clear_factor=args.clear_factor)
        _, detect = alarm_pareto.detection_under_policy(
            cases, variant, args.threshold, args.threshold * args.clear_factor,
            args.persist)
        rows[name] = {"alarms_per_h": burden["alarm_rate_per_h"],
                      "false_per_h": burden["false_alarm_rate_per_h"],
                      "n_true": burden["n_true"], "n_alarms": burden["n_alarms"],
                      "detect": detect, "lead_median_s": burden["lead_median_s"]}
        print(f"  {name:8s} alarms/h={rows[name]['alarms_per_h']:.2f} "
              f"false/h={rows[name]['false_per_h']:.2f} "
              f"detected={detect} lead={burden['lead_median_s']:.0f}s")

    change = (rows["gated"]["false_per_h"] - rows["ungated"]["false_per_h"])
    print(f"\ngating changes false alarms by {change:+.2f}/h "
          f"({rows['ungated']['false_per_h']:.2f} -> {rows['gated']['false_per_h']:.2f}) "
          f"and detection by {rows['gated']['detect'] - rows['ungated']['detect']:+d}")

    payload = {"variant": args.variant, "endpoint": args.endpoint,
               "windows": total, "usable": used,
               "usable_fraction": used / max(total, 1),
               "threshold": args.threshold, "persist": args.persist,
               "clear_factor": args.clear_factor, "policies": rows}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "results", "quality_gate.json")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write {out}: {exc}") from exc
    print(f"wrote {os.path.normpath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
