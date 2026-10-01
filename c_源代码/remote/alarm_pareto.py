"""Alarm policy Pareto front: what each (persistence, hysteresis) setting costs and saves.

The alarm layer has three knobs and they trade against each other. This sweeps them on
the real test split and reports, per setting, what the monitor would do:

  * episodes detected  - does an alarm run overlap the episode's actionable window?
  * alarms per hour    - how often the ward is disturbed
  * false alarms/hour  - alarms with no onset within the horizon
  * median lead time   - how early, on detected episodes
  * first-alarm delay  - what persistence costs on a case's first alarm

Scores come from a p0_eval `--dump-scores` dump (caseid, t, p_*), so this measures the
same fitted model the rest of the report describes.

    python alarm_pareto.py --scores <dir> --variant main --threshold 0.5
"""
import argparse
import json
import os
import sys

import alarm_policy
import build_dataset as bd
import numpy as np
import pandas as pd

STRIDE_S = bd.STRIDE_S
PERSIST = (1, 2, 3)
CLEAR = (1.0, 0.9, 0.8)


def list_parquet(directory):
    try:
        names = sorted(os.listdir(directory))
    except OSError as exc:
        raise SystemExit(f"cannot list {directory}: {exc}") from exc
    return [os.path.join(directory, n) for n in names if n.endswith(".parquet")]


def load_cases(scores_dir, variant, tracks_dir, endpoint):
    """Per-case records with the dumped scores and the true episode onsets."""
    paths = list_parquet(scores_dir)
    if not paths:
        raise SystemExit(f"no score dumps in {scores_dir}")
    scores = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    col = f"p_{variant}"
    if col not in scores.columns:
        raise SystemExit(f"{col} not in the dumps ({list(scores.columns)})")

    cases = []
    for cid, grp in scores.groupby("caseid", sort=True):
        cid = bd._i(cid)
        grp = grp.sort_values("t")
        loaded = bd.load_case_signals(cid, tracks_dir, bd.HORIZON_S)
        if loaded[0] is None:
            continue
        flags = bd.endpoint_flags(loaded[1], endpoint).astype(np.int8)
        onsets = np.flatnonzero(np.diff(flags) == 1).astype("float64") + 1.0
        cases.append({
            "caseid": cid,
            "t": grp["t"].to_numpy(dtype="float64"),
            "p": {variant: grp[col].to_numpy(dtype="float64")},
            "onset": onsets,
        })
    return cases


def detection_under_policy(cases, variant, threshold, clear, persist):
    """Episodes whose actionable window [onset-300, onset) overlaps a policy alarm."""
    n_eps = n_detected = 0
    for c in cases:
        runs = alarm_policy.alarm_runs(c["t"], c["p"][variant], threshold, clear, persist)
        t_last = bd._f(c["t"][-1], 0.0)
        for onset in c["onset"]:
            lo = onset - bd.HORIZON_S
            if lo < bd.PAST_S:
                lo = bd._f(bd.PAST_S, 300.0)
            if c["t"][0] > onset:
                continue  # no decision time precedes this episode
            n_eps += 1
            for raise_t, clear_t in runs:
                end = clear_t if np.isfinite(clear_t) else t_last
                if raise_t < onset and end >= lo:
                    n_detected += 1
                    break
    return n_eps, n_detected


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=r"D:\vitaldb_local\scores_h300")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--endpoint", default="hypotension")
    ap.add_argument("--tracks-dir", default=r"D:\vitaldb_local\numeric")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cases = load_cases(args.scores, args.variant, args.tracks_dir, args.endpoint)
    if not cases:
        raise SystemExit("no cases with both dumped scores and readable tracks")
    hours = sum(c["t"].size for c in cases) * STRIDE_S / 3600.0
    print(f"variant={args.variant} endpoint={args.endpoint} "
          f"threshold={args.threshold:g} cases={len(cases)} hours={hours:.1f}")

    rows = []
    for persist in PERSIST:
        for clear_factor in CLEAR:
            burden = alarm_policy.alarm_burden(
                cases, args.variant, args.threshold, pre=bd.HORIZON_S,
                persist_windows=persist, clear_factor=clear_factor)
            n_eps, n_det = detection_under_policy(
                cases, args.variant, args.threshold, args.threshold * clear_factor,
                persist)
            rows.append({
                "persist": persist,
                "clear_factor": clear_factor,
                "detect": n_det / n_eps if n_eps else bd.NAN,
                "alarms_per_h": burden["alarm_rate_per_h"],
                "false_per_h": burden["false_alarm_rate_per_h"],
                "lead_median_s": burden["lead_median_s"],
                "first_delay_s": burden["first_alarm_delay_median_s"],
            })

    table = pd.DataFrame(rows)
    print(f"\n{'persist':>7s} {'clear':>6s} {'detect':>7s} {'alarms/h':>9s} "
          f"{'false/h':>8s} {'lead_s':>7s} {'delay_s':>8s}")
    for _, r in table.iterrows():
        delay = bd._f(r["first_delay_s"])
        print(f"{bd._i(r['persist']):7d} {r['clear_factor']:6.2f} {r['detect']:7.4f} "
              f"{r['alarms_per_h']:9.2f} {r['false_per_h']:8.2f} "
              f"{r['lead_median_s']:7.0f} {delay:8.0f}")

    front = []
    ordered = table.sort_values(["false_per_h", "detect"], ascending=[True, False])
    for _, r in ordered.iterrows():
        if not front or r["detect"] > max(f["detect"] for f in front):
            front.append(r)
    print("\nPareto front (no setting has both more detection and fewer false alarms):")
    for r in front:
        print(f"  persist={bd._i(r['persist'])} clear={r['clear_factor']:.2f}  "
              f"detect={r['detect']:.4f}  false/h={r['false_per_h']:.2f}  "
              f"alarms/h={r['alarms_per_h']:.2f}  lead={r['lead_median_s']:.0f}s")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "results", "alarm_pareto.json")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write {out}: {exc}") from exc
    print(f"\nwrote {os.path.normpath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
