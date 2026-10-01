"""Calibration and decision-curve analysis for the monitored endpoints.

Two questions the AUROC table cannot answer:

  * **Is the probability meaningful?** AUROC only ranks. If a monitor says "12%", a
    clinician needs 12% to be about right, so this reports the Brier score and the
    expected calibration error (10 equal-width bins).
  * **Is alarming worth it at a given threshold?** Decision-curve analysis gives the
    net benefit of using the model versus "alarm on nobody" and "alarm on everybody",
    for a range of threshold probabilities. Net benefit here is
    (TP - FP * pt/(1-pt)) / n, i.e. false alarms are weighted by the odds of the
    threshold the user is willing to accept.

Reads p0_eval `--dump-scores` dumps, so it describes the same fitted models the rest of
the report uses.

    python calibration_dca.py --scores <dir> --variants main --labels hypotension
"""
import argparse
import json
import os
import sys
from itertools import pairwise

import numpy as np
import pandas as pd

NAN = np.nan
BINS = 10
THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def as_float(value, default=NAN):
    """float() that never raises (scores and counts arrive as numpy scalars)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def list_parquet(directory):
    try:
        names = sorted(os.listdir(directory))
    except OSError as exc:
        raise SystemExit(f"cannot list {directory}: {exc}") from exc
    return [os.path.join(directory, n) for n in names if n.endswith(".parquet")]


def brier(y, p):
    return as_float(np.mean((p - y) ** 2))


def ece(y, p, bins=BINS):
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(y), 1)
    err = 0.0
    for lo, hi in pairwise(edges):
        m = (p > lo) & (p <= hi)
        if not m.any():
            continue
        gap = abs(as_float(p[m].mean()) - as_float(y[m].mean()))
        err += m.sum() / total * gap
    return as_float(err)


def net_benefit(y, p, pt):
    """TP/n - FP/n * pt/(1-pt); the alarm-all strategy is the no-model reference."""
    pred = p >= pt
    tp = as_float(np.sum(pred & (y == 1)))
    fp = as_float(np.sum(pred & (y == 0)))
    n = as_float(max(len(y), 1))
    nb = tp / n - fp / n * (pt / (1.0 - pt))
    prev = as_float(y.mean())
    nb_all = prev - (1.0 - prev) * (pt / (1.0 - pt))
    return as_float(nb), as_float(nb_all)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=r"D:\vitaldb_local\scores_h300")
    ap.add_argument("--variants", nargs="+", default=["main"])
    ap.add_argument("--labels", default="hypotension")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    paths = list_parquet(args.scores)
    if not paths:
        raise SystemExit(f"no score dumps in {args.scores}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    y = df["label"].to_numpy(dtype="float64")
    prevalence = as_float(y.mean())
    print(f"{args.labels}: windows={len(df)} prevalence={prevalence:.4f} "
          f"(reference Brier of a constant predictor = "
          f"{prevalence * (1 - prevalence):.4f})")

    report = {"label": args.labels, "windows": len(df),
              "prevalence": prevalence, "variants": {}}
    print(f"\n{'variant':16s} {'Brier':>7s} {'ECE':>7s}  decision-curve net benefit "
          f"(model | alarm-all) at threshold probability")
    for name in args.variants:
        col = f"p_{name}"
        if col not in df.columns:
            print(f"  {name}: {col} not in the dumps")
            continue
        p = df[col].to_numpy(dtype="float64")
        nb_model, nb_all, rows = [], [], []
        for pt in THRESHOLDS:
            nb, nba = net_benefit(y, p, pt)
            nb_model.append(round(nb, 5))
            nb_all.append(round(nba, 5))
            rows.append({"pt": pt, "nb_model": nb, "nb_alarm_all": nba})
        entry = {"brier": brier(y, p), "ece": ece(y, p), "dca": rows,
                 "best_pt_by_nb": THRESHOLDS[max(range(len(nb_model)),
                                                 key=nb_model.__getitem__)]}
        report["variants"][name] = entry
        curve = " ".join(f"{m:+.4f}|{a:+.4f}" for m, a in zip(nb_model, nb_all,
                                                             strict=True))
        print(f"{name:16s} {entry['brier']:7.4f} {entry['ece']:7.4f}  {curve}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "results", f"calibration_{args.labels}.json")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write {out}: {exc}") from exc
    print(f"\nwrote {os.path.normpath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
