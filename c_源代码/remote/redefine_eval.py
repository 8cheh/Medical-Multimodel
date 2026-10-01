"""Evaluate the current model against the EVALUATION PROTOCOL the objective should use.

Motivation: with prevalence pi = 0.125,
    Acc = pi*TPR + (1-pi)*TNR = (1-pi) + pi*(TPR-FPR)
so accuracy is only an affine transform of the Youden index J with slope pi=0.125.
A constant predictor scores 1-pi = 0.875. Accuracy therefore cannot certify value.

This script computes the quantities a defensible objective needs:
  1. Clustered (by caseid) bootstrap CIs. Windows within a case are strongly
     correlated, so resampling windows would understate uncertainty ~sqrt(n_win/n_case).
  2. The two baselines that must actually be beaten:
       - constant predictor
       - the single best feature alone (map_last60, i.e. "read the current MAP")
  3. Alarm-constrained sensitivity: max TPR s.t. FPR <= alpha.
  4. Partial AUROC over the low-alarm region, AUPRC, Brier and ECE (calibration).
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

NAN = np.nan
DROP = ("caseid", "t", "label")
ALPHAS = (0.01, 0.05, 0.10)
N_BOOT = 400
SINGLE_FEATURE = "map_last60"


def to_float(x, default=NAN):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def to_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def load_lightgbm():
    """lightgbm exists on the compute host, not on the dev laptop."""
    try:
        return importlib.import_module("lightgbm")
    except ImportError as exc:
        raise SystemExit("lightgbm is required on the compute host") from exc


def split_cases(caseids, test_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(caseids))
    perm = rng.permutation(len(uniq))
    n_test = max(1, to_int(round(len(uniq) * test_frac)))
    return set(uniq[perm[n_test:]].tolist()), set(uniq[perm[:n_test]].tolist())


def carve_valid(train_ids, frac, seed):
    rng = np.random.default_rng(seed + 1)
    uniq = np.array(sorted(train_ids))
    perm = rng.permutation(len(uniq))
    n_val = max(1, to_int(round(len(uniq) * frac)))
    return set(uniq[perm[n_val:]].tolist()), set(uniq[perm[:n_val]].tolist())


def sens_at_fpr(y, p, alpha):
    """Highest TPR achievable with FPR <= alpha (Neyman-Pearson operating point)."""
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= alpha
    return to_float(tpr[ok].max()) if ok.any() else 0.0


def ece(y, p, bins=10):
    """Expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if not m.any():
            continue
        total += to_float(m.mean()) * abs(to_float(p[m].mean()) - to_float(y[m].mean()))
    return to_float(total)


def core_metrics(y, p, thr=0.5):
    pred = p >= thr
    tp = to_int(np.sum(pred & (y == 1)))
    fp = to_int(np.sum(pred & (y == 0)))
    fn = to_int(np.sum(~pred & (y == 1)))
    tn = to_int(np.sum(~pred & (y == 0)))
    out = {
        "accuracy": (tp + tn) / max(len(y), 1),
        "auroc": to_float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else NAN,
        "auprc": to_float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else NAN,
        "pauc_fpr10": to_float(roc_auc_score(y, p, max_fpr=0.10)) if len(np.unique(y)) > 1 else NAN,
        "brier": to_float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }
    for a in ALPHAS:
        out[f"sens@fpr<={a:.2f}"] = sens_at_fpr(y, p, a)
    return out


def clustered_bootstrap(df, pred_col, thr, n_boot=N_BOOT, seed=7):
    """Resample CASES (not windows) with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("caseid", sort=False)]
    n = len(groups)
    keys = ("accuracy", "auroc", "auprc", "sensitivity", "sens@fpr<=0.10")
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        try:
            sub = pd.concat([groups[int(i)] for i in pick], ignore_index=True)
        except (ValueError, KeyError):
            continue
        y = sub["label"].to_numpy()
        p = sub[pred_col].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        m = core_metrics(y, p, thr)
        for k in keys:
            acc[k].append(m[k])
    out = {}
    for k, vals in acc.items():
        arr = np.array([v for v in vals if np.isfinite(v)], dtype="float64")
        if arr.size:
            out[k] = {"lo": round(to_float(np.percentile(arr, 2.5)), 4),
                      "hi": round(to_float(np.percentile(arr, 97.5)), 4),
                      "mean": round(to_float(arr.mean()), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/processed/windows.parquet")
    ap.add_argument("--out", default="/root/autodl-tmp/models/redefined")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args(argv)
    lgb = load_lightgbm()

    df = pd.read_parquet(args.data)
    feats = [c for c in df.columns if c not in DROP]
    train_ids, test_ids = split_cases(df.caseid.unique(), 0.10, args.seed)
    fit_ids, val_ids = carve_valid(train_ids, 0.10, args.seed)
    print(f"rows={len(df)} feats={len(feats)} "
          f"cases fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": 6, "verbosity": -1, "seed": args.seed}
    m_fit = df.caseid.isin(fit_ids)
    m_test = df.caseid.isin(test_ids)
    ds_fit = lgb.Dataset(df.loc[m_fit, feats], label=df.loc[m_fit, "label"])
    booster = lgb.train(params, ds_fit, num_boost_round=args.rounds)

    test = df.loc[m_test, ["caseid", "t", "label", SINGLE_FEATURE]].copy()
    test["p_model"] = booster.predict(df.loc[m_test, feats])
    # "read the current MAP": lower MAP -> higher risk
    test["p_maponly"] = -test[SINGLE_FEATURE].to_numpy()

    y = test["label"].to_numpy()
    pi = to_float(y.mean())
    print(f"\ntest n={len(y)} cases={test.caseid.nunique()} prevalence={pi:.4f} "
          f"constant_baseline_acc={max(pi, 1 - pi):.4f}")

    rows = {}
    for name, col, thr in (("model", "p_model", 0.5),
                           ("map_only_baseline", "p_maponly", 0.0)):
        m = core_metrics(y, test[col].to_numpy(), thr)
        rows[name] = m
        print(f"\n=== {name} ===")
        for k, v in m.items():
            print(f"  {k:22s} {v:.4f}")

    print("\n=== clustered bootstrap 95% CI (resampled by caseid) ===")
    ci = clustered_bootstrap(test, "p_model", 0.5)
    for k, v in ci.items():
        print(f"  {k:18s} mean={v['mean']:.4f}  95% CI [{v['lo']:.4f}, {v['hi']:.4f}]")

    print("\n=== must-beat-baselines test (is the model actually adding value?) ===")
    deltas = {}
    for k in ("accuracy", "auroc", "auprc", "sensitivity", "sens@fpr<=0.10"):
        d = rows["model"][k] - rows["map_only_baseline"][k]
        deltas[k] = to_float(d)
        print(f"  {k:22s} model-map_only = {d:+.4f}")

    report = {
        "test_n": to_int(len(y)),
        "test_cases": to_int(test.caseid.nunique()),
        "prevalence": round(pi, 4),
        "constant_baseline_acc": round(to_float(max(pi, 1 - pi)), 4),
        "model": {k: round(to_float(v), 4) for k, v in rows["model"].items()},
        "map_only_baseline": {k: round(to_float(v), 4) for k, v in rows["map_only_baseline"].items()},
        "delta_model_minus_map_only": {k: round(v, 4) for k, v in deltas.items()},
        "clustered_bootstrap_95ci": ci,
    }
    try:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "redefined_eval.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write report: {exc}") from exc
    print(f"\nwrote {args.out}/redefined_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
