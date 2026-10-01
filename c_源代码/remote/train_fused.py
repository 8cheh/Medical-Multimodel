"""Does the PPG waveform actually help? Train both variants on one fixed split.

Model A: numeric features only
Model B: numeric + PPG waveform features

Both use the SAME caseid-disjoint split and the SAME seed, so the only difference
is the feature set. The comparison is only run on cases that have PPG, so neither
model gets a data advantage.
"""
import argparse
import json
import os
import sys

try:
    import lightgbm as lgb
except ImportError as exc:  # remote-only dependency; absent on the dev laptop
    raise SystemExit("lightgbm is required on the compute host: pip install lightgbm") from exc

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

NAN = np.nan
DROP = ("caseid", "t", "label")


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


def evaluate(y, p, thr=0.5):
    pred = (p >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": to_float(accuracy_score(y, pred)),
        "balanced_accuracy": to_float(balanced_accuracy_score(y, pred)),
        "auroc": to_float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else NAN,
        "auprc": to_float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else NAN,
        "sensitivity": to_float(tp / max(tp + fn, 1)),
        "specificity": to_float(tn / max(tn + fp, 1)),
        "precision": to_float(tp / max(tp + fp, 1)),
        "n": to_int(len(y)),
        "positive_rate": to_float(y.mean()),
        "majority_baseline_acc": to_float(max(y.mean(), 1 - y.mean())),
    }


def fit_predict(df, feats, fit_ids, val_ids, test_ids, seed, rounds=2000):
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "num_threads": 16, "verbosity": -1, "seed": seed,
    }
    m_fit = df.caseid.isin(fit_ids)
    m_val = df.caseid.isin(val_ids)
    m_test = df.caseid.isin(test_ids)
    ds_fit = lgb.Dataset(df.loc[m_fit, feats], label=df.loc[m_fit, "label"])
    ds_val = lgb.Dataset(df.loc[m_val, feats], label=df.loc[m_val, "label"], reference=ds_fit)
    booster = lgb.train(params, ds_fit, num_boost_round=rounds, valid_sets=[ds_val],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    p_test = booster.predict(df.loc[m_test, feats], num_iteration=booster.best_iteration)
    return booster, p_test, df.loc[m_test, "label"].to_numpy(), to_int(booster.best_iteration)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="/root/autodl-tmp/processed/windows.parquet")
    ap.add_argument("--ppg", default="/root/autodl-tmp/processed/ppg_features.parquet")
    ap.add_argument("--out", default="/root/autodl-tmp/models/fused")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    num = pd.read_parquet(args.windows)
    ppg = pd.read_parquet(args.ppg)
    merged = num.merge(ppg, on=["caseid", "t"], how="inner")
    print(f"numeric rows={len(num)} ppg rows={len(ppg)} merged rows={len(merged)}")
    print(f"merged cases={merged.caseid.nunique()} positive_rate={merged.label.mean():.4f}")
    if merged.empty:
        print("nothing to compare")
        return 1

    num_feats = [c for c in num.columns if c not in DROP]
    ppg_feats = [c for c in ppg.columns if c not in ("caseid", "t")]
    print(f"numeric feats={len(num_feats)}  ppg feats={len(ppg_feats)}")

    train_ids, test_ids = split_cases(merged.caseid.unique(), args.test_frac, args.seed)
    fit_ids, val_ids = carve_valid(train_ids, args.val_frac, args.seed)
    print(f"cases fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    report = {"merged_rows": to_int(len(merged)), "merged_cases": to_int(merged.caseid.nunique()),
              "n_numeric_feats": len(num_feats), "n_ppg_feats": len(ppg_feats),
              "cases": {"fit": len(fit_ids), "val": len(val_ids), "test": len(test_ids)}}

    results = {}
    for name, feats in (("numeric_only", num_feats), ("numeric_plus_ppg", num_feats + ppg_feats)):
        booster, p, y, best_it = fit_predict(merged, feats, fit_ids, val_ids, test_ids, args.seed)
        m = evaluate(y, p)
        m["best_iteration"] = best_it
        m["n_features"] = len(feats)
        results[name] = m
        print(f"\n=== {name} ({len(feats)} feats, {best_it} trees) ===")
        for k in ("accuracy", "balanced_accuracy", "auroc", "auprc", "sensitivity",
                  "specificity", "precision", "majority_baseline_acc"):
            print(f"  {k:22s} {m[k]:.4f}")
        imp = sorted(zip(feats, booster.feature_importance("gain"), strict=True),
                     key=lambda x: -x[1])[:12]
        print("  top gain:", ", ".join(n for n, _ in imp))

    a, b = results["numeric_only"], results["numeric_plus_ppg"]
    print("\n=== DELTA (numeric+ppg  -  numeric only) ===")
    for k in ("accuracy", "auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy"):
        print(f"  {k:22s} {b[k] - a[k]:+.4f}   ({a[k]:.4f} -> {b[k]:.4f})")

    report["results"] = results
    report["delta"] = {k: to_float(b[k] - a[k]) for k in
                       ("accuracy", "auroc", "auprc", "sensitivity", "specificity",
                        "balanced_accuracy")}
    try:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write report: {exc}") from exc
    print(f"\nwrote {args.out}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
