"""Train + evaluate the intraoperative hypotension predictor.

Split is by caseid (90/10) so no window from a test case can leak into training.
Threshold selection happens on a validation split carved out of TRAIN, never on
test.

Usage:
    python train_model.py --data /root/autodl-tmp/processed/windows.parquet
"""
import argparse
import json
import os
import sys

import lightgbm as lgb  # noqa: E402  (remote-only dependency)
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

DROP = ("caseid", "t", "label")
NAN = np.nan


def _f(x, default=NAN):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def split_cases(caseids, test_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(caseids))
    perm = rng.permutation(len(uniq))
    n_test = max(1, _i(round(len(uniq) * test_frac)))
    test_ids = set(uniq[perm[:n_test]].tolist())
    train_ids = set(uniq[perm[n_test:]].tolist())
    return train_ids, test_ids


def make_valid(train_ids, frac, seed):
    """Carve a caseid-disjoint validation split out of the training cases."""
    rng = np.random.default_rng(seed + 1)
    uniq = np.array(sorted(train_ids))
    perm = rng.permutation(len(uniq))
    n_val = max(1, _i(round(len(uniq) * frac)))
    val_ids = set(uniq[perm[:n_val]].tolist())
    fit_ids = set(uniq[perm[n_val:]].tolist())
    return fit_ids, val_ids


def metrics(y, p, thr):
    pred = (p >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "threshold": _f(thr),
        "accuracy": _f(accuracy_score(y, pred)),
        "balanced_accuracy": _f(balanced_accuracy_score(y, pred)),
        "auroc": _f(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else NAN,
        "auprc": _f(average_precision_score(y, p)) if len(np.unique(y)) > 1 else NAN,
        "sensitivity": _f(sens),
        "specificity": _f(spec),
        "precision": _f(tp / max(tp + fp, 1)),
        "tp": _i(tp), "fp": _i(fp), "tn": _i(tn), "fn": _i(fn),
        "n": _i(len(y)),
        "positive_rate": _f(y.mean()),
    }


def best_threshold(y, p):
    """Threshold maximising accuracy on the validation split."""
    order = np.argsort(-p)
    ys = y[order]
    cum_tp = np.cumsum(ys)
    total_pos = ys.sum()
    idx = np.arange(1, len(ys) + 1)
    acc = (cum_tp + (len(ys) - idx) - (total_pos - cum_tp)) / len(ys)
    k = _i(np.argmax(acc))
    thr = _f(p[order][k]) if k < len(ys) else 0.5
    return thr, _f(acc[k])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/root/autodl-tmp/models")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.data)
    feats = [c for c in df.columns if c not in DROP]
    print(f"rows={len(df)} cases={df.caseid.nunique()} features={len(feats)}", flush=True)
    print(f"overall positive_rate={df.label.mean():.4f}", flush=True)

    train_ids, test_ids = split_cases(df.caseid.unique(), args.test_frac, args.seed)
    fit_ids, val_ids = make_valid(train_ids, args.val_frac, args.seed)
    print(f"cases: fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    m_fit = df.caseid.isin(fit_ids)
    m_val = df.caseid.isin(val_ids)
    m_test = df.caseid.isin(test_ids)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": 16,          # container is limited to 16 vCPU (nproc lies)
        "verbosity": -1,
        "seed": args.seed,
    }

    ds_fit = lgb.Dataset(df.loc[m_fit, feats], label=df.loc[m_fit, "label"])
    ds_val = lgb.Dataset(df.loc[m_val, feats], label=df.loc[m_val, "label"], reference=ds_fit)
    booster = lgb.train(
        params, ds_fit, num_boost_round=args.rounds, valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)],
    )
    print(f"best_iteration={booster.best_iteration}", flush=True)

    y_val = df.loc[m_val, "label"].to_numpy()
    p_val = booster.predict(df.loc[m_val, feats], num_iteration=booster.best_iteration)
    thr, val_acc = best_threshold(y_val, p_val)
    print(f"val-tuned threshold={thr:.4f} (val acc={val_acc:.4f})", flush=True)

    y_test = df.loc[m_test, "label"].to_numpy()
    p_test = booster.predict(df.loc[m_test, feats], num_iteration=booster.best_iteration)

    report = {
        "task": "intraoperative hypotension, MAP<65 for >=1min within next 5min",
        "split": f"caseid-disjoint {1 - args.test_frac:.0%}/{args.test_frac:.0%}",
        "n_features": len(feats),
        "cases": {"fit": len(fit_ids), "val": len(val_ids), "test": len(test_ids)},
        "rows": {"fit": _i(m_fit.sum()), "val": _i(m_val.sum()), "test": _i(m_test.sum())},
        "best_iteration": _i(booster.best_iteration),
        "test_at_0.5": metrics(y_test, p_test, 0.5),
        "test_at_val_threshold": metrics(y_test, p_test, thr),
        "test_majority_baseline_acc": _f(max(y_test.mean(), 1 - y_test.mean())),
    }

    print("\n=== TEST (threshold 0.50) ===")
    for k, v in report["test_at_0.5"].items():
        print(f"  {k:20s} {v}")
    print(f"\n=== TEST (val-tuned threshold {thr:.4f}) ===")
    for k, v in report["test_at_val_threshold"].items():
        print(f"  {k:20s} {v}")
    print(f"\n  majority_baseline_acc {report['test_majority_baseline_acc']:.4f}")

    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc
    booster.save_model(os.path.join(args.out, "lgbm_hypotension.txt"), num_iteration=booster.best_iteration)
    try:
        with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write report: {exc}") from exc

    imp = sorted(zip(feats, booster.feature_importance("gain"), strict=True),
                 key=lambda x: -x[1])[:20]
    print("\n=== top features (gain) ===")
    for name, gain in imp:
        print(f"  {name:22s} {gain:12.1f}")
    print(f"\nwrote {args.out}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
