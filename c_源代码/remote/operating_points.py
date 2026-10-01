"""Report clinically meaningful operating points, not just accuracy at 0.5.

Why: the positive rate is ~12.5%, so "accuracy > 85%" is nearly free (the majority
class scores 87.5%). A hypotension alarm is only useful if you know the trade-off
you are buying. This reports, for target sensitivities, the specificity / PPV /
NPV / likelihood ratios you actually get - with every threshold chosen on the
VALIDATION split, never on test.

Usage:
    python operating_points.py --data /root/autodl-tmp/processed/windows.parquet
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

NAN = np.nan


def load_lightgbm():
    """Import lightgbm lazily.

    It lives on the compute host but not on the dev laptop, so a static import
    would be an unresolved-import error for the whole file.
    """
    try:
        return importlib.import_module("lightgbm")
    except ImportError as exc:
        raise SystemExit("lightgbm is required on the compute host: pip install lightgbm") from exc
DROP = ("caseid", "t", "label")
TARGET_SENS = (0.90, 0.80, 0.70, 0.60)


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


def confusion(y, p, thr):
    pred = p >= thr
    tp = to_int(np.sum(pred & (y == 1)))
    fp = to_int(np.sum(pred & (y == 0)))
    fn = to_int(np.sum(~pred & (y == 1)))
    tn = to_int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def point(y, p, thr, label):
    tp, fp, fn, tn = confusion(y, p, thr)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "name": label,
        "threshold": round(to_float(thr), 4),
        "accuracy": round((tp + tn) / max(len(y), 1), 4),
        "balanced_accuracy": round((sens + spec) / 2, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "lr_pos": round(sens / max(1 - spec, 1e-9), 3),
        "lr_neg": round((1 - sens) / max(spec, 1e-9), 3),
        "alarms_per_100_windows": round(100.0 * (tp + fp) / max(len(y), 1), 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def thr_for_sensitivity(y, p, target):
    """Highest threshold whose validation sensitivity still reaches `target`.

    p is sorted descending, so sensitivity rises with k. The eligible k are the
    TAIL of that array (small k = high threshold); taking the smallest eligible k
    keeps specificity as high as the sensitivity target allows. (Taking ok[-1]
    would return the lowest threshold, i.e. alarm on everything.)
    """
    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]
    tp = np.cumsum(ys)
    pos = max(to_int(ys.sum()), 1)
    sens = tp / pos
    ok = np.nonzero(sens >= target)[0]
    if ok.size == 0:
        return 0.0
    k = to_int(ok[0])
    return to_float(ps[k])


def thr_max_balanced(y, p):
    """Threshold maximising balanced accuracy on the validation split."""
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    idx = np.arange(1, len(ys) + 1)
    pos = max(to_int(ys.sum()), 1)
    neg = max(len(ys) - pos, 1)
    sens = tp / pos
    fp = idx - tp
    spec = (neg - fp) / neg          # specificity, NOT false-positive rate
    bal = (sens + spec) / 2
    k = to_int(np.argmax(bal))
    return to_float(p[order][k]), to_float(bal[k])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/processed/windows.parquet")
    ap.add_argument("--out", default="/root/autodl-tmp/models/operating_points")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=3000)
    args = ap.parse_args(argv)
    lgb = load_lightgbm()

    df = pd.read_parquet(args.data)
    feats = [c for c in df.columns if c not in DROP]
    print(f"rows={len(df)} cases={df.caseid.nunique()} feats={len(feats)}", flush=True)

    train_ids, test_ids = split_cases(df.caseid.unique(), args.test_frac, args.seed)
    fit_ids, val_ids = carve_valid(train_ids, args.val_frac, args.seed)
    print(f"cases fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
              "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": 6, "verbosity": -1, "seed": args.seed}
    m_fit = df.caseid.isin(fit_ids)
    m_val = df.caseid.isin(val_ids)
    m_test = df.caseid.isin(test_ids)
    ds_fit = lgb.Dataset(df.loc[m_fit, feats], label=df.loc[m_fit, "label"])
    ds_val = lgb.Dataset(df.loc[m_val, feats], label=df.loc[m_val, "label"], reference=ds_fit)
    booster = lgb.train(params, ds_fit, num_boost_round=args.rounds, valid_sets=[ds_val],
                        callbacks=[lgb.early_stopping(100, verbose=False)])

    y_val = df.loc[m_val, "label"].to_numpy()
    p_val = booster.predict(df.loc[m_val, feats], num_iteration=booster.best_iteration)
    y = df.loc[m_test, "label"].to_numpy()
    p = booster.predict(df.loc[m_test, feats], num_iteration=booster.best_iteration)

    points = [point(y, p, 0.5, "default_0.5")]
    thr_bal, _ = thr_max_balanced(y_val, p_val)
    points.append(point(y, p, thr_bal, "max_balanced_acc(val)"))
    for t in TARGET_SENS:
        thr = thr_for_sensitivity(y_val, p_val, t)
        points.append(point(y, p, thr, f"sens>={t:.2f}(val)"))

    header = f"{'operating point':>24} {'thr':>7} {'acc':>7} {'bal':>7} {'sens':>7} {'spec':>7} {'PPV':>7} {'NPV':>7} {'alarm%':>7}"
    print("\n" + header)
    print("-" * len(header))
    for q in points:
        print(f"{q['name']:>24} {q['threshold']:>7.3f} {q['accuracy']:>7.4f} "
              f"{q['balanced_accuracy']:>7.4f} {q['sensitivity']:>7.4f} {q['specificity']:>7.4f} "
              f"{q['ppv']:>7.4f} {q['npv']:>7.4f} {q['alarms_per_100_windows']:>7.2f}")

    base = max(y.mean(), 1 - y.mean())
    report = {
        "n_features": len(feats),
        "cases": {"fit": len(fit_ids), "val": len(val_ids), "test": len(test_ids)},
        "best_iteration": to_int(booster.best_iteration),
        "test_n": to_int(len(y)),
        "test_positive_rate": round(to_float(y.mean()), 4),
        "test_majority_baseline_acc": round(to_float(base), 4),
        "auroc": round(to_float(roc_auc_score(y, p)), 4),
        "auprc": round(to_float(average_precision_score(y, p)), 4),
        "points": points,
    }
    try:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "operating_points.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write report: {exc}") from exc

    print(f"\ntest majority baseline: {base:.4f}   AUROC {report['auroc']}   AUPRC {report['auprc']}")
    print(f"wrote {args.out}/operating_points.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
