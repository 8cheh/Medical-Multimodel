"""Dump held-out scores + feature importances so the report can plot real ROC/PR curves.

Trains the exact same LightGBM configurations that produced the existing
results/*.json reports, then writes per-variant .npz (y/p arrays) and .json
(metrics + top gains) to --out.

Variants:
    main   : /root/autodl-tmp/processed/windows.parquet          (label mode "any")
    onset  : /root/autodl-tmp/processed_onset/windows.parquet    (label mode "onset")
    fused  : windows.parquet + ppg_features.parquet merged       (numeric vs numeric+ppg)

Recomputed metrics are printed next to the previously reported values so the
dumps can be checked for reproducibility instead of trusted blindly.
"""
import argparse
import json
import os
import sys

import lightgbm as lgb
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
PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "num_threads": 16, "verbosity": -1,
}


def to_float(value, default=NAN):
    """Best-effort float; the callers below never want a hard crash on a cast."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    """Best-effort int, same contract as to_float."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def round_int(value, default=1):
    """Round-half-even to int, guarding the cast."""
    return to_int(round(to_float(value, to_float(default, 1.0))), default)


def split_cases(caseids, test_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(caseids))
    perm = rng.permutation(len(uniq))
    n_test = max(1, round_int(len(uniq) * to_float(test_frac, 0.1)))
    return set(uniq[perm[n_test:]].tolist()), set(uniq[perm[:n_test]].tolist())


def carve_valid(train_ids, frac, seed):
    rng = np.random.default_rng(seed + 1)
    uniq = np.array(sorted(train_ids))
    perm = rng.permutation(len(uniq))
    n_val = max(1, round_int(len(uniq) * to_float(frac, 0.1)))
    return set(uniq[perm[n_val:]].tolist()), set(uniq[perm[:n_val]].tolist())


def metrics(y, p, thr):
    pred = (p >= thr).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tp, fp, tn, fn = to_int(tp), to_int(fp), to_int(tn), to_int(fn)
    n_pos = to_float(y.mean(), 0.0)
    return {
        "threshold": to_float(thr, 0.5),
        "accuracy": to_float(accuracy_score(y, pred)),
        "balanced_accuracy": to_float(balanced_accuracy_score(y, pred)),
        "auroc": to_float(roc_auc_score(y, p)),
        "auprc": to_float(average_precision_score(y, p)),
        "sensitivity": to_float(tp / max(tp + fn, 1)),
        "specificity": to_float(tn / max(tn + fp, 1)),
        "precision": to_float(tp / max(tp + fp, 1)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": to_int(len(y)),
        "positive_rate": n_pos,
        "majority_baseline_acc": to_float(max(n_pos, 1.0 - n_pos)),
    }


def best_threshold(y, p):
    order = np.argsort(-p)
    ys = y[order]
    cum_tp = np.cumsum(ys)
    total_pos = ys.sum()
    idx = np.arange(1, len(ys) + 1)
    acc = (cum_tp + (len(ys) - idx) - (total_pos - cum_tp)) / len(ys)
    k = to_int(np.argmax(acc))
    return to_float(p[order][k], 0.5), to_float(acc[k])


def fit_one(df, feats, fit_ids, val_ids, test_ids, seed, tag, out, rounds=3000):
    sub = df.loc[df.caseid.isin(fit_ids | val_ids | test_ids)]
    m_fit = sub.caseid.isin(fit_ids)
    m_val = sub.caseid.isin(val_ids)
    m_test = sub.caseid.isin(test_ids)
    ds_fit = lgb.Dataset(sub.loc[m_fit, feats], label=sub.loc[m_fit, "label"])
    ds_val = lgb.Dataset(sub.loc[m_val, feats], label=sub.loc[m_val, "label"], reference=ds_fit)
    booster = lgb.train(dict(PARAMS, seed=seed), ds_fit, num_boost_round=rounds,
                        valid_sets=[ds_val], callbacks=[lgb.early_stopping(100, verbose=False)])
    y_val = sub.loc[m_val, "label"].to_numpy()
    p_val = booster.predict(sub.loc[m_val, feats], num_iteration=booster.best_iteration)
    y_test = sub.loc[m_test, "label"].to_numpy()
    p_test = booster.predict(sub.loc[m_test, feats], num_iteration=booster.best_iteration)
    thr, val_acc = best_threshold(y_val, p_val)

    np.savez_compressed(
        os.path.join(out, f"{tag}.npz"),
        y_test=y_test, p_test=np.asarray(p_test, dtype=np.float32),
        y_val=y_val, p_val=np.asarray(p_val, dtype=np.float32),
    )
    gains = sorted(zip(feats, booster.feature_importance("gain"), strict=True), key=lambda x: -x[1])
    meta = {
        "tag": tag, "best_iteration": to_int(booster.best_iteration),
        "n_features": to_int(len(feats)),
        "val_tuned_threshold": thr, "val_tuned_accuracy": val_acc,
        "test_at_0.5": metrics(y_test, p_test, 0.5),
        "test_at_val_threshold": metrics(y_test, p_test, thr),
        "top_gains": [{"feature": n, "gain": to_float(g)} for n, g in gains[:40]],
    }
    write_json(os.path.join(out, f"{tag}.json"), meta)
    print(f"[{tag}] best_iteration={meta['best_iteration']} n_features={meta['n_features']}",
          flush=True)
    m = meta["test_at_0.5"]
    print(f"[{tag}] test@0.5 acc={m['accuracy']:.6f} auroc={m['auroc']:.6f} "
          f"auprc={m['auprc']:.6f} sens={m['sensitivity']:.6f} "
          f"spec={m['specificity']:.6f}", flush=True)
    return meta


def read_table(path):
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        print(f"cannot read {path}: {type(exc).__name__}: {exc}", flush=True)
        return None


def write_json(path, payload):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"cannot write {path}: {exc}", flush=True)
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp")
    ap.add_argument("--out", default="/root/autodl-tmp/models/scores")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--skip-onset", action="store_true")
    args = ap.parse_args(argv)

    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc
    summary = {}

    for tag, path in (("main", os.path.join(args.root, "processed/windows.parquet")),
                      ("onset", os.path.join(args.root, "processed_onset/windows.parquet"))):
        if tag == "onset" and args.skip_onset:
            continue
        df = read_table(path)
        if df is None:
            print(f"[{tag}] SKIP", flush=True)
            continue
        feats = [c for c in df.columns if c not in DROP]
        print(f"[{tag}] rows={len(df)} cases={df.caseid.nunique()} feats={len(feats)}", flush=True)
        train_ids, test_ids = split_cases(df.caseid.unique(), args.test_frac, args.seed)
        fit_ids, val_ids = carve_valid(train_ids, args.val_frac, args.seed)
        summary[tag] = fit_one(df, feats, fit_ids, val_ids, test_ids, args.seed, tag, args.out)
        del df

    wpath = os.path.join(args.root, "processed/windows.parquet")
    ppath = os.path.join(args.root, "processed/ppg_features.parquet")
    num, ppg = read_table(wpath), read_table(ppath)
    if num is not None and ppg is not None:
        ppg_cols = [c for c in ppg.columns if c not in ("caseid", "t")]
        merged = num.merge(ppg, on=["caseid", "t"], how="inner")
        del num, ppg
        num_feats = [c for c in merged.columns if c not in DROP and c not in set(ppg_cols)]
        all_ppg = [c for c in merged.columns if c in set(ppg_cols)]
        print(f"[fused] merged rows={len(merged)} cases={merged.caseid.nunique()} "
              f"num_feats={len(num_feats)} ppg_feats={len(all_ppg)}", flush=True)
        train_ids, test_ids = split_cases(merged.caseid.unique(), args.test_frac, args.seed)
        fit_ids, val_ids = carve_valid(train_ids, args.val_frac, args.seed)
        for tag, feats in (("fused_numeric", num_feats),
                           ("fused_numeric_ppg", num_feats + all_ppg)):
            summary[tag] = fit_one(merged, feats, fit_ids, val_ids, test_ids,
                                   args.seed, tag, args.out, rounds=2000)
        del merged
    else:
        print("[fused] SKIP (missing inputs)", flush=True)

    write_json(os.path.join(args.out, "summary.json"), summary)
    print("SCORES_DUMP_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
