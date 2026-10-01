"""P0 evaluation: baselines, event-level metrics and paired significance tests.

Window-level accuracy cannot answer the three questions this script exists for:

1. **Does the model beat the alarm it would replace?**  Baselines are the
   standard `MAP < 65` alarm, two smoother MAP rules, and a LightGBM trained on
   MAP-only features (the ceiling of "current MAP + trend"). HPI is NOT
   reproduced here: it consumes the arterial *waveform* (SNUADC/ART), which this
   cohort never downloaded, so a faithful reproduction is out of reach and a
   fake one would be worse than none.
2. **Per-episode metrics.** How many hypotension episodes were caught, how many
   seconds of warning, how many alarms per hour of case time. Episodes come from
   `build_dataset.hypotension_flags`, i.e. the same rule the labels were built
   from, so "episode" means one thing in this project.
3. **Is a delta real?**  Paired case-level bootstrap (patients, not windows, are
   the independent unit) plus exact McNemar on the accuracy pairing.

Two modes:
    significance   paired test over existing score dumps (no dataset needed)
    dataset        rebuilds the whole comparison from windows.parquet

    python p0_eval.py significance --scores-dir <dir> --pairs fused_numeric:fused_numeric_ppg
    python p0_eval.py dataset --root <dir> --tracks-dir <dir with track CSVs> --out <dir>
"""
import argparse
import json
import os
import sys

import build_dataset as bd
import lightgbm as lgb
import numpy as np
import pandas as pd
from dump_scores import (
    DROP,
    PARAMS,
    best_threshold,
    carve_valid,
    metrics,
    split_cases,
    to_float,
    to_int,
)
from scipy.stats import binomtest

HORIZON_S = bd.HORIZON_S      # the label looks this far ahead; also the alarm lead window
STRIDE_S = bd.STRIDE_S
SENS_TARGET = 0.80            # operating point used for the baseline comparison
METRIC_KEYS = ("accuracy", "balanced_accuracy", "auroc", "auprc",
               "sensitivity", "specificity", "precision")
EVENT_KEYS = ("detection_rate", "lead_median_s", "alarm_rate_per_h",
              "false_alarm_rate_per_h")
NAN = np.nan

# Baseline rules: score = how far below the alarm threshold the signal sits, so a
# higher score means "more alarming" and the as-designed threshold is exactly 0.
RULES = {
    "rule_map65": ("map_last60", 65.0, False),       # the standard monitor alarm
    "rule_map65_mean": ("map_mean60", 65.0, False),  # same, on the 60 s mean
    "rule_map65_min": ("map_min", 65.0, False),      # "has it dipped at all in 5 min"
    # As-designed alarms for the functional endpoints (need --func-tracks data):
    "rule_rr_low": ("fn_rr_last60", 6.0, False),          # respiratory rate below 6/min
    "rule_petco2_high": ("fn_petco2_last60", 50.0, True),  # ETCO2 above 50 mmHg
}
# Prefixes of the optional feature groups added on top of the 67-feature baseline:
# `l_*` is the second, longer observation window, the others are tracks the baseline
# never used. They are recognised by name so `main` stays the published baseline.
EXTRA_PREFIXES = ("bt", "nibp", "cuff_gap")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def episode_lead_times(cases, name, thr, pre=HORIZON_S):
    """Lead time per detected episode, keyed by (caseid, onset) so variants can be paired.

    `event_metrics` reports a per-variant median, which cannot be compared between two
    variants: the +1.65 min headline is a difference of two medians, not a paired
    statistic. This returns individual episodes so the same episode can be compared
    under both variants.
    """
    out = {}
    for c in cases:
        t, pred, T = c["t"], c["p"][name] >= thr, c["onset"]
        for onset in T:
            lo = onset - pre
            lo = max(lo, bd.PAST_S)
            win = (t >= lo) & (t < onset)
            if not win.any():
                continue
            hit = np.flatnonzero(pred & win)
            if hit.size:
                out[(c["caseid"], to_float(onset))] = to_float(onset - t[hit[0]])
    return out


def paired_lead_summary(cases, a_name, a_thr, b_name, b_thr, pre=HORIZON_S):
    """Paired lead-time comparison on the episodes BOTH variants detect.

    lead_delta > 0 means variant A warned earlier than variant B on the same episode.
    Episodes detected by only one variant are counted separately rather than dropped
    silently, since they are the asymmetric part of the comparison.
    """
    la = episode_lead_times(cases, a_name, a_thr, pre)
    lb = episode_lead_times(cases, b_name, b_thr, pre)
    common = set(la) & set(lb)
    deltas = np.array([la[k] - lb[k] for k in common], dtype="float64")
    return {
        "n_episodes_a": len(la),
        "n_episodes_b": len(lb),
        "n_paired": len(common),
        "only_a_detected": len(set(la) - set(lb)),
        "only_b_detected": len(set(lb) - set(la)),
        "lead_delta_median_s": to_float(np.median(deltas)) if deltas.size else NAN,
        "lead_delta_mean_s": to_float(deltas.mean()) if deltas.size else NAN,
        "frac_a_earlier": to_float((deltas > 0).mean()) if deltas.size else NAN,
    }


def blank_metrics(thr, n):
    """Same keys as dump_scores.metrics, all NaN (a bootstrap draw can be one-class)."""
    out = dict.fromkeys(
        ("threshold", "accuracy", "balanced_accuracy", "auroc", "auprc", "sensitivity",
         "specificity", "precision", "tp", "fp", "tn", "fn", "n", "positive_rate",
         "majority_baseline_acc"), NAN)
    out["threshold"] = to_float(thr)
    out["n"] = to_int(n)
    return out


def safe_metrics(y, p, thr):
    """dump_scores.metrics with a single-class guard."""
    try:
        return metrics(y, p, thr)
    except ValueError:
        return blank_metrics(thr, len(y))


def threshold_at_sensitivity(y, p, target=SENS_TARGET):
    """Highest threshold whose sensitivity still reaches `target` on this split.

    Sensitivity is non-decreasing as the threshold falls, so the ranks meeting the
    target form a suffix. The FIRST of them is the highest threshold, i.e. the most
    specific way to still reach the target - which is what a fair comparison
    against a fixed alarm needs.
    """
    order = np.argsort(-p)
    ys = y[order]
    pos = to_int(ys.sum())
    if pos == 0:
        return NAN
    ok = np.flatnonzero(np.cumsum(ys) / pos >= target)
    return to_float(p[order][to_int(ok[0])]) if ok.size else to_float(p[order][-1])


def mcnemar(y, pa, pb, thr_a, thr_b):
    """Exact paired test on accuracy: only the windows where A and B disagree."""
    ca = (pa >= thr_a) == (y == 1)
    cb = (pb >= thr_b) == (y == 1)
    b = to_int(np.sum(ca & ~cb))
    c = to_int(np.sum(~ca & cb))
    if b + c == 0:
        return {"a_right_b_wrong": 0, "a_wrong_b_right": 0, "p": 1.0}
    return {"a_right_b_wrong": b, "a_wrong_b_right": c,
            "p": to_float(binomtest(b, b + c, 0.5).pvalue)}


# --------------------------------------------------------------------------- #
# event level
# --------------------------------------------------------------------------- #
def onsets_for(caseid, tracks_dir, horizon=HORIZON_S, endpoint="hypotension",
               func_tracks=False):
    """Onset seconds of every episode of one case for this endpoint, or [] if unreadable."""
    loaded = bd.load_case_signals(caseid, tracks_dir, horizon, func_tracks)
    if loaded[0] is None:
        return np.zeros(0, dtype=float)
    flags = bd.endpoint_flags(loaded[1], endpoint)
    if flags.size < 2:
        return np.zeros(0, dtype=float)
    return np.flatnonzero(flags[1:] & ~flags[:-1]).astype(float) + 1.0


def event_metrics(cases, name, thr, pre=HORIZON_S):
    """Detection, warning time and alarm burden for one variant at one threshold.

    An episode is *scoreable* when at least one decision time falls in
    [onset - pre, onset) - the actionable look-ahead of the label. An episode is
    *detected* when any of those windows is predicted positive; the lead time is
    measured from the first such window, i.e. from when the alarm would actually
    have fired.

    Alarm events are rising edges of `p >= thr` on the 30 s decision grid: a
    sustained high score is one alarm, not one alarm per window. An alarm is
    counted as true when an onset follows it within `pre`.
    """
    n_ep = n_scoreable = detected = 0
    lead = []
    alarms = true_alarms = 0
    hours = 0.0
    for c in cases:
        t, pred, T = c["t"], c["p"][name] >= thr, c["onset"]
        if t.size == 0:
            continue
        hours += t.size * STRIDE_S / 3600.0
        starts = np.flatnonzero(pred & ~np.concatenate([[False], pred[:-1]]))
        alarms += to_int(starts.size)
        if starts.size and T.size:
            st = t[starts]
            j = np.searchsorted(T, st, side="right")   # first onset strictly after
            hit = j < T.size
            if hit.any():
                nxt = T[np.minimum(j, T.size - 1)]
                true_alarms += to_int(np.sum(hit & (nxt <= st + pre)))
        for onset in T:
            n_ep += 1
            lo = onset - pre
            lo = max(lo, bd.PAST_S)
            win = (t >= lo) & (t < onset)
            if not win.any():
                continue
            n_scoreable += 1
            hit = np.flatnonzero(pred & win)
            if hit.size:
                detected += 1
                lead.append(to_float(onset - t[hit[0]]))
    return {
        "n_episodes": n_ep,
        "n_scoreable": n_scoreable,
        "detected": detected,
        "detection_rate": detected / n_scoreable if n_scoreable else NAN,
        "lead_median_s": to_float(np.median(lead)) if lead else NAN,
        "lead_mean_s": to_float(np.mean(lead)) if lead else NAN,
        "lead_p25_s": to_float(np.percentile(lead, 25)) if lead else NAN,
        "alarm_events": alarms,
        "alarm_rate_per_h": alarms / hours if hours else NAN,
        "false_alarm_rate_per_h": (alarms - true_alarms) / hours if hours else NAN,
        "hours": hours,
    }


# --------------------------------------------------------------------------- #
# bootstraps
# --------------------------------------------------------------------------- #
def boot_cases(cases, stat, reps, seed=42):
    """Percentile bootstrap resampling CASES, so within-patient correlation counts."""
    if not cases or reps <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(cases)
    draws = [stat([cases[i] for i in rng.integers(0, n, n)]) for _ in range(reps)]
    out = {}
    for key in sorted(draws[0]):
        v = np.array([to_float(d.get(key)) for d in draws], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[key] = {"mean": NAN, "lo": NAN, "hi": NAN, "p_le0": NAN}
            continue
        out[key] = {"mean": to_float(v.mean()),
                    "lo": to_float(np.percentile(v, 2.5)),
                    "hi": to_float(np.percentile(v, 97.5)),
                    "p_le0": to_float((v <= 0).mean())}
    return out


def window_metrics_of(cases, name, thr):
    y = np.concatenate([c["y"] for c in cases])
    p = np.concatenate([c["p"][name] for c in cases])
    return safe_metrics(y, p, thr)


def delta_stat(a_name, a_thr, b_name, b_thr, keys=METRIC_KEYS):
    """A - B over the same windows. For paired AUC/AUPRC the threshold is inert."""
    def stat(cases):
        a = window_metrics_of(cases, a_name, a_thr)
        b = window_metrics_of(cases, b_name, b_thr)
        return {k: to_float(a[k]) - to_float(b[k]) for k in keys}
    return stat


def boot_windows(y, pa, pb, reps, thr_a, thr_b, seed=42):
    """Window-level paired bootstrap, for score dumps that carry no caseid.

    The thresholds must be the same ones the point estimate used, otherwise the
    interval describes a different operating point than the delta it brackets.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        a, b = safe_metrics(y[i], pa[i], thr_a), safe_metrics(y[i], pb[i], thr_b)
        draws.append({k: to_float(a[k]) - to_float(b[k]) for k in METRIC_KEYS})
    out = {}
    for key in sorted(draws[0]):
        v = np.array([d[key] for d in draws], dtype=float)
        v = v[np.isfinite(v)]
        out[key] = {"mean": to_float(v.mean()), "lo": to_float(np.percentile(v, 2.5)),
                    "hi": to_float(np.percentile(v, 97.5)),
                    "p_le0": to_float((v <= 0).mean())} if v.size else \
                   {"mean": NAN, "lo": NAN, "hi": NAN, "p_le0": NAN}
    return out


# --------------------------------------------------------------------------- #
# mode: significance over existing score dumps
# --------------------------------------------------------------------------- #
def sidecar_threshold(scores_dir, name, default=0.5):
    """The val-tuned threshold the dump itself recorded, if the sidecar is there."""
    path = os.path.join(scores_dir, f"{name}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return to_float(json.load(fh).get("val_tuned_threshold"), default)
    except (OSError, ValueError):
        return default


def cmd_significance(args):
    report = {"mode": "significance", "level": "window",
              "note": "these dumps carry no caseid, so windows are resampled; "
                      "case-level intervals need a re-dump from `dataset` mode",
              "reps": args.reps, "pairs": {}}
    for spec in args.pairs:
        a_name, b_name = spec.split(":")
        A = np.load(os.path.join(args.scores_dir, f"{a_name}.npz"))
        B = np.load(os.path.join(args.scores_dir, f"{b_name}.npz"))
        for split in ("test", "val"):
            ya, yb = A[f"y_{split}"], B[f"y_{split}"]
            if not np.array_equal(ya, yb):
                print(f"[{a_name} vs {b_name}] {split}: rows not paired -> skipped", flush=True)
                continue
            pa = A[f"p_{split}"].astype(float)
            pb = B[f"p_{split}"].astype(float)
            thr_a = sidecar_threshold(args.scores_dir, a_name)
            thr_b = sidecar_threshold(args.scores_dir, b_name)
            ma, mb = safe_metrics(ya, pa, thr_a), safe_metrics(ya, pb, thr_b)
            entry = {
                "split": split, "n": to_int(len(ya)),
                "positive_rate": to_float(ya.mean()),
                "thresholds": {a_name: thr_a, b_name: thr_b},
                a_name: ma,
                b_name: mb,
                "delta": boot_windows(ya, pa, pb, args.reps, thr_a, thr_b, args.seed),
                "mcnemar": mcnemar(ya, pa, pb, thr_a, thr_b),
                "delta_point": {k: to_float(ma[k]) - to_float(mb[k]) for k in METRIC_KEYS},
            }
            report["pairs"].setdefault(spec, {})[split] = entry
            print_pair(spec, split, entry, a_name, b_name)
            if args.test_only and split == "test":
                continue
    return report


def print_pair(spec, split, entry, a_name, b_name):
    print(f"\n=== {spec}  [{split}] n={entry['n']} pos={entry['positive_rate']:.4f} "
          f"thr {a_name}={entry['thresholds'][a_name]:.4f} "
          f"{b_name}={entry['thresholds'][b_name]:.4f} ===")
    print(f"  delta = {a_name} - {b_name}; p(<=0) is the bootstrap tail, so the "
          f"two-sided p = 2*min(p, 1-p)")
    print(f"  {'metric':22s} {a_name:>16s} {b_name:>16s} {'delta':>9s}  "
          f"{'95% CI':>19s} {'p(<=0)':>7s}")
    for key in METRIC_KEYS:
        d = entry["delta"][key]
        print(f"  {key:22s} {to_float(entry[a_name][key]):16.4f} "
              f"{to_float(entry[b_name][key]):16.4f} "
              f"{entry['delta_point'][key]:+9.4f}  "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}] {d['p_le0']:7.3f}")
    m = entry["mcnemar"]
    print(f"  McNemar exact: a_right_b_wrong={m['a_right_b_wrong']} "
          f"a_wrong_b_right={m['a_wrong_b_right']} p={m['p']:.4g}")


# --------------------------------------------------------------------------- #
# mode: dataset (baselines + events + paired tests)
# --------------------------------------------------------------------------- #
def predict_chunked(booster, frame, n_iter, chunk=200_000):
    """Predict in slices: a full 1.4M x 67 float copy is what kills the laptop."""
    out = np.empty(len(frame), dtype=float)
    for start in range(0, len(frame), chunk):
        out[start:start + chunk] = booster.predict(
            frame.iloc[start:start + chunk], num_iteration=n_iter)
    return out


def fit_variant(df, feats, fit_ids, val_ids, seed, threads, rounds=3000):
    """Train on fit, early-stop on val. Returns (booster, y_val, p_val)."""
    m_fit = (df.caseid.isin(fit_ids)).to_numpy()
    m_val = (df.caseid.isin(val_ids)).to_numpy()
    x_fit = df.loc[m_fit, feats]
    ds_fit = lgb.Dataset(x_fit, label=df.loc[m_fit, "label"])
    del x_fit
    x_val = df.loc[m_val, feats]
    ds_val = lgb.Dataset(x_val, label=df.loc[m_val, "label"], reference=ds_fit)
    del x_val
    booster = lgb.train(dict(PARAMS, seed=seed, num_threads=threads), ds_fit,
                        num_boost_round=rounds, valid_sets=[ds_val],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    p_val = predict_chunked(booster, df.loc[m_val, feats].reset_index(drop=True),
                            booster.best_iteration)
    return booster, df.loc[m_val, "label"].to_numpy(), p_val


def build_cases(sub, names, tracks_dir, horizon=HORIZON_S, endpoint="hypotension",
                func_tracks=False):
    """Per-case records: decision times, labels, one score array per variant, onsets."""
    cases = []
    for cid, g in sub.groupby("caseid", sort=True):
        g = g.sort_values("t")
        cases.append({
            "caseid": to_int(cid),
            "t": g["t"].to_numpy(dtype=float),
            "y": g["label"].to_numpy(dtype=np.int8),
            "p": {n: g[f"p_{n}"].to_numpy(dtype=float) for n in names},
            "onset": onsets_for(to_int(cid), tracks_dir, horizon, endpoint, func_tracks),
        })
    return cases


def evaluate_group(label, df, variants, tracks_dir, args):
    """Fit every variant on one split, then score windows and episodes."""
    train_ids, test_ids = split_cases(df.caseid.unique(), args.test_frac, args.seed)
    fit_ids, val_ids = carve_valid(train_ids, args.val_frac, args.seed)
    print(f"\n[{label}] rows={len(df)} cases={df.caseid.nunique()} "
          f"fit={len(fit_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    m_test = (df.caseid.isin(test_ids)).to_numpy()
    sub = df.loc[m_test].reset_index(drop=True)
    thr, meta = {}, {}

    for name, feats in variants.items():
        if name in RULES:
            feat, cut, above = RULES[name]
            if feat not in sub.columns:
                continue  # this rule needs a channel the dataset does not carry
            score = (sub[feat].to_numpy(dtype=float) - cut) if above else (
                cut - sub[feat].to_numpy(dtype=float))
            sub[f"p_{name}"] = np.nan_to_num(score, nan=-1e9)
            thr[name] = 0.0
            meta[name] = {"kind": "rule", "feature": feat, "cut": cut, "above": above}
            print(f"  [{name}] rule {feat} {'>' if above else '<'} {cut:.0f}", flush=True)
            continue
        booster, y_val, p_val = fit_variant(df, feats, fit_ids, val_ids,
                                            args.seed, args.threads, args.rounds)
        sub[f"p_{name}"] = predict_chunked(
            booster, df.loc[m_test, feats].reset_index(drop=True), booster.best_iteration)
        y_val = y_val[np.isfinite(p_val)]
        p_val = p_val[np.isfinite(p_val)]
        thr[name] = threshold_at_sensitivity(y_val, p_val, args.sens_target)
        meta[name] = {"kind": "model", "n_features": len(feats),
                      "best_iteration": to_int(booster.best_iteration),
                      "val_threshold_at_sens": to_float(thr[name]),
                      "val_threshold_acc_opt": to_float(best_threshold(y_val, p_val)[0])}
        if args.save_booster:
            # The streaming monitor needs the same fitted model the offline numbers
            # came from; saving it here keeps the two paths on one artifact.
            try:
                os.makedirs(args.save_booster, exist_ok=True)
                path = os.path.join(args.save_booster, f"lgbm_{name}.txt")
                booster.save_model(path, num_iteration=booster.best_iteration)
                print(f"  saved booster {name} -> {path}", flush=True)
            except OSError as exc:
                print(f"  ! cannot save booster {name}: {exc}", flush=True)
        gains = sorted(zip(feats, booster.feature_importance("gain"), strict=True),
                       key=lambda x: -x[1])[:10]
        meta[name]["top_gains"] = [{"feature": f, "gain": to_float(g)} for f, g in gains]
        print(f"  [{name}] feats={len(feats)} best_iter={booster.best_iteration} "
              f"thr@>= {args.sens_target:.2f} sens={thr[name]:.4f}", flush=True)

    cases = build_cases(sub, list(variants), tracks_dir, args.horizon, args.endpoint,
                        args.func_tracks)
    if args.dump_scores:
        # Per-window scores with their caseid/t, so a monitor display (or any later
        # analysis) can be drawn on a real case timeline instead of aggregate metrics.
        try:
            os.makedirs(args.dump_scores, exist_ok=True)
            dump = sub[["caseid", "t", "label"] + [f"p_{n}" for n in variants]].copy()
            dtype = {f"p_{n}": "float32" for n in variants}
            dump = dump.astype(dtype)
            path = os.path.join(args.dump_scores, f"scores_{label}.parquet")
            dump.to_parquet(path, index=False)
            print(f"  dumped {len(dump)} scored windows -> {path}", flush=True)
        except OSError as exc:
            print(f"  ! cannot dump scores: {exc}", flush=True)
    n_ep = to_int(sum(len(c["onset"]) for c in cases))
    print(f"  test cases={len(cases)} episodes={n_ep} hours="
          f"{sum(c['t'].size for c in cases) * STRIDE_S / 3600:.1f}", flush=True)

    y = np.concatenate([c["y"] for c in cases])
    out = {"cases": {"fit": len(fit_ids), "val": len(val_ids), "test": len(test_ids)},
           "test_windows": to_int(len(y)), "test_episodes": n_ep,
           "test_positive_rate": to_float(y.mean()), "variants": {}}

    for name in variants:
        w = window_metrics_of(cases, name, thr[name])
        e = event_metrics(cases, name, thr[name], args.horizon)
        ci = boot_cases(cases, lambda cs, n=name, t=thr[name]:
                        event_metrics(cs, n, t, args.horizon),
                        args.reps, args.seed)
        out["variants"][name] = {"threshold": to_float(thr[name]), **meta[name],
                                 "window": w, "event": e,
                                 "event_ci": {k: ci.get(k) for k in EVENT_KEYS}}
        ev = e
        print(f"  {name:22s} auroc={to_float(w['auroc']):.4f} acc={to_float(w['accuracy']):.4f} "
              f"sens={to_float(w['sensitivity']):.4f} | episodes {ev['detected']}/{ev['n_scoreable']}"
              f" ({to_float(ev['detection_rate']):.3f}) lead={to_float(ev['lead_median_s']) / 60:.1f}min "
              f"alarms/h={to_float(ev['alarm_rate_per_h']):.1f} "
              f"false/h={to_float(ev['false_alarm_rate_per_h']):.1f}")

    out["deltas"] = {}
    for spec in args.compare:
        if "-" not in spec:
            print(f"  (skipping --compare entry {spec!r}: expected A-B)")
            continue
        a, b = spec.split("-", 1)
        if a not in variants or b not in variants:
            continue
        key = spec
        d = boot_cases(cases, delta_stat(a, thr[a], b, thr[b]), args.reps, args.seed)
        ya = np.concatenate([c["y"] for c in cases])
        pa = np.concatenate([c["p"][a] for c in cases])
        pb = np.concatenate([c["p"][b] for c in cases])
        out["deltas"][key] = {"boot": d, "mcnemar": mcnemar(ya, pa, pb, thr[a], thr[b])}
        # Paired lead time: the headline "model warns earlier by X minutes" claim was a
        # difference of two per-variant medians; this is the paired version of it.
        out["deltas"][key]["lead"] = paired_lead_summary(cases, a, thr[a], b, thr[b],
                                                         args.horizon)
        lead_ci = boot_cases(
            cases,
            lambda cs, n1=a, t1=thr[a], n2=b, t2=thr[b]: paired_lead_summary(
                cs, n1, t1, n2, t2, args.horizon),
            args.reps, args.seed)
        out["deltas"][key]["lead_ci"] = {
            k: lead_ci.get(k) for k in ("lead_delta_median_s", "lead_delta_mean_s",
                                        "frac_a_earlier")}
        print(f"\n  --- {key} (paired, case-level bootstrap, {args.reps} reps) ---")
        for m in ("accuracy", "auroc", "auprc", "sensitivity", "specificity"):
            dd = d.get(m, {})
            print(f"    {m:22s} {dd.get('mean', NAN):+.4f} "
                  f"[{dd.get('lo', NAN):+.4f},{dd.get('hi', NAN):+.4f}] "
                  f"p(<=0)={dd.get('p_le0', NAN):.3f}")
        mc = out["deltas"][key]["mcnemar"]
        print(f"    McNemar exact p={mc['p']:.4g} "
              f"(a_right_b_wrong={mc['a_right_b_wrong']}, a_wrong_b_right={mc['a_wrong_b_right']})")
        ld = out["deltas"][key]["lead"]
        lc = out["deltas"][key]["lead_ci"]
        med = lc.get("lead_delta_median_s") or {}
        print(f"    paired lead time: n_paired={ld['n_paired']} "
              f"(only {a}={ld['only_a_detected']}, only {b}={ld['only_b_detected']}) "
              f"median delta={ld['lead_delta_median_s']:.0f}s "
              f"[{med.get('lo', NAN):.0f},{med.get('hi', NAN):.0f}] "
              f"a earlier in {ld['frac_a_earlier'] * 100:.1f}% of episodes")
    return out


def add_extra_features(df, windows, paths, mode="strict"):
    """Merge extra feature columns that a second build produced on the same grid.

    mode="strict": the extra build must cover exactly the same windows, and any column
    the two builds share must agree value for value. That is the guard rail for a
    same-grid build (e.g. a second observation window).
    mode="inner":  the extra build may cover only a subset; rows it does not cover are
    dropped, which is the fused-style comparison restricted to cases that carry the
    extra signal (a left join would train on all-NaN features for the rest and dilute
    the very effect being measured).
    """
    for path in paths:
        if not os.path.isfile(path):
            print(f"[extra] {path} not found - skipped")
            continue
        other = pd.read_parquet(path)
        keep = [c for c in other.columns
                if c not in ("caseid", "t", "label") and c not in set(df.columns)]
        if not keep:
            print(f"[extra] {path} adds no new columns - skipped")
            continue

        if mode == "inner":
            before_rows, before_cases = len(df), df.caseid.nunique()
            df = df.merge(other[["caseid", "t"] + keep], on=["caseid", "t"],
                          how="inner", validate="one_to_one")
            filled = to_int(df[keep].notna().to_numpy().sum())
            if df.empty or filled == 0:
                raise SystemExit(f"{path} shares no usable rows with {windows}")
            print(f"[extra] {os.path.basename(os.path.dirname(path))}: +{len(keep)} "
                  f"features, inner join kept {len(df)}/{before_rows} rows and "
                  f"{df.caseid.nunique()}/{before_cases} cases")
            continue

        # Align the two builds by (caseid, t) with numpy ordering rather than a pandas
        # sort: the point is to compare VALUES positionally, and pandas' sort overloads
        # are not resolvable statically here.
        a_keys = np.lexsort((np.asarray(df["t"], dtype="float64"),
                             np.asarray(df["caseid"], dtype="float64")))
        b_keys = np.lexsort((np.asarray(other["t"], dtype="float64"),
                             np.asarray(other["caseid"], dtype="float64")))
        key_a = np.asarray(df[["caseid", "t"]], dtype="float64")[a_keys]
        key_b = np.asarray(other[["caseid", "t"]], dtype="float64")[b_keys]
        if key_a.shape != key_b.shape or not (key_a == key_b).all():
            raise SystemExit(
                f"{path} covers a different window set than {windows} "
                f"({key_b.shape[0]} vs {key_a.shape[0]} rows) - refusing a misaligned "
                "comparison; rebuild it with the same --horizon")
        shared = [c for c in other.columns
                  if c in set(df.columns) and c not in ("caseid", "t", "label")]
        if shared:
            lhs = np.asarray(df[shared], dtype="float64")[a_keys]
            rhs = np.asarray(other[shared], dtype="float64")[b_keys]
            same_nan = np.isnan(lhs) == np.isnan(rhs)
            close = np.isclose(lhs, rhs, rtol=0, atol=0, equal_nan=True)
            if not (same_nan & close).all():
                n_bad = to_int((~(same_nan & close)).sum())
                raise SystemExit(f"{path} disagrees with {windows} on {n_bad} shared "
                                 "feature values - refusing the comparison")
        df = df.merge(other[["caseid", "t"] + keep], on=["caseid", "t"],
                      how="left", validate="one_to_one")
        print(f"[extra] {os.path.basename(os.path.dirname(path))}: +{len(keep)} features "
              f"({len(shared)} shared base features verified identical)")
    return df


def cmd_dataset(args):
    tracks_dir = args.tracks_dir or os.path.join(args.root, "numeric")
    windows = args.windows or os.path.join(args.root, "processed", "windows.parquet")
    df = pd.read_parquet(windows)
    num_feats = [c for c in df.columns if c not in DROP]
    map_feats = [c for c in num_feats if c.startswith("map")]
    print(f"windows={len(df)} cases={df.caseid.nunique()} numeric_feats={len(num_feats)} "
          f"map_only_feats={len(map_feats)} horizon={args.horizon}s "
          f"endpoint={args.endpoint}")

    report = {"mode": "dataset", "root": args.root, "windows": windows,
              "tracks_dir": tracks_dir, "seed": args.seed, "reps": args.reps,
              "horizon": args.horizon, "endpoint": args.endpoint,
              "sens_target": args.sens_target, "groups": {},
                  "notes": [
                      ("Episode onset is the first second whose trailing 60 s was >=90% below "
                       "MAP 65 (build_dataset.hypotension_flags). So at the onset the patient "
                       "has already been hypotensive for about a minute by construction, which "
                       "gives a MAP-keyed alarm such as rule_map65 a structural head start."),
                      ("Event metrics use each variant's validation-tuned threshold for "
                       f"sensitivity >= {args.sens_target:.2f}, except the rule_map65* "
                       "variants, whose as-designed threshold is 0 (MAP <= 65)."),
                      ("Alarms are rising edges on the 30 s decision grid, so a sustained high "
                       "score counts once."),
                      ("Mode `significance` resamples windows and is therefore "
                       "anti-conservative; the case-level intervals here are the ones to "
                       "quote."),
                      (f"The label looks {args.horizon} s ahead, and the same span is the "
                       "actionable alarm window of an episode (see the `pre` argument of "
                       "event_metrics). Event metrics are therefore only comparable "
                       "between runs that share --horizon."),
                  ]}

    extra_paths = list(args.extra_features or [])
    if extra_paths:
        df = add_extra_features(df, windows, extra_paths, args.extra_features_mode)
    long_feats = [c for c in df.columns if c.startswith("l_")]
    add_feats = [c for c in df.columns
                 if any(c.startswith(p + "_") for p in EXTRA_PREFIXES)]
    ppg_feats = [c for c in df.columns if c.startswith("ppg_")]
    abp_feats = [c for c in df.columns if c.startswith("abp_")]
    func_feats = [c for c in df.columns if c.startswith("fn_")]
    grouped = (set(long_feats) | set(add_feats) | set(ppg_feats) | set(abp_feats)
               | set(func_feats))
    base_feats = [c for c in df.columns if c not in DROP and c not in grouped]
    variants = {"main": base_feats,
                "map_only": [c for c in base_feats if c.startswith("map")],
                **{r: [] for r in RULES}}
    if long_feats:
        variants["main_long"] = base_feats + long_feats
    if add_feats:
        variants["main_extra"] = base_feats + add_feats
    if long_feats and add_feats:
        variants["main_all"] = base_feats + long_feats + add_feats
    if ppg_feats:
        variants["main_ppg"] = base_feats + ppg_feats
        if long_feats or add_feats:
            variants["main_all_ppg"] = base_feats + long_feats + add_feats + ppg_feats
    if abp_feats:
        variants["main_abp"] = base_feats + abp_feats
    if func_feats:
        # Respiration / pulmonary mechanics / depth channels (Primus, BIS)
        variants["main_func"] = base_feats + func_feats
    if ppg_feats and abp_feats:
        # the multimodal stack: every waveform modality fused on one feature matrix
        variants["main_multimodal"] = base_feats + ppg_feats + abp_feats
        if long_feats or add_feats:
            variants["main_multimodal_all"] = (base_feats + long_feats + add_feats
                                               + ppg_feats + abp_feats)
    print("  variants: " + ", ".join(f"{k}={len(v)}" for k, v in variants.items()))
    report["groups"]["numeric"] = evaluate_group("numeric", df, variants, tracks_dir, args)

    ppg_path = args.ppg or ""
    if ppg_path and os.path.isfile(ppg_path):
        ppg = pd.read_parquet(ppg_path)
        merged = df.merge(ppg, on=["caseid", "t"], how="inner")
        ppg_cols = [c for c in ppg.columns if c not in ("caseid", "t")]
        fused = {"fused_numeric": num_feats, "fused_numeric_ppg": num_feats + ppg_cols}
        report["groups"]["fused"] = evaluate_group("fused", merged, fused, tracks_dir, args)
    else:
        print(f"\n[fused] skipped: {ppg_path or '(no --ppg given)'} not found. "
              "PPG waveform features live on the GPU box (~125 GB of .npy); "
              "the PPG significance question is answered by `significance` mode instead.")

    try:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "p0_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        raise SystemExit(f"cannot write report: {exc}") from exc
    print(f"\nwrote {os.path.join(args.out, 'p0_report.json')}")
    return report


# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    sig = sub.add_parser("significance", help="paired test over existing score dumps")
    sig.add_argument("--scores-dir", default="scores")
    sig.add_argument("--pairs", nargs="+", default=["fused_numeric:fused_numeric_ppg"])
    sig.add_argument("--reps", type=int, default=1000)
    sig.add_argument("--seed", type=int, default=42)
    sig.add_argument("--test-only", action="store_true")
    sig.add_argument("--out", default="")

    ds = sub.add_parser("dataset", help="baselines + event metrics + paired tests")
    ds.add_argument("--root", default="/root/autodl-tmp")
    ds.add_argument("--horizon", type=int, default=HORIZON_S,
                    help="label look-ahead of the dataset in seconds; must match the "
                         "--horizon used to build --windows, since it defines the "
                         "actionable alarm window per episode")
    ds.add_argument("--func-tracks", action="store_true",
                    help="the dataset was built with functional channels; required for the "
                         "episode/onset side of the functional endpoints")
    ds.add_argument("--endpoint", choices=tuple(bd.ENDPOINTS), default="hypotension",
                    help="monitored risk this dataset encodes; must match the rebuild")
    ds.add_argument("--windows", default="")
    ds.add_argument("--ppg", default="")
    ds.add_argument("--tracks-dir", default="")
    ds.add_argument("--out", default="")
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--test-frac", type=float, default=0.10)
    ds.add_argument("--val-frac", type=float, default=0.10)
    ds.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ds.add_argument("--rounds", type=int, default=3000)
    ds.add_argument("--reps", type=int, default=1000)
    ds.add_argument("--sens-target", type=float, default=SENS_TARGET)
    ds.add_argument("--extra-features", nargs="+", default=[],
                    help="parquet(s) built on the same window grid whose extra columns "
                         "are merged in as additional paired variants")
    ds.add_argument("--extra-features-mode", choices=("strict", "inner"), default="strict",
                    help="strict: extra build must cover exactly the same windows; "
                         "inner: keep only the rows it covers (fused-style comparison)")
    ds.add_argument("--dump-scores", default="",
                    help="directory to write per-window (caseid, t, label, p_*) parquet")
    ds.add_argument("--save-booster", default="",
                    help="directory to save the fitted LightGBM models (for the streaming monitor)")
    ds.add_argument("--compare", nargs="+",
                    default=["main-map_only", "main-rule_map65", "map_only-rule_map65",
                             "main_long-main", "main_extra-main", "main_all-main",
                             "main_all-main_long", "main_ppg-main",
                             "main_all_ppg-main", "main_all_ppg-main_all",
                             "main_abp-main", "main_multimodal-main",
                             "main_multimodal-main_ppg", "main_multimodal-main_abp",
                             "main_func-main", "main_func-map_only",
                             "fused_numeric_ppg-fused_numeric"])
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "significance":
        report = cmd_significance(args)
        if args.out:
            try:
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
            except OSError as exc:
                raise SystemExit(f"cannot write {args.out}: {exc}") from exc
            print(f"\nwrote {args.out}")
    else:
        if not args.out:
            args.out = os.path.join(args.root, "models", "p0")
        cmd_dataset(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
