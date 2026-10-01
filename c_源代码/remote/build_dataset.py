"""Build the intraoperative-hypotension dataset from VitalDB numeric tracks.

Task
----
At each decision time t (every 30 s) predict whether a sustained hypotension
episode (MAP < 65 mmHg for >= 1 min) will occur in the next 5 minutes.

Inputs  : per-track CSVs in /root/autodl-tmp/numeric (complete tracks, not samples)
Output  : /root/autodl-tmp/processed/windows.parquet

The label comes only from the FUTURE window; features only from the PAST window.
"""
import argparse
import glob
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import pandas as pd

NUM = "/root/autodl-tmp/numeric"
OUT = "/root/autodl-tmp/processed"

# Physiologically plausible ranges; Solar8000 emits -8/-9 style artifacts when a
# transducer is disconnected, so anything outside the range becomes NaN.
RANGES = {
    "Solar8000/ART_MBP": (20.0, 200.0),
    "Solar8000/ART_SBP": (30.0, 300.0),
    "Solar8000/ART_DBP": (10.0, 200.0),
    "Solar8000/NIBP_MBP": (20.0, 200.0),
    "Solar8000/HR": (20.0, 250.0),
    "Solar8000/PLETH_SPO2": (50.0, 100.0),
    "Solar8000/ETCO2": (5.0, 70.0),
    "Solar8000/BT": (25.0, 42.0),
}

# Optional functional channels (respiration, pulmonary mechanics, anaesthesia depth).
# Deliberately NOT part of RANGES: RANGES drives the case duration and therefore the
# window grid, so adding to it would silently change every existing dataset. These are
# read only with --func-tracks and emitted as fn_* features.
FUNC_CHANNELS = {
    "petco2": ("Primus/ETCO2", (5.0, 70.0)),
    "rr": ("Primus/RR_CO2", (0.0, 60.0)),
    "mv": ("Primus/MV", (0.0, 30.0)),
    "compl": ("Primus/COMPLIANCE", (0.0, 200.0)),
    "pip": ("Primus/PIP_MBAR", (0.0, 100.0)),
    "bis": ("BIS/BIS", (0.0, 100.0)),
}

HYPOTENSION_MAP = 65.0
SUSTAIN_S = 60            # length of a "hypotension minute"
SUSTAIN_MIN_SAMPLES = 54  # >= 90% of that minute must be below threshold

# One monitored risk = one physiological series crossing a threshold for >= SUSTAIN_S.
# The machinery is identical, only the signal and the cut differ, which is what makes
# this a monitoring task (several risks watched at once) instead of a single-endpoint
# classifier. The 67 features already summarise every one of these series, so one
# feature matrix serves every endpoint.
ENDPOINTS = {
    "hypotension": ("map", 65.0),    # MAP below 65 mmHg
    "hypoxemia": ("spo2", 90.0),     # SpO2 below 90 %
    "bradycardia": ("hr", 50.0),     # HR below 50 bpm
    "tachycardia": ("hr", 120.0),    # HR above 120 bpm
    "hypoventilation": ("rr", 6.0),  # respiratory rate below 6 /min (needs --func-tracks)
    "hypercapnia": ("petco2", 50.0),  # ETCO2 above 50 mmHg (needs --func-tracks)
}
ENDPOINT_ABOVE = {"tachycardia", "hypercapnia"}
HORIZON_S = 300           # predict 5 minutes ahead
PAST_S = 300              # observation window
LONG_PAST_S = 900         # optional second, longer observation window
STRIDE_S = 30
MAX_GAP_S = 15.0          # max zero-order-hold staleness when gridding
MIN_MAP_COVERAGE = 0.5    # fraction of the case that must have a valid MAP
MAX_DURATION_S = 24 * 3600  # reject outlier timestamps / absurd case lengths

NAN = np.nan

# "any"   : hypotension is present at any point in the future window (standard alarm)
# "onset" : a NEW hypotension episode begins in the future window. Removes the
#           boundary overlap between the trailing 60 s of the label and the
#           observation window, so it cannot be solved by reading current MAP.
LABEL_MODE = "any"


def _f(x, default=NAN):
    """float() that never raises."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default=0):
    """int() that never raises."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def track_path(caseid, tname, num_dir=NUM):
    return os.path.join(num_dir, f"{caseid}_{re.sub(r'[^A-Za-z0-9_.-]', '_', tname)}.csv")


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_track(caseid, tname, num_dir=NUM, limits=None):
    """Return (t, v) arrays with implausible values set to NaN, or None."""
    path = track_path(caseid, tname, num_dir)
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if df.shape[1] < 2 or len(df) < 2:
        return None
    # asarray, not Series.to_numpy: pd.to_numeric is typed as a scalar, which makes
    # the attribute lookup unresolvable for static checkers.
    t = np.asarray(pd.to_numeric(df.iloc[:, 0], errors="coerce"), dtype=float)
    v = np.asarray(pd.to_numeric(df.iloc[:, 1], errors="coerce"), dtype=float)
    lo, hi = limits if limits is not None else RANGES[tname]
    v = np.where(np.isfinite(v) & (v >= lo) & (v <= hi), v, NAN)
    return t, v


def to_grid(t, v, grid, max_gap=MAX_GAP_S, min_samples=2):
    """Zero-order-hold resample onto `grid` seconds, leaving long gaps as NaN.

    `min_samples` guards against building a series out of a single sample. The offline
    builder passes the whole case's samples, so it keeps the default; the streaming
    monitor passes a window-sized buffer where one sample is legitimately all the
    channel has, and must pass 1 to stay bit-identical to the offline path.
    """
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < max(1, min_samples):
        return np.full(grid.shape, NAN)
    order = np.argsort(t)
    t, v = t[order], v[order]
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.full(grid.shape, NAN)
    valid = idx >= 0
    if not valid.any():
        return out
    clip = np.clip(idx, 0, len(t) - 1)
    age = np.where(valid, grid - t[clip], np.inf)
    use = valid & (age <= max_gap)
    out[use] = v[clip[use]]
    return out


def stat_block(name, x, out):
    """Summary stats of one past-window signal -> dict."""
    fin = np.isfinite(x)
    n = _i(fin.sum())
    out[f"{name}_cov"] = n / max(len(x), 1)
    if n == 0:
        for suffix in ("mean", "std", "min", "max", "last", "delta", "last60", "mean60"):
            out[f"{name}_{suffix}"] = NAN
        return
    xf = x[fin]
    out[f"{name}_mean"] = _f(xf.mean())
    out[f"{name}_std"] = _f(xf.std())
    out[f"{name}_min"] = _f(xf.min())
    out[f"{name}_max"] = _f(xf.max())
    out[f"{name}_last"] = _f(xf[-1])
    out[f"{name}_delta"] = _f(xf[-1] - xf[0]) / max(len(xf) - 1, 1) * 60.0
    tail = x[-60:][np.isfinite(x[-60:])]
    out[f"{name}_last60"] = _f(tail[-1]) if len(tail) else NAN
    out[f"{name}_mean60"] = _f(tail.mean()) if len(tail) else NAN


def event_flags(series, threshold, above=False):
    """Per grid second: is the trailing 60 s at least 90% beyond the threshold?

    This is the "event minute" every label is built from, and the reason the P0 event
    evaluation (onset, lead time) imports it rather than restating the definition.
    """
    n = len(series)
    breach = np.isfinite(series) & ((series > threshold) if above else (series < threshold))
    csum = np.concatenate([[0], np.cumsum(breach.astype(np.int64))])
    flags = np.zeros(n, dtype=bool)
    if n > SUSTAIN_S:
        win = csum[SUSTAIN_S:] - csum[:-SUSTAIN_S]
        flags[SUSTAIN_S - 1:] = win >= SUSTAIN_MIN_SAMPLES
    return flags


def endpoint_flags(sig, endpoint="hypotension"):
    """Event flags for one endpoint, given a case's signal dict."""
    series_name, threshold = ENDPOINTS[endpoint]
    return event_flags(sig[series_name], threshold, endpoint in ENDPOINT_ABOVE)


def hypotension_flags(mapg):
    """Backwards-compatible alias for the default endpoint's flags."""
    return event_flags(mapg, HYPOTENSION_MAP)


def window_row(t0, win, low_win, keys, long_win=None, low_long=None, func_win=None):
    """One decision time's features, in the exact key order the dataset uses.

    Shared by the offline builder and the streaming monitor (remote/stream_monitor.py):
    the online path must import this rather than re-derive the statistics, otherwise
    the two implementations drift and the offline numbers stop describing the monitor.

    `win` maps signal name -> its 1 Hz samples over [t0-PAST_S, t0); `low_win` is the
    boolean "MAP below threshold" series over the same window. `long_win`/`low_long`
    are the optional second window (pass a dict of empty arrays when the record is not
    long enough yet, which is what the dataset does).
    """
    out = {}
    for name, seq in win.items():
        stat_block(name, seq, out)
    out["map_below65_frac"] = _f(low_win.mean())
    out["map_gap"] = _f(out["map_last"] - HYPOTENSION_MAP)
    out["t_min"] = _f(t0) / 60.0
    out["n_tracks"] = _i(sum(1 for k in keys if np.isfinite(win[k]).any()))
    if long_win is not None:
        for name, seq in long_win.items():
            stat_block("l_" + name, seq, out)
        out["l_map_below65_frac"] = _f(low_long.mean()) if low_long is not None else NAN
    if func_win is not None:
        # Respiration / pulmonary mechanics / depth channels, prefixed fn_ so they stay
        # a separate feature group rather than leaking into the published baseline.
        for name, seq in func_win.items():
            stat_block("fn_" + name, seq, out)
    return out


def case_windows(caseid, grid, sig, label_mode=LABEL_MODE, stride=STRIDE_S,
                 horizon=HORIZON_S, long_past=0, extra_tracks=False,
                 endpoint="hypotension", func_tracks=False):
    """Return list of (t_sec, label, feature_dict) for one case.

    `endpoint` picks which monitored risk the label describes. The features never
    change with it: the same 67 (or more) numbers describe the same past window, so
    every endpoint is scored on one shared representation.

    `long_past` > 0 adds a second, longer observation window per signal, prefixed
    `l_`. Windows too early for that window keep NaN there, so a model with and
    without the long features is trained and scored on exactly the SAME rows.

    `extra_tracks` appends the two tracks that build_dataset has always read but
    the baseline never used (body temperature and the intermittent cuff MAP),
    plus their difference from the invasive MAP. Off by default so the published
    67-feature baseline stays byte-comparable.
    """
    mapg = sig["map"]
    low = np.isfinite(mapg) & (mapg < HYPOTENSION_MAP)
    n = len(grid)

    hyp = endpoint_flags(sig, endpoint)
    hyp_c = np.concatenate([[0], np.cumsum(hyp.astype(np.int64))])
    onset = np.zeros(n, dtype=bool)
    if n > 1:
        onset[1:] = hyp[1:] & ~hyp[:-1]
    onset_c = np.concatenate([[0], np.cumsum(onset.astype(np.int64))])
    cum = onset_c if label_mode == "onset" else hyp_c

    keys = ("sbp", "dbp", "hr", "spo2", "etco2")
    series = {"map": mapg, "sbp": sig["sbp"], "dbp": sig["dbp"], "hr": sig["hr"],
              "spo2": sig["spo2"], "etco2": sig["etco2"],
              "pp": sig["sbp"] - sig["dbp"]}
    if extra_tracks:
        series["bt"] = sig["bt"]
        series["nibp"] = sig["nibp"]
        series["cuff_gap"] = mapg - sig["nibp"]
    empty = np.full(1, NAN)
    rows = []
    for t0 in range(PAST_S, max(PAST_S, n - horizon), stride):
        end = min(t0 + horizon, n)
        label = 1 if cum[end] - cum[t0] > 0 else 0
        past = slice(t0 - PAST_S, t0)
        win = {name: seq[past] for name, seq in series.items()}
        if long_past:
            long = slice(t0 - long_past, t0) if t0 >= long_past else None
            long_win = ({name: seq[long] for name, seq in series.items()}
                        if long is not None else dict.fromkeys(series, empty))
            low_long = low[long] if long is not None else None
        else:
            long_win, low_long = None, None
        rows.append((t0, label,
                     window_row(t0, win, low[past], keys, long_win, low_long,
                                func_win={k: sig[k][past] for k in FUNC_CHANNELS}
                                if func_tracks else None)))
    return rows


def load_case_signals(caseid, num_dir=NUM, horizon=HORIZON_S, func_tracks=False):
    """Read one case's numeric tracks -> (grid, sig, duration_s), or (None, reason).

    Factored out of build_case so the P0 event evaluation can rebuild the exact
    same 1 Hz grid and MAP series the labels came from.
    """
    raws = {}
    duration = 0.0
    for tname in RANGES:
        got = read_track(caseid, tname, num_dir)
        if got is None:
            continue
        t, v = got
        # A single stray timestamp would otherwise blow the grid up to millions
        # of samples and make this case take minutes.
        sane = np.isfinite(t) & (t >= 0) & (t <= MAX_DURATION_S)
        t = np.where(sane, t, NAN)
        raws[tname] = (t, v)
        fin = np.isfinite(t) & np.isfinite(v)
        if fin.any():
            duration = max(duration, _f(t[fin].max()))
    if "Solar8000/ART_MBP" not in raws or duration < PAST_S + horizon:
        return None, "no_abp_or_too_short"

    abp_t, abp_v = raws["Solar8000/ART_MBP"]
    grid = np.arange(0.0, duration + 1.0, 1.0)
    sig = {"map": to_grid(abp_t, abp_v, grid)}
    for key, tname in (("sbp", "Solar8000/ART_SBP"), ("dbp", "Solar8000/ART_DBP"),
                       ("hr", "Solar8000/HR"), ("spo2", "Solar8000/PLETH_SPO2"),
                       ("etco2", "Solar8000/ETCO2"), ("bt", "Solar8000/BT"),
                       ("nibp", "Solar8000/NIBP_MBP")):
        if tname in raws:
            trk_t, trk_v = raws[tname]
            sig[key] = to_grid(trk_t, trk_v, grid)
        else:
            sig[key] = np.full(grid.shape, NAN)
    if func_tracks:
        for key, (tname, limits) in FUNC_CHANNELS.items():
            got = read_track(caseid, tname, num_dir, limits)
            if got is None:
                sig[key] = np.full(grid.shape, NAN)
            else:
                fn_t, fn_v = got
                sig[key] = to_grid(fn_t, fn_v, grid)
    return grid, sig, duration


def build_case(caseid, num_dir=NUM, label_mode=LABEL_MODE, horizon=HORIZON_S, long_past=0,
               extra_tracks=False, endpoint="hypotension", func_tracks=False):
    """Return (DataFrame, reason) for one case."""
    loaded = load_case_signals(caseid, num_dir, horizon, func_tracks)
    if loaded[0] is None:
        return None, loaded[1]
    grid, sig, duration = loaded

    cov = _f(np.isfinite(sig["map"]).mean())
    if cov < MIN_MAP_COVERAGE:
        return None, "low_map_coverage"

    rows = case_windows(caseid, grid, sig, label_mode, horizon=horizon,
                        long_past=long_past, extra_tracks=extra_tracks, endpoint=endpoint,
                        func_tracks=func_tracks)
    if not rows:
        return None, "no_windows"
    recs = [{"caseid": caseid, "t": t, "label": lab, **feat} for t, lab, feat in rows]
    return pd.DataFrame(recs), f"ok(map_cov={cov:.2f},dur={duration / 60.0:.0f}min)"


def list_caseids(num_dir=NUM):
    pattern = os.path.join(num_dir, "*_Solar8000_ART_MBP.csv")
    try:
        files = glob.glob(pattern)
    except OSError:
        return []
    out = set()
    for path in files:
        head = os.path.basename(path).split("_", 1)[0]
        try:
            out.add(int(head))
        except ValueError:
            continue
    return sorted(out)


def parse_caseids(text):
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError as exc:
            raise SystemExit(f"bad caseid {chunk!r}") from exc
    return out


def _work(job):
    """Pool worker: one (caseid, num_dir, label_mode, horizon, long_past, extra, endpoint).

    The job carries its own arguments because Windows spawns workers instead of
    forking them, so globals set in main() would not reach the children.
    """
    caseid, num_dir, label_mode, horizon, long_past, extra, endpoint, func = job
    try:
        return build_case(caseid, num_dir, label_mode, horizon, long_past, extra, endpoint,
                          func)
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the pool
        return None, f"error:{type(exc).__name__}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tracks-dir", default=NUM)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--caseids", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--label-mode", choices=("any", "onset"), default="any")
    ap.add_argument("--horizon", type=int, default=HORIZON_S,
                    help="label look-ahead in seconds (300 = 5 min, 600 = 10 min, 900 = 15 min)")
    ap.add_argument("--long-past", type=int, default=0,
                    help="also emit l_* features summarising this many seconds back (0 = off)")
    ap.add_argument("--extra-tracks", action="store_true",
                    help="also emit features from Solar8000/BT and NIBP_MBP (unused by the baseline)")
    ap.add_argument("--func-tracks", action="store_true",
                    help="also read the functional channels (Primus respiration/pulmonary, "
                         "BIS depth) and emit them as fn_* features")
    ap.add_argument("--endpoint", choices=tuple(ENDPOINTS), default="hypotension",
                    help="which monitored risk the label describes; the features are identical")
    args = ap.parse_args(argv)

    global LABEL_MODE
    LABEL_MODE = args.label_mode
    if args.horizon <= 0:
        raise SystemExit("--horizon must be positive")
    if args.long_past and args.long_past <= PAST_S:
        raise SystemExit("--long-past must exceed the base window")
    print(f"label_mode={LABEL_MODE} horizon={args.horizon}s past={PAST_S}s "
          f"long_past={args.long_past or 'off'} extra_tracks={args.extra_tracks} "
          f"endpoint={args.endpoint}", flush=True)

    caseids = parse_caseids(args.caseids) if args.caseids else list_caseids(args.tracks_dir)
    if args.limit:
        caseids = caseids[: args.limit]
    print(f"cases to process: {len(caseids)} workers={args.workers}", flush=True)

    jobs = [(cid, args.tracks_dir, LABEL_MODE, args.horizon, args.long_past,
             args.extra_tracks, args.endpoint, args.func_tracks) for cid in caseids]
    frames = []
    reasons = {}
    with mp.Pool(processes=args.workers) as pool:
        for i, (df, why) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4), 1):
            if df is None:
                key = why.split("(")[0]
                reasons[key] = reasons.get(key, 0) + 1
            else:
                frames.append(df)
            if i % 500 == 0:
                print(f"  {i}/{len(caseids)} kept={len(frames)} {reasons}", flush=True)

    if not frames:
        print("no data built", flush=True)
        return 1
    allw = pd.concat(frames, ignore_index=True)
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc
    path = os.path.join(args.out, "windows.parquet")
    allw.to_parquet(path, index=False)

    ncase = allw.caseid.nunique()
    pos = _f(allw.label.mean())
    feats = [c for c in allw.columns if c not in ("caseid", "t", "label")]
    print(f"windows={len(allw)} cases={ncase} cols={allw.shape[1]} feats={len(feats)}")
    print(f"horizon={args.horizon}s long_past={args.long_past or 'off'} "
          f"extra_tracks={args.extra_tracks} label_mode={LABEL_MODE} "
          f"endpoint={args.endpoint}")
    print(f"positive_rate={pos:.4f} majority_baseline_acc={1 - pos:.4f}")
    print(f"per_case_windows_mean={len(allw) / max(ncase, 1):.1f}")
    print(f"dropped={reasons}")
    print(f"wrote {path} ({file_size(path) / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
