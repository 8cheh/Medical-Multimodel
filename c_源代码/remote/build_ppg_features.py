"""Extract PPG waveform features aligned to the numeric decision times.

Design note: computing features per decision time would redo the same 30 s of
signal up to 10 times (windows overlap by 30 s inside a 300 s lookback). Instead
each non-overlapping 30 s block is characterised ONCE, then the 10 blocks covering
[t0-300, t0) are aggregated per window. That is a ~10x saving.

Input : /root/autodl-tmp/processed/windows.parquet  (caseid, t, label, numeric feats)
        /root/autodl-tmp/ppg_npy/<caseid>.npy       (PPG @500 Hz, index 0 = t0)
Output: /root/autodl-tmp/processed/ppg_features.parquet  (caseid, t, ppg_*)
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

PPG_DIR = "/root/autodl-tmp/ppg_npy"
WINDOWS = "/root/autodl-tmp/processed/windows.parquet"
OUT = "/root/autodl-tmp/processed"

FS = 500.0
BLOCK_S = 30
LOOKBACK_BLOCKS = 10          # 10 x 30 s = 300 s observation window
NAN = np.nan

# caseid -> np.ndarray of decision times; populated before the fork so workers
# inherit it copy-on-write instead of re-reading the parquet.
T0_BY_CASE = {}


def to_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def to_float(x, default=NAN):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_meta(caseid, ppg_dir=PPG_DIR):
    mp_path = os.path.join(ppg_dir, f"{caseid}.json")
    if not os.path.isfile(mp_path):
        return {}
    try:
        with open(mp_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def block_features(x, fs=FS):
    """Characterise one 30 s block. Returns a dict (values may be NaN)."""
    out = {}
    n = to_int(x.size)
    fin = np.isfinite(x)
    out["n_valid"] = to_float(fin.mean()) if n else 0.0
    if to_int(fin.sum()) < fs:
        for k in ("mean", "std", "min", "max", "amp", "skew", "kurt", "zcr",
                  "hr", "prom", "ac_amp", "ac_lag", "slope"):
            out[k] = NAN
        return out

    v = x[fin]
    v = v - v.mean()
    out["mean"] = to_float(x[fin].mean())
    out["std"] = to_float(v.std())
    out["min"] = to_float(v.min())
    out["max"] = to_float(v.max())
    out["amp"] = to_float(v.max() - v.min())
    s = to_float(v.std(), 0.0) or 1e-12
    out["skew"] = to_float(((v / s) ** 3).mean())
    out["kurt"] = to_float(((v / s) ** 4).mean())
    out["zcr"] = to_float(np.mean(np.abs(np.diff(np.signbit(v).astype(np.int8)))))

    # spectral heart rate
    w = v * np.hanning(v.size)
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(v.size, d=1.0 / fs)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if band.any():
        bs = spec[band]
        bf = freqs[band]
        k = to_int(np.argmax(bs))
        med = to_float(np.median(bs), 0.0) or 1e-12
        out["hr"] = to_float(bf[k]) * 60.0
        out["prom"] = to_float(bs[k]) / med
    else:
        out["hr"] = NAN
        out["prom"] = NAN

    # autocorrelation: pulse amplitude and period.
    # FFT-based (O(n log n)); np.correlate is O(n^2) and dominated the runtime.
    vv = np.where(np.isfinite(v), v, 0.0)
    nv = to_int(vv.size)
    fv = np.fft.rfft(vv, n=2 * nv)
    ac = np.fft.irfft(fv * np.conj(fv))[:nv]
    ac = ac / (ac[0] if ac[0] > 0 else 1e-12)
    lo = to_int(0.5 * fs)
    hi = min(to_int(2.5 * fs), ac.size - 1)
    if hi > lo:
        seg = ac[lo:hi]
        j = to_int(np.argmax(seg))
        out["ac_amp"] = to_float(seg[j])
        out["ac_lag"] = to_float(lo + j) / fs
    else:
        out["ac_amp"] = NAN
        out["ac_lag"] = NAN

    idx = np.arange(v.size, dtype="float64")
    out["slope"] = to_float(np.polyfit(idx, v, 1)[0] * fs * 60.0)
    return out


BLOCK_KEYS = ("amp", "std", "skew", "kurt", "zcr", "hr", "prom", "ac_amp", "ac_lag",
              "slope", "n_valid")


def needed_blocks(t0s, n_samples, block_s=BLOCK_S, lookback=LOOKBACK_BLOCKS):
    """Block indices covering the lookback of every decision time.

    Block b covers [b*block_s, (b+1)*block_s) seconds and decision times sit on the
    same grid, so the lookback of t0 is exactly b = t0/block_s - lookback .. -1.
    Shared with the ABP extractor so both modalities index windows identically.
    """
    max_block = to_int(n_samples) // block_s
    needed = set()
    for t0 in t0s:
        b0 = to_int(t0) // block_s
        for b in range(b0 - lookback, b0):
            if 0 <= b < max_block:
                needed.add(b)
    return needed


def block_window(b, fs=FS, block_s=BLOCK_S):
    """Sample range [lo, hi) of one block."""
    lo = b * block_s * to_int(fs)
    return lo, lo + block_s * to_int(fs)


def aggregate_blocks(caseid, t0s, blocks, keys, prefix, lookback=LOOKBACK_BLOCKS):
    """One row per decision time: mean/std/min/max/last of each block feature.

    Any block whose index is missing (record too short) is skipped rather than
    counted as zero. Shared by the PPG and ABP extractors.
    """
    rows = []
    for t0 in t0s:
        b0 = to_int(t0) // BLOCK_S
        seq = [blocks[b] for b in range(b0 - lookback, b0) if b in blocks]
        if not seq:
            continue
        row = {"caseid": caseid, "t": to_int(t0)}
        for k in keys:
            vals = np.array([to_float(d.get(k)) for d in seq], dtype="float64")
            fin = np.isfinite(vals)
            row[f"{prefix}{k}_mean"] = to_float(vals[fin].mean()) if fin.any() else NAN
            row[f"{prefix}{k}_std"] = to_float(vals[fin].std()) if fin.any() else NAN
            row[f"{prefix}{k}_last"] = to_float(vals[fin][-1]) if fin.any() else NAN
            row[f"{prefix}{k}_min"] = to_float(vals[fin].min()) if fin.any() else NAN
            row[f"{prefix}{k}_max"] = to_float(vals[fin].max()) if fin.any() else NAN
        rows.append(row)
    return rows


def case_features(caseid, ppg_dir=PPG_DIR, t0s=None):
    """Return a DataFrame of PPG features for every decision time of one case.

    `t0s` is passed explicitly because Windows spawns pool workers instead of
    forking them, so the T0_BY_CASE global filled in main() never reaches them.
    """
    if t0s is None:
        t0s = T0_BY_CASE.get(caseid)
    if t0s is None or t0s.size == 0:
        return None

    path = os.path.join(ppg_dir, f"{caseid}.npy")
    if not os.path.isfile(path):
        return None
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if to_int(arr.size) < to_int(FS * BLOCK_S * 2):
        return None

    sentinel = load_meta(caseid, ppg_dir).get("nan_sentinel", -32768)
    n = to_int(arr.size)

    blocks = {}
    for b in sorted(needed_blocks(t0s, n)):
        lo, hi = block_window(b)
        chunk = np.asarray(arr[lo:hi], dtype="float64")
        if arr.dtype == np.int16:
            chunk = np.where(chunk == sentinel, NAN, chunk)
        blocks[b] = block_features(chunk)

    rows = aggregate_blocks(caseid, t0s, blocks, BLOCK_KEYS, "ppg_")
    if not rows:
        return None
    return pd.DataFrame(rows)


def _work(job):
    """Pool worker: (caseid, ppg_dir, t0s). Jobs carry their arguments because
    Windows spawns workers instead of forking them, so main()'s globals would not
    arrive (this silently produced zero features before it was fixed)."""
    caseid, ppg_dir, t0s = job
    try:
        return case_features(caseid, ppg_dir, t0s)
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the pool
        return f"error:{type(exc).__name__}:{caseid}"


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=WINDOWS)
    ap.add_argument("--ppg-dir", default=PPG_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        win = pd.read_parquet(args.windows, columns=["caseid", "t"])
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {args.windows}: {exc}") from exc

    for cid, grp in win.groupby("caseid", sort=False):
        T0_BY_CASE[to_int(cid)] = np.asarray(grp["t"], dtype="int64")

    have = {to_int(os.path.basename(p).split(".")[0])
            for p in glob.glob(os.path.join(args.ppg_dir, "*.npy"))}
    todo = sorted(have & set(T0_BY_CASE))
    if args.limit:
        todo = todo[: args.limit]
    print(f"cases with both windows and PPG: {len(todo)}", flush=True)

    frames = []
    errors = 0
    with mp.Pool(processes=args.workers) as pool:
        jobs = [(cid, args.ppg_dir, T0_BY_CASE[cid]) for cid in todo]
        for i, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=2), 1):
            if isinstance(res, str):
                errors += 1
                if errors <= 5:
                    print(f"  {res}", flush=True)
            elif res is not None:
                frames.append(res)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} frames={len(frames)} errors={errors}", flush=True)

    if not frames:
        print("no PPG features built", flush=True)
        return 1
    out = pd.concat(frames, ignore_index=True)
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc
    path = os.path.join(args.out, "ppg_features.parquet")
    out.to_parquet(path, index=False)
    print(f"ppg feature rows={len(out)} cases={out.caseid.nunique()} cols={out.shape[1]}")
    print(f"errors={errors}  wrote {path} ({os.path.getsize(path) / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
