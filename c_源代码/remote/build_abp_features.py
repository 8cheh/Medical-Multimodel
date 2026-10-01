"""Arterial-line waveform features (SNUADC/ART) on the same 30 s block grid as PPG.

Why the waveform and not just the numeric ART_MBP/SBP/DBP tracks: those are the
monitor's own per-minute values, so they cannot express beat-level morphology -
pulse pressure variation, upstroke slope, or the shape changes that precede a fall
in pressure. That is the feature family the published hypotension-prediction
indices are built on, which is why the arterial line is the second modality worth
adding rather than a third copy of the same numbers.

The block grid, the lookback indexing and the per-window aggregation are imported
from build_ppg_features so both modalities index windows identically.

Input : <windows>.parquet (decision times) + <wave dir>/<caseid>.npy @500 Hz
Output: <out>/abp_features.parquet (caseid, t, abp_*)

Usage:
    python build_abp_features.py --windows <windows.parquet> --wave-dir <art_npy> \
        --out <dir> --workers 8
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from build_ppg_features import (
    BLOCK_S,
    FS,
    NAN,
    aggregate_blocks,
    block_window,
    load_meta,
    needed_blocks,
    to_float,
    to_int,
)
from scipy import signal as sg

WAVE_DIR = "/root/autodl-tmp/art_npy"
WINDOWS = "/root/autodl-tmp/processed/windows.parquet"
OUT = "/root/autodl-tmp/processed"

# Per-block morphology summary. Key names avoid embedding an aggregation word
# (mean/std/last/min/max) because aggregate_blocks appends one: a key "pp_mean"
# would yield the column abp_pp_mean_mean.
BLOCK_KEYS = ("hr", "pp", "pp_sd", "ppv", "dpdt", "rise", "sys", "dia",
              "amp", "skew", "kurt", "n_beats", "n_valid")

T0_BY_CASE = {}


def blank_block(n_valid=0.0):
    out = dict.fromkeys(BLOCK_KEYS, NAN)
    out["n_valid"] = n_valid
    return out


def abp_block_features(x, fs=FS):
    """Characterise one 30 s arterial block. Returns a dict (values may be NaN).

    Detection runs on a 0.5-10 Hz band-pass, but every amplitude is measured on the
    raw (mean-removed) signal, so pp/dpdt keep their physiological units.
    """
    n = to_int(x.size)
    if n == 0:
        return blank_block(0.0)
    fin = np.isfinite(x)
    n_valid = to_float(fin.mean())
    if n_valid < 0.8:
        return blank_block(n_valid)

    v = np.asarray(x, dtype="float64")
    if not fin.all():
        idx = np.arange(n)
        v = np.interp(idx, idx[fin], v[fin])
    v = v - to_float(np.mean(v), 0.0)

    ny = 0.5 * fs
    try:
        b, a = sg.butter(3, [0.5 / ny, 10.0 / ny], btype="band")
        det = sg.filtfilt(b, a, v)
    except ValueError:
        return blank_block(n_valid)
    std = to_float(np.std(det), 0.0)
    if std <= 0:
        return blank_block(n_valid)
    try:
        peaks, _ = sg.find_peaks(det, distance=max(1, to_int(0.4 * fs)),
                                 prominence=0.3 * std)
    except ValueError:
        return blank_block(n_valid)
    if peaks.size < 8:
        return blank_block(n_valid)

    # Per beat: use the trough that PRECEDES each systolic peak, so pp is this
    # beat's pulse pressure, the upstroke is this beat's upstroke, and the interval
    # is this beat's period. (An earlier version measured the upstroke from the
    # following trough, which sliced v[trough:p0+1] with trough > p0 - always
    # empty - so every beat was discarded and all morphology came out NaN.)
    pps, dts, slopes, rises, sys_v, dia_v = [], [], [], [], [], []
    for i in range(1, peaks.size):
        p_prev, p0 = to_int(peaks[i - 1]), to_int(peaks[i])
        if p0 - p_prev < 3:
            continue
        trough = p_prev + to_int(np.argmin(v[p_prev:p0]))
        sys_value = to_float(v[p0])
        dia_value = to_float(v[trough])
        pp = sys_value - dia_value
        if not (pp > 0):
            continue
        up = v[trough:p0 + 1]
        if up.size < 3:
            continue
        slopes.append(to_float(np.gradient(up).max(), 0.0) * fs)
        pps.append(pp)
        dts.append((p0 - p_prev) / fs)
        rises.append((p0 - trough) / max(p0 - p_prev, 1))
        sys_v.append(sys_value)
        dia_v.append(dia_value)
    if len(pps) < 5:
        return blank_block(n_valid)

    pp = np.array(pps)
    dt_med = to_float(np.median(dts), 0.0)
    pp_mean = to_float(pp.mean(), 0.0)
    pp_std = to_float(pp.std(), 0.0)
    scale = to_float(np.std(v), 0.0) or 1e-12
    return {
        "hr": 60.0 / dt_med if dt_med > 0 else NAN,
        "pp": pp_mean,
        "pp_sd": pp_std,
        "ppv": 100.0 * pp_std / pp_mean if pp_mean > 0 else NAN,
        "dpdt": to_float(np.median(slopes)),
        "rise": to_float(np.median(rises)),
        "sys": to_float(np.median(sys_v)),
        "dia": to_float(np.median(dia_v)),
        "amp": to_float(v.max() - v.min()),
        "skew": to_float(((v / scale) ** 3).mean()),
        "kurt": to_float(((v / scale) ** 4).mean()),
        "n_beats": to_float(len(pps)),
        "n_valid": n_valid,
    }


def case_features(caseid, wave_dir=WAVE_DIR, t0s=None):
    """One row per decision time of one case, or None when the track is missing."""
    if t0s is None:
        t0s = T0_BY_CASE.get(caseid)
    if t0s is None or t0s.size == 0:
        return None

    path = os.path.join(wave_dir, f"{caseid}.npy")
    if not os.path.isfile(path):
        return None
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if to_int(arr.size) < to_int(FS * BLOCK_S * 2):
        return None

    sentinel = load_meta(caseid, wave_dir).get("nan_sentinel", -32768)
    n = to_int(arr.size)

    blocks = {}
    for b in sorted(needed_blocks(t0s, n)):
        lo, hi = block_window(b)
        chunk = np.asarray(arr[lo:hi], dtype="float64")
        if arr.dtype == np.int16:
            chunk = np.where(chunk == sentinel, NAN, chunk)
        blocks[b] = abp_block_features(chunk)

    rows = aggregate_blocks(caseid, t0s, blocks, BLOCK_KEYS, "abp_")
    if not rows:
        return None
    return pd.DataFrame(rows)


def _work(job):
    caseid, wave_dir, t0s = job
    try:
        return case_features(caseid, wave_dir, t0s)
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the pool
        return f"error:{type(exc).__name__}:{caseid}"


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=WINDOWS)
    ap.add_argument("--wave-dir", default=WAVE_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=8)
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
            for p in glob.glob(os.path.join(args.wave_dir, "*.npy"))}
    todo = sorted(have & set(T0_BY_CASE))
    if args.limit:
        todo = todo[: args.limit]
    print(f"cases with both windows and ABP: {len(todo)}", flush=True)

    frames = []
    errors = 0
    with mp.Pool(processes=args.workers) as pool:
        jobs = [(cid, args.wave_dir, T0_BY_CASE[cid]) for cid in todo]
        for i, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=2), 1):
            if isinstance(res, str):
                errors += 1
                if errors <= 5:
                    print(f"  {res}", flush=True)
            elif res is not None:
                frames.append(res)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} frames={len(frames)} errors={errors}",
                      flush=True)

    if not frames:
        print("no ABP features built", flush=True)
        return 1
    out = pd.concat(frames, ignore_index=True)
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc
    path = os.path.join(args.out, "abp_features.parquet")
    out.to_parquet(path, index=False)
    print(f"abp feature rows={len(out)} cases={out.caseid.nunique()} cols={out.shape[1]}")
    print(f"errors={errors}  wrote {path} ({os.path.getsize(path) / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
