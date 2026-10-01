"""Verify that the PPG .npy (index 0 = recording start) lines up in absolute time
with the numeric tracks, and report download progress.

If the two timelines were offset, PPG pulse rate and the monitor's own PLETH_HR /
HR track would disagree. Agreement is the alignment proof.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

PPG_DIR = "/root/autodl-tmp/ppg_npy"
NUM_DIR = "/root/autodl-tmp/numeric"
FS = 500.0
NAN = np.nan


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


def load_meta(path):
    mp = path.replace(".npy", ".json")
    if not os.path.isfile(mp):
        return {}
    try:
        with open(mp, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def read_numeric(caseid, tname, lo=20.0, hi=250.0):
    """Return (t, v) for a numeric track, implausible values as NaN."""
    path = os.path.join(NUM_DIR, f"{caseid}_{tname.replace('/', '_')}.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    if df.shape[1] < 2:
        return None
    t = np.asarray(df.iloc[:, 0], dtype="float64")
    v = np.asarray(df.iloc[:, 1], dtype="float64")
    v = np.where(np.isfinite(v) & (v >= lo) & (v <= hi), v, NAN)
    return t, v


def ppg_rate(x, fs, lo=0.7, hi=3.0):
    """Heart rate in bpm from the dominant spectral peak, plus peak prominence."""
    if x.size < fs * 4:
        return NAN, NAN
    y = x - np.nanmean(x)
    y = np.where(np.isfinite(y), y, 0.0)
    y = y * np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / fs)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return NAN, NAN
    bs = spec[band]
    bf = freqs[band]
    k = to_int(np.argmax(bs))
    med = to_float(np.median(bs), 0.0)
    if med <= 0:
        med = 1e-12
    return to_float(bf[k]) * 60.0, to_float(bs[k]) / med


def load_segment(path, seconds=300.0):
    """First `seconds` of a compacted PPG file, sentinel decoded to NaN."""
    arr = np.load(path, mmap_mode="r")
    n0 = min(to_int(arr.size), to_int(FS * seconds))
    seg = np.asarray(arr[:n0], dtype="float64")
    if arr.dtype == np.int16:
        sent = load_meta(path).get("nan_sentinel", -32768)
        seg = np.where(seg == sent, NAN, seg)
    return seg


def monitor_rate(caseid, seconds=300.0):
    """Median of the monitor's own rate tracks over the same interval."""
    for tname in ("Solar8000/PLETH_HR", "Solar8000/HR"):
        got = read_numeric(caseid, tname)
        if got is None:
            continue
        t, v = got
        m = (t >= 0) & (t <= seconds) & np.isfinite(v)
        if to_int(m.sum()) >= 5:
            return to_float(np.median(v[m])), tname
    return NAN, "-"


def main():
    files = sorted(glob.glob(os.path.join(PPG_DIR, "*.npy")))
    print(f"=== download progress: {len(files)} / 6157 cases ===")
    if not files:
        return 0

    rng = np.random.default_rng(1)
    idx = rng.choice(len(files), size=min(8, len(files)), replace=False)
    print(f"\n{'case':>6} {'ppg_bpm':>8} {'mon_bpm':>8} {'diff':>6} {'prom':>6}  verdict")
    agree = 0
    for i in idx:
        path = str(files[to_int(i)])
        cid = os.path.basename(path).split(".")[0]
        seg = load_segment(path)
        ppg_bpm, prom = ppg_rate(seg, FS)
        mon, src = monitor_rate(cid)
        diff = abs(ppg_bpm - mon) if np.isfinite(ppg_bpm) and np.isfinite(mon) else NAN
        ok = bool(np.isfinite(diff) and diff <= 15)
        agree += 1 if ok else 0
        print(f"{cid:>6} {ppg_bpm:>8.1f} {mon:>8.1f} {diff:>6.1f} {prom:>6.1f}  "
              f"{'ALIGNED' if ok else 'MISMATCH'} ({src})")

    verdict = "timelines aligned" if agree >= len(idx) * 0.6 else "ALIGNMENT SUSPECT"
    print(f"\n{agree}/{len(idx)} cases agree within 15 bpm -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
