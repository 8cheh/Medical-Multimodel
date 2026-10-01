"""Sanity-verify the compacted PPG arrays: are they genuine pulsatile waveforms?

A real PPG has a dominant spectral peak at the heart rate (0.5-3 Hz ~ 30-180 bpm).
If a file is all NaN, constant, or noise, this catches it before we build on it.
"""
import glob
import json
import os
import sys

import numpy as np

D = "/root/autodl-tmp/ppg_npy"
FS = 500.0  # SNUADC waveform tracks are stored at 500 Hz (dt = 0.002 s)
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


def band_peak_hz(x, fs):
    """Dominant frequency in 0.5-3 Hz plus its prominence over the band median.

    A share-of-total metric is useless here: the 0.5-3.0 Hz band holds ~2500 FFT
    bins, so even a sharp pulse peak is ~0.3% of the band sum. Prominence
    (peak / median band power) is the right shape test.
    """
    if x.size < fs * 4:
        return NAN, 0.0
    x = x - np.nanmean(x)
    x = np.where(np.isfinite(x), x, 0.0)
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    band = (freqs >= 0.5) & (freqs <= 3.0)
    if not band.any():
        return NAN, 0.0
    band_spec = spec[band]
    band_freqs = freqs[band]
    k = to_int(np.argmax(band_spec))
    peak = to_float(band_freqs[k])
    med = to_float(np.median(band_spec))
    prom = to_float(band_spec.max()) / med if med > 0 else 0.0
    return peak, prom


def load_meta(path):
    meta_path = path.replace(".npy", ".json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def load_head(path, meta, limit=500_000):
    arr = np.load(path, mmap_mode="r")
    n = min(arr.size, limit)
    if arr.dtype == np.int16:
        sentinel = meta.get("nan_sentinel", -32768)
        v = np.asarray(arr[:n], dtype="float64")
        v = np.where(v == sentinel, NAN, v)
    else:
        v = np.asarray(arr[:n], dtype="float64")
    return arr, v


def main():
    files = sorted(glob.glob(os.path.join(D, "*.npy")))
    if not files:
        print("no npy files yet")
        return 1
    print(f"total npy files: {len(files)}")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), size=min(12, len(files)), replace=False)
    print(f"{'case':>7} {'n':>10} {'dtype':>8} {'hours':>7} {'valid%':>7} "
          f"{'range':>16} {'peakHz':>7} {'bpm':>6} {'prom':>8}")
    ok = 0
    for i in pick:
        path = str(files[to_int(i)])
        cid = os.path.basename(path).split(".")[0]
        meta = load_meta(path)
        arr, v = load_head(path, meta)
        valid = to_float(np.isfinite(v).mean()) if v.size else 0.0
        fin = v[np.isfinite(v)]
        rng_s = f"{fin.min():.1f}..{fin.max():.1f}" if fin.size else "n/a"
        peak, prom = band_peak_hz(v, FS)
        hours = arr.size / FS / 3600.0
        good = (valid > 0.9 and np.isfinite(peak) and 0.5 <= peak <= 3.0 and prom > 20)
        if good:
            ok += 1
        print(f"{cid:>7} {arr.size:>10} {str(arr.dtype):>8} {hours:>7.2f} {valid * 100:>6.1f}% "
              f"{rng_s:>16} {peak:>7.2f} {peak * 60:>6.0f} {prom:>8.1f}  {'OK' if good else 'SUSPECT'}")

    print(f"\n{ok}/{len(pick)} sampled files look like genuine pulsatile PPG")
    return 0 if ok >= len(pick) * 0.75 else 1


if __name__ == "__main__":
    sys.exit(main())
