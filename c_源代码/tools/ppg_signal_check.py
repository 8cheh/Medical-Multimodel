"""Validate downloaded PPG (.npy) samples and extract the numbers the report plots.

Checks, per sample:
  * shape / dtype / NaN fraction (the sidecar claims valid_frac)
  * amplitude range on the finite part
  * hand-checked pulse rate from a band-limited Welch PSD, plus window-to-window
    stability so a single lucky window cannot masquerade as a clean pulse

Writes scratch/ppg_samples/ppg_signal_check.json for the figure step to consume.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import signal as sg

FS_HZ = 500.0
BAND_HZ = (0.5, 4.0)   # 30-240 bpm
NAN = np.nan


def to_float(value, default=NAN):
    """Best-effort float; analysis code prefers a sentinel over a hard crash."""
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


def load_sample(path):
    try:
        arr = np.load(path)
    except (OSError, ValueError) as exc:
        print(f"cannot load {path}: {type(exc).__name__}: {exc}")
        return None
    sidecar = os.path.splitext(path)[0] + ".json"
    meta = {}
    if os.path.exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"cannot read {sidecar}: {exc}")
    return arr.astype(np.float64), meta


def clean(arr):
    """Replace NaN with the median so downstream filters stay finite."""
    med = to_float(np.nanmedian(arr))
    if not np.isfinite(med):
        med = 0.0
    return np.nan_to_num(arr, nan=med), med


def bandpass(x):
    sos = sg.butter(4, [BAND_HZ[0] / (FS_HZ / 2), BAND_HZ[1] / (FS_HZ / 2)],
                    btype="band", output="sos")
    return sg.sosfiltfilt(sos, x)


def welch_peak(x):
    """Dominant pulse rate over the whole record plus peak-to-median prominence."""
    try:
        fw, pw = sg.welch(x, fs=FS_HZ, nperseg=to_int(20 * FS_HZ),
                          noverlap=to_int(10 * FS_HZ))
    except (ValueError, TypeError) as exc:
        print(f"welch failed: {exc}")
        return NAN, NAN
    band = np.asarray((fw >= BAND_HZ[0]) & (fw <= BAND_HZ[1]))
    if not band.any():
        return NAN, NAN
    f_band, p_band = fw[band], pw[band]
    k = to_int(np.argmax(p_band))
    med = to_float(np.median(p_band))
    peak = to_float(f_band[k]) * 60.0
    prominence = to_float(p_band[k]) / med if med > 0 else NAN
    return peak, to_float(prominence)


def window_rates(x, win_s=30.0, step_s=60.0, max_s=900.0):
    """Pulse rate per window; a real pulse stays in a narrow band across windows."""
    n_win = to_int(win_s * FS_HZ)
    limit = min(len(x), to_int(max_s * FS_HZ))
    step = to_int(step_s * FS_HZ)
    rates = []
    for start in range(0, max(limit - n_win, 0), step):
        seg = x[start:start + n_win]
        if len(seg) < n_win:
            break
        try:
            f2, p2 = sg.welch(seg, fs=FS_HZ, nperseg=to_int(10 * FS_HZ),
                              noverlap=to_int(5 * FS_HZ))
        except (ValueError, TypeError) as exc:
            print(f"welch(window) failed: {exc}")
            continue
        b2 = np.asarray((f2 >= BAND_HZ[0]) & (f2 <= BAND_HZ[1]))
        if not b2.any():
            continue
        rates.append(to_float(f2[b2][to_int(np.argmax(p2[b2]))]) * 60.0)
    return rates


def summarise(arr, meta, path):
    nan_frac = to_float(np.isnan(arr).mean())
    v, med = clean(arr)
    filt = bandpass(v)
    hr, prominence = welch_peak(filt)
    rates = window_rates(filt)
    finite = v[np.isfinite(v)]
    return {
        "file": os.path.basename(path),
        "caseid": meta.get("caseid"),
        "n_samples": to_int(arr.size),
        "duration_s": to_float(arr.size) / FS_HZ,
        "sidecar_valid_frac": meta.get("valid_frac"),
        "nan_frac_measured": nan_frac,
        "median": med,
        "p01": to_float(np.percentile(finite, 1)),
        "p99": to_float(np.percentile(finite, 99)),
        "std": to_float(finite.std()),
        "hr_bpm_welch": hr,
        "spectral_prominence": prominence,
        "window_hr_bpm": rates,
        "window_hr_median": to_float(np.median(rates)) if rates else NAN,
        "window_hr_spread": to_float(np.max(rates) - np.min(rates)) if rates else NAN,
    }


def analyse(path):
    loaded = load_sample(path)
    if loaded is None:
        return None
    arr, meta = loaded
    return summarise(arr, meta, path)


def write_json(path, payload):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"cannot write {path}: {exc}")
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="scratch/ppg_samples")
    ap.add_argument("--out", default="scratch/ppg_samples/ppg_signal_check.json")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.dir, "*.npy")))
    if not files:
        print(f"no .npy samples in {args.dir}")
        return 1

    rows = []
    for path in files:
        row = analyse(path)
        if row is None:
            continue
        rows.append(row)
        print(f"{row['file']}: n={row['n_samples']} ({row['duration_s']:.0f}s) "
              f"nan={row['nan_frac_measured']:.5f} median={row['median']:.2f} "
              f"p01={row['p01']:.1f} p99={row['p99']:.1f}")
        print(f"    Welch HR={row['hr_bpm_welch']:.1f} bpm "
              f"prominence={row['spectral_prominence']:.1f} "
              f"30s-window median={row['window_hr_median']:.1f} "
              f"spread={row['window_hr_spread']:.1f}")
        print(f"    windows: {' '.join(f'{h:.0f}' for h in row['window_hr_bpm'][:10])}")

    payload = {"fs_hz": FS_HZ, "band_hz": list(BAND_HZ), "samples": rows}
    if not write_json(args.out, payload):
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
