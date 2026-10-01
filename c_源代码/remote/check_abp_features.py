"""Synthetic self-check for the ABP extractor.

Real waveforms cannot distinguish "the signal has no beats" from "the code threw
every beat away", which is exactly how an always-empty upstroke slice produced an
all-NaN feature block that looked like a physiological finding. So the extractor is
also tested on a waveform whose heart rate and pulse pressure are known by
construction.
"""
import sys

import build_abp_features as abp
import numpy as np
from build_ppg_features import FS, to_float, to_int


def synth_abp(hr_bpm=60.0, fs=FS, seconds=30.0, sys_v=120.0, dia_v=80.0):
    """Arterial-like beats: fast upstroke, exponential decay to the diastolic value."""
    n = to_int(seconds * fs)
    period = to_int(60.0 / hr_bpm * fs)
    rise = max(3, to_int(0.15 * fs))
    x = np.full(n, to_float(dia_v, 0.0))
    idx = np.arange(rise)
    upshape = (1.0 - np.cos(np.pi * idx / rise)) / 2.0
    k = np.arange(max(1, period - rise))
    decay = np.exp(-k / (0.25 * fs))
    for start in range(0, n - period, period):
        x[start:start + rise] = dia_v + (sys_v - dia_v) * upshape[: max(0, n - start)]
        tail = start + rise
        seg = decay[: max(0, min(len(decay), n - tail))]
        x[tail:tail + len(seg)] = dia_v + (sys_v - dia_v) * seg
    return x


def check(hr, expect_pp=40.0, tol_hr=3.0, tol_pp=6.0):
    x = synth_abp(hr_bpm=hr)
    out = abp.abp_block_features(x)
    assert np.isfinite(out["hr"]), f"HR not detected at {hr} bpm"
    assert abs(out["hr"] - hr) <= tol_hr, f"HR {out['hr']:.1f} vs {hr}"
    assert abs(out["pp"] - expect_pp) <= tol_pp, f"pp {out['pp']:.1f} vs {expect_pp}"
    assert out["n_beats"] >= 20, f"only {out['n_beats']} beats found at {hr} bpm"
    assert np.isfinite(out["dpdt"]) and out["dpdt"] > 0, f"dpdt {out['dpdt']}"
    assert 0.0 < out["rise"] < 1.0, f"rise fraction {out['rise']}"
    print(f"  OK {hr:5.1f} bpm -> hr={out['hr']:.1f} pp={out['pp']:.1f} "
          f"beats={out['n_beats']:.0f} dpdt={out['dpdt']:.0f} rise={out['rise']:.2f}")
    return out


def main():
    for hr in (50.0, 60.0, 90.0, 120.0, 150.0):
        check(hr)

    # the raw tracks are coarsely quantised, so coarse input must still work
    x = np.round(synth_abp(hr_bpm=75.0) / 4.0) * 4.0
    out = abp.abp_block_features(x)
    assert np.isfinite(out["pp"]) and abs(out["pp"] - 40.0) <= 8.0, \
        f"quantised pp {out['pp']}"
    print(f"  OK quantised (4-unit steps) -> pp={out['pp']:.1f} hr={out['hr']:.1f}")

    # flat and too-few-beat blocks must report absence rather than invent values
    for label, sig in (("flat", np.full(to_int(30 * FS), 100.0)),
                       ("too slow (5 bpm)", synth_abp(hr_bpm=5.0))):
        out = abp.abp_block_features(sig)
        assert not np.isfinite(out["hr"]) and not np.isfinite(out["pp"]), \
            f"{label}: expected NaN, got hr={out['hr']} pp={out['pp']}"
        print(f"  OK {label} -> NaN (no beats invented)")
    print("ABP extractor self-check passed")


if __name__ == "__main__":
    sys.exit(main())
