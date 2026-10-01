"""Self-check: the extracted episode definition still reproduces the labels.

Runs on whatever track CSVs sit in --tracks-dir (the 20-case pilot is enough).
Recomputes the "any" label from `hypotension_flags` with an INDEPENDENT loop and
compares it against what `case_windows` actually recorded, per case. If the
extraction of hypotension_flags/load_case_signals changed semantics, this fails.

    python check_build_refactor.py --tracks-dir <dir with *_Solar8000_ART_MBP.csv>
"""
import argparse

import build_dataset as bd
import numpy as np


def independent_labels(mapg, t0, horizon):
    """Label for window t0, written the slow/obvious way (no cumsum tricks)."""
    n = len(mapg)
    end = min(t0 + horizon, n)
    low = np.isfinite(mapg) & (mapg < bd.HYPOTENSION_MAP)
    for t in range(t0, end):
        lo = max(0, t - bd.SUSTAIN_S + 1)
        window = low[lo: t + 1]
        if window.size >= bd.SUSTAIN_MIN_SAMPLES and window.sum() >= bd.SUSTAIN_MIN_SAMPLES:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    caseids = bd.list_caseids(args.tracks_dir)[: args.limit]
    assert caseids, f"no ART_MBP tracks in {args.tracks_dir}"
    checked = frames = dropped = 0
    for cid in caseids:
        loaded = bd.load_case_signals(cid, args.tracks_dir)
        assert loaded[0] is not None or isinstance(loaded[1], str), "bad failure contract"
        if loaded[0] is None:
            continue
        grid, sig, _dur = loaded
        assert grid[0] == 0.0 and grid[-1] >= bd.PAST_S + bd.HORIZON_S, "grid shape"
        # hyp must equal the per-sample rule stated in its docstring
        hyp = bd.hypotension_flags(sig["map"])
        low = np.isfinite(sig["map"]) & (sig["map"] < bd.HYPOTENSION_MAP)
        for t in (700, 1200, 2500):
            if t >= len(hyp):
                continue
            win = low[t - bd.SUSTAIN_S + 1: t + 1]
            assert bool(hyp[t]) == (win.sum() >= 54), f"hyp mismatch at {t} ({cid})"
        rows = bd.case_windows(cid, grid, sig, "any")
        assert rows, "no windows"
        for t0, lab, _feat in rows[:40]:
            exp = independent_labels(sig["map"], t0, bd.HORIZON_S)
            assert lab == exp, f"label mismatch case={cid} t={t0}: {lab} != {exp}"
            checked += 1
        df, why = bd.build_case(cid, args.tracks_dir, "any")
        assert df is not None or isinstance(why, str), "build_case contract"
        if df is None:  # legitimately dropped (low coverage / too short / no windows)
            dropped += 1
            continue
        n_feats = len([c for c in df.columns if c not in ("caseid", "t", "label")])
        assert n_feats == 67, f"expected 67 features, got {n_feats}"
        frames += 1
    assert frames > 0, "no case produced windows"
    print(f"OK cases={frames} labels_verified={checked} dropped={dropped}")


if __name__ == "__main__":
    main()
