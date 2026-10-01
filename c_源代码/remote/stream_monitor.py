"""Online (streaming) monitoring: feed samples, get risk and alarms as they happen.

The offline pipeline reads whole cases and computes features for every decision time.
A monitor has to do the same thing causally, with no future samples and a bounded
buffer. Three properties make that exact rather than approximate:

  * **Causality.** `to_grid` is a backward zero-order hold with a 15 s staleness limit,
    so a feature for a window ending at t0 only ever reads samples with t <= t0.
    Restricting the buffer to the past therefore cannot change any value.
  * **No drift.** The feature row comes from `build_dataset.window_row` - the same
    function the offline dataset uses - and the buffer is pruned to exactly what the
    next decision still needs.
  * **One policy.** Alarm state comes from `alarm_policy.alarm_runs`, re-run over the
    accumulated scores, so there is a single implementation of persistence/hysteresis.

    python stream_monitor.py --check --caseid <id> --windows <windows.parquet> \
        --scores <dir> --booster <lgbm_main.txt>
"""
import argparse
import os
import sys
import time

import alarm_policy
import build_dataset as bd
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # only the scored forms need it
    lgb = None

SIGNALS = ("map", "sbp", "dbp", "hr", "spo2", "etco2")
KEYS = ("sbp", "dbp", "hr", "spo2", "etco2")


class MonitorStream:
    """Incremental monitor. `push()` samples, receive decisions as they become due."""

    def __init__(self, booster=None, feature_names=None, endpoint="hypotension",
                 threshold=0.5, persist_windows=1, clear_factor=1.0,
                 past_s=bd.PAST_S, stride_s=bd.STRIDE_S, max_gap=bd.MAX_GAP_S):
        self.booster = booster
        self.names = list(feature_names if feature_names is not None
                          else (booster.feature_name() if booster is not None else []))
        self.endpoint = endpoint
        self.threshold = bd._f(threshold, 0.5)
        self.persist = bd._i(persist_windows, 1)
        self.clear = self.threshold * bd._f(clear_factor, 1.0)
        self.past_s = bd._i(past_s, bd.PAST_S)
        self.stride = bd._i(stride_s, bd.STRIDE_S)
        self.max_gap = bd._f(max_gap, bd.MAX_GAP_S)
        self.buf = {name: ([], []) for name in SIGNALS}
        self.next_t = self.past_s
        self.scores = []
        self.rows = []
        self.latencies = []
        self.t_last = -1.0
        self._n_raised = 0
        self._n_cleared = 0

    # ------------------------------------------------------------------ input
    def push(self, t, channel, value):
        """Feed one sample; returns the decisions that became due (possibly none)."""
        if channel not in self.buf:
            raise KeyError(f"unknown channel {channel!r}; expected one of {SIGNALS}")
        t = bd._f(t, -1.0)
        ts, vs = self.buf[channel]
        ts.append(t)
        vs.append(bd._f(value, bd.NAN))
        self.t_last = max(self.t_last, t)
        return self.advance(t)

    def advance(self, now=None):
        """Emit every decision time that `now` has reached."""
        if now is None:
            now = self.t_last
        out = []
        while self.next_t <= now:
            out.append(self._decide(self.next_t))
            self.next_t += self.stride
        self._prune()
        return out

    # ------------------------------------------------------------------ internals
    def _decide(self, t0):
        started = time.perf_counter()
        grid = np.arange(t0 - self.past_s, t0, 1.0)
        win = {}
        for name in SIGNALS:
            ts, vs = self.buf[name]
            if ts:
                # min_samples=1: a window-sized buffer may legitimately hold a single
                # sample of a sparse channel, which is exactly what the offline path
                # (holding the whole case) would use for those grid points.
                win[name] = bd.to_grid(np.asarray(ts, dtype="float64"),
                                       np.asarray(vs, dtype="float64"),
                                       grid, self.max_gap, min_samples=1)
            else:
                win[name] = np.full(grid.shape, bd.NAN)
        win["pp"] = win["sbp"] - win["dbp"]
        low = np.isfinite(win["map"]) & (win["map"] < bd.HYPOTENSION_MAP)
        row = bd.window_row(t0, win, low, KEYS, None, None)

        score = bd.NAN
        if self.booster is not None and self.names:
            x = np.array([[bd._f(row.get(n)) for n in self.names]], dtype="float64")
            try:
                score = bd._f(self.booster.predict(x)[0])
            except (ValueError, TypeError):
                score = bd.NAN
        self.scores.append((t0, score))
        self.rows.append({"t": t0, "score": score, **row})
        events = self._policy_events()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.latencies.append(elapsed_ms)
        return {"t": t0, "score": score, "features": row, "events": events,
                "latency_ms": elapsed_ms}

    def _policy_events(self):
        """Raise/clear events since the last call, from the one policy implementation."""
        t = np.array([s[0] for s in self.scores], dtype="float64")
        s = np.array([s[1] for s in self.scores], dtype="float64")
        runs = alarm_policy.alarm_runs(t, s, self.threshold, self.clear, self.persist)
        events = []
        while self._n_raised < len(runs):
            events.append("raise")
            self._n_raised += 1
        closed = sum(1 for _raise, clear in runs if np.isfinite(clear))
        while self._n_cleared < closed:
            events.append("clear")
            self._n_cleared += 1
        return events

    def _prune(self):
        """Drop samples the next decision can no longer read (bounded memory)."""
        keep_from = self.next_t - self.past_s - self.max_gap
        for ts, vs in self.buf.values():
            if ts and ts[0] < keep_from:
                drop = 0
                while drop < len(ts) and ts[drop] < keep_from:
                    drop += 1
                del ts[:drop]
                del vs[:drop]

    def stats(self):
        lat = np.array(self.latencies, dtype="float64")
        return {
            "decisions": len(self.scores),
            "latency_median_ms": bd._f(np.median(lat)) if lat.size else bd.NAN,
            "latency_p95_ms": bd._f(np.percentile(lat, 95)) if lat.size else bd.NAN,
            "buffer_samples": sum(len(ts) for ts, _ in self.buf.values()),
        }


# ---------------------------------------------------------------------- replay
def replay(caseid, booster_path, tracks_dir, threshold=0.5, persist=1,
           clear_factor=1.0, limit_s=0):
    """Replay one stored case through the monitor, sample by sample, in time order."""
    loaded = bd.load_case_signals(caseid, tracks_dir)
    if loaded[0] is None:
        raise SystemExit(f"case {caseid} unreadable: {loaded[1]}")
    _grid, _sig, duration = loaded
    tracks = {name: bd.read_track(caseid, tname, tracks_dir)
              for name, tname in (("map", "Solar8000/ART_MBP"),
                                  ("sbp", "Solar8000/ART_SBP"),
                                  ("dbp", "Solar8000/ART_DBP"),
                                  ("hr", "Solar8000/HR"),
                                  ("spo2", "Solar8000/PLETH_SPO2"),
                                  ("etco2", "Solar8000/ETCO2"))}

    booster = None
    if booster_path:
        if lgb is None:
            raise SystemExit("lightgbm is required to score a replay")
        booster = lgb.Booster(model_file=booster_path)

    stream = MonitorStream(booster=booster, threshold=threshold, persist_windows=persist,
                           clear_factor=clear_factor)
    # Samples must be replayed in TIME order across channels: pushing one channel's
    # whole series first would let early decisions see only that channel, which is
    # not what a monitor does (and would make the comparison meaningless).
    samples = []
    for name, got in tracks.items():
        if got is None:
            continue
        t, v = got
        for ti, vi in zip(t, v, strict=True):
            if not (np.isfinite(ti) and np.isfinite(vi)):
                continue
            if limit_s and ti > limit_s:
                continue
            samples.append((bd._f(ti, -1.0), name, bd._f(vi, bd.NAN)))
    samples.sort(key=lambda item: item[0])
    events = []
    for ti, name, vi in samples:
        events.extend((ev, name) for ev in stream.push(ti, name, vi))
    # decisions whose window closed after the last sample
    for ev in stream.advance(duration):
        events.append((ev, ""))
    return stream, events


def check(caseid, windows, scores_dir, booster_path, tracks_dir, tol=1e-5):
    """Do the streaming features and scores match the offline run for this case?"""
    stream, _events = replay(caseid, booster_path, tracks_dir)
    emitted = pd.DataFrame(stream.rows)

    offline = pd.read_parquet(windows)
    offline = offline.loc[offline["caseid"] == caseid]
    offline = offline.sort_values("t").reset_index(drop=True)
    if offline.empty:
        raise SystemExit(f"case {caseid} not in {windows}")

    merged = emitted.merge(offline, on="t", how="outer", suffixes=("_on", "_off"),
                           indicator=True)
    only_on = bd._i((merged["_merge"] == "left_only").sum())
    only_off = bd._i((merged["_merge"] == "right_only").sum())
    # The offline dataset drops decision times whose 5 min label window would run past
    # the end of the record (range(PAST_S, n - horizon, stride)); a live monitor has no
    # such notion, so it legitimately emits those trailing windows. Anything else is a
    # real disagreement.
    expected_tail = bd.HORIZON_S // bd.STRIDE_S
    if only_off or only_on > expected_tail:
        print(f"  DECISION MISMATCH: only-stream={only_on} (expected <= {expected_tail}), "
              f"only-offline={only_off} (expected 0)")
        return 1

    feats = [c for c in offline.columns if c not in ("caseid", "t", "label")]
    # Compare ONLY the decisions both sides have: the trailing only-stream windows have
    # no offline counterpart at all, so their _off side is NaN by construction.
    shared = emitted.merge(offline, on="t", how="inner", suffixes=("_on", "_off"))
    bad = []
    for col in feats:
        a = shared[f"{col}_on"].to_numpy(dtype="float64")
        b = shared[f"{col}_off"].to_numpy(dtype="float64")
        same_nan = np.isnan(a) == np.isnan(b)
        close = np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
        if not (same_nan & close).all():
            n_bad = bd._i((~(same_nan & close)).sum())
            first = bd._i(np.flatnonzero(~(same_nan & close))[0])
            bad.append((col, f"{n_bad} mismatched, first t={shared['t'].iloc[first]:.0f}"))
    print(f"case {caseid}: streamed decisions={len(emitted)} offline windows={len(offline)} "
          f"shared={len(shared)} (only-stream={only_on} = trailing windows the offline "
          f"dataset drops by construction, only-offline={only_off})")
    if bad:
        for col, why in bad[:10]:
            print(f"  FEATURE MISMATCH {col}: {why}")
        return 1
    print(f"  OK all {len(feats)} feature columns identical between streamed and offline")

    if scores_dir and booster_path:
        path = os.path.join(scores_dir, "scores_numeric.parquet")
        if os.path.isfile(path):
            sc = pd.read_parquet(path)
            sc = sc.loc[sc["caseid"] == caseid, ["t", "p_main"]]
            if sc.empty:
                # The dump holds the test split only; a case outside it has no offline
                # score to compare against, which is not a disagreement.
                print("  score check: skipped (case is not in the scored split)")
            else:
                cmp = emitted.merge(sc, on="t", how="inner")
                d = np.abs(cmp["score"].to_numpy() - cmp["p_main"].to_numpy())
                worst = bd._f(np.nanmax(d)) if d.size else bd.NAN
                print(f"  score check on {len(cmp)} windows: max |diff| = {worst:.3e} "
                      f"(tolerance {tol:g}) -> {'OK' if worst <= tol else 'MISMATCH'}")
                if not (worst <= tol):
                    return 1
    st = stream.stats()
    print(f"  streaming stats: {st['decisions']} decisions, median latency "
          f"{st['latency_median_ms']:.2f} ms, p95 {st['latency_p95_ms']:.2f} ms, "
          f"buffer {st['buffer_samples']} samples")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify streamed == offline")
    ap.add_argument("--caseid", type=int, default=0)
    ap.add_argument("--windows", default=r"D:\vitaldb_local\processed\windows.parquet")
    ap.add_argument("--scores", default=r"D:\vitaldb_local\scores_h300")
    ap.add_argument("--booster", default=r"D:\vitaldb_local\models_stream\lgbm_main.txt")
    ap.add_argument("--tracks-dir", default=r"D:\vitaldb_local\numeric")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--clear-factor", type=float, default=0.8)
    args = ap.parse_args(argv)

    if not args.check:
        print(__doc__)
        return 0
    caseid = args.caseid
    if not caseid:
        w = pd.read_parquet(args.windows, columns=["caseid", "t"])
        caseid = bd._i(w.caseid.iloc[0])
        # Prefer a case that has dumped scores, so the score comparison actually runs.
        path = os.path.join(args.scores, "scores_numeric.parquet")
        if os.path.isfile(path):
            dumped = pd.read_parquet(path, columns=["caseid"])
            ids = sorted(bd._i(c) for c in dumped.caseid.unique())
            if ids:
                caseid = ids[0]
        print(f"(no --caseid given; picked {caseid})")
    return check(caseid, args.windows, args.scores, args.booster, args.tracks_dir)


if __name__ == "__main__":
    sys.exit(main())
