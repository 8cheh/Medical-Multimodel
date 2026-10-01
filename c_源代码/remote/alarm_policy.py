"""Alarm layer: turn a per-window risk score into monitor alarms.

p0_eval measures alarm *burden* by counting rising edges of score >= threshold. That
is the right metric but it is not a policy: a real monitor additionally needs

  * persistence - the condition must hold for N consecutive decision windows before
    the alarm fires, so one noisy window cannot wake the ward; and
  * hysteresis - clearing happens at a lower score than onset, so a score hovering
    on the threshold does not produce an alarm burst.

Both cost something (delay to the first alarm) and buy something (fewer alarms and
fewer false alarms). This module implements the policy and quantifies that trade, so
the report can state the price of each setting instead of assuming one.

    python alarm_policy.py            # self-check on synthetic sequences
"""
import sys

import numpy as np

# np.nan is already a float, so this needs no call at import time.
NAN = np.nan
STRIDE_S = 30.0


def as_float(value, default=NAN):
    """float() that never raises (times and scores arrive as numpy scalars)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def alarm_runs(t, score, threshold, clear_threshold=None, persist_windows=1):
    """Alarm intervals as (raise_time, clear_time) for one case.

    `t` are decision times in seconds (30 s grid), `score` the risk per window.
    A run starts once `persist_windows` consecutive windows are >= threshold and
    ends at the first window below `clear_threshold` (default: the same threshold,
    i.e. no hysteresis). clear_time is NaN while a run is still open at the end.
    """
    t = np.asarray(t, dtype="float64")
    s = np.asarray(score, dtype="float64")
    if clear_threshold is None:
        clear_threshold = threshold
    n = min(t.size, s.size)
    runs = []
    armed = 0
    open_at = NAN
    for i in range(n):
        if not np.isfinite(s[i]):
            armed = 0
            if np.isfinite(open_at):
                runs.append((open_at, t[i]))
                open_at = NAN
            continue
        if not np.isfinite(open_at):
            if s[i] >= threshold:
                armed += 1
                if armed >= max(1, persist_windows):
                    open_at = t[i - armed + 1]
                    armed = 0
            else:
                armed = 0
        elif s[i] < clear_threshold:
            runs.append((open_at, t[i]))
            open_at = NAN
            armed = 0
    if np.isfinite(open_at):
        runs.append((open_at, NAN))
    return runs


def alarm_burden(cases, name, threshold, pre=300.0, persist_windows=1,
                 clear_factor=1.0, onset_key="onset"):
    """Alarm burden under one policy, scored against real episode onsets.

    A raise is a true alarm when an onset follows it within `pre` seconds; otherwise
    it is a false alarm - the same convention p0_eval uses, so the two are comparable.
    """
    n_alarms = n_true = 0
    hours = 0.0
    delays = []
    leads = []
    for c in cases:
        t = np.asarray(c["t"], dtype="float64")
        if t.size == 0:
            continue
        hours += t.size * STRIDE_S / 3600.0
        score = c["p"][name]
        clear = threshold * clear_factor
        # The policy is always applied; with clear_factor=1.0 and persist=1 this is
        # identical to the raw detector, so the two settings stay independent. (An
        # earlier version branched on persist_windows==1 and silently dropped the
        # hysteresis factor, which made hysteresis look like it did nothing.)
        raw = alarm_runs(t, score, threshold, None, 1)
        runs = alarm_runs(t, score, threshold, clear, persist_windows)
        onsets = np.asarray(c[onset_key], dtype="float64")
        for raise_t, _clear_t in runs:
            n_alarms += 1
            window = (onsets > raise_t) & (onsets <= raise_t + pre)
            if onsets.size and window.any():
                n_true += 1
                leads.append(as_float(onsets[window].min() - raise_t))
        if raw and runs and (persist_windows > 1 or clear_factor != 1.0):
            # price of the policy: how much later the first alarm now fires
            delays.append(as_float(runs[0][0] - raw[0][0]))
    finite_delays = np.array([d for d in delays if np.isfinite(d)])
    lead_arr = np.array(leads, dtype="float64")
    return {
        "n_alarms": n_alarms,
        "n_true": n_true,
        "alarm_rate_per_h": n_alarms / hours if hours else NAN,
        "false_alarm_rate_per_h": (n_alarms - n_true) / hours if hours else NAN,
        "lead_median_s": as_float(np.median(lead_arr)) if lead_arr.size else NAN,
        "first_alarm_delay_median_s": (as_float(np.median(finite_delays))
                                       if finite_delays.size else NAN),
    }


def compare_policies(cases, name, thresholds, pre=300.0):
    """Burden of the raw detector vs persistence/hysteresis variants."""
    out = []
    for spec in thresholds:
        out.append({"policy": spec, **alarm_burden(
            cases, name, spec["threshold"], pre=pre,
            persist_windows=spec.get("persist", 1),
            clear_factor=spec.get("clear_factor", 1.0))})
    return out


# --------------------------------------------------------------------------- #
def _self_check():
    t = np.arange(0, 3600, 30.0)

    # one isolated spike: must not alarm once persistence is required
    s = np.full(t.size, 0.1)
    s[40] = 0.9
    cases = [{"t": t, "p": {"m": s}, "onset": np.array([t[40] + 120.0])}]
    raw = alarm_burden(cases, "m", 0.5, persist_windows=1)
    per2 = alarm_burden(cases, "m", 0.5, persist_windows=2)
    assert raw["n_alarms"] == 1, f"raw={raw}"
    assert per2["n_alarms"] == 0, f"persist=2 should suppress the spike: {per2}"
    print(f"  OK isolated spike: raw alarms={raw['n_alarms']}, "
          f"persist=2 alarms={per2['n_alarms']}")

    # sustained high: exactly one alarm, not one per window. The onset sits inside
    # the 5 min after the raise, so this alarm is a true one.
    cases2 = [{"t": t, "p": {"m": np.where((t >= 600) & (t < 1200), 0.9, 0.1)},
               "onset": np.array([750.0])}]
    run = alarm_burden(cases2, "m", 0.5, persist_windows=1)
    assert run["n_alarms"] == 1, f"sustained high should be one alarm: {run}"
    assert run["n_true"] == 1, f"the alarm precedes an onset: {run}"
    assert run["lead_median_s"] == 150.0, f"lead should be onset-raise: {run}"
    print(f"  OK sustained 600 s of high score -> {run['n_alarms']} alarm, "
          f"lead={run['lead_median_s']:.0f}s")

    # an onset beyond the horizon must NOT count as a true alarm
    far = alarm_burden([{"t": t, "p": {"m": np.where((t >= 600) & (t < 1200), 0.9, 0.1)},
                         "onset": np.array([1500.0])}], "m", 0.5, persist_windows=1)
    assert far["n_true"] == 0 and far["false_alarm_rate_per_h"] > 0, far
    print(f"  OK onset beyond the horizon -> false alarm "
          f"({far['false_alarm_rate_per_h']:.1f}/h)")

    # Chatter that crosses the alarm threshold but stays ABOVE the clear level:
    # that is the situation hysteresis exists for. The precondition is asserted,
    # because both earlier versions of this test silently broke it (a signal that
    # never dips below the threshold, then one that dips below the clear level -
    # in both cases hysteresis correctly had nothing to suppress).
    clear_level = 0.5 * 0.8
    chat = 0.5 + 0.08 * np.sin(np.arange(t.size) * 0.7)
    assert chat.min() < 0.5, "precondition: must cross the alarm threshold"
    assert chat.min() > clear_level, "precondition: must stay above the clear level"
    cases3 = [{"t": t, "p": {"m": chat}, "onset": np.array([], dtype=float)}]
    no_h = alarm_burden(cases3, "m", 0.5, persist_windows=1, clear_factor=1.0)
    with_h = alarm_burden(cases3, "m", 0.5, persist_windows=1, clear_factor=0.8)
    assert no_h["n_alarms"] > 1, f"precondition: should chatter, got {no_h['n_alarms']}"
    assert with_h["n_alarms"] < no_h["n_alarms"], \
        f"hysteresis did not help: {no_h['n_alarms']} vs {with_h['n_alarms']}"
    print(f"  OK chatter at the threshold: alarms={no_h['n_alarms']} -> "
          f"{with_h['n_alarms']} with 20% hysteresis")
    print("alarm policy self-check passed")


if __name__ == "__main__":
    sys.exit(_self_check() or 0)
