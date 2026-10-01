"""Self-check for the non-trivial logic in p0_eval.py. No dataset required.

Three things were easy to get wrong, so each gets an assertion:
  * threshold_at_sensitivity must return the MOST specific threshold that still
    reaches the target (a first version returned the least specific one, which
    would have inflated every baseline's alarm burden).
  * event_metrics on a hand-built case whose answer is known by inspection.
  * mcnemar / safe_metrics edge cases: no disagreements, single-class draw.
"""
import numpy as np
import p0_eval as p0


def check_threshold_is_most_specific():
    pos, neg = 20, 80
    y = np.array([1] * pos + [0] * neg)
    p = np.concatenate([np.linspace(0.9, 0.5, pos), np.linspace(0.49, 0.1, neg)])
    for target in (0.5, 0.8, 0.9, 1.0):
        thr = p0.threshold_at_sensitivity(y, p, target)
        order = np.argsort(-p)
        ys = y[order]
        rank = np.flatnonzero(p[order] >= thr)[-1]
        sens = ys[: rank + 1].sum() / pos
        assert sens >= target - 1e-12, f"target={target} thr={thr} sens={sens}"
        # one step higher must miss the target, else a higher threshold existed
        if rank > 0:
            prev = ys[:rank].sum() / pos
            assert prev < target - 1e-12, f"target={target} not the most specific: {thr}"
    # unreachable target falls back to alarming on everything
    assert p0.threshold_at_sensitivity(y, p, 1.5) == p[-1]
    print("  threshold_at_sensitivity: most-specific property holds")


def case(t, pred, onset, name="v"):
    return {"caseid": 1, "t": np.asarray(t, dtype=float),
            "y": np.ones(len(t), dtype=np.int8),
            "p": {name: np.asarray(pred, dtype=float)},
            "onset": np.asarray(onset, dtype=float)}


def check_event_metrics():
    t = np.arange(300.0, 601.0, 30.0)          # 11 decision times

    e = p0.event_metrics([case(t, np.ones_like(t), [600.0])], "v", 0.5)
    assert e["n_episodes"] == 1 and e["n_scoreable"] == 1 and e["detected"] == 1, e
    assert e["detection_rate"] == 1.0, e
    assert e["lead_median_s"] == 300.0, e      # first alarm window is t=300
    assert e["alarm_events"] == 1, e           # rising edge only, not one per window
    assert e["false_alarm_rate_per_h"] == 0.0, e
    assert abs(e["alarm_rate_per_h"] - 1 / (11 * 30 / 3600)) < 1e-9, e

    e = p0.event_metrics([case(t, np.zeros_like(t), [600.0])], "v", 0.5)
    assert e["detected"] == 0 and e["detection_rate"] == 0.0, e
    assert np.isnan(e["lead_median_s"]) and e["alarm_rate_per_h"] == 0.0, e

    # an alarm with no onset inside the next 5 min is a false alarm
    t2 = np.arange(300.0, 1201.0, 30.0)
    far = np.zeros_like(t2)
    far[t2 >= 900] = 1.0                       # alarm at 900, onset at 2400
    e = p0.event_metrics([case(t2, far, [2400.0])], "v", 0.5)
    assert e["alarm_events"] == 1 and e["false_alarm_rate_per_h"] > 0, e
    assert e["detected"] == 0, e               # onset is 25 min after the alarm

    # an episode that starts before the first decision time is not scoreable
    e = p0.event_metrics([case(t, np.ones_like(t), [250.0])], "v", 0.5)
    assert e["n_episodes"] == 1 and e["n_scoreable"] == 0, e
    assert np.isnan(e["detection_rate"]), e
    print("  event_metrics: detection, lead time, alarm burden verified")


def check_mcnemar_and_metrics():
    # identical predictors: nothing to test
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    same = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9, 0.1])
    m = p0.mcnemar(y, same, same, 0.5, 0.5)
    assert m["a_right_b_wrong"] == 0 and m["a_wrong_b_right"] == 0 and m["p"] == 1.0, m

    # a perfect predictor vs one that alarms on nothing:
    #   a is right on all 9 positives, b is wrong on all 9 -> (b=9, c=0)
    #   exact two-sided p = 2 * 0.5**9 = 0.0039
    y9 = np.array([1] * 9 + [0] * 9)
    perfect = np.array([1.0] * 9 + [0.0] * 9)
    silent = np.zeros(18)
    m = p0.mcnemar(y9, perfect, silent, 0.5, 0.5)
    assert m["a_right_b_wrong"] == 9 and m["a_wrong_b_right"] == 0, m
    assert abs(m["p"] - 0.00390625) < 1e-9, m
    flipped = p0.mcnemar(y9, silent, perfect, 0.5, 0.5)   # direction must flip
    assert flipped["a_right_b_wrong"] == 0 and flipped["a_wrong_b_right"] == 9, flipped
    assert flipped["p"] == m["p"], f"p must not depend on argument order: {flipped} vs {m}"

    # a single-class bootstrap draw must not raise out of roc_auc_score
    blank = p0.safe_metrics(np.zeros(10, dtype=np.int8), np.linspace(0, 1, 10), 0.5)
    assert np.isnan(blank["auroc"]) and blank["n"] == 10, blank
    print("  mcnemar / safe_metrics: edge cases verified")


if __name__ == "__main__":
    check_threshold_is_most_specific()
    check_event_metrics()
    check_mcnemar_and_metrics()
    print("OK p0_eval self-check")
