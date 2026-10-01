"""Test the two --extra-features merge modes on tiny synthetic frames.

The PPG comparison depends on `inner` keeping exactly the intersection, while the
same-grid ablations depend on `strict` refusing a misaligned build. Both are cheap
to pin down here, so a later refactor cannot silently weaken the guard rail.
"""
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "remote"))
import p0_eval as p0  # noqa: E402

TMP = tempfile.mkdtemp(prefix="extra_merge_")


def frame(cases, times, prefix, n=3):
    rows = []
    for c in cases:
        for t in times:
            rows.append({"caseid": c, "t": t,
                         **{f"{prefix}_f{i}": c * 10 + t + i for i in range(n)}})
    return pd.DataFrame(rows)


def expect_exit(fn, why):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(f"expected a refusal: {why}")


def main():
    base = frame([1, 2, 3], [300, 330], "map")
    same = frame([1, 2, 3], [300, 330], "l", n=2)
    subset = frame([1, 2], [300, 330], "ppg", n=2)
    disjoint = frame([9], [300, 330], "ppg", n=2)
    paths = {}
    for name, df in (("same", same), ("subset", subset), ("disjoint", disjoint)):
        paths[name] = os.path.join(TMP, f"{name}.parquet")
        df.to_parquet(paths[name], index=False)

    out = p0.add_extra_features(base, "base", [paths["same"]], "strict")
    assert len(out) == len(base) and "l_f0" in out.columns, "strict same-grid merge"
    print("OK strict: same window set merges and preserves every row")

    expect_exit(lambda: p0.add_extra_features(base, "base", [paths["subset"]], "strict"),
                "strict must refuse a build covering fewer windows")
    print("OK strict: a partial window set is refused (no silent left-join NaNs)")

    out = p0.add_extra_features(base, "base", [paths["subset"]], "inner")
    assert sorted(out.caseid.unique()) == [1, 2], out.caseid.unique()
    assert len(out) == 4, len(out)
    assert out["ppg_f0"].notna().all(), "inner join must actually carry values"
    print("OK inner: keeps exactly the intersection (2/3 cases, 4/6 rows)")

    expect_exit(lambda: p0.add_extra_features(base, "base", [paths["disjoint"]], "inner"),
                "inner must refuse a build with no usable rows")
    print("OK inner: a disjoint build is refused instead of returning empty features")
    print("all merge-mode checks passed")


if __name__ == "__main__":
    main()
