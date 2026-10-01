"""Download selected VitalDB tracks as per-track CSV.

The packed .vital file (17.5MB/case) contains every track, but the deprecated
per-track CSV endpoint serves a single track (14KB for a numeric track). Since
we only need a handful of numeric tracks, this cuts the download by ~100x.

Usage:
    python dl_tracks.py --workers 32 --out /root/autodl-tmp/numeric \
        --tracks Solar8000/ART_MBP,Solar8000/ART_SBP,Solar8000/ART_DBP,Solar8000/HR
"""
import argparse
import gzip
import io
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

API = "https://api.vitaldb.net"
TRKS = "/root/autodl-tmp/cache/trks.csv.gz"
TRKS_PARQUET = "/root/autodl-tmp/cache/trks.parquet"
# `--index` overrides both; a `.parquet` next to whatever index is given is used as
# the fast path, so the same downloader serves the AutoDL box and a local machine.

_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0}


def load_index(path=TRKS, path_parquet=TRKS_PARQUET):
    """Read the track index (track name -> tid per caseid).

    Prefers the baked parquet copy: parsing the 486k-row gzipped CSV takes ~39 s
    and every job pays it.
    """
    if os.path.isfile(path_parquet):
        try:
            return pd.read_parquet(path_parquet)
        except (OSError, ValueError):
            pass
    if not os.path.isfile(path):
        raise SystemExit(f"track index not found: {path}")
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def safe_name(tname):
    """Track name -> filesystem-safe component (no separators, no traversal)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tname)


def to_int(x, default=0):
    """int() that never raises (groupby keys are typed as Hashable)."""
    try:
        return int(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def fetch(sess, tid, path):
    try:
        r = sess.get(f"{API}/{tid}", headers={"Accept-Encoding": "identity"}, timeout=90)
        if r.status_code != 200 or not r.content:
            return "fail", 0
        body = r.content
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, path)
        return "ok", len(body)
    except Exception:
        return "fail", 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--index", default=TRKS)
    ap.add_argument("--index-parquet", default="")
    ap.add_argument("--cohort-track", default="Solar8000/ART_MBP")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--caseids", default="")
    return ap.parse_args(argv)


def build_cohort(idx, args):
    if not args.caseids:
        return sorted(idx.loc[idx.tname == args.cohort_track, "caseid"].unique().tolist())
    out = []
    for chunk in args.caseids.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError as exc:
            raise SystemExit(f"bad caseid {chunk!r}") from exc
    return out


def main(argv=None):
    args = parse_args(argv)
    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc

    idx = load_index(args.index, args.index_parquet or args.index + ".parquet")
    cohort = build_cohort(idx, args)
    if args.limit:
        cohort = cohort[: args.limit]

    # Group once: scanning the 486k-row index per case is O(cases x rows) and
    # takes minutes on the full cohort.
    by_case = {}
    for cid, sub in idx.groupby("caseid", sort=False):
        by_case[to_int(cid)] = dict(zip(sub.tname.tolist(), sub.tid.tolist(), strict=False))

    jobs = []
    for cid in cohort:
        have = by_case.get(cid)
        if not have:
            continue
        for t in tracks:
            if t not in have:
                continue
            path = os.path.join(args.out, f"{cid}_{safe_name(t)}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with _lock:
                    _stats["skip"] += 1
                continue
            jobs.append((have[t], path))

    print(f"cohort={len(cohort)} tracks={len(tracks)} jobs={len(jobs)} "
          f"skip={_stats['skip']}", flush=True)

    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    sess.mount("https://", adapter)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, sess, tid, path): path for tid, path in jobs}
        for fut in as_completed(futs):
            status, n = fut.result()
            with _lock:
                _stats[status] += 1
                _stats["bytes"] += n
                done += 1
                if done % 200 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} ok={_stats['ok']} fail={_stats['fail']} "
                          f"MB={_stats['bytes'] / 1048576:.1f}", flush=True)

    print(f"DONE ok={_stats['ok']} fail={_stats['fail']} skip={_stats['skip']} "
          f"MB={_stats['bytes'] / 1048576:.1f}", flush=True)
    return 0 if _stats["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
