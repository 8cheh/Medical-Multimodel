"""Download a VitalDB waveform track, compacted to binary on arrival.

Written for SNUADC/PLETH (PPG) but the endpoint is the same for every waveform, so
--track selects any of them (SNUADC/ART for the arterial line, SNUADC/ECG_II, ...).
The per-track CSV endpoint serves the waveform at its native ~500 Hz as text (~48 MB
per case), so this downloads each case, converts the amplitude column to int16 (or
float32 when the samples are not integral), writes <caseid>.npy plus a tiny
<caseid>.json sidecar, and deletes the CSV immediately. Peak disk stays near one
file per worker.

Usage:
    python dl_ppg.py --out /root/autodl-tmp/ppg_npy --workers 10
    python dl_ppg.py --track SNUADC/ART --out <dir> --caseids 1,2,3
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

API = "https://api.vitaldb.net"
TRKS_PARQUET = "/root/autodl-tmp/cache/trks.parquet"
LEGACY_CSV_DIR = "/root/autodl-tmp/ppg"
NAN_SENTINEL = -32768

_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0, "csv_bytes": 0}


def to_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# np.nan is a plain Python float, so using it as the default keeps to_float's
# contract identical while removing the call-in-default (and its noqa).
NAN = np.nan


def to_float(x, default=NAN):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return 0.0


def load_cohort(path=TRKS_PARQUET, track="SNUADC/PLETH"):
    """caseid -> tid for every case that has the requested waveform track."""
    df = pd.read_parquet(path)
    sub = df.loc[df.tname == track, ["caseid", "tid"]]
    return [(to_int(c), str(t)) for c, t in zip(sub.caseid, sub.tid, strict=True)]


def csv_to_npy(csv_path, npy_path, meta_path, track="SNUADC/PLETH"):
    """Convert one downloaded CSV into a compact array + sidecar. Returns bytes."""
    frame = pd.read_csv(csv_path, header=0, dtype="float64")
    raw = np.asarray(frame.iloc[:, -1], dtype="float64")
    n = to_int(raw.size)
    if n == 0:
        raise ValueError("empty track")

    finite = np.isfinite(raw)
    vals = raw[finite]
    integral = bool(vals.size > 0 and np.all(vals == np.round(vals)))
    lo = to_float(vals.min()) if vals.size else NAN
    hi = to_float(vals.max()) if vals.size else NAN
    fits_i16 = integral and np.isfinite(lo) and lo > NAN_SENTINEL and hi < 32767

    if fits_i16:
        arr = np.full(n, NAN_SENTINEL, dtype="int16")
        arr[finite] = np.round(vals).astype("int16")
        dtype = "int16"
    else:
        arr = raw.astype("float32")
        dtype = "float32"

    np.save(npy_path, arr)
    payload = {
        "caseid": to_int(os.path.basename(npy_path).split(".")[0]),
        "track": track,
        "n_samples": n,
        "dtype": dtype,
        "nan_sentinel": NAN_SENTINEL if dtype == "int16" else None,
        "valid_frac": round(to_float(finite.mean(), 0.0), 6),
        "unit": "raw ADC" if fits_i16 else "native",
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError as exc:
        raise ValueError(f"cannot write sidecar: {exc}") from exc
    return to_int(os.path.getsize(npy_path))


def one_case(sess, caseid, tid, outdir, tmpdir, keep_csv=False, track="SNUADC/PLETH"):
    npy = os.path.join(outdir, f"{caseid}.npy")
    meta = os.path.join(outdir, f"{caseid}.json")
    if os.path.isfile(npy) and os.path.isfile(meta):
        return "skip", 0, 0

    cached = os.path.join(LEGACY_CSV_DIR, f"{caseid}_SNUADC_PLETH.csv")
    tmp_csv = os.path.join(tmpdir, f"{caseid}.csv")
    src = cached if os.path.isfile(cached) else tmp_csv

    if not os.path.isfile(src):
        try:
            with sess.get(f"{API}/{tid}", stream=True, timeout=(20, 300)) as resp:
                if resp.status_code != 200:
                    return "fail", 0, 0
                with open(tmp_csv, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        except Exception:  # noqa: BLE001 - one failed request must not stop the run
            _unlink(tmp_csv)
            return "fail", 0, 0

    csv_bytes = to_int(os.path.getsize(src))
    try:
        npy_bytes = csv_to_npy(src, npy, meta, track)
    except Exception:  # noqa: BLE001 - a bad file must not kill the batch
        _unlink(npy)
        _unlink(meta)
        return "fail", 0, csv_bytes
    finally:
        if src == tmp_csv and not keep_csv:
            _unlink(tmp_csv)
        elif src == cached and not keep_csv:
            _unlink(cached)
    return "ok", npy_bytes, csv_bytes


def _unlink(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/ppg_npy")
    ap.add_argument("--tmp-dir", default="/root/autodl-tmp/tmp",
                    help="scratch dir for the intermediate CSV that is deleted on arrival")
    ap.add_argument("--index", default=TRKS_PARQUET,
                    help="track index parquet (same file dl_tracks.py uses)")
    ap.add_argument("--caseids", default="",
                    help="comma-separated caseids to fetch instead of the whole cohort")
    ap.add_argument("--track", default="SNUADC/PLETH",
                    help="waveform track name, e.g. SNUADC/PLETH (PPG) or SNUADC/ART (ABP)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--keep-csv", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for d in (args.out, args.tmp_dir):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as exc:
            raise SystemExit(f"cannot create {d}: {exc}") from exc

    cohort = load_cohort(args.index, args.track)
    if args.caseids:
        want = {to_int(x.strip()) for x in args.caseids.split(",") if x.strip()}
        cohort = [(c, t) for c, t in cohort if c in want]
        missing = want - {c for c, _ in cohort}
        if missing:
            print(f"WARNING: {len(missing)} requested caseid(s) have no PPG track: "
                  f"{sorted(missing)[:10]}", flush=True)
    if args.limit:
        cohort = cohort[: args.limit]
    print(f"track={args.track} cases={len(cohort)} workers={args.workers} out={args.out}",
          flush=True)

    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(pool_connections=args.workers * 2,
                                       pool_maxsize=args.workers * 2))
    t0 = time.time()
    done = 0
    for start in range(0, len(cohort), args.chunk):
        fg = free_gb(args.out)
        if fg < args.min_free_gb:
            print(f"STOP: free={fg:.1f}GB < {args.min_free_gb}GB", flush=True)
            break
        batch = cohort[start:start + args.chunk]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(one_case, sess, cid, tid, args.out,
                              args.tmp_dir, args.keep_csv, args.track)
                    for cid, tid in batch]
            for fut in as_completed(futs):
                status, npy_bytes, csv_bytes = fut.result()
                with _lock:
                    _stats[status] += 1
                    _stats["bytes"] += npy_bytes
                    _stats["csv_bytes"] += csv_bytes
                    done += 1
        el = max(time.time() - t0, 1e-9)
        print(f"  {done}/{len(cohort)} ok={_stats['ok']} skip={_stats['skip']} "
              f"fail={_stats['fail']} npy={_stats['bytes'] / 2**30:.2f}GB "
              f"csv_read={_stats['csv_bytes'] / 2**30:.1f}GB "
              f"free={free_gb(args.out):.0f}GB "
              f"rate={done / el * 60:.0f}/min", flush=True)

    print(f"DONE ok={_stats['ok']} skip={_stats['skip']} fail={_stats['fail']} "
          f"npy={_stats['bytes'] / 2**30:.2f}GB "
          f"compression={_stats['csv_bytes'] / max(_stats['bytes'], 1):.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
