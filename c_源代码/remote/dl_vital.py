"""Download the complete packed .vital file for every case (the authoritative,
all-tracks binary form of the open dataset).

Size  : ~13.4 MB/case average -> ~86 GB for all 6388 cases
Disk  : refuses to continue below --min-free-gb so the data disk cannot fill up

Usage:
    python dl_vital.py --out /root/autodl-tmp/raw --workers 24 --min-free-gb 40
"""
import argparse
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter

API = "https://api.vitaldb.net"
VERSION = "1.0.1"

_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0}
_stop = threading.Event()


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return 0.0


def fetch(sess, cid, outdir):
    """Download one case. Returns (status, bytes)."""
    path = os.path.join(outdir, f"{cid}.vital")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return "skip", os.path.getsize(path)
    tmp = path + ".part"
    url = f"{API}/{VERSION}/{cid}.vital"
    try:
        with sess.get(url, stream=True, timeout=(20, 180)) as resp:
            if resp.status_code != 200:
                return "fail", 0
            n = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    n += len(chunk)
        if n == 0:
            return "fail", 0
        os.replace(tmp, path)
        return "ok", n
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return "fail", 0


def parse_caseids(text, hi):
    if not text:
        return list(range(1, hi + 1))
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise SystemExit(f"bad caseid {tok!r}") from exc
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/raw")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-caseid", type=int, default=6388)
    ap.add_argument("--caseids", default="")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--chunk", type=int, default=400)
    args = ap.parse_args(argv)

    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create {args.out}: {exc}") from exc

    caseids = parse_caseids(args.caseids, args.max_caseid)
    print(f"cases={len(caseids)} workers={args.workers} out={args.out}", flush=True)

    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=args.workers * 2, pool_maxsize=args.workers * 2)
    sess.mount("https://", adapter)

    done = 0
    total = len(caseids)
    for start in range(0, total, args.chunk):
        if _stop.is_set():
            print("STOPPED early (disk guard)", flush=True)
            break
        fg = free_gb(args.out)
        if fg < args.min_free_gb:
            print(f"STOP: free={fg:.1f}GB < {args.min_free_gb}GB", flush=True)
            break
        batch = caseids[start:start + args.chunk]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch, sess, cid, args.out) for cid in batch]
            for fut in as_completed(futs):
                status, n = fut.result()
                with _lock:
                    _stats[status] += 1
                    _stats["bytes"] += n
                    done += 1
        print(f"  {done}/{total} ok={_stats['ok']} skip={_stats['skip']} fail={_stats['fail']} "
              f"GB={_stats['bytes'] / (1024 ** 3):.1f} free={free_gb(args.out):.1f}GB", flush=True)

    print(f"DONE ok={_stats['ok']} skip={_stats['skip']} fail={_stats['fail']} "
          f"GB={_stats['bytes'] / (1024 ** 3):.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
