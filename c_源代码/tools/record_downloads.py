"""Record what the functional-channel download actually produced.

The architecture figure and the report both cite "41 channels / N files / N GB", so
that number needs an artifact of its own rather than a memory of a log line. Counts
the files that are on disk for the selected channels and folds in the downloader's own
totals.

    python tools/record_downloads.py
"""
import json
import os
import sys

NUM = r"D:\vitaldb_local\numeric"
TRACKS = r"D:\vitaldb_local\func_tracks.txt"
LOG = r"D:\vitaldb_local\logs\func_dl.log"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "func_download.json")


def safe_name(tname):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tname)


def to_int(node, default=0):
    """The downloader's log is an external artifact; a malformed field must not crash."""
    try:
        return int(node)
    except (TypeError, ValueError):
        return default


def to_float(node, default=0.0):
    try:
        return float(node)
    except (TypeError, ValueError):
        return default


def main():
    try:
        with open(TRACKS, encoding="utf-8") as fh:
            channels = [t for t in fh.read().split(",") if t.strip()]
    except OSError as exc:
        raise SystemExit(f"cannot read {TRACKS}: {exc}") from exc

    try:
        listing = os.listdir(NUM)
    except OSError as exc:
        raise SystemExit(f"cannot list {NUM}: {exc}") from exc
    present = set(listing)

    per_channel, files, total = {}, 0, 0
    for tname in channels:
        suffix = "_" + safe_name(tname) + ".csv"
        hits = [n for n in present if n.endswith(suffix)]
        size = 0
        for n in hits:
            try:
                size += os.path.getsize(os.path.join(NUM, n))
            except OSError:
                continue
        per_channel[tname] = {"files": len(hits), "bytes": size}
        files += len(hits)
        total += size

    ok = fail = None
    mb = None
    # PowerShell's `*>` redirection writes UTF-16LE, so the byte order mark decides how
    # to decode: reading it as UTF-8 silently yields text that matches nothing.
    raw = b""
    try:
        with open(LOG, "rb") as fh:
            raw = fh.read()
    except OSError:
        pass
    text = raw.decode("utf-16", errors="replace") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") \
        else raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("DONE"):
            parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
            ok = to_int(parts.get("ok"), 0)
            fail = to_int(parts.get("fail"), 0)
            mb = to_float(parts.get("MB"), 0.0)

    payload = {
        "channels_requested": len(channels),
        "channels": channels,
        "per_channel": per_channel,
        "files_on_disk": files,
        "bytes_on_disk": total,
        "gb_on_disk": round(total / 1024 ** 3, 2),
        "downloader_ok": ok,
        "downloader_fail": fail,
        "downloader_mb": mb,
        "source": {"dir": NUM, "tracks": TRACKS, "log": LOG},
    }
    try:
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise SystemExit(f"cannot write {OUT}: {exc}") from exc

    print(f"channels={len(channels)} files={files} size={payload['gb_on_disk']} GB "
          f"downloader ok={ok} fail={fail}")
    for tname, info in sorted(per_channel.items(),
                              key=lambda kv: -kv[1]["files"])[:6]:
        print(f"  {tname:34s} {info['files']:5d} files "
              f"{info['bytes'] / 1024 ** 2:8.1f} MB")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
