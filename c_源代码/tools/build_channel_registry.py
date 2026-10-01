"""Device/channel registry: every device in the project, and what is actually local.

The pipeline used to be hardcoded to 8 numeric tracks plus two waveforms. This turns
that into a registry derived from the VitalDB track index, classified by the clinical
function the request names (circulation, respiration, pulmonary mechanics, depth,
neuromuscular block, infusion, fluids, cerebral oximetry), and cross-referenced with
what is actually on disk so the capability matrix cannot overstate itself.

Writes results/channel_registry.json and prints a readable summary.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

IDX = r"D:\vitaldb_local\cache\trks.parquet"
NUM = r"D:\vitaldb_local\numeric"
ABP = r"D:\vitaldb_local\abp_npy"
PPG_FEAT = r"D:\vitaldb_local\ppg_features_full\ppg_features.parquet"
OUT = r"D:\github\VitalDB\results\channel_registry.json"

# Which clinical function each device family serves, and the keyword families that
# identify the channels implementing it.
FUNCTIONS = {
    "circulation": ("Solar8000", "SNUADC", "EV1000", "Vigileo", "CardioQ", "FMS"),
    "respiration": ("Primus", "SNUADC"),
    "pulmonary_mechanics": ("Primus", "SNUADC"),
    "anesthesia_depth": ("BIS",),
    "neuromuscular_block": ("TOF", "NMT", "Orchestra"),
    "infusion": ("Orchestra",),
    "cerebral_oximetry": ("Invos",),
    "temperature": ("Solar8000", "FMS", "Primus"),
}

KEYWORDS = {
    "respiration": ("ETCO2", "CO2", "RR", "RESP", "SPONT", "TV", "MV", "FLOW", "AW_"),
    "pulmonary_mechanics": ("PEEP", "PEEP", "COMPL", "RESIST", "PPLAT", "PMAX", "VTE",
                            "VTI", "MV_", "FIO2", "O2_"),
    "neuromuscular_block": ("TOF", "NMT", "PTC", "TRAIN", "STIM", "TWITCH"),
    "anesthesia_depth": ("BIS", "SEF", "EMG", "SR", "EEG"),
    "infusion": ("RATE", "VOL", "DOSE", "DRUG"),
    "cerebral_oximetry": ("SCO2", "INVOS"),
    "temperature": ("TEMP", "BT", "T_"),
}

WAVE_HINT = re.compile(r"_WAV$|/WAV|_WAVE$")


def as_int(value, default=0):
    """int() that never raises (counts come from filenames and pandas sums)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def local_counts():
    """What is actually on disk, per channel, right now."""
    out = {}
    if os.path.isdir(NUM):
        for p in glob.glob(os.path.join(NUM, "*.csv")):
            m = re.match(r"(\d+)_(.+)\.csv$", os.path.basename(p))
            if m:
                out.setdefault(m.group(2).replace("_", "/", 1), set()).add(
                    as_int(m.group(1)))
    # the numeric filenames were written with '/' replaced by '_'
    fixed = defaultdict(set)
    for k, v in out.items():
        device, _, chan = k.partition("/")
        fixed[f"{device}/{chan}"].update(v)
    counts = {k: len(v) for k, v in fixed.items()}
    if os.path.isdir(ABP):
        n = len(glob.glob(os.path.join(ABP, "*.npy")))
        if n:
            counts["SNUADC/ART"] = max(counts.get("SNUADC/ART", 0), n)
    if os.path.isfile(PPG_FEAT):
        cases = pd.read_parquet(PPG_FEAT, columns=["caseid"]).caseid.nunique()
        counts["SNUADC/PLETH"] = max(counts.get("SNUADC/PLETH", 0), as_int(cases))
    return counts


def main():
    idx = pd.read_parquet(IDX)
    per_channel = (idx.groupby("tname")
                   .agg(cases=("caseid", "nunique"), tracks=("tid", "size"))
                   .reset_index())
    per_channel["device"] = per_channel.tname.str.split("/").str[0]
    per_channel["is_wave"] = per_channel.tname.str.contains(WAVE_HINT)
    local = local_counts()

    registry = {
        "source_index": IDX,
        "channels_total": as_int(len(per_channel)),
        "devices": {},
        "functions": {},
        "locally_available": {},
    }
    for dev, sub in per_channel.groupby("device"):
        registry["devices"][dev] = {
            "channels": as_int(len(sub)),
            "tracks": as_int(sub.tracks.sum()),
            "cases_max": as_int(sub.cases.max()),
            "waveforms": as_int(sub.is_wave.sum()),
        }
    for fn, devs in FUNCTIONS.items():
        hits = []
        for dev in devs:
            sub = per_channel[per_channel.device == dev]
            kws = KEYWORDS.get(fn, ())
            if kws:
                mask = sub.tname.str.upper().str.contains("|".join(kws), regex=True)
                sub = sub[mask]
            hits.extend(sub.tname.tolist())
        registry["functions"][fn] = sorted(set(hits))
    registry["locally_available"] = dict(sorted(local.items(), key=lambda kv: -kv[1]))

    try:
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise SystemExit(f"cannot write {OUT}: {exc}") from exc

    print(f"channels in index: {registry['channels_total']}")
    print("\n=== devices (channels / tracks / max cases) ===")
    for dev, info in sorted(registry["devices"].items(), key=lambda kv: -kv[1]["tracks"]):
        print(f"  {dev:12s} channels={info['channels']:4d} tracks={info['tracks']:7d} "
              f"waveforms={info['waveforms']:3d} cases<={info['cases_max']}")
    print("\n=== requested functions -> channels present in the index ===")
    for fn, chans in registry["functions"].items():
        print(f"  {fn:22s} {len(chans):3d} channels  e.g. {', '.join(chans[:4])}")
    print("\n=== locally available channel data ===")
    for chan, n in registry["locally_available"].items():
        print(f"  {chan:28s} {n:5d} cases")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
