"""Pick the functional channels worth downloading, from the registry index.

The project can currently see circulation only. These are the channels that carry the
functions it cannot see, chosen by name from the index and restricted to the existing
2,500-case cohort (numeric + PPG + ABP) so every new endpoint is directly comparable
with the work already done:

  respiration / ventilation : Primus CO2/ETCO2/INCO2/RR_CO2/FIO2/FEO2/MV/TV
  pulmonary mechanics       : Primus COMPLIANCE/PEEP/PPLAT/PIP/AWP/PAMB/MAWP
  depth of anaesthesia      : BIS BIS/EMG/SEF/SQI/SR/TOTPOW, Primus MAC + agent conc.
  drug effect (intervention): Orchestra per-drug RATE/VOL plus CE/CP/CT where present

Writes the two argument lists to disk (they are far too long for a command line) and
prints what the download will cost. The BIS EEG waveforms are deliberately excluded:
they are ~100 MB per case each and this project has no EEG use for them yet.

    python tools/pick_functional_channels.py
"""
import os
import sys

import pandas as pd

IDX = r"D:\vitaldb_local\cache\trks.parquet"
COHORT = r"D:\vitaldb_local\abp_features_full\abp_features.parquet"
OUT_DIR = r"D:\vitaldb_local"
TRACKS_TXT = os.path.join(OUT_DIR, "func_tracks.txt")
CASEIDS_TXT = os.path.join(OUT_DIR, "func_caseids.txt")

PRIMUS = ("CO2", "ETCO2", "INCO2", "RR_CO2", "FIO2", "FEO2", "MV", "TV",
          "COMPLIANCE", "PEEP_MBAR", "PPLAT_MBAR", "PIP_MBAR", "AWP", "MAC")
BIS = ("BIS/BIS", "BIS/EMG", "BIS/SEF", "BIS/SQI", "BIS/SR", "BIS/TOTPOW")
# Orchestra names every drug as <DRUG>_<KIND>. Curated by clinical meaning: the two
# main anaesthetics (remifentanil, propofol), a second remifentanil dilution, the
# neuromuscular blocker, and the only vasopressors present at all.
#
# Coverage found in this cohort (out of 2500): RFTN20 2092, PPF20 1346, RFTN50 39,
# ROC 70, PHEN 74, NEPI 54. Two consequences worth recording:
#   * an "requires a vasopressor" endpoint is NOT feasible here - hand-given boluses
#     are not recorded and pump infusions cover <3% of cases;
#   * neuromuscular block exists but on only 70 cases, so it can be demonstrated,
#     not modelled.
ORCHESTRA_DRUGS = ("RFTN20", "PPF20", "RFTN50", "ROC", "PHEN", "NEPI")
ORCHESTRA_KINDS = ("_RATE", "_VOL", "_CE", "_CP", "_CT")
EXCLUDE_SUBSTR = ("_WAV",)


def main():
    idx = pd.read_parquet(IDX)
    cohort = sorted(pd.read_parquet(COHORT, columns=["caseid"]).caseid.unique().astype(int))
    cohort_set = set(cohort)

    wanted = [f"Primus/{c}" for c in PRIMUS] + list(BIS)
    orchestra = [n for n in idx.loc[idx.tname.str.startswith("Orchestra/"), "tname"].unique()
                 if n.startswith(tuple(f"Orchestra/{d}" for d in ORCHESTRA_DRUGS))
                 and n.endswith(ORCHESTRA_KINDS)]
    wanted += sorted(orchestra)
    wanted = [w for w in wanted if not any(x in w for x in EXCLUDE_SUBSTR)]

    present = set(idx.tname.unique())
    missing = [w for w in wanted if w not in present]
    if missing:
        print(f"WARNING: not in the index, skipped: {missing}")
    wanted = [w for w in wanted if w in present]

    sub = idx[idx.tname.isin(wanted) & idx.caseid.isin(cohort_set)]
    cov = sub.groupby("tname").caseid.nunique().sort_values(ascending=False)
    jobs = len(sub)

    for path, payload in ((TRACKS_TXT, ",".join(wanted)),
                          (CASEIDS_TXT, ",".join(str(c) for c in cohort))):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(payload)
        except OSError as exc:
            print(f"cannot write {path}: {exc}")
            return 1

    print(f"cohort={len(cohort)} cases, channels={len(wanted)}, jobs={jobs:,}")
    print(f"estimated size: {jobs * 0.10 / 1024:.1f} GB (at ~100 KB per numeric track)")
    print(f"at the measured ~580 files/min -> {jobs / 580 / 60:.1f} h")
    print("\nper-channel coverage in the cohort:")
    for name, n in cov.items():
        print(f"  {name:34s} {n:5d}")
    print(f"\nwrote {TRACKS_TXT} and {CASEIDS_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
