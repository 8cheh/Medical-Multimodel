"""Does report/REPORT.md actually contain the numbers the artifacts contain?

Part four of the report was transcribed from results/p0_report*.json by hand, so
this re-derives every tabulated value from the JSON and asserts the formatted
string appears in the report. A transcription slip (or a later re-run that moves
a number) fails here instead of shipping.
"""
import json
import os
import re
import sys

ROOT = r"D:\github\VitalDB"
MD = os.path.join(ROOT, "report", "REPORT.md")
SUMMARY = os.path.join(ROOT, "report", "SUMMARY.md")
HORIZONS = {"5 min": "results/p0_report.json",
            "10 min": "results/p0_report_h600.json",
            "15 min": "results/p0_report_h900.json"}
OPT = "results/p0_report_h900opt.json"


def read_text(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def group(rel):
    path = os.path.join(ROOT, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    try:
        return rep["groups"]["numeric"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"{path} has no groups.numeric section") from exc


def ppg_pair(rel="results/p0_significance.json"):
    """The fused PPG-vs-numeric paired-test dump, one entry per split."""
    path = os.path.join(ROOT, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    try:
        return rep["pairs"]["fused_numeric:fused_numeric_ppg"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"{path} has no fused_numeric:fused_numeric_ppg pair") from exc


def as_float(node):
    try:
        return float(node)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"not a number: {node!r}") from exc


def neg(node):
    """The dump stores (numeric - numeric+ppg); the report quotes (+ppg - numeric)."""
    return -as_float(node)


def num(g, variant, *path):
    node = g["variants"][variant]
    for key in path:
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"not a number: {variant}/{path}: {node!r}") from exc


def delta(g, pair, metric, which):
    node = g["deltas"][pair]["boot"][metric][which]
    try:
        return float(node)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"not a number: {pair}/{metric}/{which}: {node!r}") from exc


# --- the mid-term report and the delivery README ------------------------------
# Both restate the headline numbers by hand. The checks above only assert that each
# derived literal appears *somewhere* in the report pair, so a typo in a third document
# would survive them: these are required in each named document instead.
EXTRA_DOCS = (("MIDTERM", "report/MIDTERM.md"),
              ("DELIVERY", "算法实现/d_运行结果/README.md"),
              ("LATEX", "report/latex/MIDTERM.tex"))
ENDPOINT_DOCS = [
    # label, dump, rule baseline, matched-channel variant, also require its detection
    ("低血压", "results/p0_report.json", "rule_map65", None, False),
    ("低氧", "results/p0_report_hypoxemia.json", "rule_map65", None, False),
    ("心动过缓", "results/p0_report_bradycardia.json", "rule_map65", None, False),
    ("通气不足", "results/p0_report_hypoventilation.json", "rule_rr_low", "main_func", True),
    ("高碳酸", "results/p0_report_hypercapnia.json", "rule_petco2_high", "main_func", False),
]


def read_json(rel):
    path = os.path.join(ROOT, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc


def latex_to_text(text):
    """Make a .tex source comparable to the rendered text the other checks match against.

    A literal like ``12.48%`` is written ``12.48\\%`` in LaTeX, and ``1.25--2.16`` renders
    as an en dash, so a raw substring check would report the numbers missing when they are
    plainly on the page. Only the escapes that occur in this document are handled - this is
    not a LaTeX parser, and it must not silently rewrite anything else.
    """
    for src, dst in (("\\%", "%"), ("\\_", "_"), ("\\&", "&"), ("\\#", "#"),
                     ("$\\to$", "->"), ("$\\rightarrow$", "->"), ("$\\geq$", ">="),
                     ("$\\times$", "x"), ("$-$0", "-0"), ("$-$1", "-1"),
                     ("$-$2", "-2"), ("$-$3", "-3"), ("$-$4", "-4"), ("$-$5", "-5"),
                     ("$-$6", "-6"), ("$-$7", "-7"), ("$-$8", "-8"), ("$-$9", "-9"),
                     ("{,}", ","), ("\\,", " "), ("\\ ", " "), ("~", " "),
                     ("---", "\u2014"), ("--", "\u2013")):
        text = text.replace(src, dst)
    return text


def extra_doc_checks():
    """Numbers the mid-term report and the delivery README must quote, per artifact."""
    docs, problems = {}, []
    for key, rel in EXTRA_DOCS:
        path = os.path.join(ROOT, rel)
        if os.path.isfile(path):
            text = read_text(path).replace("\u2212", "-")
            docs[key] = latex_to_text(text) if rel.endswith(".tex") else text
        else:
            problems.append(f"missing document: {rel}")
    both = tuple(key for key, _ in EXTRA_DOCS)
    checked = 0

    def want(literal, why, where=both):
        nonlocal checked
        checked += 1
        lit = literal.replace("\u2212", "-")
        for key in where:
            if key in docs and lit not in docs[key]:
                problems.append(f"{key} lacks {lit!r} ({why})")

    def claim(holds, why, where=both):
        """A qualitative claim ("no positive gain") is mechanically checkable too."""
        nonlocal checked
        checked += 1
        if not holds:
            for key in where:
                if key in docs:
                    problems.append(f"{key} states {why}, but the artifact disagrees")

    for label, rel, rule, func, func_det in ENDPOINT_DOCS:
        g = group(rel)
        want(f"{g['test_positive_rate'] * 100:.2f}%", f"{label} prevalence")
        want(f"{num(g, rule, 'window', 'auroc'):.4f}", f"{label} rule baseline auroc")
        want(f"{num(g, 'main', 'window', 'auroc'):.4f}", f"{label} model auroc")
        want(f"{num(g, 'main', 'window', 'auprc'):.4f}", f"{label} model auprc")
        want(f"{num(g, 'main', 'event', 'detection_rate') * 100:.2f}%", f"{label} detection")
        want(f"{num(g, 'main', 'event', 'false_alarm_rate_per_h'):.2f}", f"{label} false/h")
        if func:
            want(f"{num(g, func, 'window', 'auroc'):.4f}", f"{label} +channels auroc")
            want(f"{num(g, func, 'window', 'auprc'):.4f}", f"{label} +channels auprc")
            want(f"{delta(g, 'main_func-main', 'auroc', 'mean'):+.4f}",
                 f"{label} channels auroc delta", where=("MIDTERM",))
            if func_det:
                want(f"{num(g, func, 'event', 'detection_rate') * 100:.1f}%",
                     f"{label} +channels detection")

    # horizons, the optimisation levers, the alarm front, calibration, seed spread
    for label, rel in HORIZONS.items():
        g = group(rel)
        want(f"{num(g, 'main', 'window', 'auroc'):.4f}", f"{label} model auroc")
        want(f"{num(g, 'rule_map65', 'window', 'auroc'):.4f}", f"{label} rule auroc")
    opt = group(OPT)
    for pair in ("main_long-main", "main_extra-main"):
        want(f"{delta(opt, pair, 'auroc', 'mean'):+.4f}", f"optimisation {pair}")
    for row in read_json("results/alarm_pareto.json"):
        if abs(as_float(row["clear_factor"]) - 0.8) < 1e-9:
            want(f"{as_float(row['detect']) * 100:.2f}%",
                 f"alarm front persist={row['persist']}")
            want(f"{as_float(row['false_per_h']):.2f}",
                 f"alarm front persist={row['persist']} false/h")
    for label in ("hypotension", "hypoxemia"):
        main = read_json(f"results/calibration_{label}.json")["variants"]["main"]
        want(f"{as_float(main['brier']):.4f}", f"{label} Brier")
        want(f"{as_float(main['ece']):.4f}", f"{label} ECE")
        for row in main["dca"]:
            if abs(as_float(row["pt"]) - 0.10) < 1e-9:
                want(f"{as_float(row['nb_model']):+.4f}", f"{label} net benefit at pt=0.10")
    for auroc in ("0.9537", "0.9519", "0.9602", "0.9552"):
        want(auroc, "multi-seed AUROC", where=("MIDTERM",))

    # the paired lead-time test that replaced the marginal-median claim
    g300 = group("results/p0_report.json")
    lead = ((g300.get("deltas") or {}).get("main-rule_map65") or {}).get("lead") or {}
    if lead:
        want(f"{lead['frac_a_earlier'] * 100:.1f}%", "paired lead fraction earlier",
             where=("MIDTERM",))
        want(f"{lead['lead_delta_mean_s']:+.1f}", "paired lead mean delta", where=("MIDTERM",))
    for which in ("lo", "hi"):
        want(f"{delta(g300, 'main-rule_map65', 'auroc', which):+.4f}",
             f"5 min model-vs-rule auroc {which}", where=("MIDTERM",))

    # the waveform claims, and the two channel claims that point the other way
    for label, rel, pair in (("低血压 ABP", "results/p0_report_multimodal_h300.json",
                              "main_multimodal-main"),
                             ("低氧 PPG", "results/p0_report_hypoxemia_mm.json",
                              "main_multimodal-main"),
                             ("低血压 PPG", "results/p0_report_multimodal_h300.json",
                              "main_multimodal-main")):
        claim(delta(group(rel), pair, "auroc", "lo") <= 0,
              f"{label} has no positive auroc gain", where=("MIDTERM",))
    for which in ("mean", "lo", "hi"):
        want(f"{delta(group('results/p0_report_multimodal_h300.json'), 'main_multimodal-main', 'auroc', which):+.4f}",
             f"ABP auroc {which}", where=("MIDTERM",))
        want(f"{delta(group('results/p0_report_hypercapnia.json'), 'main_func-main', 'auroc', which):+.4f}",
             f"hypercapnia channels auroc {which}", where=("MIDTERM",))
    claim(delta(group("results/p0_report_hypoventilation.json"), "main_func-main",
                "auroc", "lo") > 0,
          "the matched respiratory channels DO gain", where=("MIDTERM",))
    claim(delta(group("results/p0_report_hypercapnia.json"), "main_func-main",
                "auroc", "lo") <= 0,
          "a second ETCO2 channel adds nothing", where=("MIDTERM",))

    # dataset scale, the storage census, and the PDF verdict (both are artifacts)
    for literal, why in (("1,553,854", "dataset windows"), ("3,495", "dataset cases"),
                         ("67 维", "feature count"), ("11 维", "map-only feature count")):
        want(literal, why, where=("MIDTERM",))
    # the streaming check log: the latency range and the score-parity bound
    stream = read_text(os.path.join(ROOT, "results", "stream_check.log"))
    medians = [as_float(x) for x in re.findall(r"median latency ([0-9.]+) ms", stream)]
    diffs = [as_float(x) for x in re.findall(r"max \|diff\| = ([0-9.eE+-]+)", stream)]
    cases = len(re.findall(r"^=== caseid", stream, re.M))
    if medians and diffs:
        want(f"{min(medians):.2f}–{max(medians):.2f} ms", "streaming latency range")
        want(f"{max(diffs):.3e}", "streaming score-parity bound")
        want(f"{cases} 例", "streaming case count", where=("MIDTERM",))
    else:
        problems.append("results/stream_check.log carries no latency/diff lines")

    # The LaTeX report states that the whole check roster passes. Anchor that on the roster
    # itself rather than on the last verdict: a stored verdict below the total would make
    # the comparison fail no matter how often it is re-run.
    roster = read_json("results/check_all_roster.json")
    want(f"{roster['total']}/{roster['total']} 通过", "one-command check count",
         where=("LATEX",))
    inv = read_json("算法实现/inventory.json")
    manifest = read_json("算法实现/MANIFEST.json")
    want(f"{manifest['n_entries']} 个文件副本", "delivery copy count", where=("MIDTERM",))
    want(f"{inv['channels']['indexed']}", "indexed channels", where=("MIDTERM",))
    want(f"{inv['channels']['local']} 路", "local channels", where=("MIDTERM",))
    want(f"{inv['local_data']['numeric_csv']['files']:,}", "numeric track files",
         where=("MIDTERM",))
    want(f"{inv['local_data']['numeric_csv']['gb']} GB", "numeric track size",
         where=("MIDTERM",))
    want(f"{inv['local_data']['abp_npy']['gb']} GB", "ABP waveform size", where=("MIDTERM",))
    want(f"{inv['download']['files_on_disk']:,}", "functional channel files",
         where=("MIDTERM",))
    want(f"{inv['download']['gb_on_disk']} GiB", "functional channel size",
         where=("MIDTERM",))
    want(str(read_json("results/check_numbers.json")["checked"]),
         "the report-number count", where=both)
    if os.path.isfile(os.path.join(ROOT, "results", "verify_pdf.json")):
        v = read_json("results/verify_pdf.json")
        for literal, why in ((f"{v['pages']} 页", "PDF pages"),
                             (f"{v['images']} 图", "PDF figures"),
                             (f"{v['key_numbers']['checked']} 数值", "PDF numbers"),
                             (f"{v['links']['internal']} 链接", "PDF links")):
            want(literal, why, where=("MIDTERM",))
    else:
        problems.append("results/verify_pdf.json missing - run tools/verify_pdf.py")
    return checked, problems


def main():
    # The report writes negative values with a typographic minus (U+2212) while
    # formatting produces ASCII '-', so normalise both sides before comparing. The
    # readable narrative quotes the same conclusions, so it is scanned too - otherwise
    # the two documents would be free to drift apart.
    text = read_text(MD)
    if os.path.isfile(SUMMARY):
        text += "\n" + read_text(SUMMARY)
    text = text.replace("\u2212", "-")
    missing = []
    state = {"checked": 0}

    def need(literal, why):
        state["checked"] += 1
        if literal.replace("\u2212", "-") not in text:
            missing.append(f"{literal!r} ({why})")

    # --- horizon tables. The 5 min table tabulates all five variants; the 10/15 min
    # tables only carry the model, the MAP-only model and the standard rule ---
    for label, rel in HORIZONS.items():
        g = group(rel)
        variants = (("main", "map_only", "rule_map65", "rule_map65_mean", "rule_map65_min")
                    if label == "5 min" else ("main", "map_only", "rule_map65"))
        for variant in variants:
            if variant not in g["variants"]:
                continue
            need(f"{num(g, variant, 'window', 'auroc'):.4f}", f"{label} {variant} auroc")
        for variant in ("main", "rule_map65"):
            det = num(g, variant, "event", "detection_rate") * 100
            need(f"{det:.2f}%", f"{label} {variant} detection")
            need(f"{num(g, variant, 'event', 'false_alarm_rate_per_h'):.2f}",
                 f"{label} {variant} false/h")

    # --- 5 min baseline table: the full metric row set ---
    g5 = group(HORIZONS["5 min"])
    for variant in ("main", "map_only", "rule_map65", "rule_map65_mean", "rule_map65_min"):
        for metric in ("auroc", "auprc", "accuracy", "sensitivity", "specificity"):
            need(f"{num(g5, variant, 'window', metric):.4f}", f"5min {variant} {metric}")
        need(f"{num(g5, variant, 'event', 'lead_median_s') / 60:.2f}",
             f"5min {variant} lead min")
        need(f"{num(g5, variant, 'event', 'alarm_rate_per_h'):.2f}",
             f"5min {variant} alarm/h")

    # --- paired deltas quoted in the horizon comparison table ---
    for label, rel in HORIZONS.items():
        g = group(rel)
        need(f"{delta(g, 'main-rule_map65', 'auroc', 'mean'):+.4f}", f"{label} d auroc rule")
        need(f"{delta(g, 'main-map_only', 'auroc', 'mean'):+.4f}", f"{label} d auroc map_only")
    for label, rel in (("5 min", HORIZONS["5 min"]), ("15 min", HORIZONS["15 min"])):
        g = group(rel)
        need(f"{delta(g, 'main-rule_map65', 'auroc', 'lo'):+.4f}", f"{label} d auroc rule lo")
        need(f"{delta(g, 'main-rule_map65', 'auroc', 'hi'):+.4f}", f"{label} d auroc rule hi")

    # --- optimisation ablation table + paired deltas ---
    go = group(OPT)
    for variant in ("main", "main_long", "main_extra", "main_all"):
        for metric in ("auroc", "auprc", "accuracy", "sensitivity"):
            need(f"{num(go, variant, 'window', metric):.4f}", f"opt {variant} {metric}")
        need(f"{num(go, variant, 'event', 'false_alarm_rate_per_h'):.2f}",
             f"opt {variant} false/h")
        need(f"{num(go, variant, 'event', 'detection_rate') * 100:.2f}%",
             f"opt {variant} detection")
    for pair in ("main_long-main", "main_extra-main", "main_all-main"):
        for metric in ("auroc", "auprc", "sensitivity"):
            for which in ("mean", "lo", "hi"):
                need(f"{delta(go, pair, metric, which):+.4f}", f"{pair} {metric} {which}")
        need(f"{delta(go, pair, 'accuracy', 'mean'):+.4f}", f"{pair} accuracy mean")

    # --- figure 9 section: the PPG paired test, quoted in the (+ppg - numeric) sense.
    # Point differences come from delta_point; only the interval comes from the
    # bootstrap distribution (its mean drifts from the point estimate in the 4th
    # decimal, which is what caught this line the first time).
    sig = ppg_pair()
    test, val = sig["test"], sig["val"]
    for metric in ("accuracy", "auroc", "auprc"):
        d = test["delta"][metric]
        need(f"{neg(test['delta_point'][metric]):+.4f}", f"ppg {metric} point")
        need(f"{neg(d['hi']):+.4f}", f"ppg {metric} ci low")
        need(f"{neg(d['lo']):+.4f}", f"ppg {metric} ci high")
    for metric in ("specificity", "sensitivity", "precision"):
        need(f"{neg(test['delta_point'][metric]):+.4f}", f"ppg {metric} point")
    need(f"{as_float(test['mcnemar']['p']):.3f}", "ppg mcnemar p")
    need(str(test["mcnemar"]["a_right_b_wrong"]), "ppg mcnemar discordant a")
    need(str(test["mcnemar"]["a_wrong_b_right"]), "ppg mcnemar discordant b")
    for side in ("fused_numeric", "fused_numeric_ppg"):
        need(f"{as_float(test['thresholds'][side]):.4f}", f"ppg threshold {side}")
    need(f"{neg(val['delta']['auroc']['mean']):+.4f}", "ppg val-split sign flip")

    # --- part six: the two new endpoints, then waveforms-vs-endpoint -----------
    for endpoint, rel in (("低氧", "results/p0_report_hypoxemia.json"),
                          ("心动过缓", "results/p0_report_bradycardia.json")):
        g = group(rel)
        need(f"{g['test_positive_rate'] * 100:.2f}%", f"{endpoint} prevalence")
        need(f"{1 - g['test_positive_rate']:.4f}", f"{endpoint} majority baseline")
        for variant in ("main", "rule_map65"):
            need(f"{num(g, variant, 'window', 'auroc'):.4f}", f"{endpoint} {variant} auroc")
        need(f"{num(g, 'main', 'window', 'auprc'):.4f}", f"{endpoint} model auprc")
        need(f"{num(g, 'main', 'event', 'detection_rate') * 100:.2f}%",
             f"{endpoint} detection")
        need(f"{num(g, 'main', 'event', 'lead_median_s') / 60:.2f}", f"{endpoint} lead")
        need(f"{num(g, 'main', 'event', 'false_alarm_rate_per_h'):.2f}",
             f"{endpoint} false/h")

    for label, rel in (("低血压", "results/p0_report_multimodal_h300.json"),
                       ("低氧", "results/p0_report_hypoxemia_mm.json")):
        g = group(rel)
        for metric in ("auroc", "auprc"):
            for which in ("mean", "lo", "hi"):
                need(f"{delta(g, 'main_multimodal-main', metric, which):+.4f}",
                     f"{label} waveforms {metric} {which}")

    # --- PPG at three horizons (case-level paired, results/p0_report_ppg_h*.json) ---
    for tag in ("300", "600", "900"):
        g = group(f"results/p0_report_ppg_h{tag}.json")
        for which in ("mean", "lo", "hi"):
            need(f"{delta(g, 'main_ppg-main', 'auroc', which):+.4f}",
                 f"PPG at {tag}s auroc {which}")

    # --- part six: the functional (respiratory) endpoints, where matching matters ---
    for label, rel, rule in (
            ("hypoventilation", "results/p0_report_hypoventilation.json", "rule_rr_low"),
            ("hypercapnia", "results/p0_report_hypercapnia.json", "rule_petco2_high")):
        g = group(rel)
        need(f"{g['test_positive_rate'] * 100:.2f}%", f"{label} prevalence")
        for variant in ("main", "main_func", rule):
            need(f"{num(g, variant, 'window', 'auroc'):.4f}", f"{label} {variant} auroc")
            need(f"{num(g, variant, 'window', 'auprc'):.4f}", f"{label} {variant} auprc")
            need(f"{num(g, variant, 'event', 'detection_rate') * 100:.1f}%",
                 f"{label} {variant} detection")
        for metric in ("auroc", "auprc", "accuracy"):
            for which in ("mean", "lo", "hi"):
                need(f"{delta(g, 'main_func-main', metric, which):+.4f}",
                     f"{label} functional {metric} {which}")

    # --- P0-1: the paired lead-time test that replaced the marginal-median claim ---
    g300 = group("results/p0_report.json")
    for pair, label in (("main-rule_map65", "model vs rule"),
                        ("main-map_only", "model vs map_only")):
        lead = ((g300.get("deltas") or {}).get(pair) or {}).get("lead")
        if not lead:
            missing.append(f"no paired-lead block in {pair} (the claim is unverifiable)")
            state["checked"] += 1
            continue
        need(str(lead["n_paired"]), f"{label} n_paired")
        need(f"{lead['frac_a_earlier'] * 100:.1f}%", f"{label} fraction earlier")
        need(f"{lead['lead_delta_mean_s']:+.1f}", f"{label} mean lead delta")

    # --- P0-4: signal-quality gating (results/quality_gate.json) ---
    path = os.path.join(ROOT, "results", "quality_gate.json")
    try:
        with open(path, encoding="utf-8") as fh:
            qg = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    need(f"{as_float(qg['usable_fraction']) * 100:.2f}%", "quality gate usable fraction")
    for key in ("ungated", "gated"):
        need(f"{as_float(qg['policies'][key]['false_per_h']):.2f}", f"quality gate {key} false/h")
        need(str(qg["policies"][key]["detect"]), f"quality gate {key} episodes detected")

    # --- P0-2: calibration + decision curve (results/calibration_*.json) ---
    for label in ("hypotension", "hypoxemia"):
        path = os.path.join(ROOT, "results", f"calibration_{label}.json")
        try:
            with open(path, encoding="utf-8") as fh:
                rep = json.load(fh)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        main = rep["variants"]["main"]
        need(f"{as_float(main['brier']):.4f}", f"{label} Brier")
        need(f"{as_float(main['ece']):.4f}", f"{label} ECE")
        for row in main["dca"]:
            if abs(as_float(row["pt"]) - 0.10) < 1e-9:
                need(f"{as_float(row['nb_model']):+.4f}", f"{label} net benefit at pt=0.10")

    # --- P0-3: the alarm policy Pareto front (results/alarm_pareto.json) ---
    path = os.path.join(ROOT, "results", "alarm_pareto.json")
    try:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    for row in rows:
        if abs(as_float(row["clear_factor"]) - 0.8) < 1e-9:
            need(f"{as_float(row['detect']) * 100:.2f}%",
                 f"alarm pareto persist={row['persist']} detection")
            need(f"{as_float(row['false_per_h']):.2f}",
                 f"alarm pareto persist={row['persist']} false/h")

    # --- cross-run reproducibility claim: h900 and h900opt must agree on main ---
    state["checked"] += 1
    here = f"{num(go, 'main', 'window', 'auroc'):.4f}"
    there = f"{num(group(HORIZONS['15 min']), 'main', 'window', 'auroc'):.4f}"
    if here != there:
        missing.append(f"reproducibility claim: h900 main={there} != h900opt main={here}")

    extra_checked, extra_problems = extra_doc_checks()

    print(f"checked {state['checked']} values against the artifacts; missing {len(missing)}")
    print(f"checked {extra_checked} values against the mid-term report and the delivery "
          f"README; missing {len(extra_problems)}")
    # Leave the counts behind as artifacts: the architecture figure cites the report one,
    # and a figure that cites a number has to cite a file, not a terminal's scrollback.
    for name, payload in (("check_numbers.json",
                           {"checked": state["checked"], "missing": len(missing)}),
                          ("check_extra_docs.json",
                           {"checked": extra_checked, "missing": len(extra_problems),
                            "docs": [rel for _, rel in EXTRA_DOCS]})):
        try:
            with open(os.path.join(ROOT, "results", name), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
        except OSError as exc:
            print(f"  (cannot record {name}: {exc})")
    for item in missing:
        print("  MISSING", item)
    for item in extra_problems:
        print("  PROBLEM", item)
    if missing or extra_problems:
        print("FAIL report numbers do not match the artifacts")
        return 1
    print("OK every checked report number matches results/p0_report*.json, "
          "results/p0_significance.json, report/MIDTERM.md and the delivery README")
    return 0


if __name__ == "__main__":
    sys.exit(main())
