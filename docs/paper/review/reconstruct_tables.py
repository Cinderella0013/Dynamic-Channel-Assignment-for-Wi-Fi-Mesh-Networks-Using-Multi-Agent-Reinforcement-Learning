"""Regenerate manuscript v5 Tables 5-9 (and the 0.093 / 0.094 check) from the recorded suite outputs, full precision.

python docs/paper/review/reconstruct_tables.py > docs/paper/review/reconstruct_tables.out.md

Reads runs/v10 (simulator) and runs/ns3/{v10,v10base} (ns-3). No new simulation. Every number goes through
mesh.eval.composite (the code that produced the reports), so the aggregation order is the one documented in
docs/paper/review/review_v5_c5_c7.md.
"""
import json
from pathlib import Path

import numpy as np

from mesh.eval.composite import (COUNT_METRICS, FILE, LOWER_BETTER, WEIGHTS, _load, _mean_score, _means, delta,
                                 score, seed_matrix_ci)

R = Path("runs/v10")
AB = [f"S{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 6)]
T5 = range(5)
OURS = "egat_nogru_h"
LEARNED = ["egat_h2", "li", "yuan", "q_neutral", "li_h", "yuan_h"]
NAMES = {"ko": "Ko", "li": "Li", "yuan": "Yuan", "m0": "M0", "q_neutral": "Q-neutral", "li_h": "Li+H",
         "yuan_h": "Yuan+H", "egat_h2": "EGAT+GRU+H", OURS: "EGAT+H"}


def files(d, s, methods, seeds=T5, heur=True):
    """Suite files of one table cell. The heuristic file (M0, Ko: deterministic, one file) is repeated once per
    training seed so every method has the same number of runs per scenario; this is what the reports did."""
    out = [str(d / f"t{k}" / s / f"suite_{m}.json") for m in methods for k in seeds]
    if heur:
        out += [str(d / "t0" / s / "suite_heur.json")] * len(list(seeds))
    return out


def provenance(paths):
    """Commit, scenario overrides and agent overrides recorded in each distinct suite file."""
    seen = {}
    for p in dict.fromkeys(paths):
        d = json.load(open(p))
        key = (d["commit"][:12], json.dumps(d.get("overrides", {}), sort_keys=True),
               json.dumps(d.get("agent_overrides", {}), sort_keys=True), f"seeds {d['seeds'][0]}-{d['seeds'][-1]}")
        seen.setdefault(key, []).append(p)
    return [f"- commit {c}, overrides {o}, agent {a}, eval {s}: {len(v)} files, e.g. `{v[0]}`"
            for (c, o, a, s), v in seen.items()]


def ci_cell(res, ref, scen):
    r = seed_matrix_ci(res, OURS, ref, scen)
    return f"{r['mean']:+.6f} [{r['ci'][0]:+.6f}, {r['ci'][1]:+.6f}] (df {r['df']:.2f})"


def table5():
    print("## Table 5: seed-pair composite and 95% t-interval, setting (a), A+B\n")
    sets = {"0-19": R, "20-39": R / "fresh", "40-59": R / "fresh2"}
    loaded = {k: _load(files(d, "a", [OURS] + LEARNED)) for k, d in sets.items()}
    print("| reference | " + " | ".join(sets) + " |\n|---|---|---|---|")
    for ref in ["ko", "li", "yuan", "m0", "q_neutral", "li_h", "yuan_h", "egat_h2"]:
        print(f"| {NAMES[ref]} | " + " | ".join(ci_cell(loaded[k][0], ref, AB) for k in sets) + " |")
    for k, d in sets.items():
        print(f"\nRun/config, topologies {k}:")
        print("\n".join(provenance(files(d, "a", [OURS] + LEARNED))))
    print()
    return loaded["0-19"][0]


def table6(res):
    print("## Table 6: raw means over 11 scenarios (scenario mean of all runs, then mean over scenarios), "
          "setting (a), topologies 0-19\n")
    keys = ["aggregate", "jain", "p5", "loss_rate", "n_channel_switch", "n_route_change", "stability"]
    print("| method | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1))
    for m in ["m0", "ko", "li", "yuan", "li_h", "yuan_h", "egat_h2", OURS, "q_neutral"]:
        v = {k: np.mean([_means(res, m, s)[k] for s in AB]) for k in keys}
        print(f"| {NAMES[m]} | " + " | ".join(f"{v[k]:.6f}" for k in keys) + " |")
    print()


def table7(res):
    print("## Table 7: per-term contributions of the pooled-mean composite, topologies 0-19 (full precision)\n")
    print("| reference | " + " | ".join(WEIGHTS) + " | total (sum of terms) | report 'mean' column |\n|" + "---|" * 8)
    for ref in ["ko", "li", "yuan", "li_h", "yuan_h", "m0", "egat_h2"]:
        parts = {k: np.mean([w * delta(_means(res, OURS, s)[k], _means(res, ref, s)[k], k in LOWER_BETTER,
                                       k in COUNT_METRICS) for s in AB]) for k, w in WEIGHTS.items()}
        print(f"| {NAMES[ref]} | " + " | ".join(f"{v:+.6f}" for v in parts.values())
              + f" | {sum(parts.values()):+.6f} | {_mean_score(res, OURS, ref, AB):+.6f} |")
    print()


def table7_rounded_inputs():
    """Reproduces how 0.094 arose: make_figures.decomposition() read the per-metric means from the markdown
    report, where they are printed to 3 decimals, and recomputed the deltas from those rounded means."""
    import sys
    sys.path.insert(0, "docs/paper/figures")
    import make_figures as mf
    t = {k: mf.metric(mf.TEST, k) for k in WEIGHTS}
    parts = {k: np.mean([w * delta(t[k][OURS][s], t[k]["yuan_h"][s], k in LOWER_BETTER, k in COUNT_METRICS)
                         for s in AB]) for k, w in WEIGHTS.items()}
    print("## Yuan+H: 0.093 vs 0.094\n")
    print("Terms recomputed from the 3-decimal means printed in `runs/v10/final/composite_a_nogru_test.md` "
          "(the input `make_figures.decomposition()` used):\n")
    print(" | ".join(f"{k} {v:+.6f}" for k, v in parts.items()) + f" | total {sum(parts.values()):+.6f}\n")


def table8():
    print("## Table 8: three-seed ablation (training seeds 0-2, margin 2), setting (a), A+B, seed-pair mean\n")
    variants = {"egat_h2": "Full GRU", "egat_noedge": "No edge features", "egat_nogru": "No GRU",
                "egat_noko": "No Ko prior", "egat_nobc": "No behaviour cloning"}
    print("| variant | vs Ko | vs Yuan+H | vs full GRU [95% CI] |\n|---|---|---|---|")
    for v, lab in variants.items():
        ps = files(R, "a", [v], seeds=range(3)) + files(R, "a", ["yuan_h"], heur=False)
        cells = []
        for ref in ("ko", "yuan_h"):
            res, _ = _load(ps)
            r = seed_matrix_ci(res, v, ref, AB)
            cells.append(f"{r['mean']:+.6f}")
        if v == "egat_h2":
            cells.append("reference")
        else:
            res, _ = _load(files(R, "a", [v, "egat_h2"], seeds=range(3), heur=False))
            r = seed_matrix_ci(res, v, "egat_h2", AB)
            cells.append(f"{r['mean']:+.6f} [{r['ci'][0]:+.6f}, {r['ci'][1]:+.6f}]")
        print(f"| {lab} | " + " | ".join(cells) + " |")
    print("\nRun/config:")
    print("\n".join(provenance(files(R, "a", list(variants), seeds=range(3)))))
    print()


def table9():
    print("## Table 9: S2/B1/B3, seed-pair mean [95% t-interval]\n")
    sc = ["S2", "B1", "B3"]
    sim = _load(files(R, "a", [OURS] + LEARNED))[0]
    ns = {}
    for lab, d in (("ns-3 RTS/CTS", "v10"), ("ns-3 base MAC", "v10base")):
        nd = Path("runs/ns3") / d
        ps = [str(nd / f"t{k}" / f"suite_{m}.json") for m in (OURS, "egat_h2", "rl", "li_h", "yuan_h") for k in T5]
        ps += [str(nd / "t0" / "suite_base.json")] * 5
        ns[lab] = (_load(ps)[0], ps)
    print("| reference | simulator (topologies 0-19) | " + " | ".join(ns) + " |\n|---|---|---|---|")
    for ref in ["ko", "li", "yuan", "m0", "q_neutral", "li_h", "yuan_h", "egat_h2"]:
        print(f"| {NAMES[ref]} | {ci_cell(sim, ref, sc)} | " + " | ".join(ci_cell(r, ref, sc) for r, _ in ns.values()) + " |")
    for lab, (_, ps) in ns.items():
        print(f"\nRun/config, {lab} (topologies 0-4):")
        print("\n".join(provenance(ps)))
    print("\nRaw means, ns-3 S2 (ours vs Ko):\n")
    for lab, (res, _) in ns.items():
        for m in (OURS, "ko"):
            x = _means(res, m, "S2")
            print(f"- {lab} {NAMES[m]}: " + ", ".join(f"{k} {x[k]:.6f}" for k in ("aggregate", "jain", "p5", "loss_rate", "stability")))
    print()


def zero_reference(res):
    print("## Zero-reference check (Q-neutral P5)\n")
    v = {s: _means(res, "q_neutral", s)["p5"] for s in AB}
    print("Q-neutral mean P5 per scenario (topologies 0-19): " + ", ".join(f"{s} {x:.3e}" for s, x in v.items()) + "\n")


if __name__ == "__main__":
    res = table5()
    table6(res)
    table7(res)
    table7_rounded_inputs()
    zero_reference(res)
    table8()
    table9()
