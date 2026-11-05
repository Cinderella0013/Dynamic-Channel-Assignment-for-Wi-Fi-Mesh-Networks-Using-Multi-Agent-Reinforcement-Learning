"""ns-3 validation report: python ns3/report.py --ns3 runs/ns3/default --sim runs/v8/fresh --sim-sub a

Layout (both sides): <dir>/t<k>/[<sub>/]suite_<name>.json, one or more methods per file, schema of
mesh.eval.suite.run_suite. Sim runs are cut to the eval seeds present in the ns-3 files, so both sides are paired
on the same scenarios and seeds. Scores, seed-pair CI and Holm come from mesh.eval.composite.
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mesh.eval import composite as C  # noqa: E402

SCENARIOS = ["S2", "B1", "B3"]
METHODS = ["m0", "ko", "li", "yuan", "q_neutral"]
OURS = "egat_h3"
GATE = 0.20


def _tdirs(root, sub):
    ds = [d for d in glob.glob(os.path.join(root, "t*")) if re.fullmatch(r"t\d+", os.path.basename(d))]
    return [os.path.join(d, sub) if sub else d for d in sorted(ds, key=lambda d: int(os.path.basename(d)[1:]))]


def load(root, sub="", scenarios=SCENARIOS, seeds=None):
    """-> (results in composite format with FILE tags, eval seeds found). One merged file per training seed t<k>."""
    tmp, paths, found = tempfile.mkdtemp(), [], set()
    for i, d in enumerate(_tdirs(root, sub)):
        merged = {"commit": "", "results": {}}
        for f in sorted(glob.glob(os.path.join(d, "suite_*.json"))):
            j = json.load(open(f))
            merged["commit"] = merged["commit"] or j["commit"]
            sd = list(j["seeds"])
            keep = [n for n, s in enumerate(sd) if seeds is None or s in seeds]
            found |= {sd[n] for n in keep}
            for m, by_s in j["results"].items():
                for s in scenarios:
                    if s in by_s:
                        merged["results"].setdefault(m, {})[s] = {k: [v[n] for n in keep] for k, v in by_s[s].items()}
        if merged["results"]:
            p = os.path.join(tmp, f"t{i}.json")
            json.dump(merged, open(p, "w"))
            paths.append(p)
    if not paths:
        raise SystemExit(f"no suite_*.json under {root}")
    return C._load(paths)[0], sorted(found)


def _mean(res, m, scen, key):
    return float(np.mean([C._means(res, m, s)[key] for s in scen]))


def _lead(res, ref, scen):
    """(mean composite lead, seed-pair t-interval, scenarios better after Holm on the bootstrap p)"""
    sm = C.seed_matrix_ci(res, OURS, ref, scen)
    sg = C.significance(res, OURS, ref, scen)
    return C._mean_score(res, OURS, ref, scen), sm["ci"], sg["frac_sig_boot"]


def _cell(x):
    lead, ci, sig = x
    return f"{lead:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] ({sig:.0%})"


def build(ns3, sim, scen=SCENARIOS):
    avail = [s for s in scen if all(s in r.get(m, {}) for r in (ns3, sim) for m in (OURS,))]
    refs = [m for m in METHODS if m in ns3 and m in sim and all(s in ns3[m] and s in sim[m] for s in avail)]
    out = ["| method | ns-3 lead of EGAT+H | sim lead of EGAT+H | thr ns-3 / sim (Mbps) | loss ns-3 / sim | "
           "ch. switches ns-3 / sim | route changes ns-3 / sim |", "|---|---|---|---|---|---|---|"]
    for m in refs + [OURS]:
        lead = ("-", "-") if m == OURS else (_cell(_lead(ns3, m, avail)), _cell(_lead(sim, m, avail)))
        f = lambda key, fmt: f"{_mean(ns3, m, avail, key):{fmt}} / {_mean(sim, m, avail, key):{fmt}}"
        out.append(f"| {m} | {lead[0]} | {lead[1]} | {f('aggregate', '.2f')} | {f('loss_rate', '.3f')} | "
                   f"{f('n_channel_switch', '.0f')} | {f('n_route_change', '.1f')} |")
    out += ["", f"Lead = mean composite over {', '.join(avail)} (+0.10 = 10% better), [seed-pair 95% t-interval], "
            "(share of scenarios better after Holm, bootstrap p)."]
    # calibration: Ko and M0 on S2, mean aggregate throughput
    cal = ["", "## Calibration gate (S2, mean aggregate throughput)", "",
           f"| method | ns-3 (Mbps) | sim (Mbps) | difference | gate {GATE:.0%} |", "|---|---|---|---|---|"]
    ok = True
    for m in ("ko", "m0"):
        if m in ns3 and m in sim and "S2" in ns3[m] and "S2" in sim[m]:
            a, b = _mean(ns3, m, ["S2"], "aggregate"), _mean(sim, m, ["S2"], "aggregate")
            d = (a - b) / b
            ok &= abs(d) <= GATE
            cal.append(f"| {m} | {a:.2f} | {b:.2f} | {d:+.1%} | {'pass' if abs(d) <= GATE else 'FAIL'} |")
        else:
            ok = False
            cal.append(f"| {m} | - | - | - | missing |")
    cal.append(f"\nGate: {'passed' if ok else 'NOT passed: stop, ns-3 stays future work'}.")
    return "\n".join(out + cal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3", required=True, help="e.g. runs/ns3/default")
    ap.add_argument("--sim", required=True, help="e.g. runs/v8/fresh")
    ap.add_argument("--sim-sub", default="a", help="subfolder below t<k> of the simulator results ('' for none)")
    a = ap.parse_args()
    ns3, seeds = load(a.ns3)
    sim, found = load(a.sim, a.sim_sub, seeds=set(seeds))
    if set(found) != set(seeds):
        raise SystemExit(f"simulator results lack eval seeds {sorted(set(seeds) - set(found))}: rerun the sim with those seeds")
    print(f"# ns-3 vs simulator (eval seeds {seeds}, training seeds {len(set(ns3[OURS]['S2'][C.FILE]))})\n")
    print(build(ns3, sim))


if __name__ == "__main__":
    main()
