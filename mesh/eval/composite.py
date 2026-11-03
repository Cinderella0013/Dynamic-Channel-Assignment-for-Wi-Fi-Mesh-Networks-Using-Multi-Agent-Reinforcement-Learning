"""Pre-registered composite score (spec addendum 5.1): python -m mesh.eval.composite runs/v6/suite_*.json --ours egat"""
import argparse
import json

import numpy as np

from mesh.sim.scenarios import SUITE_A, SUITE_B, SUITE_C

WEIGHTS = {"aggregate": 0.25, "jain": 0.20, "p5": 0.20, "loss_rate": 0.20, "stability": 0.15}
LOWER_BETTER = {"loss_rate", "stability"}


def delta(ours, ref, lower_better):
    """Relative improvement of ours over ref, clipped to [-1, 1]."""
    if lower_better:
        d = (ref - ours) / (abs(ref) + 1.0)  # +1 smooths count metrics that can be 0
    elif ref == 0:
        d = 0.0 if ours == 0 else float(np.sign(ours))
    else:
        d = (ours - ref) / abs(ref)
    return float(np.clip(d, -1.0, 1.0))


def _means(results, method, scenario):
    r = results[method][scenario]
    m = {k: float(np.mean(v)) for k, v in r.items()}
    m["stability"] = m["n_channel_switch"] + m["n_route_change"]
    return m


def score(results, method, ref, scenario, weights=WEIGHTS):
    ours, base = _means(results, method, scenario), _means(results, ref, scenario)
    return sum(w * delta(ours[k], base[k], k in LOWER_BETTER) for k, w in weights.items())


def _load(paths):
    results, commits = {}, set()
    for p in [paths] if isinstance(paths, str) else paths:
        d = json.load(open(p))
        commits.add(d["commit"])
        for method, by_s in d["results"].items():
            results.setdefault(method, {}).update(by_s)
    return results, commits


def _mean_score(results, ours, ref, scenarios, weights=WEIGHTS):
    return float(np.mean([score(results, ours, ref, s, weights) for s in scenarios]))


def report(paths, ours="egat"):
    results, commits = _load(paths)
    methods = list(results)
    scen = list(results[ours])
    ab = [s for s in scen if s in SUITE_A + SUITE_B]
    c = [s for s in scen if s in SUITE_C]
    refs = [m for m in methods if m != ours]
    out = [f"# Composite report - ours = {ours} (commits: {', '.join(sorted(commits))})", ""]

    out += ["## Composite vs each baseline (mean over scenarios, +0.10 = 10% better)", "",
            "| reference | A+B | C |", "|---|---|---|"]
    for ref in refs:
        cell_c = f"{_mean_score(results, ours, ref, c):+.3f}" if c else "-"
        out.append(f"| {ref} | {_mean_score(results, ours, ref, ab):+.3f} | {cell_c} |")

    if "ko" in refs:
        out += ["", "## Per scenario vs ko", "", "| scenario | composite |", "|---|---|"]
        out += [f"| {s} | {score(results, ours, 'ko', s):+.3f} |" for s in scen]

    out += ["", "## Average rank over 5 metrics (1 = best, weight-free)", "", "| method | rank |", "|---|---|"]
    ranks = {m: [] for m in methods}
    for s in scen:
        means = {m: _means(results, m, s) for m in methods}
        for k in WEIGHTS:
            order = sorted(methods, key=lambda m: means[m][k], reverse=k not in LOWER_BETTER)
            for i, m in enumerate(order):
                ranks[m].append(i + 1)
    out += [f"| {m} | {np.mean(r):.2f} |" for m, r in sorted(ranks.items(), key=lambda kv: np.mean(kv[1]))]

    out += ["", "## Weight sensitivity (each weight x0.5 / x1.5, renormalised; A+B)", "",
            "| reference | min | max |", "|---|---|---|"]
    for ref in refs:
        vals = []
        for k in WEIGHTS:
            for f in (0.5, 1.5):
                w = dict(WEIGHTS)
                w[k] *= f
                tot = sum(w.values())
                vals.append(_mean_score(results, ours, ref, ab, {kk: v / tot for kk, v in w.items()}))
        out.append(f"| {ref} | {min(vals):+.3f} | {max(vals):+.3f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ours", default="egat")
    a = ap.parse_args()
    print(report(a.paths, a.ours))


if __name__ == "__main__":
    main()
