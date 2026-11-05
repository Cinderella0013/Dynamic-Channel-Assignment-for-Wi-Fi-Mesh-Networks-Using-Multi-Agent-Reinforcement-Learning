"""Pre-registered composite score (spec addendum 5.1): python -m mesh.eval.composite runs/v6/suite_*.json --ours egat"""
import argparse
import json

import numpy as np

from mesh.sim.scenarios import SUITE_A, SUITE_B, SUITE_C

WEIGHTS = {"aggregate": 0.25, "jain": 0.20, "p5": 0.20, "loss_rate": 0.20, "stability": 0.15}
LOWER_BETTER = {"loss_rate", "stability"}
COUNT_METRICS = {"stability"}  # counts that can be 0: smoothed with +1
FILE = "_file"  # per-run tag added by _load: index of the file (= training seed) the run came from


def delta(ours, ref, lower_better, count=True):
    """Relative improvement of ours over ref, clipped to [-1, 1]."""
    if lower_better and count:
        d = (ref - ours) / (abs(ref) + 1.0)  # +1 smooths count metrics that can be 0
    elif lower_better:
        d = (ref - ours) / max(abs(ref), 0.01)  # rates: plain relative change
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
    return sum(w * delta(ours[k], base[k], k in LOWER_BETTER, k in COUNT_METRICS) for k, w in weights.items())


def _ranks(values, lower_better):
    """Mid-ranks (1 = best); ties share the average of their positions."""
    v = np.asarray(values, dtype=float) * (1 if lower_better else -1)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks.tolist()


def _load(paths):
    results, commits = {}, set()
    for f, p in enumerate([paths] if isinstance(paths, str) else paths):
        d = json.load(open(p))
        commits.add(d["commit"])
        for method, by_s in d["results"].items():
            for scen, r in by_s.items():  # repeated method/scenario = another training seed: pool its runs
                cur = results.setdefault(method, {}).setdefault(scen, {})
                for k, v in r.items():
                    cur[k] = cur.get(k, []) + list(v)
                cur[FILE] = cur.get(FILE, []) + [f] * len(r["aggregate"])  # which training-seed file each run is from
    return results, commits


def holm(pvals, alpha=0.05):
    """Holm step-down over a family {name: p}: reject while sorted p_(i) <= alpha / (m - i)."""
    out, m = {k: False for k in pvals}, len(pvals)
    for i, (k, v) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        if v > alpha / (m - i):
            break
        out[k] = True
    return out


def _pick(r, idx):
    return {k: [v[i] for i in idx] for k, v in r.items()}


def significance(results, ours, ref, scenarios, n_boot=2000, alpha=0.05, seed=0):
    """Paired runs (same eval seed, same training-seed file), two per-scenario tests reported side by side:
    - Wilcoxon signed-rank on per-run composites (tests their median; noisy when a reference varies a lot run to run);
    - bootstrap of the pre-registered composite itself (computed on metric means, runs resampled with their pairs),
      two-sided bootstrap p.
    Both are Holm-corrected across the scenarios (spec 5.3); a scenario counts when it is rejected and the composite > 0.
    Also bootstrap CIs of the mean composite over the scenarios: "ci" resamples runs only; "ci_h" is hierarchical,
    resampling training-seed files of `ours` first (shared across scenarios), then runs within each drawn file."""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(seed)
    p, pb, boot, lo = {}, {}, np.zeros(n_boot), {}
    files = sorted(set(results[ours][scenarios[0]].get(FILE, [0])))
    draws = [rng.choice(files, size=len(files)) for _ in range(n_boot)]
    boot_h = np.zeros(n_boot)
    for s in scenarios:
        a, b = results[ours][s], results[ref][s]
        n = len(a["aggregate"])
        assert n == len(b["aggregate"]), f"{ours}/{ref} {s}: unpaired run counts {n} vs {len(b['aggregate'])}"
        per = [score({"o": {s: _pick(a, [i])}, "r": {s: _pick(b, [i])}}, "o", "r", s) for i in range(n)]
        p[s] = float(wilcoxon(per).pvalue) if np.any(per) else 1.0
        bs = np.array([score({"o": {s: _pick(a, idx)}, "r": {s: _pick(b, idx)}}, "o", "r", s)
                       for idx in (rng.integers(n, size=n) for _ in range(n_boot))])
        boot += bs
        tag = np.array(a.get(FILE, [0] * n))
        pos = {f: np.flatnonzero(tag == f) for f in files}
        for j, fs in enumerate(draws):
            idx = np.concatenate([rng.choice(pos[f], size=len(pos[f])) for f in fs])
            boot_h[j] += score({"o": {s: _pick(a, idx)}, "r": {s: _pick(b, idx)}}, "o", "r", s)
        lo[s] = float(np.percentile(bs, 100 * alpha / 2))
        pb[s] = min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))
    boot /= len(scenarios)
    boot_h /= len(scenarios)
    better = {s: score(results, ours, ref, s) > 0 for s in scenarios}
    rw, rb = holm(p, alpha), holm(pb, alpha)
    return {"p": p, "p_boot": pb, "boot_lo": lo,
            "frac_sig": sum(rw[s] and better[s] for s in scenarios) / len(scenarios),
            "frac_sig_boot": sum(rb[s] and better[s] for s in scenarios) / len(scenarios),
            "ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            "ci_h": (float(np.percentile(boot_h, 2.5)), float(np.percentile(boot_h, 97.5)))}


def seed_matrix_ci(results, ours, ref, scenarios, alpha=0.05):
    """Composite of every (ours training-seed file, ref training-seed file) pair, runs paired by eval seed
    within the pair; mean over the matrix and a t-interval whose variance counts both methods' training seeds
    (row-mean and column-mean variances, Welch-Satterthwaite df). Training seeds of different methods are
    independent, so no file pairing is privileged."""
    from scipy.stats import t as student_t
    fo = sorted(set(results[ours][scenarios[0]][FILE]))
    fr = sorted(set(results[ref][scenarios[0]][FILE]))
    m = np.zeros((len(fo), len(fr)))
    for s in scenarios:
        a, b = results[ours][s], results[ref][s]
        ta, tb = np.array(a[FILE]), np.array(b[FILE])
        for i, f in enumerate(fo):
            ia = np.flatnonzero(ta == f)
            for j, g in enumerate(fr):
                ib = np.flatnonzero(tb == g)
                assert len(ia) == len(ib), f"{ours}/{ref} {s}: files {f}/{g} have {len(ia)} vs {len(ib)} runs"
                m[i, j] += score({"o": {s: _pick(a, ia)}, "r": {s: _pick(b, ib)}}, "o", "r", s)
    m /= len(scenarios)
    no, nr = m.shape
    vo = m.mean(1).var(ddof=1) / no if no > 1 else 0.0
    vr = m.mean(0).var(ddof=1) / nr if nr > 1 else 0.0
    terms = [(v, n) for v, n in ((vo, no), (vr, nr)) if v > 0]
    df = (sum(v for v, _ in terms) ** 2 / sum(v * v / (n - 1) for v, n in terms)) if terms else max(no - 1, 1)
    half = student_t.ppf(1 - alpha / 2, df) * np.sqrt(vo + vr)
    return {"matrix": m, "mean": float(m.mean()), "ci": (float(m.mean() - half), float(m.mean() + half)),
            "df": float(df)}


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

    out += ["", "## Significance (A+B; per-scenario tests Holm-corrected at 0.05; bootstrap 95% CI of the mean)", "",
            "| reference | mean | 95% CI | 95% CI (training seeds resampled) | seed-pair mean "
            "| 95% t-interval (both methods' training seeds) | scenarios better (Wilcoxon, Holm) "
            "| scenarios better (bootstrap, Holm) |",
            "|---|---|---|---|---|---|---|---|"]
    sigs = {ref: significance(results, ours, ref, ab) for ref in refs}
    for ref, sg in sigs.items():
        sm = seed_matrix_ci(results, ours, ref, ab)
        out.append(f"| {ref} | {_mean_score(results, ours, ref, ab):+.3f} | [{sg['ci'][0]:+.3f}, {sg['ci'][1]:+.3f}] "
                   f"| [{sg['ci_h'][0]:+.3f}, {sg['ci_h'][1]:+.3f}] "
                   f"| {sm['mean']:+.3f} | [{sm['ci'][0]:+.3f}, {sm['ci'][1]:+.3f}] (df {sm['df']:.1f}) "
                   f"| {sg['frac_sig']:.0%} | {sg['frac_sig_boot']:.0%} |")
    out += ["", "Per-scenario tests treat runs as independent: they are conditional on the trained models and "
            "do not count training-seed variance; the t-interval does."]

    if "ko" in refs:
        sk = significance(results, ours, "ko", scen)
        out += ["", "## Per scenario vs ko", "", "| scenario | composite | p (Wilcoxon) | p (bootstrap) | bootstrap 2.5% |",
                "|---|---|---|---|---|"]
        out += [f"| {s} | {score(results, ours, 'ko', s):+.3f} | {sk['p'][s]:.4f} | {sk['p_boot'][s]:.4f} "
                f"| {sk['boot_lo'][s]:+.3f} |" for s in scen]

    out += ["", "## Average rank over 5 metrics (1 = best, weight-free)", "", "| method | rank |", "|---|---|"]
    ranks = {m: [] for m in methods}
    for s in scen:
        means = {m: _means(results, m, s) for m in methods}
        for k in WEIGHTS:
            for m, r in zip(methods, _ranks([means[m][k] for m in methods], k in LOWER_BETTER)):
                ranks[m].append(r)
    out += [f"| {m} | {np.mean(r):.2f} |" for m, r in sorted(ranks.items(), key=lambda kv: np.mean(kv[1]))]

    out += ["", "## Per-metric means (5 metrics x method x scenario)"]
    for k in WEIGHTS:
        out += ["", f"### {k}", "", "| method | " + " | ".join(scen) + " |", "|---" * (len(scen) + 1) + "|"]
        out += ["| " + m + " | " + " | ".join(f"{_means(results, m, s)[k]:.3f}" for s in scen) + " |"
                for m in methods if all(s in results[m] for s in scen)]

    out += ["", "## Weight sensitivity (each weight x0.5 / x1.5, renormalised; A+B)", "",
            "| reference | min | max | +10% holds for all weights |", "|---|---|---|---|"]
    for ref in refs:
        vals = []
        for k in WEIGHTS:
            for f in (0.5, 1.5):
                w = dict(WEIGHTS)
                w[k] *= f
                tot = sum(w.values())
                vals.append(_mean_score(results, ours, ref, ab, {kk: v / tot for kk, v in w.items()}))
        out.append(f"| {ref} | {min(vals):+.3f} | {max(vals):+.3f} | {'yes' if min(vals) >= 0.10 else 'no'} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ours", default="egat")
    a = ap.parse_args()
    print(report(a.paths, a.ours))


if __name__ == "__main__":
    main()
