"""Results figures from the composite reports: python docs/paper/figures/make_figures.py
Reads runs/v8/final and runs/v8/fresh/final; writes fig_*.pdf/png next to this file and prints A+B metric means."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AB = ["S1", "S2", "S3", "S4", "S5", "S6", "B1", "B2", "B3", "B4", "B5"]
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#8a8a8a"  # Okabe-Ito; validated, second series also hatched
NAMES = {"ko": "Ko", "li": "Li", "yuan": "Yuan", "m0": "M0", "q_neutral": "Q-neutral",
         "li_h": "Li+H", "yuan_h": "Yuan+H", "egat_h3": "EGAT+H (ours)", "egat": "EGAT (no H)"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True})


def tables(path):
    """{section heading: [row cells]} for every markdown table in a report."""
    out, head = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            head = line.lstrip("# ").strip()
        elif line.startswith("|") and not line.startswith("|---"):
            out.setdefault(head, []).append([c.strip() for c in line.strip("|").split("|")])
    return out


def num(s):
    return float(s.replace("+", ""))


def significance(path):
    rows = tables(path)[next(k for k in tables(path) if k.startswith("Significance"))]
    hdr = rows[0]
    i_mean, i_ci = hdr.index("seed-pair mean"), [j for j, h in enumerate(hdr) if h.startswith("95% t-interval")][0]
    res = {}
    for r in rows[1:]:
        lo, hi = re.findall(r"[+-]\d\.\d+", r[i_ci])[:2]
        res[r[0]] = (num(r[i_mean]), num(lo), num(hi))
    return res


def per_scenario(path):
    return {r[0]: num(r[1]) for r in tables(path)["Per scenario vs ko"][1:]}


def metric(path, name):
    rows = tables(path)[name]
    cols = rows[0][1:]
    return {r[0]: dict(zip(cols, map(float, r[1:]))) for r in rows[1:]}


def fig_main():
    refs = ["ko", "li", "yuan", "li_h", "yuan_h"]
    a = significance(ROOT / "runs/v8/final/composite_a_egat_h3_fair.md")
    f = significance(ROOT / "runs/v8/fresh/final/composite_a_egat_h3.md")
    x, w = np.arange(len(refs)), 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for k, (d, lab, col, hatch) in enumerate([(a, "test seeds 0-19", BLUE, None), (f, "fresh seeds 20-39", ORANGE, "///")]):
        m = np.array([d[r][0] for r in refs])
        err = np.array([[d[r][0] - d[r][1] for r in refs], [d[r][2] - d[r][0] for r in refs]])
        ax.bar(x + (k - 0.5) * w, m, w * 0.92, yerr=err, color=col, hatch=hatch, edgecolor="white",
               linewidth=0.5, capsize=2, error_kw={"elinewidth": 0.8}, label=lab)
    ax.axhline(0.10, color="black", ls="--", lw=0.8)
    ax.text(len(refs) - 0.5, 0.105, "+10% target", ha="right", va="bottom", fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [NAMES[r] for r in refs])
    ax.set_ylabel("Composite improvement")
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.93))
    fig.tight_layout()
    save(fig, "fig_main")


def fig_scenarios():
    a = per_scenario(ROOT / "runs/v8/final/composite_a_egat_h3_fair.md")
    b = per_scenario(ROOT / "runs/v8/final/composite_b_egat_h3_fair.md")
    x, w = np.arange(len(AB)), 0.38
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(x - w / 2, [a[s] for s in AB], w * 0.92, color=BLUE, label="setting (a)")
    ax.bar(x + w / 2, [b[s] for s in AB], w * 0.92, color=ORANGE, hatch="///", edgecolor="white", linewidth=0.5,
           label="setting (b): queue-preserving")
    ax.axhline(-0.05, color="black", ls="--", lw=0.8)
    ax.text(-0.5, -0.06, "-5% floor", va="top", fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, AB)
    ax.set_ylabel("Composite vs Ko")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "fig_scenarios")


def fig_switching():
    p = ROOT / "runs/v8/final/composite_a_egat_h3_fair.md"
    st, ag = metric(p, "stability"), metric(p, "aggregate")
    q = ROOT / "runs/v8/final/composite_a_egat.md"  # EGAT without hysteresis
    st["egat"], ag["egat"] = metric(q, "stability")["egat"], metric(q, "aggregate")["egat"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for m in ["ko", "li", "yuan", "li_h", "yuan_h", "egat", "egat_h3"]:
        s, g = np.mean([st[m][c] for c in AB]), np.mean([ag[m][c] for c in AB])
        ours = m.startswith("egat")
        ax.scatter(s, g, s=36 if ours else 22, color=BLUE if ours else GRAY, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(NAMES[m], (s, g), xytext=(4, -3 if m != "li_h" else -9), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([30, 100, 300, 600], ["30", "100", "300", "600"])
    ax.minorticks_off()
    ax.set_xlabel("Channel + route switches per episode (log)")
    ax.set_ylabel("Aggregate throughput (Mb/s)")
    fig.tight_layout()
    save(fig, "fig_switching")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def summary():
    p = ROOT / "runs/v8/final/composite_a_egat_h3_fair.md"
    for name in ("aggregate", "jain", "p5", "loss_rate", "stability"):
        t = metric(p, name)
        print(name, {m: round(float(np.mean([t[m][c] for c in AB])), 3) for m in t})


def decomposition():
    """Table 3: weighted contribution of each metric to the composite of EGAT+H vs each reference (A+B, seeds 0-19)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from mesh.eval.composite import COUNT_METRICS, LOWER_BETTER, WEIGHTS, delta
    p = ROOT / "runs/v8/final/composite_a_egat_h3_fair.md"
    t = {k: metric(p, k) for k in WEIGHTS}
    for ref in ("ko", "li", "yuan", "li_h", "yuan_h", "m0"):
        parts = {k: np.mean([w * delta(t[k]["egat_h3"][s], t[k][ref][s], k in LOWER_BETTER, k in COUNT_METRICS)
                             for s in AB]) for k, w in WEIGHTS.items()}
        print(ref, " ".join(f"{k} {v:+.3f}" for k, v in parts.items()), f"total {sum(parts.values()):+.3f}")


def score100(path, sources=18, nodes=20, steps=200):
    """Descriptive 0-100 score (not the pre-registered composite): each part is a fraction of its best possible value,
    averaged over A+B. Delivered traffic 45% (1 - loss; throughput and loss both measure it), Jain 20%,
    worst flows 20% (p5 / mean per-flow throughput), stability 15% (1 - switches / possible switches)."""
    m = {k: metric(path, k) for k in ("aggregate", "jain", "p5", "loss_rate", "stability")}
    out = {}
    for meth in m["aggregate"]:
        parts = []
        for s in AB:
            agg = m["aggregate"][meth][s]
            tail = min(m["p5"][meth][s] / (agg / sources), 1.0) if agg > 0 else 0.0
            parts.append([1 - m["loss_rate"][meth][s], m["jain"][meth][s], tail,
                          max(0.0, 1 - m["stability"][meth][s] / (nodes * steps))])
        a = np.mean(parts, axis=0)
        out[meth] = 100 * (0.45 * a[0] + 0.20 * a[1] + 0.20 * a[2] + 0.15 * a[3])
    return out


if __name__ == "__main__":
    fig_main(); fig_scenarios(); fig_switching(); summary(); decomposition()
    for f in ("runs/v8/final/composite_a_egat_h3_fair.md", "runs/v8/fresh/final/composite_a_egat_h3.md"):
        print("score100", f, {k: round(v, 1) for k, v in score100(ROOT / f).items()})
