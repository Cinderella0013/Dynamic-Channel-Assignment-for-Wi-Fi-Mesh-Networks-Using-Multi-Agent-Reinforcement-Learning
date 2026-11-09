"""EGAT-MAPPO architecture diagram: python docs/paper/figures/make_architecture.py -> fig_architecture.{pdf,png}

Numbered stages left to right, tensor shapes on every arrow (n = number of nodes).
Dimensions match mesh/model/egat.py (hidden 64, 4 heads, 2 layers) and mesh/rl/agents_io.py."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
MODEL, KO, EXEC, TRAIN = "#dbe9f6", "#fbe8c8", "#ececec", "#d9efe6"  # pale Okabe-Ito blue / orange / gray / green
EDGE = {MODEL: "#0072B2", KO: "#E69F00", EXEC: "#6e6e6e", TRAIN: "#009E73"}
plt.rcParams.update({"font.size": 6.6, "font.family": "serif", "mathtext.fontset": "dejavuserif"})


def box(ax, x, y, w, h, text, fill, bold=False, title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=EDGE[fill], lw=0.9))
    if title:
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontweight="bold", color=EDGE[fill])
        ax.text(x + w / 2, y + (h - 0.45) / 2, text, ha="center", va="center", linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontweight="bold" if bold else "normal",
                linespacing=1.3)


def arrow(ax, a, b, label=None, dashed=False, color="#333333", off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=1, shrinkB=1))
    if label:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], label, ha="center", va="bottom",
                fontsize=5.6, color="#444444")


def main():
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 9.0)
    ax.axis("off")

    # stage labels
    for x, t in [(1.35, "(1) Input"), (5.45, "(2) Encoder"), (9.6, "(3) Decision heads + (4) Ko prior"),
                 (13.75, "(5) Execution  ->  (6) Output")]:
        ax.text(x, 8.65, t, ha="center", fontsize=7.2, fontweight="bold")

    # (1) inputs, per node i / per edge (i, j)
    box(ax, 0.1, 5.5, 2.5, 2.6, "own state (18):\nload, unmet, backlog,\nhop, channel, parent\n"
        "+ 4 candidates (20)\n\n$x_i \\in \\mathbb{R}^{38}$", MODEL, title="Node features")
    box(ax, 0.1, 2.7, 2.5, 2.4, "link, conflict, SNR, rate,\nload, same channel,\ndelivery ratio\n\n"
        "$e_{ij} \\in \\mathbb{R}^{7}$", MODEL, title="Edge features")

    # (2) encoder (no recurrent memory: the GRU was removed, docs/paper/protocol_nogru.md)
    box(ax, 3.4, 3.4, 4.1, 4.0, "Linear 38 -> 64\n\nEdge-aware GATv2, 2 layers,\n4 heads x 16\n"
        "$s_{ij}=a^\\top\\sigma(W_s h_i+W_t h_j+W_e e_{ij})$\nattend over link + conflict\nneighbours and self\n\n"
        "residual + LayerNorm", MODEL, title="Shared GNN encoder")
    arrow(ax, (2.6, 6.8), (3.4, 6.2), "$n \\times 38$")
    arrow(ax, (2.6, 3.9), (3.4, 4.5), "$n \\times n \\times 7$", off=(0, -0.38))

    # (3) heads and (4) Ko prior
    box(ax, 8.15, 6.0, 2.9, 1.6, "MLP 64 -> 64 -> 4\nscore per parent slot\nevery step\n+ progress mask",
        MODEL, title="Route head")
    box(ax, 8.15, 2.6, 2.9, 1.6, "MLP 64 -> 64 -> 6\nscore per channel\nevery $t_c=10$ steps", MODEL,
        title="Channel head")
    box(ax, 8.15, 4.45, 2.9, 1.3, "logits $-$ softplus($\\beta$)\n$\\times\\ z(\\mathrm{cost}_{Ko})$, $\\beta$ learned",
        KO, title="Ko residual prior")
    arrow(ax, (7.5, 5.8), (8.15, 6.7), "$h_i \\in \\mathbb{R}^{64}$", off=(-0.35, 0.05))
    arrow(ax, (7.5, 4.9), (8.15, 3.5), "$h_i$", off=(-0.25, -0.1))
    arrow(ax, (9.6, 5.75), (9.6, 6.0), color=EDGE[KO])
    arrow(ax, (9.6, 4.45), (9.6, 4.2), color=EDGE[KO])

    # (5) execution and (6) output
    box(ax, 11.6, 3.2, 4.3, 4.4,
        "a. hysteresis: +m to current\n    parent and channel (m = 2)\n\n"
        "b. sequential channel decoding:\n    parents first, avoid channels\n    of decided conflict neighbours\n\n"
        "c. fallback to Ko if\n    entropy > 2 nats", EXEC, title="Decentralised execution")
    box(ax, 11.85, 1.5, 3.8, 1.0, "Action per node $i$:\n(downlink channel, parent)", EXEC, bold=True)
    arrow(ax, (11.05, 6.8), (11.6, 6.2), "$n \\times 4$")
    arrow(ax, (11.05, 3.4), (11.6, 4.0), "$n \\times 6$", off=(0, -0.38))
    arrow(ax, (13.75, 3.2), (13.75, 2.5))

    # training-only parts
    box(ax, 3.0, 0.3, 3.4, 1.55, "Behaviour cloning from Ko\nfirst 10k steps: actor CE,\ncritic on Ko returns",
        TRAIN)
    box(ax, 6.7, 0.3, 4.7, 1.55, "Centralised critic $V([h_i,\\ \\bar h])$\nMLP 128 -> 128 -> 1\n"
        "MAPPO: PPO-clip + GAE, shared actor\n$\\lambda$: 0.5 $\\to$ 0.1, EMA + validation", TRAIN)
    arrow(ax, (7.0, 3.4), (8.6, 1.85), dashed=True, color=EDGE[TRAIN])
    arrow(ax, (4.7, 1.85), (4.7, 3.4), dashed=True, color=EDGE[TRAIN])
    ax.text(0.15, 1.05, "training only\n(dashed)", fontsize=6, color=EDGE[TRAIN], va="center")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_architecture.{ext}", dpi=300)


if __name__ == "__main__":
    main()
