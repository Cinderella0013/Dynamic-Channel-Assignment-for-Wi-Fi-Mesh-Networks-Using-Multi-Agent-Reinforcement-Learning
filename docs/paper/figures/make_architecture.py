"""EGAT-MAPPO architecture diagram: python docs/paper/figures/make_architecture.py -> fig_architecture.{pdf,png}"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
MODEL, KO, EXEC, TRAIN = "#dbe9f6", "#fbe8c8", "#ececec", "#d9efe6"  # pale Okabe-Ito blue / orange / gray / green
EDGE = {MODEL: "#0072B2", KO: "#E69F00", EXEC: "#6e6e6e", TRAIN: "#009E73"}
plt.rcParams.update({"font.size": 6.6, "font.family": "serif", "mathtext.fontset": "dejavuserif"})


def box(ax, x, y, w, h, text, fill, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=EDGE[fill], lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontweight="bold" if bold else "normal",
            linespacing=1.3)


def arrow(ax, a, b, dashed=False, color="#333333"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # inputs
    box(ax, 0.1, 4.6, 2.5, 1.6, "Node features\nload, unmet, backlog,\nhop, channel, parent\n+ 4 parent candidates", MODEL)
    box(ax, 0.1, 2.6, 2.5, 1.5, "Edge features $e_{ij}$\nlink, conflict, SNR,\nrate, load, same ch.,\ndelivery ratio", MODEL)
    # encoder and memory
    box(ax, 3.2, 3.0, 2.6, 2.8, "Edge-aware\nGATv2 x 2\n\n$s_{ij}=a^\\top\\sigma(W_s h_i$\n$+\\,W_t h_j+W_e e_{ij})$"
        "\n\nresidual + LayerNorm", MODEL)
    box(ax, 6.35, 3.6, 1.3, 1.6, "GRU\nmemory\n$h_i$", MODEL)
    arrow(ax, (2.6, 5.4), (3.2, 4.9))
    arrow(ax, (2.6, 3.35), (3.2, 3.9))
    arrow(ax, (5.8, 4.4), (6.35, 4.4))

    # heads and Ko prior
    box(ax, 8.25, 5.0, 2.6, 1.3, "Route head\n4 parent slots\n(every step)", MODEL)
    box(ax, 8.25, 2.5, 2.6, 1.3, "Channel head\n6 channels\n(every $t_c=10$ steps)", MODEL)
    box(ax, 8.25, 3.95, 2.6, 0.85, "Ko residual prior\n$-\\,\\mathrm{softplus}(\\beta)\\,z(\\mathrm{cost}_{Ko})$", KO)
    arrow(ax, (7.65, 4.6), (8.25, 5.5))
    arrow(ax, (7.65, 4.2), (8.25, 3.2))
    arrow(ax, (9.55, 4.8), (9.55, 5.0), color=EDGE[KO])
    arrow(ax, (9.55, 3.95), (9.55, 3.8), color=EDGE[KO])
    ax.text(9.55, 6.45, "progress mask: parent strictly closer to a gateway", ha="center", fontsize=6, style="italic")

    # execution
    box(ax, 11.45, 3.0, 3.65, 3.3,
        "Decentralised execution\n\n1. hysteresis: +m on current\n    parent and channel (m = 3)\n"
        "2. sequential channel decoding\n    (parents first)\n3. fallback to Ko if\n    entropy > 2 nats", EXEC)
    box(ax, 11.6, 1.7, 3.35, 0.85, "Action per node:\n(downlink channel, parent)", EXEC, bold=True)
    arrow(ax, (10.85, 5.6), (11.45, 5.2))
    arrow(ax, (10.85, 3.2), (11.45, 3.8))
    arrow(ax, (13.27, 3.0), (13.27, 2.55))

    # training-only parts
    box(ax, 3.0, 0.35, 3.3, 1.35, "Behaviour cloning from Ko\n(first 10k steps): actor CE,\ncritic on Ko returns", TRAIN)
    box(ax, 6.8, 0.35, 4.05, 1.35, "Centralised critic $V([h_i,\\ \\bar h])$\nMAPPO: PPO-clip + GAE,\n"
        "$\\lambda$: 0.5 $\\to$ 0.1, EMA + validation", TRAIN)
    arrow(ax, (7.0, 3.6), (8.6, 1.7), dashed=True, color=EDGE[TRAIN])
    arrow(ax, (4.6, 1.7), (4.6, 3.0), dashed=True, color=EDGE[TRAIN])
    ax.text(0.15, 0.95, "training only\n(dashed)", fontsize=6, color=EDGE[TRAIN], va="center")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_architecture.{ext}", dpi=300)


if __name__ == "__main__":
    main()
