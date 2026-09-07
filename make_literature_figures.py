"""
Regenerates the two literature-derived supplementary figures:

    Images/Chart 2.png  - reported efficiency improvements across domains
    Images/Chart 4.png  - multi-agent coordination results (Ripple Effect Protocol)

Every value below is quoted from a cited source and is annotated with that source.
No value in this file is estimated, interpolated, or illustrative. Figures follow the
project figure style specification (9 to 9.5 pt text, flat 2D, print-safe palette,
300 dpi, true print width).

Sources
-------
anthropic2025code  Anthropic, "Code execution with MCP". 150,000 tokens -> 2,000
                   tokens, a 98.7 percent reduction.
wu2025instructmpc  Wu, Ai and Li, "InstructMPC", IEEE CDC 2025. 58.29 percent
                   electricity cost reduction under predictive control.
li2025energyplus   Li, Xu and Hong, "EnergyPlus-MCP", SoftwareX 32:102367.
                   30 percent reduction in interior lighting energy.
catchmetrics2025   Catch Metrics, "A brief introduction to MCP server performance
                   optimisation". Up to 100x faster cached responses, that is a
                   99 percent reduction in response time.
chopra2025ripple   Chopra et al., "Ripple Effect Protocol", arXiv:2510.16572.
                   Total supply chain cost 7,300 (A2A) -> 4,251 (REP), 41.8 percent.
                   At 200 agents: sensitivity sharing 3 percent of runtime,
                   LLM inference 38 percent, wait time 59 percent.
"""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Project figure style (Part 3 of the writing instructions)
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#2563EB"
ORANGE = "#E8710A"
GREEN = "#059669"
GREY = "#6B7280"
TEXT = "#111827"
GRID = "#E5E7EB"

# IEEE double-column text width is 7.16 in. The .tex includes these figures at
# 0.74 and 0.71 of \textwidth, so design at exactly that width.
W_CHART2 = 0.74 * 7.16
W_CHART4 = 0.71 * 7.16


def _style_axes(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT, length=3, width=0.8)


def chart2_reported_improvements():
    """Reported efficiency improvements across domains, all as percentage reductions."""
    labels = [
        "Response time,\ncached (Catch Metrics)",
        "Interior lighting energy\n(EnergyPlus-MCP)",
        "Electricity cost,\npredictive control (InstructMPC)",
        "Token consumption,\ncomplex task (Anthropic)",
    ]
    values = [99.0, 30.0, 58.29, 98.7]
    notes = ["99% (100x faster)", "30%", "58.29%", "98.7%"]
    # Same series colours as the rest of the paper; lightness varies for greyscale.
    colors = [GREY, GREEN, ORANGE, BLUE]

    fig, ax = plt.subplots(figsize=(W_CHART2, 2.55))
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.62, color=colors, edgecolor="none")

    for yi, v, note in zip(y, values, notes):
        ax.text(v + 1.5, yi, note, va="center", ha="left",
                fontsize=9, color=TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=TEXT)
    ax.set_xlabel("Reduction reported by the cited source (percent)", color=TEXT)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    _style_axes(ax, grid_axis="x")

    fig.tight_layout()
    fig.savefig("Images/Chart 2.png", bbox_inches="tight", pad_inches=0.05, dpi=300,
                facecolor="white")
    fig.savefig("Images/Chart 2.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote Images/Chart 2.png")


def chart4_multi_agent_coordination():
    """Ripple Effect Protocol results: supply chain cost, and runtime composition."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_CHART4, 2.45))

    # ---- Left: total supply chain cost, A2A vs REP -------------------
    methods = ["A2A\nbaseline", "Ripple Effect\nProtocol"]
    costs = [7300, 4251]
    bars = ax1.bar(methods, costs, width=0.55, color=[GREY, BLUE],
                   edgecolor="none")
    ax1.text(bars[0].get_x() + bars[0].get_width() / 2, costs[0] + 200,
             f"{costs[0]:,}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax1.text(bars[1].get_x() + bars[1].get_width() / 2, costs[1] + 200,
             f"{costs[1]:,}\n41.8% lower", ha="center", va="bottom",
             fontsize=9, color=ORANGE)

    ax1.set_ylabel("Total supply chain cost", color=TEXT)
    ax1.set_ylim(0, 9400)
    ax1.set_title("Supply chain coordination", color=TEXT, pad=6)
    _style_axes(ax1, grid_axis="y")

    # ---- Right: runtime composition at 200 agents --------------------
    parts = ["Wait\ntime", "LLM\ninference", "Sensitivity\nsharing"]
    share = [59, 38, 3]
    colors = [GREY, BLUE, GREEN]
    y = np.arange(len(parts))
    ax2.barh(y, share, height=0.6, color=colors, edgecolor="none")
    for yi, s in zip(y, share):
        ax2.text(s + 1.5, yi, f"{s}%", va="center", ha="left",
                 fontsize=9, color=TEXT)

    ax2.set_yticks(y)
    ax2.set_yticklabels(parts, fontsize=9, color=TEXT)
    ax2.set_xlabel("Share of total runtime (percent)", color=TEXT)
    ax2.set_xlim(0, 76)
    ax2.set_title("Runtime composition at 200 agents", color=TEXT, pad=6)
    _style_axes(ax2, grid_axis="x")

    fig.tight_layout()
    fig.savefig("Images/Chart 4.png", bbox_inches="tight", pad_inches=0.05, dpi=300,
                facecolor="white")
    fig.savefig("Images/Chart 4.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote Images/Chart 4.png")


if __name__ == "__main__":
    chart2_reported_improvements()
    chart4_multi_agent_coordination()
