"""Figure 1: overall system framework. The test-time de-synchronised scheduling layer sits on a frozen
multi-robot explorer; it decides WHEN each robot broadcasts (phase) under a fixed rate, over a
range-limited lossy channel, which sets how stale teammates' shared belief is and hence travel/overlap.
The bottom strip shows the synchrony-tax causal chain. Run locally: py gen_system_fig.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import house_style as hs
hs.apply()
P = hs.PAL
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "mdpi_template", "figures")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(10.4, 4.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec, tcol="#222", fs=10, bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.5",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tcol, zorder=3, fontweight=("bold" if bold else "normal"))


def arrow(x1, y1, x2, y2, col=P["ink"], lw=2.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=col, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def tint(hexc, a):  # light fill from a palette colour
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hexc)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


# top row of pipeline boxes
y0, h = 56, 26
box(2,  y0, 21, h, "Frozen multi-robot\nexplorer\n(graph-attention policy\nplans on the shared belief)",
    tint(P["gray"], 0.12), P["gray"], fs=9.5)
box(28, y0, 23, h, "De-synchronised\nscheduling layer\n($\\bf{when}$ to broadcast,\nfixed per-robot rate)",
    tint(P["ow"], 0.16), P["ow"], fs=9.5, bold=False)
box(56, y0, 19, h, "Range-limited,\nlossy broadcast\nchannel",
    tint(P["desync"], 0.12), P["desync"], fs=9.5)
box(80, y0, 18, h, "Teammates' belief\n(fresh or stale,\nage $\\tau_i$)",
    tint(P["purple"], 0.12), P["purple"], fs=9.5)

arrow(23, y0 + h / 2, 28, y0 + h / 2)
arrow(51, y0 + h / 2, 56, y0 + h / 2)
arrow(75, y0 + h / 2, 80, y0 + h / 2)
ax.text(25.5, y0 + h / 2 + 4.5, "pose +\nembedding", ha="center", va="bottom", fontsize=7.5, color=P["gray"])

# "frozen" tag and "our contribution" tag
ax.text(12.5, y0 + h + 2.5, "frozen, pretrained", ha="center", fontsize=8.5, style="italic", color=P["gray"])
ax.text(39.5, y0 + h + 2.5, "our test-time layer", ha="center", fontsize=9, fontweight="bold", color=P["ow"])
# scheduler options
ax.text(39.5, y0 - 4.5, "synchronised  /  even-staggered  /  OW-Desync", ha="center", fontsize=8, color=P["ow"])

# feedback arrow: belief informs the next plan
arrow(89, y0, 12.5, y0, col=P["ink"], lw=1.6, rad=-0.32)
ax.text(50, 40.5, "the belief each robot acts on is only as fresh as the last update it received",
        ha="center", fontsize=8.5, style="italic", color=P["ink"])

# bottom: synchrony-tax causal chain
cy = 16
chain = ["limited\nbandwidth", "synchronised\nupdates", "correlated\nstale beliefs",
         "duplicated\nfrontiers", "overlap /\ntravel waste"]
cw, gap = 15.5, 3.0; x = 4
for k, t in enumerate(chain):
    fc = tint(P["sync"], 0.10) if k >= 1 else tint(P["gray"], 0.10)
    box(x, cy, cw, 13, t, fc, P["sync"] if k >= 1 else P["gray"], fs=8.2, lw=1.2)
    if k < len(chain) - 1:
        arrow(x + cw, cy + 6.5, x + cw + gap, cy + 6.5, col=P["sync"], lw=1.5)
    x += cw + gap
ax.text(50, cy - 5.5, "the synchrony tax;  de-synchronising the broadcast phase breaks the second link and removes it",
        ha="center", fontsize=8.5, color=P["sync"], fontweight="bold")

fig.suptitle("A test-time de-synchronised scheduling layer on a frozen multi-robot explorer",
             y=0.99, fontsize=12)
fig.tight_layout()
for d in [OUT, PAPER]:
    fig.savefig(os.path.join(d, "fig_system.png"), dpi=300, bbox_inches="tight")
plt.close(fig); print("saved fig_system")
