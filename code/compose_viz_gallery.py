"""Compose the synchronised-vs-de-synchronised qualitative gallery (replaces the off-message
rich-vs-frozen compare_m0). Reads the six rendered trajectory frames in jeit/_vizframes, crops the
baked title, and lays them out 2 rows (sync / de-sync) x 3 example maps with unified labels.
Run locally: `py compose_viz_gallery.py`.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import house_style as hs
hs.apply()
P = hs.PAL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FR = os.path.join(ROOT, "_vizframes")
PAPER = os.path.join(ROOT, "paper", "mdpi_template", "figures")
OUT = os.path.join(HERE, "results", "figures")
os.makedirs(OUT, exist_ok=True)

# (map label, travel sync, cov sync, travel desync, cov desync); de-sync wins (travel down, cov >= sync)
MAPS = [
    ("1", 133, 99.2, 127, 99.5),
    ("3", 125, 95.4, 113, 96.0),
    ("8", 121, 95.3,  95, 95.0),
]


def load_crop(name):
    im = mpimg.imread(os.path.join(FR, name))
    if im.dtype != np.uint8 and im.max() <= 1.0:
        im = (im * 255).astype(np.uint8)
    g = im[..., :3].mean(-1) if im.ndim == 3 else im
    # crop the top title strip (~13%) then autocrop white margins
    h = g.shape[0]
    im = im[int(0.125 * h):]
    g = g[int(0.125 * h):]
    rows = np.where((g < 248).any(1))[0]
    cols = np.where((g < 248).any(0))[0]
    if len(rows) and len(cols):
        im = im[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return im


ncol = len(MAPS)
fig, axes = plt.subplots(2, ncol, figsize=(3.7 * ncol, 7.2))
if ncol == 1:
    axes = axes.reshape(2, 1)
for j, (mid, ts, cs, td, cd) in enumerate(MAPS):
    for i, (mode, fn, tr, cov, col) in enumerate([
        ("Synchronised", f"m{mid}_sync.png", ts, cs, P["sync"]),
        ("De-synchronised", f"m{mid}_desync.png", td, cd, P["desync"]),
    ]):
        ax = axes[i, j]
        ax.imshow(load_crop(fn)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_edgecolor(col); s.set_linewidth(2.4)
        ax.set_xlabel(f"travel {tr},  coverage {cov:.0f}%", color=col, fontsize=10, labelpad=4)
    red = 100 * (td - ts) / ts
    axes[0, j].set_title(f"Example map {mid}\n(de-sync travel {red:+.0f}%)", fontsize=11)
# row labels
for i, lab, col in [(0, "Synchronised", P["sync"]), (1, "De-synchronised", P["desync"])]:
    axes[i, 0].text(-0.07, 0.5, lab, transform=axes[i, 0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=12, fontweight="bold", color=col)
fig.suptitle("Synchronised vs de-synchronised exploration at matched bandwidth "
             "($N=8$, broadcast period $T=12$, three example maps)", y=1.0, fontsize=12)
fig.tight_layout()
for d in [OUT, PAPER]:
    fig.savefig(os.path.join(d, "fig_syncviz.png"), dpi=200)
plt.close(fig)
print("saved fig_syncviz.png")
