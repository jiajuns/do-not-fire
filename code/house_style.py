"""Unified publication house style for all Sensors figures (old + new + OW-Desync).
Import and call apply() at the top of every figure generator so the whole paper is visually
consistent: clean despined axes, subtle grid, sans-serif, 300 dpi, and one semantic palette.
"""
import matplotlib.pyplot as plt

# One semantic palette used everywhere.
PAL = {
    "desync": "#2E5EAA",   # our de-synchronisation principle (good)
    "sync":   "#C0504D",   # synchronised baseline / "bad"
    "ow":     "#16A085",   # OW-Desync (the new method)
    "purple": "#7A4FA0",   # secondary / third series
    "accent": "#E8973A",   # warm accent (e.g. "de-sync hurts")
    "gray":   "#8A8A8A",   # neutral reference
    "ink":    "#33373D",   # axis / text ink
}
# convenience ordered cycle
CYCLE = [PAL["desync"], PAL["purple"], PAL["ow"], PAL["accent"], PAL["sync"], PAL["gray"]]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.titleweight": "semibold",
        "axes.linewidth": 1.0, "axes.edgecolor": PAL["ink"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "axes.grid.axis": "y",
        "grid.color": "#C7CCD4", "grid.linewidth": 0.7, "grid.alpha": 0.45,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
        "text.color": PAL["ink"], "axes.labelcolor": PAL["ink"],
        "legend.fontsize": 9, "legend.frameon": False,
        "lines.linewidth": 2.2, "lines.markersize": 6.5,
        "lines.markeredgecolor": "white", "lines.markeredgewidth": 0.8,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=CYCLE),
    })
