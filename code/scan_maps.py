"""Scan the map set for free-area (arena size) distribution, to find harder/bigger maps."""
import glob, os, sys
import numpy as np
from skimage import io, color, util
from skimage.measure import block_reduce

maps = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "maps", "*.png")))
n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 400
step = max(1, len(maps) // n_sample)
samp = maps[::step]
rows = []
for p in samp:
    raw = io.imread(p)
    if raw.ndim == 3:
        if raw.shape[2] == 4:
            raw = color.rgba2rgb(raw)
        g = util.img_as_ubyte(color.rgb2gray(raw)).astype(int)
    else:
        g = raw.astype(int)
    g = block_reduce(g, 2, np.min)
    free = int((g > 150).sum())
    rows.append((free, g.shape, os.path.basename(p)))
rows.sort(reverse=True)
a = np.array([r[0] for r in rows])
print(f"sampled {len(samp)} of {len(maps)} maps")
print(f"free-cell area: min={a.min()} p25={int(np.percentile(a,25))} median={int(np.median(a))} "
      f"p75={int(np.percentile(a,75))} max={a.max()}  (ratio max/median={a.max()/max(1,np.median(a)):.1f}x)")
print("dims seen:", set(r[1] for r in rows))
print("top 12 biggest:", [(r[0], r[2]) for r in rows[:12]])
print("smallest 5:", [(r[0], r[2]) for r in rows[-5:]])
