"""Symlink the top-K largest-free-area maps into maps_big/ (hardest available subset)."""
import glob, os, sys, shutil
import numpy as np
from skimage import io, color, util
from skimage.measure import block_reduce

K = int(sys.argv[1]) if len(sys.argv) > 1 else 25
root = os.path.join(os.path.dirname(__file__), "..")
maps = sorted(glob.glob(os.path.join(root, "maps", "*.png")))
areas = []
for p in maps:
    raw = io.imread(p)
    if raw.ndim == 3:
        if raw.shape[2] == 4:
            raw = color.rgba2rgb(raw)
        g = util.img_as_ubyte(color.rgb2gray(raw)).astype(int)
    else:
        g = raw.astype(int)
    g = block_reduce(g, 2, np.min)
    areas.append((int((g > 150).sum()), p))
areas.sort(reverse=True)
out = os.path.join(root, "maps_big")
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
for free, p in areas[:K]:
    os.symlink(p, os.path.join(out, os.path.basename(p)))
print(f"maps_big: {K} maps, free-area {areas[K-1][0]}..{areas[0][0]} (median all={int(np.median([a for a,_ in areas]))})")
