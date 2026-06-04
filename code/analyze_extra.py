"""Extra rigor analyses on EXISTING data (no new runs):
 - bootstrap 95% CIs on the headline paired comparisons (sync, learned),
 - exploration-speed proxy (episode_steps; fewer = faster at saturated coverage).
Writes results/summary/setcomm_bootstrap_ci.csv and prints a table."""
import os, glob, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)


def load(g):
    fs = glob.glob(os.path.join(HERE, "results", "raw", g, "*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio", "episode_steps", "map_id", "seed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def boot_ci(diff, nb=10000):
    diff = np.asarray(diff, float); diff = diff[~np.isnan(diff)]
    if len(diff) < 4:
        return (float("nan"),) * 3
    idx = RNG.integers(0, len(diff), size=(nb, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(d, akey, bkey, col):
    A = d[d["_k"] == akey].set_index(["map_id", "seed"])[col]
    B = d[d["_k"] == bkey].set_index(["map_id", "seed"])[col]
    idx = A.index.intersection(B.index)
    return (A.loc[idx] - B.loc[idx]).values


rows = []

# sync isolation: staggered - synchronised at each interval (travel & steps)
d = load("setcomm_sync")
if d is not None:
    d["_k"] = d["method_name"]
    for iv in [5, 8, 12]:
        for col, lab in [("total_travel_distance", "travel"), ("episode_steps", "steps")]:
            diff = paired(d, f"Stag_{iv}", f"Sync_{iv}", col)
            m, lo, hi = boot_ci(diff)
            rows.append(dict(comparison=f"Stag-Sync int{iv}", metric=lab, mean_diff=m, ci_lo=lo, ci_hi=hi, n=len(diff)))

# learned vs baselines (held-out maps 10-19) travel & steps
d = load("setcomm_learned")
if d is not None:
    d = d[d["map_id"] >= 10].copy()
    d["_k"] = d["method_name"]
    for B in [2, 4]:
        for base in ["VoI", "RoundRobin", "Random"]:
            for col, lab in [("total_travel_distance", "travel"), ("episode_steps", "steps")]:
                diff = paired(d, f"Learned_B{B}", f"{base}_B{B}", col)
                m, lo, hi = boot_ci(diff)
                rows.append(dict(comparison=f"Learned-{base} B{B}", metric=lab, mean_diff=m, ci_lo=lo, ci_hi=hi, n=len(diff)))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "results", "summary", "setcomm_bootstrap_ci.csv"), index=False)
pd.set_option("display.width", 160)
print("=== bootstrap 95% CIs on paired differences (negative = ours better) ===")
print(out.to_string(index=False))
