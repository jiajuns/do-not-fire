"""Full TARGETED/pairwise-communication run for the 'who-to-tell under a hard
bandwidth cap' contribution.

Eval-only sweep over per-step link budget B x scheduling mode x seeds, on the hardest
maps, N=8, targeted comm (point-to-point position+embedding under a hard per-step cap).
Tests whether value-of-information targeting (belief_error x proximity) beats blind
round-robin / random allocation at matched budget -- strongest at the scarcest B.

modes: voi (ours) | err, prox, stale (ablations) | round_robin, random (baselines)
plus a small sigma ablation for voi. Parallel, wall-clock budget capped. Group
'setcomm_targeted'. Produces a (mode x B) pivot and paired Wilcoxon stats (voi vs
round_robin/random at each B, on travel & overlap).
"""

import os
import sys
import time
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
MAPS_TEST = os.path.join(ROOT, "maps_test")
GROUP = "setcomm_targeted"
OUT = os.path.join(HERE, "results", "overnight")

WORKERS = int(os.environ.get("TGT_WORKERS", "16"))
SEEDS = [int(x) for x in os.environ.get("TGT_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
NUM_MAPS = int(os.environ.get("TGT_NUM_MAPS", "20"))
BS = [int(x) for x in os.environ.get("TGT_BS", "1,2,3,4,6,8,12,16").split(",")]
N_AGENTS = int(os.environ.get("TGT_N_AGENTS", "8"))
SIGMA = float(os.environ.get("TGT_SIGMA", "80"))
VOI_FLOOR = float(os.environ.get("TGT_VOI_FLOOR", "0.5"))
BUDGET_SEC = int(os.environ.get("TGT_BUDGET_MIN", "420")) * 60


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "targeted.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd):
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def specs():
    """Return list of (method_name, mode, sigma, floor, B)."""
    out = []
    core_modes = ["voi", "voi_lif", "err", "prox", "stale", "round_robin", "random"]
    for b in BS:
        for m in core_modes:
            fl = VOI_FLOOR if m == "voi_lif" else 0.0
            out.append((f"T_{m}_B{b}", m, SIGMA, fl, b))
    # sigma ablation for voi (only the proximity-weighted mode depends on sigma)
    for b in [2, 4, 8]:
        for sg in [40.0, 160.0]:
            out.append((f"T_voi_s{sg:g}_B{b}", "voi", sg, 0.0, b))
    return out


def job_cmd(method, mode, sigma, floor, b, seed):
    return [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", "periodic",
            "--seed", str(seed), "--num-maps", str(NUM_MAPS), "--group", GROUP,
            "--map-dir", MAPS_TEST, "--device", "cpu", "--n-agents", str(N_AGENTS),
            "--targeted", "--link-budget", str(b), "--target-mode", mode,
            "--voi-sigma", str(sigma), "--voi-floor", str(floor)]


def csv_exists(method, seed):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{method}_seed{seed}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(MAPS_TEST):
        log(f"ERROR {MAPS_TEST} missing")
        return 1
    t0 = time.time()
    jobs = []
    for (nm, mode, sigma, fl, b) in specs():
        for s in SEEDS:
            if not csv_exists(nm, s):
                jobs.append((nm, mode, sigma, fl, b, s))
    log(f"targeted: {len(specs())} specs x {len(SEEDS)} seeds = {len(jobs)} jobs, "
        f"B={BS}, N={N_AGENTS}, sigma={SIGMA}, {WORKERS} workers, budget {BUDGET_SEC//60}min")

    done = [0]
    total = len(jobs)

    def one(j):
        nm, mode, sigma, fl, b, s = j
        code, out = run(job_cmd(nm, mode, sigma, fl, b, s))
        done[0] += 1
        if code != 0:
            log(f"  FAIL {nm} seed{s}: {out[-200:]}")
        elif done[0] % 25 == 0:
            log(f"  {done[0]}/{total} (elapsed {int((time.time()-t0)/60)}min)")
        return code

    ex = ThreadPoolExecutor(max_workers=WORKERS)
    futures = set()
    i = 0
    while i < len(jobs) or futures:
        while len(futures) < WORKERS and i < len(jobs) and time.time() - t0 < BUDGET_SEC:
            futures.add(ex.submit(one, jobs[i]))
            i += 1
        if not futures:
            break
        _, futures = wait(futures, return_when=FIRST_COMPLETED)
        if time.time() - t0 >= BUDGET_SEC and i < len(jobs):
            log(f"BUDGET reached; launched {i}/{len(jobs)}, draining {len(futures)}")
    ex.shutdown(wait=True)
    log(f"eval done/stopped: {i}/{len(jobs)} launched in {int((time.time()-t0)/60)}min")

    run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
         "--group", GROUP, "--output_dir", "results/summary"])

    _analyze()
    open(os.path.join(OUT, "TARGETED_DONE"), "w").close()
    log(f"TARGETED DONE in {(time.time()-t0)/60:.1f} min")
    return 0


def _analyze():
    """(mode x B) pivot + paired Wilcoxon: voi vs round_robin/random at each B."""
    try:
        import glob
        import re
        import pandas as pd
        import numpy as np
        from scipy import stats
        fs = glob.glob(f"results/raw/{GROUP}/*.csv")
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        for c in ["final_coverage_ratio", "total_travel_distance", "overlap_ratio",
                  "total_communication_volume", "episode_steps", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        def parse(nm):
            mm = re.match(r"T_(voi_lif|voi|err|prox|stale|round_robin|random)(?:_s[\d.]+)?_B(\d+)$", str(nm))
            if not mm:
                return (None, None, None)
            sg = re.search(r"_s([\d.]+)_", str(nm))
            return (mm.group(1), int(mm.group(2)), float(sg.group(1)) if sg else SIGMA)
        d["mode"], d["B"], d["sigma"] = zip(*d.method_name.map(parse))
        d = d.dropna(subset=["mode", "B"])

        # main pivot: only the default-sigma rows
        main = d[(d.sigma == SIGMA)]
        piv = main.groupby(["B", "mode"]).agg(
            comm=("total_communication_volume", "mean"),
            travel=("total_travel_distance", "mean"),
            overlap=("overlap_ratio", "mean"),
            cov=("final_coverage_ratio", "mean"),
            steps=("episode_steps", "mean"),
            n=("total_travel_distance", "size")).reset_index().sort_values(["B", "mode"])
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_targeted_pivot.csv"), index=False)
        log("targeted pivot (mode x B):\n" + piv.to_string(index=False))

        rows = []
        for b in sorted(main.B.unique()):
            sub = main[main.B == b]
            for ours in ["voi", "voi_lif"]:
                oo = sub[sub.mode == ours].set_index(["map_id", "seed"])
                for base in ["round_robin", "random"]:
                    bb = sub[sub.mode == base].set_index(["map_id", "seed"])
                    idx = oo.index.intersection(bb.index)
                    if len(idx) < 8:
                        continue
                    for met in ["total_travel_distance", "overlap_ratio"]:
                        a = oo.loc[idx, met].values
                        c = bb.loc[idx, met].values
                        try:
                            _, p = stats.wilcoxon(a, c)
                        except Exception:
                            p = float("nan")
                        rows.append({"B": int(b), "ours": ours, "metric": met, "baseline": base,
                                     "ours_mean": float(a.mean()), "base_mean": float(c.mean()),
                                     "diff_pct": float(100 * (a.mean() - c.mean()) / c.mean()),
                                     "ours_better_frac": float((a < c).mean()),
                                     "n_pairs": int(len(idx)), "wilcoxon_p": float(p)})
        ps = pd.DataFrame(rows)
        ps.to_csv(os.path.join(HERE, "results", "summary", "setcomm_targeted_paired.csv"), index=False)
        log("targeted paired stats (voi vs baselines):\n" + ps.to_string(index=False))
    except Exception as e:
        import traceback
        log(f"analyze failed: {e}\n{traceback.format_exc()[-500:]}")


if __name__ == "__main__":
    sys.exit(main())
