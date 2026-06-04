"""Cross-backbone synchronisation-isolation (generality of the Pillar-A finding).

Runs the decisive control -- SYNCHRONISED periodic vs STAGGERED periodic at the SAME interval
(identical per-robot rate AND total comm volume; only the broadcast phase differs) -- on THREE
different planners that all share the same env, node graph and communication-scheduling layer:
  dars  : the learned attention policy (DARS-2024 frozen backbone)
  fnear : classical Yamauchi nearest-frontier explorer
  futil : classical highest-utility (info-greedy) frontier explorer
If staggered beats synchronised on ALL THREE, "synchronisation is the controllable cost" is a
general principle of multi-robot communication, not an artifact of the learned policy.

Group setcomm_3bb. Paired Wilcoxon (staggered vs synchronised) per backbone x interval, on
travel and overlap. Wall-clock budget capped (overnight-safe).
"""
import os
import sys
import time
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
MAPS = os.path.join(ROOT, "maps_test")
GROUP = "setcomm_3bb"
INTERVALS = [int(x) for x in os.environ.get("TBB_INTERVALS", "5,8,12").split(",")]
SEEDS = list(range(int(os.environ.get("TBB_SEEDS", "5"))))
N = int(os.environ.get("TBB_N", "8"))
NUM = int(os.environ.get("TBB_NUM_MAPS", "20"))
WORKERS = int(os.environ.get("TBB_WORKERS", "5"))
BUDGET_SEC = int(os.environ.get("TBB_BUDGET_MIN", "560")) * 60

# (label, per-job env overrides) -- the only thing that changes is the "decide where to go" rule
BACKBONES = [
    ("dars", {}),
    ("fnear", {"BLMRE_FRONTIER": "1", "BLMRE_FRONTIER_MODE": "nearest"}),
    ("futil", {"BLMRE_FRONTIER": "1", "BLMRE_FRONTIER_MODE": "utility"}),
]


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    method, mode, iv, s, bbenv = j
    env = dict(os.environ)
    env["BLMRE_GATE_POSITIONS"] = "1"  # broadcast model: position sharing is gated
    env.update(bbenv)
    cmd = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", mode, "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", str(N), "--interval", str(iv)]
    return subprocess.run(cmd, cwd=HERE, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    with open(os.path.join(HERE, "results", "overnight", "threebb.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    jobs = []
    for bb, bbenv in BACKBONES:
        for iv in INTERVALS:
            for mode, tag in [("periodic", "Sync"), ("staggered", "Stag")]:
                m = f"{bb}_{tag}_{iv}"
                for s in SEEDS:
                    if not csv_ok(m, s):
                        jobs.append((m, mode, iv, s, bbenv))
    t0 = time.time()
    log(f"3bb: {len(BACKBONES)} backbones x intervals {INTERVALS} x sync/stag x {len(SEEDS)} seeds "
        f"= {len(jobs)} jobs, {WORKERS} workers, budget {BUDGET_SEC//60}min")
    ex = ThreadPoolExecutor(max_workers=WORKERS)
    futs, i, done = [], 0, [0]

    def wrap(j):
        rc = one(j)
        done[0] += 1
        if done[0] % 10 == 0:
            log(f"  {done[0]}/{len(jobs)} ({int((time.time()-t0)/60)}min)")
        return rc

    for j in jobs:
        if time.time() - t0 > BUDGET_SEC:
            log(f"BUDGET reached at submission; launched {i}/{len(jobs)}")
            break
        futs.append(ex.submit(wrap, j))
        i += 1
    for f in futs:
        try:
            f.result()
        except Exception:
            pass
    ex.shutdown(wait=True)
    log(f"eval done: {done[0]}/{len(jobs)} in {int((time.time()-t0)/60)}min")

    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    _analyze()
    open(os.path.join(HERE, "results", "overnight", "TBB_DONE"), "w").close()
    log(f"3BB DONE in {int((time.time()-t0)/60)}min")
    return 0


def _analyze():
    try:
        import re
        import pandas as pd
        from scipy import stats
        fs = glob.glob(f"results/raw/{GROUP}/*.csv")
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "total_communication_volume",
                  "final_coverage_ratio", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["bb"] = d.method_name.str.extract(r"^(dars|fnear|futil)_")
        d["tag"] = d.method_name.str.extract(r"_(Sync|Stag)_")
        d["iv"] = pd.to_numeric(d.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")
        d = d.dropna(subset=["bb", "tag", "iv"])
        piv = d.groupby(["bb", "iv", "tag"]).agg(
            travel=("total_travel_distance", "mean"), overlap=("overlap_ratio", "mean"),
            cov=("final_coverage_ratio", "mean"), comm=("total_communication_volume", "mean")).round(3)
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_3bb_pivot.csv"))
        log("3bb pivot (Sync vs Stag by backbone x interval):\n" + piv.to_string())
        rows = []
        for bb, _ in BACKBONES:
            for iv in INTERVALS:
                S = d[(d.bb == bb) & (d.iv == iv) & (d.tag == "Sync")].set_index(["map_id", "seed"])
                G = d[(d.bb == bb) & (d.iv == iv) & (d.tag == "Stag")].set_index(["map_id", "seed"])
                idx = S.index.intersection(G.index)
                if len(idx) < 8:
                    continue
                for met in ["total_travel_distance", "overlap_ratio"]:
                    a = G.loc[idx, met].values
                    c = S.loc[idx, met].values  # staggered vs synchronised
                    try:
                        _, p = stats.wilcoxon(a, c)
                    except Exception:
                        p = float("nan")
                    rows.append(dict(backbone=bb, interval=int(iv), metric=met,
                                     sync=float(c.mean()), stag=float(a.mean()),
                                     stag_minus_sync_pct=float(100 * (a.mean() - c.mean()) / c.mean()),
                                     stag_better_frac=float((a < c).mean()), n=int(len(idx)), p=float(p)))
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "summary", "setcomm_3bb_paired.csv"), index=False)
        log("staggered vs synchronised per backbone (neg = de-sync better):\n" + pd.DataFrame(rows).to_string(index=False))
    except Exception as e:
        import traceback
        log(f"analysis failed: {e}\n{traceback.format_exc()[-400:]}")


if __name__ == "__main__":
    sys.exit(main())
