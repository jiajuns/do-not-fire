"""Clean nearest-frontier (fnear) cross-interval control to complete the cross-backbone gradient.

The original 3-backbone run truncated fnear under the shared wall-clock budget (it is ~5x more
expensive per episode), leaving only 27-35 paired cells at intervals 5 and 8. Here we re-run ONLY
the myopic nearest-frontier planner, with its own budget and worker pool, at intervals 5, 8 and 12
(staggered vs synchronised periodic, N=8, broadcast model, default deconfliction). This gives a
full-n cross-interval fnear control to sit beside the clean DARS and utility-frontier results.
Group setcomm_fnearclean. Paired Wilcoxon staggered vs synchronised per interval.
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
GROUP = "setcomm_fnearclean"
INTERVALS = [int(x) for x in os.environ.get("FN_INTERVALS", "5,8,12").split(",")]
SEEDS = list(range(int(os.environ.get("FN_SEEDS", "6"))))
N = int(os.environ.get("FN_N", "8"))
NUM = 20
WORKERS = int(os.environ.get("FN_WORKERS", "10"))
BUDGET_SEC = int(os.environ.get("FN_BUDGET_MIN", "240")) * 60


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    method, mode, iv, s = j
    env = dict(os.environ)
    env["BLMRE_GATE_POSITIONS"] = "1"
    env["BLMRE_FRONTIER"] = "1"
    env["BLMRE_FRONTIER_MODE"] = "nearest"
    cmd = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", mode, "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", str(N), "--interval", str(iv)]
    return subprocess.run(cmd, cwd=HERE, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    with open(os.path.join(HERE, "results", "overnight", "fnearclean.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    jobs = []
    for iv in INTERVALS:
        for mode, tag in [("periodic", "Sync"), ("staggered", "Stag")]:
            m = f"fnear_{tag}_{iv}"
            for s in SEEDS:
                if not csv_ok(m, s):
                    jobs.append((m, mode, iv, s))
    t0 = time.time()
    log(f"fnearclean: nearest-frontier x intervals {INTERVALS} x sync/stag x {len(SEEDS)} seeds "
        f"= {len(jobs)} jobs, {WORKERS} workers, budget {BUDGET_SEC//60}min")
    ex = ThreadPoolExecutor(max_workers=WORKERS)
    futs, i = [], 0
    for j in jobs:
        if time.time() - t0 > BUDGET_SEC:
            log(f"BUDGET reached; launched {i}/{len(jobs)}")
            break
        futs.append(ex.submit(one, j))
        i += 1
    for f in futs:
        try:
            f.result()
        except Exception:
            pass
    ex.shutdown(wait=True)
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    _analyze()
    open(os.path.join(HERE, "results", "overnight", "FNEARCLEAN_DONE"), "w").close()
    log(f"FNEARCLEAN DONE in {int((time.time()-t0)/60)}min")
    return 0


def _analyze():
    try:
        import pandas as pd
        from scipy import stats
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["tag"] = d.method_name.str.extract(r"_(Sync|Stag)_")
        d["iv"] = pd.to_numeric(d.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")
        d = d.dropna(subset=["tag", "iv"])
        rows = []
        for iv in INTERVALS:
            S = d[(d.iv == iv) & (d.tag == "Sync")].set_index(["map_id", "seed"])
            G = d[(d.iv == iv) & (d.tag == "Stag")].set_index(["map_id", "seed"])
            idx = S.index.intersection(G.index)
            if len(idx) < 8:
                rows.append(dict(interval=iv, n=int(len(idx)), note="too few"))
                continue
            for met in ["total_travel_distance", "overlap_ratio"]:
                a = G.loc[idx, met].values
                c = S.loc[idx, met].values
                try:
                    _, p = stats.wilcoxon(a, c)
                except Exception:
                    p = float("nan")
                rows.append(dict(interval=iv, metric=met.split("_")[0],
                                 sync=round(float(c.mean()), 2), stag=round(float(a.mean()), 2),
                                 stag_minus_sync_pct=round(float(100 * (a.mean() - c.mean()) / c.mean()), 2),
                                 n=int(len(idx)), p=round(float(p), 5)))
        import pandas as pd2
        out = pd2.DataFrame(rows)
        out.to_csv(os.path.join(HERE, "results", "summary", "setcomm_fnearclean_paired.csv"), index=False)
        log("clean fnear (nearest-frontier) staggered vs synchronised:\n" + out.to_string(index=False))
    except Exception as e:
        import traceback
        log(f"analysis failed: {e}\n{traceback.format_exc()[-400:]}")


if __name__ == "__main__":
    sys.exit(main())
