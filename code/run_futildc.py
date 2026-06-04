"""Causal mechanism test for Pillar A: does the de-synchronisation benefit grow with how much a
planner couples on the shared teammate belief? We hold the planner fixed (utility frontier) and
sweep ONLY its deconfliction weight FRONTIER_DECONFLICT_W in {0,1,3,6} -- the single knob that sets
how strongly it avoids believed-teammate-occupied nodes (a belief refreshed only by communication).
At each weight we run the decisive phase-only control (staggered vs synchronised periodic, interval
12, N=8) and measure the de-sync travel/overlap reduction. If the reduction increases monotonically
with the deconfliction weight, the de-sync benefit is *causally* tied to belief-coupling strength,
upgrading the 3-backbone gradient from correlational to causal. Group setcomm_futildc.
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
GROUP = "setcomm_futildc"
IV = int(os.environ.get("FDC_IV", "12"))
WEIGHTS = [float(x) for x in os.environ.get("FDC_WEIGHTS", "0,1,3,6").split(",")]
SEEDS = list(range(int(os.environ.get("FDC_SEEDS", "8"))))
N = 8
NUM = 20
WORKERS = int(os.environ.get("FDC_WORKERS", "8"))
BUDGET_SEC = int(os.environ.get("FDC_BUDGET_MIN", "120")) * 60


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    method, mode, w, s = j
    env = dict(os.environ)
    env["BLMRE_GATE_POSITIONS"] = "1"
    env["BLMRE_FRONTIER"] = "1"
    env["BLMRE_FRONTIER_MODE"] = "utility"
    env["BLMRE_FRONTIER_DECONFLICT_W"] = str(w)
    cmd = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", mode, "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", str(N), "--interval", str(IV)]
    return subprocess.run(cmd, cwd=HERE, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    with open(os.path.join(HERE, "results", "overnight", "futildc.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    jobs = []
    for w in WEIGHTS:
        wtag = str(w).replace(".", "p")
        for mode, tag in [("periodic", "Sync"), ("staggered", "Stag")]:
            m = f"futil_dc{wtag}_{tag}_{IV}"
            for s in SEEDS:
                if not csv_ok(m, s):
                    jobs.append((m, mode, w, s))
    t0 = time.time()
    log(f"futildc: utility frontier x deconflict {WEIGHTS} x sync/stag x {len(SEEDS)} seeds @ iv{IV} "
        f"= {len(jobs)} jobs, {WORKERS} workers")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    _analyze()
    open(os.path.join(HERE, "results", "overnight", "FUTILDC_DONE"), "w").close()
    log(f"FUTILDC DONE in {int((time.time()-t0)/60)}min")
    return 0


def _analyze():
    try:
        import pandas as pd
        from scipy import stats
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["w"] = pd.to_numeric(d.method_name.str.extract(r"_dc([0-9p]+)_")[0].str.replace("p", ".", regex=False),
                               errors="coerce")
        d["tag"] = d.method_name.str.extract(r"_(Sync|Stag)_")
        d = d.dropna(subset=["w", "tag"])
        rows = []
        for w in WEIGHTS:
            S = d[(d.w == w) & (d.tag == "Sync")].set_index(["map_id", "seed"])
            G = d[(d.w == w) & (d.tag == "Stag")].set_index(["map_id", "seed"])
            idx = S.index.intersection(G.index)
            if len(idx) < 8:
                rows.append(dict(deconflict_w=w, n=int(len(idx)), note="too few"))
                continue
            for met in ["total_travel_distance", "overlap_ratio"]:
                a = G.loc[idx, met].values
                c = S.loc[idx, met].values
                try:
                    _, p = stats.wilcoxon(a, c)
                except Exception:
                    p = float("nan")
                rows.append(dict(deconflict_w=w, metric=met.split("_")[0],
                                 sync=round(float(c.mean()), 2), stag=round(float(a.mean()), 2),
                                 desync_pct=round(float(100 * (a.mean() - c.mean()) / c.mean()), 2),
                                 n=int(len(idx)), p=round(float(p), 5)))
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(HERE, "results", "summary", "setcomm_futildc_doseresp.csv"), index=False)
        log("DOSE-RESPONSE: de-sync benefit vs deconfliction weight (neg = de-sync better):\n" +
            out.to_string(index=False))
    except Exception as e:
        import traceback
        log(f"analysis failed: {e}\n{traceback.format_exc()[-400:]}")


if __name__ == "__main__":
    sys.exit(main())
