"""Powered confirmatory test of decision-aware VoI on TRAVEL (B=4, last-known belief, N=8).

The additions sweep showed decision-aware VoI vs geometric VoI cut travel by 3.2% at B=4 but at
n=100 this was not significant (paired Wilcoxon p=0.26); a power analysis on the observed paired
difference (mean -32.7, sd 195) puts the n needed for p<0.05 at ~280. This runner PRE-COMMITS to
n=400 (20 maps x 20 seeds) for the SAME comparison (no method change, no cell-fishing: B=4 is the
only cell the power analysis flagged as reachable) and reports whatever p results. round_robin is
included as the practical baseline. Group setcomm_dvoipow; last-known belief throughout.
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
GROUP = "setcomm_dvoipow"
B = int(os.environ.get("DV_B", "4"))
N = int(os.environ.get("DV_N", "8"))
NUM = int(os.environ.get("DV_NUM_MAPS", "20"))
SEEDS = list(range(int(os.environ.get("DV_SEEDS", "20"))))
WORKERS = int(os.environ.get("DV_WORKERS", "8"))
SIGMA = os.environ.get("DV_SIGMA", "80")
BUDGET_SEC = int(os.environ.get("DV_BUDGET_MIN", "120")) * 60
MODES = ["voi", "decision_voi", "round_robin"]
HOLD_ENV = {"BLMRE_PREDICTIVE_BELIEF": "0", "BLMRE_KALMAN_BELIEF": "0"}


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    mode, s = j
    env = dict(os.environ)
    env.update(HOLD_ENV)
    cmd = [PY, "run_eval_one.py", "--method-name", mode, "--comm-mode", "periodic",
           "--seed", str(s), "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS,
           "--device", "cpu", "--n-agents", str(N), "--targeted", "--link-budget", str(B),
           "--target-mode", mode, "--voi-sigma", SIGMA, "--voi-floor", "0.0"]
    return subprocess.run(cmd, cwd=HERE, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    with open(os.path.join(HERE, "results", "overnight", "dvoipow.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    jobs = [(m, s) for m in MODES for s in SEEDS if not csv_ok(m, s)]
    t0 = time.time()
    log(f"dvoipow: modes {MODES} x {len(SEEDS)} seeds @ B={B}, N={N}, {NUM} maps = {len(jobs)} jobs, {WORKERS} workers")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    _analyze()
    open(os.path.join(HERE, "results", "overnight", "DVOIPOW_DONE"), "w").close()
    log(f"DVOIPOW DONE in {int((time.time()-t0)/60)}min")
    return 0


def _analyze():
    try:
        import pandas as pd
        from scipy import stats
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        rows = []
        for base in ["voi", "round_robin"]:
            for met in ["total_travel_distance", "overlap_ratio"]:
                A = d[d.method_name == "decision_voi"].set_index(["map_id", "seed"])
                C = d[d.method_name == base].set_index(["map_id", "seed"])
                idx = A.index.intersection(C.index)
                if len(idx) < 8:
                    continue
                a = A.loc[idx, met].values
                c = C.loc[idx, met].values
                try:
                    _, p = stats.wilcoxon(a, c)
                except Exception:
                    p = float("nan")
                rows.append(dict(decision_voi_vs=base, metric=met, n=int(len(idx)),
                                 mean_dvoi=round(float(a.mean()), 2), mean_base=round(float(c.mean()), 2),
                                 pct=round(float(100 * (a.mean() - c.mean()) / c.mean()), 2),
                                 better_frac=round(float((a < c).mean()), 2), p=round(float(p), 5)))
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(HERE, "results", "summary", "setcomm_dvoipow_paired.csv"), index=False)
        log("decision_voi powered test (neg = decision_voi better):\n" + out.to_string(index=False))
    except Exception as e:
        import traceback
        log(f"analysis failed: {e}\n{traceback.format_exc()[-400:]}")


if __name__ == "__main__":
    sys.exit(main())
