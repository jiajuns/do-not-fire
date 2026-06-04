"""Clean multi-N decisive control (resolves the genN/ANCOVA per-N conflict with data).

The published fig_multiN regresses an `async` dummy on the genN sweep with a *linear* log-comm
covariate; because travel vs. communication saturates, that covariate does not fully remove the
comm-regime difference between async and sync methods, so the per-N "all p<0.05" is suspect, and
strict matched-comm Periodic-vs-Random pairs are null at N>=4. The genuinely clean test (used at
N=8 in run_sync.py) is STAGGERED vs SYNCHRONISED periodic at the SAME interval: identical per-robot
rate AND identical total communication volume, the only difference being broadcast phase.

This sweeps that decisive control across team sizes N in {4,6,8,10,12} on the frozen DARS backbone.
If staggered still beats synchronised per N, fig_multiN stands on clean data; if it only holds at
some N, we report that honestly. Group setcomm_syncN. Paired Wilcoxon stag vs sync per N x interval.
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
GROUP = "setcomm_syncN"
NS = [int(x) for x in os.environ.get("SYNCN_NS", "4,6,8,10,12").split(",")]
INTERVALS = [int(x) for x in os.environ.get("SYNCN_INTERVALS", "8,12").split(",")]
SEEDS = list(range(int(os.environ.get("SYNCN_SEEDS", "6"))))
NUM = int(os.environ.get("SYNCN_NUM_MAPS", "20"))
WORKERS = int(os.environ.get("SYNCN_WORKERS", "12"))
BUDGET_SEC = int(os.environ.get("SYNCN_BUDGET_MIN", "180")) * 60


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    method, mode, n, iv, s = j
    env = dict(os.environ)
    env["BLMRE_GATE_POSITIONS"] = "1"  # broadcast model: position sharing gated
    cmd = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", mode, "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", str(n), "--interval", str(iv)]
    return subprocess.run(cmd, cwd=HERE, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    with open(os.path.join(HERE, "results", "overnight", "syncN.log"), "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main():
    jobs = []
    for n in NS:
        for iv in INTERVALS:
            for mode, tag in [("periodic", "Sync"), ("staggered", "Stag")]:
                m = f"N{n}_{tag}_{iv}"
                for s in SEEDS:
                    if not csv_ok(m, s):
                        jobs.append((m, mode, n, iv, s))
    t0 = time.time()
    log(f"syncN: Ns {NS} x intervals {INTERVALS} x sync/stag x {len(SEEDS)} seeds = {len(jobs)} jobs, "
        f"{WORKERS} workers, budget {BUDGET_SEC//60}min")
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
            log(f"BUDGET reached; launched {i}/{len(jobs)}")
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
    open(os.path.join(HERE, "results", "overnight", "SYNCN_DONE"), "w").close()
    log(f"SYNCN DONE in {int((time.time()-t0)/60)}min")
    return 0


def _analyze():
    try:
        import pandas as pd
        from scipy import stats
        fs = glob.glob(f"results/raw/{GROUP}/*.csv")
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "total_communication_volume",
                  "final_coverage_ratio", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["N"] = pd.to_numeric(d.method_name.str.extract(r"^N(\d+)_")[0], errors="coerce")
        d["tag"] = d.method_name.str.extract(r"_(Sync|Stag)_")
        d["iv"] = pd.to_numeric(d.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")
        d = d.dropna(subset=["N", "tag", "iv"])
        piv = d.groupby(["N", "iv", "tag"]).agg(
            travel=("total_travel_distance", "mean"), overlap=("overlap_ratio", "mean"),
            cov=("final_coverage_ratio", "mean"), comm=("total_communication_volume", "mean")).round(3)
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_syncN_pivot.csv"))
        log("syncN pivot (Sync vs Stag by N x interval):\n" + piv.to_string())
        rows = []
        for n in NS:
            for iv in INTERVALS:
                S = d[(d.N == n) & (d.iv == iv) & (d.tag == "Sync")].set_index(["map_id", "seed"])
                G = d[(d.N == n) & (d.iv == iv) & (d.tag == "Stag")].set_index(["map_id", "seed"])
                idx = S.index.intersection(G.index)
                if len(idx) < 8:
                    rows.append(dict(N=n, interval=iv, n=int(len(idx)), note="too few pairs"))
                    continue
                for met in ["total_travel_distance", "overlap_ratio"]:
                    a = G.loc[idx, met].values
                    c = S.loc[idx, met].values  # staggered vs synchronised
                    try:
                        _, p = stats.wilcoxon(a, c)
                    except Exception:
                        p = float("nan")
                    rows.append(dict(N=n, interval=iv, metric=met,
                                     sync=round(float(c.mean()), 2), stag=round(float(a.mean()), 2),
                                     stag_minus_sync_pct=round(float(100 * (a.mean() - c.mean()) / c.mean()), 2),
                                     stag_better_frac=round(float((a < c).mean()), 2), n=int(len(idx)),
                                     p=round(float(p), 5)))
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "summary", "setcomm_syncN_paired.csv"), index=False)
        log("clean per-N decisive control (stag vs sync, neg = de-sync better):\n" +
            pd.DataFrame(rows).to_string(index=False))
    except Exception as e:
        import traceback
        log(f"analysis failed: {e}\n{traceback.format_exc()[-400:]}")


if __name__ == "__main__":
    sys.exit(main())
