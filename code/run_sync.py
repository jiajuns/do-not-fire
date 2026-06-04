"""Synchronisation-isolation experiment (the decisive Pillar-A control).

Broadcast model (GATE_POSITIONS=1). Compares SYNCHRONISED periodic vs STAGGERED periodic at
the SAME interval (=identical per-robot rate AND identical total comm volume) -> the ONLY
difference is whether the team broadcasts in lockstep. If staggered beats synchronised, the
efficiency cost is specifically SYNCHRONISATION (not comm volume, not content-adaptivity,
not random's extra variance). Intervals {3,5,8,12} span scarce->medium budgets, N=8.
Paired Wilcoxon staggered vs periodic per interval on travel & overlap.
"""
import os, sys, time, subprocess, glob, re
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable
MAPS = os.path.join(ROOT, "maps_test"); GROUP = "setcomm_sync"
INTERVALS = [int(x) for x in os.environ.get("SYNC_INTERVALS", "3,5,8,12").split(",")]
SEEDS = list(range(int(os.environ.get("SYNC_SEEDS", "8")))); N = 8; NUM = 20
WORKERS = int(os.environ.get("SYNC_WORKERS", "16"))
os.environ["BLMRE_GATE_POSITIONS"] = "1"  # broadcast model: position sharing is gated (children inherit)


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    method, mode, iv, s = j
    env = dict(os.environ); env["BLMRE_GATE_POSITIONS"] = "1"
    cmd = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", mode, "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", str(N), "--interval", str(iv)]
    return subprocess.run(cmd, cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def main():
    jobs = []
    for iv in INTERVALS:
        for mode, tag in [("periodic", "Sync"), ("staggered", "Stag")]:
            m = f"{tag}_{iv}"
            for s in SEEDS:
                if not csv_ok(m, s):
                    jobs.append((m, mode, iv, s))
    t0 = time.time(); print(f"sync run: {len(jobs)} jobs (intervals {INTERVALS} x sync/stag x {len(SEEDS)} seeds)", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    try:
        import pandas as pd
        from scipy import stats
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "total_communication_volume",
                  "final_coverage_ratio", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["tag"] = d["method_name"].str.replace(r"_\d+$", "", regex=True)
        d["iv"] = d["method_name"].str.extract(r"_(\d+)$").astype(int)
        piv = d.groupby(["iv", "tag"]).agg(comm=("total_communication_volume", "mean"),
                                           travel=("total_travel_distance", "mean"),
                                           overlap=("overlap_ratio", "mean"),
                                           cov=("final_coverage_ratio", "mean")).round(3)
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_sync_pivot.csv"))
        print("=== sync-isolation pivot (Sync vs Stag by interval) ===\n" + piv.to_string())
        rows = []
        for iv in INTERVALS:
            S = d[(d.iv == iv) & (d.tag == "Sync")].set_index(["map_id", "seed"])
            G = d[(d.iv == iv) & (d.tag == "Stag")].set_index(["map_id", "seed"])
            idx = S.index.intersection(G.index)
            if len(idx) < 8:
                continue
            for met in ["total_travel_distance", "overlap_ratio"]:
                a = G.loc[idx, met].values; c = S.loc[idx, met].values  # staggered vs sync
                try:
                    _, p = stats.wilcoxon(a, c)
                except Exception:
                    p = float("nan")
                rows.append(dict(interval=iv, metric=met, sync=float(c.mean()), stag=float(a.mean()),
                                 stag_minus_sync_pct=float(100 * (a.mean() - c.mean()) / c.mean()),
                                 stag_better=float((a < c).mean()), p=float(p)))
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "summary", "setcomm_sync_paired.csv"), index=False)
        print("=== staggered vs synchronised (neg=staggered better) ===\n" + pd.DataFrame(rows).to_string(index=False))
    except Exception as e:
        print("analysis failed", e)
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    open(os.path.join(HERE, "results", "overnight", "SYNC_DONE"), "w").close()
    print(f"SYNC DONE {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
