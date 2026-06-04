"""Tier-1/2 improvement batch (eval-only, robust, budget-guarded):
 Phase A  larger teams N in {16,20}: targeted {voi,learned,round_robin,random}@B4 +
          sync/staggered @ interval {8,12} -> does the de-sync/VoI advantage scale up?
 Phase B  per-N learned weights at N in {4,12}: search 20 weight vectors on val maps 0-9,
          pick best per N, test on held-out maps 10-19 -> fix the N-transfer limitation.
 Phase C  sensor-range sensitivity {10,15,30} m at N=8: sync vs staggered + voi vs round_robin
          -> scheduling advantage robust to sensing range?
Each phase aggregates + analyses independently. Groups: setcomm_bigN / setcomm_wsearchN /
setcomm_learnedN / setcomm_srange.
"""
import os, sys, time, glob, re, subprocess
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable
MAPS = os.path.join(ROOT, "maps_test"); OUT = os.path.join(HERE, "results", "overnight")
WORKERS = int(os.environ.get("IMP_WORKERS", "16")); WW = "2,0,1,1,0"  # learned weights from overnight2
T0 = time.time()
DL_A = T0 + int(os.environ.get("IMP_A_MIN", "150")) * 60
DL_B = T0 + int(os.environ.get("IMP_B_MIN", "330")) * 60
DL_C = T0 + int(os.environ.get("IMP_C_MIN", "450")) * 60
CAND = [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0.5, 0],
        [0, 0, 1, 1, 0], [0, 0, 1, 2, 0], [1, 0, 1, 0, 0], [1, 0, 1, 1, 0], [0, 0, 1, 0, -0.5],
        [0, 0, 1, 0, 0.5], [1, 0, 0, 1, 0], [0.5, 0, 1, 0.5, 0], [0, 1, 1, 0, 0], [1, 1, 1, 1, 0],
        [0, 0, 2, 1, 0], [0.5, 0, 1, 0.5, -0.3], [1, 0, 1, 0.5, 0], [0, 0, 1, 1.5, -0.5], [2, 0, 1, 1, 0]]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "improve.log"), "a", encoding="utf-8").write(line + "\n")


def tgt(method, group, seed, B, mode, weights="", n=8, srange=None):
    a = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", "periodic", "--seed", str(seed),
         "--num-maps", "20", "--group", group, "--map-dir", MAPS, "--device", "cpu", "--n-agents", str(n),
         "--targeted", "--link-budget", str(B), "--target-mode", mode, "--voi-sigma", "80", "--voi-floor", "0.5"]
    if weights:
        a += ["--voi-weights", weights]
    if srange:
        a += ["--sensor-range", str(srange)]
    return (group, method, seed, a, {})


def bcast(method, group, seed, comm_mode, interval, n=8, srange=None):
    a = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", comm_mode, "--seed", str(seed),
         "--num-maps", "20", "--group", group, "--map-dir", MAPS, "--device", "cpu", "--n-agents", str(n),
         "--interval", str(interval)]
    if srange:
        a += ["--sensor-range", str(srange)]
    return (group, method, seed, a, {"BLMRE_GATE_POSITIONS": "1"})


def csv_ok(group, method, seed):
    p = os.path.join(HERE, "results", "raw", group, f"{group}_{method}_seed{seed}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def run_jobs(jobs, deadline, tag):
    done = [0]; tot = len(jobs)
    def one(j):
        group, method, seed, argv, envx = j
        if csv_ok(group, method, seed):
            done[0] += 1; return 0
        env = dict(os.environ); env.update(envx)
        r = subprocess.run(argv, cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        done[0] += 1
        if r.returncode != 0:
            log(f"  FAIL {group}/{method} s{seed}: {r.stderr[-140:]}")
        elif done[0] % 30 == 0:
            log(f"  {tag} {done[0]}/{tot} ({int((time.time()-T0)/60)}min)")
        return r.returncode
    ex = ThreadPoolExecutor(max_workers=WORKERS); fut = set(); i = 0
    while i < len(jobs) or fut:
        while len(fut) < WORKERS and i < len(jobs) and time.time() < deadline:
            fut.add(ex.submit(one, jobs[i])); i += 1
        if not fut:
            break
        _, fut = wait(fut, return_when=FIRST_COMPLETED)
        if time.time() >= deadline and i < len(jobs):
            log(f"  {tag} BUDGET reached {i}/{len(jobs)}")
    ex.shutdown(wait=True)
    log(f"{tag} done {i}/{len(jobs)} in {int((time.time()-T0)/60)}min")


def aggregate(group):
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{group}",
                    "--group", group, "--output_dir", "results/summary"], cwd=HERE,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load(group):
    import pandas as pd
    fs = glob.glob(os.path.join(HERE, "results", "raw", group, "*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio", "map_id", "seed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def phase_a():
    log("Phase A: larger teams N in {16,20}")
    jobs = []
    for n in [16, 20]:
        for s in range(8):
            for nm, mode, w in [("Learned", "wsum", WW), ("VoI", "voi", ""),
                                ("RoundRobin", "round_robin", ""), ("Random", "random", "")]:
                jobs.append(tgt(f"{nm}_B4_N{n}", "setcomm_bigN", s, 4, mode, w, n=n))
            for iv in [8, 12]:
                jobs.append(bcast(f"Sync_{iv}_N{n}", "setcomm_bigN", s, "periodic", iv, n=n))
                jobs.append(bcast(f"Stag_{iv}_N{n}", "setcomm_bigN", s, "staggered", iv, n=n))
    run_jobs(jobs, DL_A, "A")
    aggregate("setcomm_bigN")


def phase_b():
    log("Phase B: per-N learned weights at N in {4,12}")
    for n in [4, 12]:
        jobs = []
        for idx, w in enumerate(CAND):
            ws = ",".join(str(x) for x in w)
            for s in [0, 1, 2]:
                jobs.append((f"setcomm_wsearchN{n}", f"WS{idx:02d}", s,
                             [PY, "run_eval_one.py", "--method-name", f"WS{idx:02d}", "--comm-mode", "periodic",
                              "--seed", str(s), "--num-maps", "10", "--group", f"setcomm_wsearchN{n}",
                              "--map-dir", MAPS, "--device", "cpu", "--n-agents", str(n), "--targeted",
                              "--link-budget", "4", "--target-mode", "wsum", "--voi-sigma", "80", "--voi-weights", ws], {}))
        run_jobs(jobs, DL_B, f"B-search-N{n}")
        aggregate(f"setcomm_wsearchN{n}")
        best = CAND[0]
        try:
            import pandas as pd
            d = load(f"setcomm_wsearchN{n}")
            d["idx"] = d["method_name"].str.extract(r"WS(\d+)").astype(float)
            g = d.groupby("idx").agg(travel=("total_travel_distance", "mean"), overlap=("overlap_ratio", "mean")).reset_index()
            g["score"] = g["travel"] / 1000.0 + 0.5 * g["overlap"]
            bi = int(g.sort_values("score").iloc[0]["idx"]); best = CAND[bi]
            log(f"  N={n} best weights WS{bi:02d}={best} (voi WS00)")
        except Exception as e:
            log(f"  B-search-N{n} rank failed: {e}")
        ws = ",".join(str(x) for x in best)
        jobs = []
        for s in range(8):
            for nm, mode, w in [("Learned", "wsum", ws), ("VoI", "voi", ""),
                                ("RoundRobin", "round_robin", ""), ("Random", "random", "")]:
                jobs.append(tgt(f"{nm}_N{n}", "setcomm_learnedN", s, 4, mode, w, n=n))
        run_jobs(jobs, DL_B, f"B-test-N{n}")
    aggregate("setcomm_learnedN")
    _learnedN_stats()


def _learnedN_stats():
    try:
        import pandas as pd
        from scipy import stats
        d = load("setcomm_learnedN"); d = d[d["map_id"] >= 10]
        d["nm"] = d["method_name"].str.replace(r"_N\d+$", "", regex=True)
        d["N"] = d["method_name"].str.extract(r"_N(\d+)$").astype(int)
        rows = []
        for n in sorted(d["N"].unique()):
            sub = d[d["N"] == n]; L = sub[sub["nm"] == "Learned"].set_index(["map_id", "seed"])
            for base in ["VoI", "RoundRobin", "Random"]:
                B = sub[sub["nm"] == base].set_index(["map_id", "seed"]); idx = L.index.intersection(B.index)
                if len(idx) < 8:
                    continue
                a = L.loc[idx, "total_travel_distance"].values; c = B.loc[idx, "total_travel_distance"].values
                try:
                    _, p = stats.wilcoxon(a, c)
                except Exception:
                    p = float("nan")
                rows.append(dict(N=int(n), baseline=base, learned=float(a.mean()), base=float(c.mean()),
                                 diff_pct=float(100 * (a.mean() - c.mean()) / c.mean()), p=float(p)))
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "summary", "setcomm_learnedN_paired.csv"), index=False)
        log("per-N learned (held-out, travel):\n" + pd.DataFrame(rows).to_string(index=False))
    except Exception as e:
        log(f"learnedN stats failed: {e}")


def phase_c():
    log("Phase C: sensor-range sensitivity {10,15,30} at N=8")
    jobs = []
    for sr in [10, 15, 30]:
        for s in range(6):
            jobs.append(tgt(f"VoI_sr{sr}", "setcomm_srange", s, 4, "voi", "", n=8, srange=sr))
            jobs.append(tgt(f"RoundRobin_sr{sr}", "setcomm_srange", s, 4, "round_robin", "", n=8, srange=sr))
            jobs.append(bcast(f"Sync8_sr{sr}", "setcomm_srange", s, "periodic", 8, n=8, srange=sr))
            jobs.append(bcast(f"Stag8_sr{sr}", "setcomm_srange", s, "staggered", 8, n=8, srange=sr))
    run_jobs(jobs, DL_C, "C")
    aggregate("setcomm_srange")


def main():
    os.makedirs(OUT, exist_ok=True)
    log(f"improve start; workers={WORKERS}; deadlines A={int((DL_A-T0)/60)} B={int((DL_B-T0)/60)} C={int((DL_C-T0)/60)}min")
    for ph in [phase_a, phase_b, phase_c]:
        try:
            ph()
        except Exception as e:
            import traceback; log(f"{ph.__name__} crashed: {e}\n{traceback.format_exc()[-300:]}")
    open(os.path.join(OUT, "IMPROVE_DONE"), "w").close()
    log(f"IMPROVE DONE in {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()
