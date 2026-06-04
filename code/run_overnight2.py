"""8h unattended pipeline (eval-only, robust):
 Phase B1  search ~20 linear-VoI weight vectors on VALIDATION maps 0-9 (B=4) -> pick best.
 Phase B2  test best learned weights vs voi/voi_lif/round_robin/random on HELD-OUT maps
           10-19 across B in {2,4,8}; paired Wilcoxon (learned vs each baseline).
 Phase A   multi-N targeted (N in {4,8,12}) + packet-loss robustness (guaranteed-value backstop).
All phases wall-clock-guarded; each aggregates independently so partial completion still yields
clean tables. Map holdout (search 0-9, test 10-19) prevents leakage.
"""
import os, sys, time, glob, re, subprocess
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable
MAPS = os.path.join(ROOT, "maps_test"); OUT = os.path.join(HERE, "results", "overnight")
WORKERS = int(os.environ.get("O2_WORKERS", "16"))
T0 = time.time()
DL_B1 = T0 + int(os.environ.get("O2_B1_MIN", "60")) * 60
DL_B2 = T0 + int(os.environ.get("O2_B2_MIN", "240")) * 60
DL_A = T0 + int(os.environ.get("O2_A_MIN", "465")) * 60

CAND = [
    [0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0],
    [0, 0, 1, 0.5, 0], [0, 0, 1, 1, 0], [0, 0, 1, 2, 0], [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 0], [0, 0, 1, 0, -0.5], [0, 0, 1, 0, 0.5], [1, 0, 0, 1, 0],
    [0.5, 0, 1, 0.5, 0], [0, 1, 1, 0, 0], [1, 1, 1, 1, 0], [0, 0, 2, 1, 0],
    [0.5, 0, 1, 0.5, -0.3], [1, 0, 1, 0.5, 0], [0, 0, 1, 1.5, -0.5], [2, 0, 1, 1, 0],
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "overnight2.log"), "a", encoding="utf-8").write(line + "\n")


def ev(method, group, seed, nmaps, B, mode, weights="", drop=0.0, n=8):
    a = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", "periodic", "--seed", str(seed),
         "--num-maps", str(nmaps), "--group", group, "--map-dir", MAPS, "--device", "cpu",
         "--n-agents", str(n), "--targeted", "--link-budget", str(B), "--target-mode", mode,
         "--voi-sigma", "80", "--voi-floor", "0.5", "--comm-drop", str(drop)]
    if weights:
        a += ["--voi-weights", weights]
    return a


def csv_ok(group, method, seed):
    p = os.path.join(HERE, "results", "raw", group, f"{group}_{method}_seed{seed}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def run_jobs(jobs, deadline, tag):
    done = [0]; tot = len(jobs)
    def one(j):
        method, group, seed, kw = j
        if csv_ok(group, method, seed):
            done[0] += 1; return 0
        r = subprocess.run(ev(method, group, seed, **kw), cwd=HERE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        done[0] += 1
        if r.returncode != 0:
            log(f"  FAIL {group}/{method} s{seed}: {r.stderr[-160:]}")
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
            log(f"  {tag} BUDGET reached, launched {i}/{len(jobs)}");
    ex.shutdown(wait=True)
    log(f"{tag} done: launched {i}/{len(jobs)} in {int((time.time()-T0)/60)}min")


def aggregate(group):
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{group}",
                    "--group", group, "--output_dir", "results/summary"],
                   cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load(group):
    import pandas as pd
    fs = glob.glob(os.path.join(HERE, "results", "raw", group, "*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio",
              "total_communication_volume", "map_id", "seed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def phase_b1():
    log(f"Phase B1: weight search, {len(CAND)} candidates x 3 seeds on maps 0-9, B=4")
    jobs = []
    for idx, w in enumerate(CAND):
        ws = ",".join(str(x) for x in w)
        for s in [0, 1, 2]:
            jobs.append((f"WS{idx:02d}", "setcomm_wsearch", s,
                         dict(nmaps=10, B=4, mode="wsum", weights=ws)))
    run_jobs(jobs, DL_B1, "B1")
    aggregate("setcomm_wsearch")
    d = load("setcomm_wsearch")
    best = 0
    try:
        import pandas as pd
        d["idx"] = d["method_name"].str.extract(r"WS(\d+)").astype(float)
        g = d.groupby("idx").agg(travel=("total_travel_distance", "mean"),
                                 overlap=("overlap_ratio", "mean"),
                                 cov=("final_coverage_ratio", "mean")).reset_index()
        g["score"] = g["travel"] / 1000.0 + 0.5 * g["overlap"]  # lower=better
        g = g.sort_values("score")
        g.to_csv(os.path.join(HERE, "results", "summary", "setcomm_wsearch_rank.csv"), index=False)
        best = int(g.iloc[0]["idx"])
        log("Phase B1 ranking (top 6 by travel+0.5*overlap):\n" + g.head(6).to_string(index=False))
        log(f"BEST candidate WS{best:02d} weights={CAND[best]} (voi baseline = WS00 {CAND[0]})")
    except Exception as e:
        log(f"B1 rank failed: {e}; defaulting to voi weights")
    return CAND[best]


def phase_b2(best_w):
    ws = ",".join(str(x) for x in best_w)
    log(f"Phase B2: test Learned(w={best_w}) vs baselines on held-out maps 10-19, B in 2/4/8")
    methods = [("Learned", "wsum", ws), ("VoI", "voi", ""), ("VoI_lif", "voi_lif", ""),
               ("RoundRobin", "round_robin", ""), ("Random", "random", "")]
    jobs = []
    for B in [2, 4, 8]:
        for nm, mode, w in methods:
            for s in range(8):
                jobs.append((f"{nm}_B{B}", "setcomm_learned", s,
                             dict(nmaps=20, B=B, mode=mode, weights=w)))
    run_jobs(jobs, DL_B2, "B2")
    aggregate("setcomm_learned")
    _b2_stats()


def _b2_stats():
    try:
        import pandas as pd
        from scipy import stats
        d = load("setcomm_learned")
        d = d[d["map_id"] >= 10]  # HELD-OUT maps only (search used 0-9)
        d["nm"] = d["method_name"].str.replace(r"_B\d+$", "", regex=True)
        d["B"] = d["method_name"].str.extract(r"_B(\d+)$").astype(int)
        piv = d.groupby(["B", "nm"]).agg(travel=("total_travel_distance", "mean"),
                                         overlap=("overlap_ratio", "mean"),
                                         cov=("final_coverage_ratio", "mean")).reset_index()
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_learned_pivot.csv"), index=False)
        log("Phase B2 pivot (held-out maps 10-19):\n" + piv.to_string(index=False))
        rows = []
        for B in sorted(d["B"].unique()):
            sub = d[d["B"] == B]
            L = sub[sub["nm"] == "Learned"].set_index(["map_id", "seed"])
            for base in ["VoI", "RoundRobin", "Random"]:
                Bb = sub[sub["nm"] == base].set_index(["map_id", "seed"])
                idx = L.index.intersection(Bb.index)
                if len(idx) < 8:
                    continue
                for met in ["total_travel_distance", "overlap_ratio"]:
                    a = L.loc[idx, met].values; c = Bb.loc[idx, met].values
                    try:
                        _, p = stats.wilcoxon(a, c)
                    except Exception:
                        p = float("nan")
                    rows.append(dict(B=int(B), metric=met, baseline=base,
                                     learned=float(a.mean()), base=float(c.mean()),
                                     diff_pct=float(100 * (a.mean() - c.mean()) / c.mean()),
                                     learned_better=float((a < c).mean()), p=float(p)))
        pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "summary", "setcomm_learned_paired.csv"), index=False)
        log("Phase B2 paired (Learned vs baselines, held-out):\n" + pd.DataFrame(rows).to_string(index=False))
    except Exception as e:
        import traceback; log(f"B2 stats failed: {e}\n{traceback.format_exc()[-400:]}")


def phase_a(best_w):
    ws = ",".join(str(x) for x in best_w)
    log("Phase A: multi-N targeted + packet-loss robustness")
    jobs = []
    for N in [4, 8, 12]:
        for B in [2, 4, 8]:
            for nm, mode, w in [("Learned", "wsum", ws), ("VoI", "voi", ""),
                                ("RoundRobin", "round_robin", ""), ("Random", "random", "")]:
                for s in range(6):
                    jobs.append((f"{nm}_B{B}_N{N}", "setcomm_tgtN", s,
                                 dict(nmaps=20, B=B, mode=mode, weights=w, n=N)))
    run_jobs(jobs, min(DL_A, DL_B2 + 150 * 60), "A-N")
    aggregate("setcomm_tgtN")
    jobs = []
    for drop in [0.1, 0.2]:
        for nm, mode, w in [("Learned", "wsum", ws), ("VoI", "voi", ""),
                            ("RoundRobin", "round_robin", ""), ("Random", "random", "")]:
            for s in range(6):
                jobs.append((f"{nm}_d{drop}", "setcomm_tgtdrop", s,
                             dict(nmaps=20, B=4, mode=mode, weights=w, drop=drop)))
    run_jobs(jobs, DL_A, "A-drop")
    aggregate("setcomm_tgtdrop")


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(MAPS):
        log(f"ERROR {MAPS} missing"); return 1
    log(f"overnight2 start; workers={WORKERS}; deadlines B1={int((DL_B1-T0)/60)} B2={int((DL_B2-T0)/60)} A={int((DL_A-T0)/60)} min")
    best_w = CAND[0]
    try:
        best_w = phase_b1()
    except Exception as e:
        log(f"Phase B1 crashed: {e}")
    try:
        phase_b2(best_w)
    except Exception as e:
        log(f"Phase B2 crashed: {e}")
    try:
        phase_a(best_w)
    except Exception as e:
        log(f"Phase A crashed: {e}")
    open(os.path.join(OUT, "OVERNIGHT2_DONE"), "w").close()
    log(f"OVERNIGHT2 DONE in {(time.time()-T0)/60:.1f} min; best_w={best_w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
