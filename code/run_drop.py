"""Robustness rerun (after fixing COMM_DROP in the targeted path): graceful degradation
under packet loss. drop in {0,0.1,0.2,0.3} x {Learned(wsum),VoI,RoundRobin,Random}, N=8, B=4."""
import os, sys, time, subprocess, glob, re
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable
MAPS = os.path.join(ROOT, "maps_test"); GROUP = "setcomm_drop2"
DROPS = [0.0, 0.1, 0.2, 0.3]
METHODS = [("Learned", "wsum", "2,0,1,1,0"), ("VoI", "voi", ""),
           ("RoundRobin", "round_robin", ""), ("Random", "random", "")]
SEEDS = list(range(int(os.environ.get("DROP_SEEDS", "12")))); B = 4; N = 8; NUM = 20
WORKERS = int(os.environ.get("DROP_WORKERS", "16"))


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    method, mode, w, d, s = j
    cmd = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", "periodic", "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", str(N), "--targeted", "--link-budget", str(B), "--target-mode", mode,
           "--voi-sigma", "80", "--voi-floor", "0.5", "--comm-drop", str(d)]
    if w:
        cmd += ["--voi-weights", w]
    return subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def main():
    jobs = []
    for d in DROPS:
        for nm, mode, w in METHODS:
            m = f"{nm}_d{d:g}"
            for s in SEEDS:
                if not csv_ok(m, s):
                    jobs.append((m, mode, w, d, s))
    t0 = time.time(); print(f"drop run: {len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    # quick pivot
    try:
        import pandas as pd
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["nm"] = d["method_name"].str.replace(r"_d[\d.]+$", "", regex=True)
        d["drop"] = d["method_name"].str.extract(r"_d([\d.]+)$").astype(float)
        piv = d.groupby(["drop", "nm"]).agg(travel=("total_travel_distance", "mean"),
                                            overlap=("overlap_ratio", "mean"),
                                            cov=("final_coverage_ratio", "mean")).round(3)
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_drop2_pivot.csv"))
        print(piv.to_string())
        # paired significance: Learned vs each baseline, per drop, travel & overlap
        from scipy import stats
        d["map_id"] = pd.to_numeric(d["map_id"], errors="coerce")
        d["seed"] = pd.to_numeric(d["seed"], errors="coerce")
        rows = []
        for drop in sorted(d["drop"].dropna().unique()):
            sub = d[d["drop"] == drop]
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
                    rows.append(dict(drop=float(drop), metric=met, baseline=base,
                                     learned=float(a.mean()), base=float(c.mean()),
                                     diff_pct=float(100 * (a.mean() - c.mean()) / c.mean()),
                                     learned_better=float((a < c).mean()), n=int(len(idx)), p=float(p)))
        pdf = pd.DataFrame(rows)
        pdf.to_csv(os.path.join(HERE, "results", "summary", "setcomm_drop2_paired.csv"), index=False)
        print("=== drop paired (Learned vs baselines, neg=Learned better) ===\n" + pdf.to_string(index=False))
    except Exception as e:
        import traceback; print("pivot/paired failed", e, traceback.format_exc()[-300:])
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    open(os.path.join(HERE, "results", "overnight", "DROP2_DONE"), "w").close()
    print(f"DROP2 DONE {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
