"""Multi-agent-count GENERALIZATION run for the 'limit communication volume' study.

Eval-only sweep across team sizes N in {4,6,8,10,12} x fixed scheduling methods
(periodic / spiking / random / evtrig) x seeds, on the hardest maps, position-gated.
Shows the communication-vs-efficiency tradeoff AND the 'budget governs, not policy'
finding hold across team sizes. Parallel, wall-clock budget capped (~7.5h).

Learned gates are excluded (trained at N=8 -> OOD at other N). Group 'setcomm_genN'.
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
GROUP = "setcomm_genN"
OUT = os.path.join(HERE, "results", "overnight")

WORKERS = int(os.environ.get("GENN_WORKERS", "16"))
SEEDS = [int(x) for x in os.environ.get("GENN_SEEDS", "0,1,2,3,4,5,6").split(",")]
NUM_MAPS = int(os.environ.get("GENN_NUM_MAPS", "20"))
NS = [int(x) for x in os.environ.get("GENN_NS", "4,6,8,10,12").split(",")]
TH = [float(x) for x in os.environ.get("GENN_TH", "0.4,0.7,1.0,1.5,2.5").split(",")]
BUDGET_SEC = int(os.environ.get("GENN_BUDGET_MIN", "450")) * 60


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "genN.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd):
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def methods():
    m = []
    for iv in [1, 2, 3, 5, 8, 12, 20]:
        m.append((("Original" if iv == 1 else f"Periodic_{iv}"), {"mode": "periodic", "interval": iv}))
    for th in TH:
        m.append((f"SETComm_th{th:g}", {"mode": "spiking", "threshold": th}))
    for p in [0.1, 0.2, 0.35, 0.5]:
        m.append((f"Random_p{p:g}", {"mode": "random", "random_prob": p}))
    for d in [20, 40, 80]:
        m.append((f"EventTrig_d{d:g}", {"mode": "evtrig", "evtrig_delta": d}))
    return m


def job_cmd(method, seed, n, kw):
    a = [PY, "run_eval_one.py", "--method-name", method, "--seed", str(seed),
         "--num-maps", str(NUM_MAPS), "--group", GROUP, "--map-dir", MAPS_TEST, "--device", "cpu",
         "--n-agents", str(n), "--comm-mode", kw["mode"], "--interval", str(kw.get("interval", 1)),
         "--threshold", str(kw.get("threshold", 1.0)), "--evtrig-delta", str(kw.get("evtrig_delta", 40.0)),
         "--random-prob", str(kw.get("random_prob", 0.2))]
    return a


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
    for n in NS:
        for base, kw in methods():
            nm = f"{base}_N{n}"
            for s in SEEDS:
                if not csv_exists(nm, s):
                    jobs.append((nm, s, n, kw))
    log(f"genN: N={NS} x {len(methods())} methods x {len(SEEDS)} seeds = {len(jobs)} jobs, "
        f"{WORKERS} workers, budget {BUDGET_SEC//60}min")

    done = [0]
    total = len(jobs)

    def one(j):
        nm, s, n, kw = j
        code, out = run(job_cmd(nm, s, n, kw))
        done[0] += 1
        if code != 0:
            log(f"  FAIL {nm} seed{s} (N={n}): {out[-200:]}")
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

    # per-N pivot: communication vs travel/coverage by team size and base method
    try:
        import glob, pandas as pd
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")])
        for c in ["final_coverage_ratio", "total_travel_distance", "total_communication_volume", "num_agents"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["base"] = d.method_name.str.replace(r"_N\d+$", "", regex=True)
        piv = d.groupby(["num_agents", "base"]).agg(
            comm=("total_communication_volume", "mean"),
            travel=("total_travel_distance", "mean"),
            cov=("final_coverage_ratio", "mean")).reset_index().sort_values(["num_agents", "comm"])
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_genN_pivot.csv"), index=False)
        log("per-N pivot (comm/travel/cov by N x method):\n" + piv.to_string(index=False))
    except Exception as e:
        log(f"pivot failed: {e}")

    open(os.path.join(OUT, "GENN_DONE"), "w").close()
    log(f"GENN DONE in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
