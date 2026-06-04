"""Deep overnight-2 run: a large EVAL-ONLY sweep (no training) to produce a smooth,
high-significance communication-vs-travel Pareto. Reuses the gates trained by #13 and
the maps_test split. Parallel eval fills all cores; a wall-clock budget caps it ~7.5h.

Auto-launched by deepwatch after the main pipeline (#13 + supplement) finishes.
Writes group 'setcomm_deep' (separate from the main results).
"""

import os
import sys
import glob
import time
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
MAPS_TEST = os.path.join(ROOT, "maps_test")
MODEL_DIR = os.path.join(ROOT, "model")
GROUP = "setcomm_deep"
OUT = os.path.join(HERE, "results", "overnight")

WORKERS = int(os.environ.get("DEEP_WORKERS", "16"))
SEEDS = [int(x) for x in os.environ.get("DEEP_SEEDS", "0,1,2,3,4,5,6,7,8,9,10,11").split(",")]
NUM_MAPS = int(os.environ.get("DEEP_NUM_MAPS", "20"))
BUDGET_SEC = int(os.environ.get("DEEP_BUDGET_MIN", "450")) * 60


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "deep.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd):
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def job_cmd(method, seed, **kw):
    a = [PY, "run_eval_one.py", "--method-name", method, "--seed", str(seed),
         "--num-maps", str(NUM_MAPS), "--group", GROUP, "--map-dir", MAPS_TEST, "--device", "cpu",
         "--comm-mode", kw["mode"], "--interval", str(kw.get("interval", 1)),
         "--threshold", str(kw.get("threshold", 1.0)), "--evtrig-delta", str(kw.get("evtrig_delta", 40.0)),
         "--comm-drop", str(kw.get("comm_drop", 0.0)), "--random-prob", str(kw.get("random_prob", 0.2))]
    if kw.get("gate"):
        a += ["--gate-path", kw["gate"]]
    return a


def build_matrix():
    m = []
    for iv in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        m.append((("Original" if iv == 1 else f"Periodic_{iv}"), {"mode": "periodic", "interval": iv}))
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0]:
        m.append((f"SETComm_th{th:g}", {"mode": "spiking", "threshold": th}))
    for g in sorted(glob.glob(os.path.join(MODEL_DIR, "spiking_gate_b*.pth"))):
        b = os.path.basename(g)[len("spiking_gate_"):-len(".pth")]
        m.append((f"SETComm_L_{b}", {"mode": "spiking_learned", "gate": g}))
    for p in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]:
        m.append((f"Random_p{p:g}", {"mode": "random", "random_prob": p}))
    for d in [5, 10, 20, 30, 40, 60, 80, 120, 160]:
        m.append((f"EventTrig_d{d:g}", {"mode": "evtrig", "evtrig_delta": d}))
    for nm, kw in [("Abl_driftcov", {"mode": "spiking", "threshold": 1.0, "w_cov": 1.0}),
                   ("Abl_leak0.7", {"mode": "spiking", "threshold": 1.0, "leak": 0.7}),
                   ("Abl_leak0.9", {"mode": "spiking", "threshold": 1.0, "leak": 0.9}),
                   ("Abl_refr1", {"mode": "spiking", "threshold": 1.0, "refractory": 1}),
                   ("Abl_refr3", {"mode": "spiking", "threshold": 1.0, "refractory": 3})]:
        m.append((nm, kw))
    return m


def csv_exists(method, seed):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{method}_seed{seed}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(MAPS_TEST):
        log(f"ERROR {MAPS_TEST} missing")
        return 1
    t0 = time.time()
    matrix = build_matrix()
    jobs = [(nm, s, kw) for (nm, kw) in matrix for s in SEEDS if not csv_exists(nm, s)]
    log(f"deep run: {len(matrix)} methods x {len(SEEDS)} seeds = {len(jobs)} jobs, "
        f"{WORKERS} workers, budget {BUDGET_SEC//60}min")

    done = [0]
    total = len(jobs)

    def one(j):
        nm, s, kw = j
        code, out = run(job_cmd(nm, s, **kw))
        done[0] += 1
        if done[0] % 20 == 0:
            log(f"  {done[0]}/{total} (elapsed {int((time.time()-t0)/60)}min)")
        return code

    # budget-guarded parallel launch
    ex = ThreadPoolExecutor(max_workers=WORKERS)
    futures = set()
    i = 0
    while i < len(jobs) or futures:
        while len(futures) < WORKERS and i < len(jobs):
            if time.time() - t0 >= BUDGET_SEC:
                break
            futures.add(ex.submit(one, jobs[i]))
            i += 1
        if not futures:
            break
        _, futures = wait(futures, return_when=FIRST_COMPLETED)
        if time.time() - t0 >= BUDGET_SEC and i < len(jobs):
            log(f"BUDGET reached; launched {i}/{len(jobs)} jobs, draining {len(futures)} running")
    ex.shutdown(wait=True)
    log(f"eval finished/stopped: launched {i}/{len(jobs)} jobs in {int((time.time()-t0)/60)}min")

    run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
         "--group", GROUP, "--output_dir", "results/summary"])
    run([PY, "scripts/analyze_setcomm.py", "--group", GROUP, "--baseline", "Periodic_5"])
    open(os.path.join(OUT, "DEEP_DONE"), "w").close()
    log(f"DEEP DONE in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
