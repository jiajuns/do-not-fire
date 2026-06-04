"""Supplementary experiments to run AFTER the main scaled run (run_overnight) finishes.

Phase E: classic distance-triggered baseline (evtrig) over a delta-sweep, written into
         the SAME group as the main run so it joins the comm-vs-travel Pareto.
Phase R: robustness to packet loss -- a few methods x drop-rate sweep -> degradation curves.

Reuses the maps_test/ split and trained gates produced by run_overnight. Reuses run_eval_one
(parallel, CPU). Expects BLMRE_GATE_POSITIONS=1 and BLMRE_N_AGENTS=8 from the launcher.
"""

import os
import sys
import glob
import time
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
MAPS_TEST = os.path.join(ROOT, "maps_test")
MODEL_DIR = os.path.join(ROOT, "model")
GROUP_MAIN = "setcomm_full"
GROUP_ROBUST = "setcomm_robust"
OUT = os.path.join(HERE, "results", "overnight")

WORKERS = int(os.environ.get("OVERNIGHT_WORKERS", "14"))
SEEDS = [int(x) for x in os.environ.get("OVERNIGHT_SEEDS", "0,1,2,3,4,5,6,7").split(",")]
NUM_MAPS = int(os.environ.get("OVERNIGHT_NUM_MAPS", "20"))
DELTAS = [float(x) for x in os.environ.get("SUPP_EVTRIG_DELTAS", "10,20,40,80,160").split(",")]
DROPS = [float(x) for x in os.environ.get("SUPP_DROPS", "0.0,0.1,0.3,0.5").split(",")]


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "supplement.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd):
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def job_cmd(method, seed, group, **kw):
    a = [PY, "run_eval_one.py", "--method-name", method, "--seed", str(seed),
         "--num-maps", str(NUM_MAPS), "--group", group, "--map-dir", MAPS_TEST, "--device", "cpu",
         "--comm-mode", kw["mode"], "--interval", str(kw.get("interval", 1)),
         "--threshold", str(kw.get("threshold", 1.0)), "--evtrig-delta", str(kw.get("evtrig_delta", 40.0)),
         "--comm-drop", str(kw.get("comm_drop", 0.0)), "--random-prob", str(kw.get("random_prob", 0.2))]
    if kw.get("gate"):
        a += ["--gate-path", kw["gate"]]
    return a


def run_jobs(jobs):
    done = [0]
    total = len(jobs)

    def one(j):
        method, seed, group, kw = j
        code, out = run(job_cmd(method, seed, group, **kw))
        done[0] += 1
        tail = out.strip().splitlines()[-1] if out.strip() else f"code {code}"
        log(f"  [{done[0]}/{total}] {tail}")
        return code

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))


def find_learned_gate():
    cands = sorted(glob.glob(os.path.join(MODEL_DIR, "spiking_gate_b*.pth")))
    for pref in ("b1.0", "b1", "b0.5"):
        for c in cands:
            if pref in os.path.basename(c):
                return c
    return cands[0] if cands else None


def _phase_r(t0):
    # --- Phase R: packet-loss robustness (writes a separate group) ---
    gate = find_learned_gate()
    robust = [
        {"name": "Original", "mode": "periodic", "interval": 1},
        {"name": "Periodic_5", "mode": "periodic", "interval": 5},
        {"name": "EventTrig_d40", "mode": "evtrig", "evtrig_delta": 40},
        {"name": "SETComm_th1", "mode": "spiking", "threshold": 1.0},
    ]
    if gate:
        robust.append({"name": "SETComm_L", "mode": "spiking_learned", "gate": gate})
        log(f"robustness uses learned gate {os.path.basename(gate)}")
    log(f"Phase R: {len(robust)} methods x drops={DROPS} x {len(SEEDS)} seeds")
    rjobs = []
    for m in robust:
        for dr in DROPS:
            kw = {k: v for k, v in m.items() if k != "name"}
            kw["comm_drop"] = dr
            for s in SEEDS:
                rjobs.append((f"{m['name']}_drop{dr:g}", s, GROUP_ROBUST, kw))
    run_jobs(rjobs)
    run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP_ROBUST}",
         "--group", GROUP_ROBUST, "--output_dir", "results/summary"])
    try:
        import pandas as pd
        d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP_ROBUST}/*.csv")])
        for c in ["final_coverage_ratio", "total_travel_distance", "total_communication_volume"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["base"] = d.method_name.str.replace(r"_drop.*", "", regex=True)
        d["drop"] = d.method_name.str.extract(r"_drop([0-9.]+)").astype(float)
        piv = d.groupby(["base", "drop"]).agg(
            cov=("final_coverage_ratio", "mean"), travel=("total_travel_distance", "mean")).reset_index()
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_robust_pivot.csv"), index=False)
        log("robustness pivot (cov/travel vs drop):\n" + piv.to_string(index=False))
    except Exception as e:
        log(f"robust pivot failed: {e}")
    open(os.path.join(OUT, "SUPP_R_DONE"), "w").close()
    log(f"Phase R done in {(time.time()-t0)/60:.1f} min")
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=["E", "R", "both"])
    args = ap.parse_args()
    phase = args.phase

    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(MAPS_TEST):
        log(f"ERROR: {MAPS_TEST} missing (run run_overnight first)")
        return 1
    t0 = time.time()

    if phase not in ("E", "both"):
        return _phase_r(t0)

    # --- Phase E: evtrig delta-sweep into the main group (joins Pareto) ---
    log(f"Phase E: evtrig deltas={DELTAS} x {len(SEEDS)} seeds")
    ejobs = [(f"EventTrig_d{d:g}", s, GROUP_MAIN, {"mode": "evtrig", "evtrig_delta": d})
             for d in DELTAS for s in SEEDS]
    run_jobs(ejobs)
    log("re-aggregate + analyze main group (now with evtrig)")
    run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP_MAIN}",
         "--group", GROUP_MAIN, "--output_dir", "results/summary"])
    run([PY, "scripts/analyze_setcomm.py", "--group", GROUP_MAIN, "--baseline", "Periodic_5"])

    open(os.path.join(OUT, "SUPP_E_DONE"), "w").close()
    log(f"Phase E done in {(time.time()-t0)/60:.1f} min")
    if phase == "E":
        return 0
    return _phase_r(t0)


if __name__ == "__main__":
    sys.exit(main())
