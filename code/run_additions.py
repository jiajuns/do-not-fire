"""Sweep for the three method additions on top of the frozen policy (eval-only, matched budget):
  (1) predictive belief maintenance: hold | cv (constant-velocity) | kf (Kalman, mean+cov)
  (2) decision-aware (counterfactual Howard) VoI scoring          (target-mode decision_voi)
  (6) Kalman uncertainty-augmented VoI (ehat = residual + beta*sqrt(trace P))

Belief mode and the decision/KF sub-flags are passed via PER-JOB ENV (run_eval_one inherits the
parent env and never overrides these new BLMRE_* vars), so no change to run_eval_one's CLI is
needed. Everything is matched-budget: belief/scoring change only WHAT the scheduler believes/values,
never how many links fire (LINK_BUDGET B fixed), so comm volume is comparable across a fixed B.

Group 'setcomm_additions'. Parallel via ThreadPoolExecutor; wall-clock capped. Writes
results/raw/setcomm_additions/*.csv then aggregates + a (belief x mode x B) pivot.
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
GROUP = "setcomm_additions"
OUT = os.path.join(HERE, "results", "overnight")

WORKERS = int(os.environ.get("ADD_WORKERS", "48"))
SEEDS = [int(x) for x in os.environ.get("ADD_SEEDS", "0,1,2,3,4").split(",")]
NUM_MAPS = int(os.environ.get("ADD_NUM_MAPS", "20"))
BS = [int(x) for x in os.environ.get("ADD_BS", "2,4,8").split(",")]
N_AGENTS = int(os.environ.get("ADD_N_AGENTS", "8"))
SIGMA = float(os.environ.get("ADD_SIGMA", "80"))
BUDGET_SEC = int(os.environ.get("ADD_BUDGET_MIN", "600")) * 60

# belief-maintenance env presets (composed onto each job's env)
BELIEF_ENV = {
    "hold": {"BLMRE_PREDICTIVE_BELIEF": "0", "BLMRE_KALMAN_BELIEF": "0"},
    "cv":   {"BLMRE_PREDICTIVE_BELIEF": "1", "BLMRE_KALMAN_BELIEF": "0"},
    "kf":   {"BLMRE_KALMAN_BELIEF": "1", "BLMRE_A_MAX_PRED": "1.0", "BLMRE_KF_UNCERT_BETA": "1.0"},
}


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "additions.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, env=env)
    return p.returncode, p.stdout


def specs():
    """(method_name, belief, target_mode, B, extra_env_overrides)."""
    out = []
    # --- core grid: belief x mode x B (round_robin/random are belief-independent: cv only) ---
    for b in BS:
        for bel in ("hold", "cv", "kf"):
            for mode in ("voi", "decision_voi", "whittle", "wsum"):
                out.append((f"A_{bel}_{mode}_B{b}", bel, mode, b, {}))
        for mode in ("round_robin", "random"):
            out.append((f"A_cv_{mode}_B{b}", "cv", mode, b, {}))
    # --- DVoI metric ablation (B=4, cv): argmax | kl (logit is the default in the grid) ---
    for met in ("argmax", "kl"):
        out.append((f"A_cv_decision_voi_{met}_B4", "cv", "decision_voi", 4, {"BLMRE_DVOI_METRIC": met}))
    # --- KF uncertainty-weight ablation (B=4, kf, voi): beta in {0, 0.5, 2} (1.0 in grid) ---
    for beta in ("0", "0.5", "2"):
        out.append((f"A_kf_voi_beta{beta}_B4", "kf", "voi", 4, {"BLMRE_KF_UNCERT_BETA": beta}))
    # --- KF process-noise (a_max) sensitivity (B=4, kf, voi): {0.25,0.5,2} (1.0 in grid) ---
    for am in ("0.25", "0.5", "2"):
        out.append((f"A_kf_voi_amax{am}_B4", "kf", "voi", 4, {"BLMRE_A_MAX_PRED": am}))
    return out


def job_cmd(method, mode, b, seed):
    floor = "0.5" if mode == "voi_lif" else "0.0"
    return [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", "periodic",
            "--seed", str(seed), "--num-maps", str(NUM_MAPS), "--group", GROUP,
            "--map-dir", MAPS_TEST, "--device", "cpu", "--n-agents", str(N_AGENTS),
            "--targeted", "--link-budget", str(b), "--target-mode", mode,
            "--voi-sigma", str(SIGMA), "--voi-floor", floor]


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
    for (nm, bel, mode, b, ov) in specs():
        env = dict(BELIEF_ENV[bel]); env.update(ov)
        for s in SEEDS:
            if not csv_exists(nm, s):
                jobs.append((nm, mode, b, s, env))
    log(f"additions: {len(specs())} specs x {len(SEEDS)} seeds = {len(jobs)} jobs; "
        f"B={BS}, N={N_AGENTS}, sigma={SIGMA}, {WORKERS} workers, budget {BUDGET_SEC//60}min")

    done = [0]
    total = len(jobs)

    def one(j):
        nm, mode, b, s, env = j
        code, out = run(job_cmd(nm, mode, b, s), env)
        done[0] += 1
        if code != 0:
            log(f"  FAIL {nm} seed{s}: {out[-300:]}")
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
         "--group", GROUP, "--output_dir", "results/summary"], {})
    _pivot()
    open(os.path.join(OUT, "ADDITIONS_DONE"), "w").close()
    log(f"ADDITIONS DONE in {(time.time()-t0)/60:.1f} min")
    return 0


def _pivot():
    """(belief x mode x B) pivot on the core grid + a quick predictive/decision/kf readout."""
    try:
        import glob
        import re
        import pandas as pd
        fs = glob.glob(f"results/raw/{GROUP}/*.csv")
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        for c in ["final_coverage_ratio", "total_travel_distance", "overlap_ratio",
                  "total_communication_volume", "episode_steps", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        def parse(nm):
            mm = re.match(r"A_(hold|cv|kf)_(voi|decision_voi|whittle|wsum|round_robin|random)_B(\d+)$", str(nm))
            if not mm:
                return (None, None, None)
            return (mm.group(1), mm.group(2), int(mm.group(3)))
        d["belief"], d["mode"], d["B"] = zip(*d.method_name.map(parse))
        core = d.dropna(subset=["belief", "mode", "B"])
        piv = core.groupby(["B", "mode", "belief"]).agg(
            comm=("total_communication_volume", "mean"),
            travel=("total_travel_distance", "mean"),
            overlap=("overlap_ratio", "mean"),
            cov=("final_coverage_ratio", "mean"),
            steps=("episode_steps", "mean"),
            n=("total_travel_distance", "size")).reset_index().sort_values(["B", "mode", "belief"])
        piv.to_csv(os.path.join(HERE, "results", "summary", "setcomm_additions_pivot.csv"), index=False)
        log("additions pivot (B x mode x belief):\n" + piv.to_string(index=False))
    except Exception as e:
        import traceback
        log(f"pivot failed: {e}\n{traceback.format_exc()[-500:]}")


if __name__ == "__main__":
    sys.exit(main())
