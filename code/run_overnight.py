"""Unattended overnight pipeline for SET-Comm (run inside tmux).

Phases:
  1. split maps into disjoint maps_train / maps_test (symlinks)
  2. smoke: 1 episode (spiking) to confirm the stack runs
  3. train a small beta-sweep of learned gates (GPU, sequential)
  4. full eval matrix in parallel (CPU): periodic / spiking-thresholds / learned / random / ablation
  5. aggregate_results + analyze_setcomm (Pareto + paired stats)
  6. write REPORT.md with the headline verdict

Outputs under code/results/ ; logs + STATUS + REPORT under code/results/overnight/.
Config via env: OVERNIGHT_WORKERS, OVERNIGHT_TRAIN_EPISODES, OVERNIGHT_SEEDS, OVERNIGHT_NUM_MAPS.
"""

import os
import sys
import glob
import time
import shutil
import subprocess
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))          # .../jeit/code
ROOT = os.path.dirname(HERE)                                # .../jeit
PY = sys.executable
GROUP = "setcomm_full"
MAPS_ALL = os.path.join(ROOT, "maps")
MAPS_TRAIN = os.path.join(ROOT, "maps_train")
MAPS_TEST = os.path.join(ROOT, "maps_test")
MODEL_DIR = os.path.join(ROOT, "model")
OUT = os.path.join(HERE, "results", "overnight")
LOG = os.path.join(OUT, "run.log")
STATUS = os.path.join(OUT, "STATUS.txt")

WORKERS = int(os.environ.get("OVERNIGHT_WORKERS", "12"))
TRAIN_EP = int(os.environ.get("OVERNIGHT_TRAIN_EPISODES", "250"))
SEEDS = [int(x) for x in os.environ.get("OVERNIGHT_SEEDS", "0,1,2,3,4").split(",")]
NUM_MAPS = int(os.environ.get("OVERNIGHT_NUM_MAPS", "10"))
N_TEST_MAPS = NUM_MAPS
N_TRAIN_MAPS = int(os.environ.get("OVERNIGHT_TRAIN_MAPS", "200"))
BETAS = [float(x) for x in os.environ.get("OVERNIGHT_BETAS", "0.3,0.6,1.0").split(",")]
TRAIN_BUDGET_SEC = int(os.environ.get("OVERNIGHT_TRAIN_BUDGET_MIN", "360")) * 60


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_status(msg):
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()}\n{msg}\n")
    log("STATUS: " + msg)


def run(cmd, **kw):
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, **kw)
    return p.returncode, p.stdout


def split_maps():
    # rank maps by free-area and use the HARDEST (largest) ones, so coordination
    # (hence communication) actually matters. Train/test disjoint, no leakage.
    import numpy as np
    from skimage import io, color, util
    from skimage.measure import block_reduce

    def free_area(p):
        raw = io.imread(p)
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                raw = color.rgba2rgb(raw)
            g = util.img_as_ubyte(color.rgb2gray(raw)).astype(int)
        else:
            g = raw.astype(int)
        return int((block_reduce(g, 2, np.min) > 150).sum())

    maps = sorted(glob.glob(os.path.join(MAPS_ALL, "*.png")))
    need = N_TEST_MAPS + N_TRAIN_MAPS
    log(f"ranking {len(maps)} maps by free-area (hardest {need})...")
    ranked = sorted(maps, key=free_area, reverse=True)[:need]
    test = ranked[:N_TEST_MAPS]
    train = ranked[N_TEST_MAPS:need]
    for d, sel in [(MAPS_TEST, test), (MAPS_TRAIN, train)]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for src in sel:
            os.symlink(src, os.path.join(d, os.path.basename(src)))
    log(f"maps: {len(test)} test, {len(train)} train (hardest by free-area, disjoint)")


def smoke():
    code, out = run([PY, "run_eval_one.py", "--method-name", "Smoke", "--comm-mode", "spiking",
                     "--seed", "0", "--num-maps", "1", "--group", "_smoke",
                     "--map-dir", MAPS_TEST, "--threshold", "0.8", "--device", "cpu"])
    ok = code == 0 and "DONE" in out
    log(f"smoke {'OK' if ok else 'FAIL'}\n{out[-800:]}")
    return ok


def train_gates(t0):
    gates = {}
    for b in BETAS:
        if time.time() - t0 > TRAIN_BUDGET_SEC:
            log(f"train budget {TRAIN_BUDGET_SEC//60}min reached; skipping remaining betas {[x for x in BETAS if x >= b]}")
            break
        out_path = os.path.join(MODEL_DIR, f"spiking_gate_b{b:g}.pth")
        set_status(f"training gate beta={b} ({TRAIN_EP} ep) -> {out_path}")
        code, out = run([PY, "train_spiking_gate.py", "--episodes", str(TRAIN_EP),
                         "--beta", str(b), "--map-dir", MAPS_TRAIN, "--out", out_path,
                         "--seed", "0"])
        with open(os.path.join(OUT, f"train_b{b:g}.log"), "w", encoding="utf-8") as f:
            f.write(out)
        if code == 0 and os.path.exists(out_path):
            gates[b] = out_path
            log(f"gate beta={b} trained -> {out_path}\n{out[-400:]}")
        else:
            log(f"gate beta={b} FAILED (code {code})\n{out[-800:]}")
    return gates


def build_matrix(gates):
    m = []
    for iv in [1, 2, 3, 5, 8, 10, 15, 20]:
        name = "Original" if iv == 1 else f"Periodic_{iv}"
        m.append({"name": name, "mode": "periodic", "interval": iv})
    for th in [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0]:
        m.append({"name": f"SETComm_th{th:g}", "mode": "spiking", "threshold": th})
    for b, path in gates.items():
        m.append({"name": f"SETComm_L_b{b:g}", "mode": "spiking_learned", "gate": path})
    for p in [0.05, 0.1, 0.2, 0.35, 0.5, 0.7]:
        m.append({"name": f"Random_p{p:g}", "mode": "random", "random_prob": p})
    m.append({"name": "Abl_driftcov", "mode": "spiking", "threshold": 1.0, "w_drift": 1.0, "w_cov": 1.0})
    m.append({"name": "Abl_leak0.8", "mode": "spiking", "threshold": 1.0, "leak": 0.8})
    m.append({"name": "Abl_leak0.9", "mode": "spiking", "threshold": 1.0, "leak": 0.9})
    m.append({"name": "Abl_refr2", "mode": "spiking", "threshold": 1.0, "refractory": 2})
    m.append({"name": "Abl_refr3", "mode": "spiking", "threshold": 1.0, "refractory": 3})
    return m


def job_cmd(method, seed):
    a = [PY, "run_eval_one.py", "--method-name", method["name"], "--comm-mode", method["mode"],
         "--seed", str(seed), "--num-maps", str(NUM_MAPS), "--group", GROUP,
         "--map-dir", MAPS_TEST, "--device", "cpu",
         "--interval", str(method.get("interval", 1)),
         "--threshold", str(method.get("threshold", 1.0)),
         "--leak", str(method.get("leak", 1.0)),
         "--refractory", str(method.get("refractory", 0)),
         "--w-drift", str(method.get("w_drift", 1.0)),
         "--w-cov", str(method.get("w_cov", 0.0)),
         "--random-prob", str(method.get("random_prob", 0.2))]
    if method.get("gate"):
        a += ["--gate-path", method["gate"]]
    return a


def csv_exists(method, seed):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{method['name']}_seed{seed}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def run_matrix(matrix):
    jobs = [(mtd, s) for mtd in matrix for s in SEEDS]
    todo = [(m, s) for (m, s) in jobs if not csv_exists(m, s)]
    log(f"eval matrix: {len(jobs)} jobs ({len(jobs) - len(todo)} already done), {WORKERS} workers")
    done = [0]
    total = len(todo)

    def one(job):
        m, s = job
        code, out = run(job_cmd(m, s))
        done[0] += 1
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        if code != 0:
            log(f"  [{done[0]}/{total}] FAIL {m['name']} seed{s} (code {code}): {out[-300:]}")
        else:
            log(f"  [{done[0]}/{total}] {tail}")
        if done[0] % 10 == 0:
            set_status(f"eval {done[0]}/{total}")
        return code

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, todo))


def analyze():
    set_status("aggregate + analyze")
    c1, o1 = run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                  "--group", GROUP, "--output_dir", "results/summary"])
    log("aggregate:\n" + o1[-600:])
    c2, o2 = run([PY, "scripts/analyze_setcomm.py", "--group", GROUP, "--baseline", "Periodic_5"])
    log("analyze:\n" + o2[-1500:])
    return o2


def write_report():
    import pandas as pd
    rep = [f"# SET-Comm overnight report", f"_generated {datetime.datetime.now().isoformat()}_", ""]
    sdir = os.path.join(HERE, "results", "summary", GROUP)
    try:
        summ = pd.read_csv(os.path.join(sdir, f"{GROUP}_setcomm_summary.csv"))
        cols = ["method_name", "n", "final_coverage_ratio_mean", "total_communication_volume_mean",
                "mean_broadcast_interval_mean", "on_pareto_front"]
        cols = [c for c in cols if c in summ.columns]
        summ = summ.sort_values("total_communication_volume_mean")
        rep += ["## Per-method summary (sorted by communication)", "",
                summ[cols].to_string(index=False), ""]
        front = set(summ[summ.get("on_pareto_front", False)]["method_name"]) if "on_pareto_front" in summ else set()
        learned_on_front = sorted([x for x in front if x.startswith("SETComm_L")])
        rep += ["## Pareto front", "", f"on front: {sorted(front)}", "",
                f"learned gates on front: {learned_on_front or 'NONE'}", ""]
    except Exception as e:
        rep += [f"(summary read failed: {e})"]
    try:
        stats = pd.read_csv(os.path.join(sdir, f"{GROUP}_paired_stats.csv"))
        key = stats[stats["target"].astype(str).str.startswith("SETComm_L")]
        rep += ["## Headline: learned gate vs Periodic_5 (paired)", "",
                key.to_string(index=False), "",
                "Interpretation (position-gated, efficiency framing): want communication",
                "mean_diff < 0 (p<0.05) AND travel mean_diff not significantly > 0.", ""]
        # crude verdict (efficiency framing: less comm at maintained travel efficiency)
        verdict = "INCONCLUSIVE"
        try:
            comm = key[key["metric"] == "total_communication_volume"]
            trav = key[key["metric"] == "total_travel_distance"]
            less_comm = ((comm["mean_diff"] < 0) & (comm["wilcoxon_p"] < 0.05)).any()
            travel_ok = ((trav["mean_diff"] <= 0) | (trav["wilcoxon_p"] > 0.05)).any()
            verdict = ("LEARNED GATE WINS (less comm, efficiency maintained)" if (less_comm and travel_ok)
                       else ("less comm but travel worse" if less_comm
                             else "no significant comm reduction vs Periodic_5"))
        except Exception:
            pass
        rep += [f"## VERDICT: {verdict}", ""]
    except Exception as e:
        rep += [f"(stats read failed: {e})"]
    rep += ["## Artifacts",
            f"- summary CSVs: results/summary/{GROUP}_summary_paper.csv (+ numeric)",
            f"- Pareto + stats: results/summary/{GROUP}/",
            f"- raw per-episode: results/raw/{GROUP}/",
            f"- gate training logs: results/overnight/train_b*.log"]
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    log("wrote REPORT.md")


def main():
    os.makedirs(OUT, exist_ok=True)
    shutil.rmtree(os.path.join(HERE, "results", "raw", GROUP), ignore_errors=True)
    t0 = time.time()
    set_status("starting")
    log(f"workers={WORKERS} train_ep={TRAIN_EP} seeds={SEEDS} num_maps={NUM_MAPS} betas={BETAS}")

    split_maps()
    set_status("smoke")
    if not smoke():
        set_status("ABORTED at smoke (stack failed to run)")
        return 1

    gates = train_gates(t0)
    if not gates:
        log("WARN: no learned gates trained; continuing with fixed/periodic/random only")

    matrix = build_matrix(gates)
    set_status(f"eval ({len(matrix)} methods x {len(SEEDS)} seeds)")
    run_matrix(matrix)

    out = analyze()
    write_report()

    open(os.path.join(OUT, "DONE"), "w").close()
    set_status(f"DONE in {(time.time()-t0)/60:.1f} min")
    log("ALL DONE. See results/overnight/REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
