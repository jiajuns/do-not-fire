"""Mechanism probe for Pillar A: does SYNCHRONISED broadcasting make robots decide alike and herd?

Reuses the eval harness (TestWorker) unmodified. After each episode it reads the per-step team
positions the metrics tracker already stores and computes two direct mechanism observables,
contrasting synchronised vs staggered periodic broadcasting at a matched budget (interval 12, N=8,
broadcast model):
  * team dispersion  = mean over steps of the mean pairwise inter-robot distance (herding -> LOWER);
  * move correlation = mean over steps of the mean pairwise cosine similarity of one-step
    displacement vectors across robots (correlated decisions -> HIGHER).

Launcher/worker split: each (mode,seed) runs as its own subprocess so the comm mode is configured
BEFORE the simulator imports (mirrors run_eval_one). Workers write results/raw/setcomm_mechprobe/
<mode>_seed<s>.csv; the launcher aggregates + a paired sync-vs-stagger summary. Group setcomm_mechprobe.
"""
import os
import sys
import csv
import glob
import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
GROUP = "setcomm_mechprobe"
N = int(os.environ.get("MP_N", "8"))
IV = int(os.environ.get("MP_IV", "12"))
NUM = int(os.environ.get("MP_NUM_MAPS", "20"))
SEEDS = list(range(int(os.environ.get("MP_SEEDS", "6"))))
WORKERS = int(os.environ.get("MP_WORKERS", "8"))
RAW = os.path.join(HERE, "results", "raw", GROUP)
SUM = os.path.join(HERE, "results", "summary")


# ----------------------------- worker -----------------------------
def worker(mode, seed):
    os.environ["BLMRE_COMM_MODE"] = mode
    os.environ["BLMRE_COMM_INTERVAL"] = str(IV)
    os.environ["BLMRE_GATE_POSITIONS"] = "1"
    os.environ["BLMRE_MAP_DIR"] = os.path.join(ROOT, "maps_test")
    os.environ["BLMRE_N_AGENTS"] = str(N)
    os.environ["BLMRE_SEED"] = str(seed)
    os.environ["BLMRE_NUM_TEST"] = str(NUM)
    os.environ["BLMRE_EXPERIMENT_GROUP"] = GROUP
    os.environ["BLMRE_EXPERIMENT_NAME"] = f"{GROUP}_{mode}_seed{seed}"
    os.environ["BLMRE_METHOD_NAME"] = mode

    import numpy as np
    import torch
    torch.set_num_threads(1)
    import test_parameter as tp
    from model import PolicyNet
    from test_worker import TestWorker
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    net = PolicyNet(tp.NODE_INPUT_DIM, tp.EMBEDDING_DIM).to(device)
    ck = os.path.join(HERE, "..", "model", "clean", "checkpoint.pth")
    try:
        c = torch.load(ck, map_location=device)
    except Exception:
        c = torch.load(ck, map_location=device, weights_only=False)
    net.load_state_dict(c["policy_model"])
    net.eval()

    def mech(tracker):
        ph = tracker.position_history
        n = len(ph)
        T = min(len(h) for h in ph)
        if n < 2 or T < 2:
            return float("nan"), float("nan")
        P = np.stack([np.stack(ph[r][:T]) for r in range(n)])  # (n,T,2)
        disp, corr = [], []
        for t in range(T):
            pts = P[:, t, :]
            disp.append(float(np.mean([np.linalg.norm(pts[i] - pts[j])
                                       for i, j in itertools.combinations(range(n), 2)])))
            if t >= 1:
                mv = P[:, t, :] - P[:, t - 1, :]
                cs = []
                for i, j in itertools.combinations(range(n), 2):
                    a, b = mv[i], mv[j]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > 1e-6 and nb > 1e-6:
                        cs.append(float(np.dot(a, b) / (na * nb)))
                if cs:
                    corr.append(float(np.mean(cs)))
        return (float(np.mean(disp)) if disp else float("nan"),
                float(np.mean(corr)) if corr else float("nan"))

    os.makedirs(RAW, exist_ok=True)
    rows = []
    for ep in range(NUM):
        w = TestWorker(0, net, ep, device=device, save_image=False, greedy=True)
        w.run_episode()
        d, c2 = mech(w._metrics_tracker)
        rows.append(dict(map_id=ep, seed=seed, mode=mode, dispersion=d, move_corr=c2,
                         overlap=float(w.perf_metrics.get("overlap_ratio", float("nan"))),
                         travel=float(w.perf_metrics.get("total_travel_distance", float("nan")))))
    p = os.path.join(RAW, f"{mode}_seed{seed}.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"worker done {mode} seed{seed} -> {p}", flush=True)
    return 0


# ----------------------------- launcher -----------------------------
def launch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(SUM, exist_ok=True)
    jobs = [(m, s) for s in SEEDS for m in ["periodic", "staggered"]]

    def one(j):
        mode, seed = j
        if os.path.exists(os.path.join(RAW, f"{mode}_seed{seed}.csv")):
            return 0
        env = dict(os.environ)
        env["MP_WORKER"] = f"{mode},{seed}"
        return subprocess.run([PY, "run_mechprobe.py"], cwd=HERE, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    print(f"mechprobe: {len(jobs)} workers (sync/stag x {len(SEEDS)} seeds), {WORKERS} parallel", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))
    _analyze()
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    open(os.path.join(HERE, "results", "overnight", "MECHPROBE_DONE"), "w").close()
    print("MECHPROBE DONE", flush=True)


def _analyze():
    import numpy as np
    import pandas as pd
    from scipy import stats
    fs = glob.glob(os.path.join(RAW, "*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out = []
    for met in ["dispersion", "move_corr", "overlap", "travel"]:
        S = d[d["mode"] == "periodic"].set_index(["map_id", "seed"])[met]
        G = d[d["mode"] == "staggered"].set_index(["map_id", "seed"])[met]
        idx = S.index.intersection(G.index)
        a = G.loc[idx].values
        c = S.loc[idx].values
        try:
            _, p = stats.wilcoxon(a, c)
        except Exception:
            p = float("nan")
        out.append(dict(metric=met, sync=round(float(np.nanmean(c)), 4), stag=round(float(np.nanmean(a)), 4),
                        stag_minus_sync_pct=round(float(100 * (np.nanmean(a) - np.nanmean(c)) / np.nanmean(c)), 2),
                        n=int(len(idx)), p=round(float(p), 5)))
    pd.DataFrame(out).to_csv(os.path.join(SUM, "setcomm_mechprobe_paired.csv"), index=False)
    print("=== MECHANISM: synchronised vs staggered (matched budget, neg = stag lower) ===")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    w = os.environ.get("MP_WORKER")
    if w:
        mode, seed = w.split(",")
        sys.exit(worker(mode, int(seed)))
    sys.exit(launch())
