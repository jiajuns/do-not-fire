"""De-sync DOSE-RESPONSE pilot (mechanism test for the cross-backbone generality claim).

Question: does the de-synchronisation benefit grow with how much a frontier planner relies on
the *shared teammate belief* to coordinate? That reliance is exactly the deconfliction weight
FRONTIER_DECONFLICT_W (it penalises moving onto a believed-teammate-occupied node, and that
belief is refreshed only by communication). If raising the weight makes STAGGERED beat
SYNCHRONISED by progressively more, the de-sync effect is mechanistically tied to coordination
through shared belief -- which is why the learned DARS policy (which heavily uses teammate
context) shows it strongly, the utility frontier shows it weakly at the default weight, and the
myopic nearest-frontier (which barely coordinates) does not.

Group setcomm_dcpilot. Interval 12 (where the DARS effect was largest), N=8, broadcast model.
Sweeps FRONTIER_DECONFLICT_W per backbone; paired Wilcoxon staggered vs synchronised.
"""
import os
import sys
import time
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
MAPS = os.path.join(ROOT, "maps_test")
GROUP = "setcomm_dcpilot"
IV = int(os.environ.get("DC_IV", "12"))
WEIGHTS = [float(x) for x in os.environ.get("DC_WEIGHTS", "1,4,8").split(",")]
SEEDS = list(range(int(os.environ.get("DC_SEEDS", "6"))))
N = 8
NUM = 20
WORKERS = int(os.environ.get("DC_WORKERS", "12"))
BUDGET_SEC = int(os.environ.get("DC_BUDGET_MIN", "90")) * 60

BACKBONES = [("fnear", "nearest"), ("futil", "utility")]


def csv_ok(m, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{m}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    method, mode, w, s, fmode = j
    env = dict(os.environ)
    env["BLMRE_GATE_POSITIONS"] = "1"
    env["BLMRE_FRONTIER"] = "1"
    env["BLMRE_FRONTIER_MODE"] = fmode
    env["BLMRE_FRONTIER_DECONFLICT_W"] = str(w)
    cmd = [PY, "run_eval_one.py", "--method-name", method, "--comm-mode", mode, "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", str(N), "--interval", str(IV)]
    return subprocess.run(cmd, cwd=HERE, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def main():
    jobs = []
    for bb, fmode in BACKBONES:
        for w in WEIGHTS:
            wtag = str(w).replace(".", "p")
            for mode, tag in [("periodic", "Sync"), ("staggered", "Stag")]:
                m = f"{bb}_dc{wtag}_{tag}_{IV}"
                for s in SEEDS:
                    if not csv_ok(m, s):
                        jobs.append((m, mode, w, s, fmode))
    t0 = time.time()
    print(f"dcpilot: {len(BACKBONES)} bb x weights {WEIGHTS} x sync/stag x {len(SEEDS)} seeds "
          f"@ iv{IV} = {len(jobs)} jobs, {WORKERS} workers", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    _analyze()
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    open(os.path.join(HERE, "results", "overnight", "DCPILOT_DONE"), "w").close()
    print(f"DCPILOT DONE {int(time.time()-t0)}s", flush=True)
    return 0


def _analyze():
    try:
        import pandas as pd
        from scipy import stats
        fs = glob.glob(f"results/raw/{GROUP}/*.csv")
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        for c in ["total_travel_distance", "overlap_ratio", "map_id", "seed"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["bb"] = d.method_name.str.extract(r"^(fnear|futil)_")
        d["w"] = pd.to_numeric(d.method_name.str.extract(r"_dc([0-9p]+)_")[0].str.replace("p", ".", regex=False),
                               errors="coerce")
        d["tag"] = d.method_name.str.extract(r"_(Sync|Stag)_")
        d = d.dropna(subset=["bb", "w", "tag"])
        rows = []
        for bb, _ in BACKBONES:
            for w in WEIGHTS:
                S = d[(d.bb == bb) & (d.w == w) & (d.tag == "Sync")].set_index(["map_id", "seed"])
                G = d[(d.bb == bb) & (d.w == w) & (d.tag == "Stag")].set_index(["map_id", "seed"])
                idx = S.index.intersection(G.index)
                if len(idx) < 8:
                    rows.append(dict(bb=bb, w=w, n=int(len(idx)), note="too few pairs"))
                    continue
                a = G.loc[idx, "total_travel_distance"].values
                c = S.loc[idx, "total_travel_distance"].values
                try:
                    _, p = stats.wilcoxon(a, c)
                except Exception:
                    p = float("nan")
                rows.append(dict(bb=bb, w=w, sync=round(float(c.mean()), 1), stag=round(float(a.mean()), 1),
                                 stag_minus_sync_pct=round(float(100 * (a.mean() - c.mean()) / c.mean()), 2),
                                 stag_better_frac=round(float((a < c).mean()), 2), n=int(len(idx)),
                                 p=round(float(p), 4)))
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(HERE, "results", "summary", "setcomm_dcpilot_doseresp.csv"), index=False)
        print("=== de-sync DOSE-RESPONSE vs deconfliction weight (stag-sync travel %, neg=de-sync better) ===")
        print(out.to_string(index=False))
    except Exception as e:
        import traceback
        print("analysis failed:", e, traceback.format_exc()[-400:])


if __name__ == "__main__":
    sys.exit(main())
