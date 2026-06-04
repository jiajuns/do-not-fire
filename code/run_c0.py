"""Tiny c0 (LIF floor) sweep for voi_lif: does a smaller floor let the theorem-backed
voi_lif match greedy voi's performance while keeping bounded staleness?
Records max_link_staleness so we can plot the performance vs staleness-bound tradeoff.
Group setcomm_c0; compare against voi/round_robin/random already in setcomm_targeted."""
import os, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PY = sys.executable
MAPS = os.path.join(ROOT, "maps_test"); GROUP = "setcomm_c0"
C0S = [float(x) for x in os.environ.get("C0_GRID", "0.05,0.1,0.2,0.5,1.0").split(",")]
BS = [int(x) for x in os.environ.get("C0_BS", "4,8").split(",")]
SEEDS = list(range(int(os.environ.get("C0_SEEDS", "8"))))
WORKERS = int(os.environ.get("C0_WORKERS", "16")); NUM = int(os.environ.get("C0_NUMMAPS", "20"))


def csv_ok(nm, s):
    p = os.path.join(HERE, "results", "raw", GROUP, f"{GROUP}_{nm}_seed{s}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def one(j):
    nm, c0, B, s = j
    cmd = [PY, "run_eval_one.py", "--method-name", nm, "--comm-mode", "periodic", "--seed", str(s),
           "--num-maps", str(NUM), "--group", GROUP, "--map-dir", MAPS, "--device", "cpu",
           "--n-agents", "8", "--targeted", "--link-budget", str(B), "--target-mode", "voi_lif",
           "--voi-sigma", "80", "--voi-floor", str(c0)]
    return subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def main():
    jobs = []
    for c0 in C0S:
        for B in BS:
            for s in SEEDS:
                nm = f"T_voi_lif_c{c0:g}_B{B}"
                if not csv_ok(nm, s):
                    jobs.append((nm, c0, B, s))
    t0 = time.time(); print(f"c0 run: {len(jobs)} jobs ({len(C0S)} c0 x {len(BS)} B x {len(SEEDS)} seeds)", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, jobs))
    subprocess.run([PY, "scripts/aggregate_results.py", "--input_dir", f"results/raw/{GROUP}",
                    "--group", GROUP, "--output_dir", "results/summary"])
    os.makedirs(os.path.join(HERE, "results", "overnight"), exist_ok=True)
    open(os.path.join(HERE, "results", "overnight", "C0_DONE"), "w").close()
    print(f"C0 DONE {int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
