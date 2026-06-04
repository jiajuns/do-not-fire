"""Run ONE (method, seed) over a set of maps; write one per-episode CSV + coverage curves.

Ray-free, defaults to CPU so many instances run in parallel (the simulator is CPU-bound).
The gate is built automatically by TestWorker from the BLMRE_* env vars set here, so this
covers periodic / spiking / spiking_learned / random uniformly.

Used as the parallel job unit by run_overnight.py.
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-name", required=True)
    ap.add_argument("--comm-mode", required=True,
                    choices=["periodic", "staggered", "spiking", "spiking_learned", "random", "evtrig", "cwd", "submod", "vlw", "desync", "stagfix", "owdesync"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-maps", type=int, default=10)
    ap.add_argument("--n-agents", type=int, default=None, help="override team size (sets BLMRE_N_AGENTS)")
    ap.add_argument("--sensor-range", type=float, default=None, help="override sensor range (sets BLMRE_SENSOR_RANGE)")
    ap.add_argument("--group", required=True)
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--refractory", type=int, default=0)
    ap.add_argument("--w-drift", type=float, default=1.0)
    ap.add_argument("--w-cov", type=float, default=0.0)
    ap.add_argument("--random-prob", type=float, default=0.2)
    ap.add_argument("--evtrig-delta", type=float, default=40.0)
    ap.add_argument("--comm-drop", type=float, default=0.0)
    ap.add_argument("--targeted", action="store_true",
                    help="enable targeted/pairwise comm under a hard per-step link budget")
    ap.add_argument("--link-budget", type=int, default=4, help="B: directed links per step")
    ap.add_argument("--target-mode", default="voi",
                    choices=["voi", "voi_lif", "wsum", "err", "stale", "prox", "whittle", "decision_voi", "round_robin", "random"])
    ap.add_argument("--voi-sigma", type=float, default=50.0, help="proximity kernel width (meters)")
    ap.add_argument("--voi-floor", type=float, default=0.0, help="LIF floor current (voi_lif)")
    ap.add_argument("--voi-weights", default="", help="comma list of 5 weights for mode wsum")
    ap.add_argument("--gate-path", default="")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    # configure BEFORE importing test_parameter (it reads env at import time)
    os.environ["BLMRE_COMM_MODE"] = a.comm_mode
    os.environ["BLMRE_COMM_INTERVAL"] = str(a.interval)
    os.environ["BLMRE_SPIKE_THRESHOLD"] = str(a.threshold)
    os.environ["BLMRE_SPIKE_LEAK"] = str(a.leak)
    os.environ["BLMRE_SPIKE_REFRACTORY"] = str(a.refractory)
    os.environ["BLMRE_SPIKE_W_DRIFT"] = str(a.w_drift)
    os.environ["BLMRE_SPIKE_W_COV"] = str(a.w_cov)
    os.environ["BLMRE_RANDOM_COMM_PROB"] = str(a.random_prob)
    os.environ["BLMRE_EVTRIG_DELTA"] = str(a.evtrig_delta)
    os.environ["BLMRE_COMM_DROP"] = str(a.comm_drop)
    if a.targeted:
        os.environ["BLMRE_COMM_TARGETED"] = "1"
        os.environ["BLMRE_LINK_BUDGET"] = str(a.link_budget)
        os.environ["BLMRE_TARGET_MODE"] = a.target_mode
        os.environ["BLMRE_VOI_SIGMA"] = str(a.voi_sigma)
        os.environ["BLMRE_VOI_FLOOR"] = str(a.voi_floor)
        if a.voi_weights:
            os.environ["BLMRE_VOI_WEIGHTS"] = a.voi_weights
    if a.gate_path:
        os.environ["BLMRE_SPIKE_GATE_PATH"] = a.gate_path
    os.environ["BLMRE_MAP_DIR"] = a.map_dir
    if a.n_agents:
        os.environ["BLMRE_N_AGENTS"] = str(a.n_agents)
    if a.sensor_range:
        os.environ["BLMRE_SENSOR_RANGE"] = str(a.sensor_range)
    os.environ["BLMRE_SEED"] = str(a.seed)
    os.environ["BLMRE_NUM_TEST"] = str(a.num_maps)
    os.environ["BLMRE_EXPERIMENT_GROUP"] = a.group
    exp_name = f"{a.group}_{a.method_name}_seed{a.seed}"
    os.environ["BLMRE_EXPERIMENT_NAME"] = exp_name
    os.environ["BLMRE_METHOD_NAME"] = a.method_name

    import numpy as np
    import torch
    torch.set_num_threads(1)  # one sim per core; avoid OMP oversubscription across parallel jobs
    import test_parameter as tp
    from parameter import MAX_EPISODE_STEP, SENSOR_RANGE
    from model import PolicyNet
    from test_worker import TestWorker
    from experiment_logger import EvaluationLogger, build_episode_row
    from episode_metrics import save_coverage_curve

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")

    net = PolicyNet(tp.NODE_INPUT_DIM, tp.EMBEDDING_DIM).to(device)
    ck = a.checkpoint or os.path.join(os.path.dirname(__file__), "..", "model", "clean", "checkpoint.pth")
    try:
        c = torch.load(ck, map_location=device)
    except Exception:
        c = torch.load(ck, map_location=device, weights_only=False)
    net.load_state_dict(c["policy_model"])
    net.eval()

    logger = EvaluationLogger(exp_name, group=a.group)
    row_defaults = {
        "method_name": a.method_name, "seed": a.seed, "num_agents": tp.TEST_N_AGENTS,
        "sensor_range": SENSOR_RANGE, "communication_interval": a.interval,
        "overlap_penalty_weight": 0.0, "max_episode_steps": MAX_EPISODE_STEP,
        "checkpoint_path": ck,
    }
    for ep in range(a.num_maps):
        w = TestWorker(0, net, ep, device=device, save_image=False, greedy=True)
        w.run_episode()
        m = w.perf_metrics
        m["map_id"] = ep  # episode->map deterministic; use it for cross-method pairing
        logger.log_episode(build_episode_row(m, ep, row_defaults))
        cs = m.get("coverage_steps")
        if cs is not None:
            save_coverage_curve(exp_name, a.seed, ep, cs, m.get("coverage_values", []))
    print(f"DONE {a.method_name} seed{a.seed} ({a.num_maps} maps)")


if __name__ == "__main__":
    main()
