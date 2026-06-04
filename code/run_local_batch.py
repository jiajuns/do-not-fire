"""Ray-free LOCAL batch: run several comm methods over a few maps, write the standard
per-episode CSVs (results/raw/<group>/), so aggregate_results.py + analyze_setcomm.py
can produce the Pareto front + paired stats locally on CPU.

This is for quick local validation of the whole pipeline, NOT paper-scale numbers.

Example:
    python run_local_batch.py --group setcomm_local --num-maps 4
    python scripts/aggregate_results.py --input_dir results/raw/setcomm_local --group setcomm_local
    python scripts/analyze_setcomm.py --group setcomm_local --baseline Periodic_5
"""

import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="setcomm_local")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--num-maps", type=int, default=4, help="episodes per (method, seed)")
    ap.add_argument("--map-dir", default="../maps")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gate-path", default="model/spiking_gate.pth")
    args = ap.parse_args()

    os.environ["BLMRE_MAP_DIR"] = args.map_dir
    os.environ["BLMRE_NUM_TEST"] = str(args.num_maps)

    import torch
    import numpy as np
    import test_parameter as tp
    from parameter import MAX_EPISODE_STEP, SENSOR_RANGE
    from model import PolicyNet
    from test_worker import TestWorker
    from spiking_comm import CommGate, GateNet
    from experiment_logger import EvaluationLogger, build_episode_row
    from episode_metrics import save_coverage_curve

    # method matrix: (name, mode, kwargs)
    methods = [
        ("Original",      "periodic",        {"interval": 1}),
        ("Periodic_5",    "periodic",        {"interval": 5}),
        ("Random_p0.3",   "random",          {"random_prob": 0.3}),
        ("SETComm_th0.8", "spiking",         {"threshold": 0.8}),
        ("SETComm_th1.5", "spiking",         {"threshold": 1.5}),
        ("SETComm_th3",   "spiking",         {"threshold": 3.0}),
    ]

    device = torch.device("cpu")
    net = PolicyNet(tp.NODE_INPUT_DIM, tp.EMBEDDING_DIM).to(device)
    ckpt_path = args.checkpoint or os.path.join(
        os.path.dirname(__file__), "..", "model", "clean", "checkpoint.pth")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["policy_model"])
    net.eval()

    # fresh raw dir for this group (avoid append-on-rerun)
    raw_dir = os.path.join(os.path.dirname(__file__), "results", "raw", args.group)
    shutil.rmtree(raw_dir, ignore_errors=True)

    n = tp.TEST_N_AGENTS

    def make_gate(mode, kw):
        if mode == "spiking_learned":
            gn = GateNet()
            if os.path.exists(args.gate_path):
                gn.load_state_dict(torch.load(args.gate_path, map_location="cpu"))
            gn.eval()
            return CommGate(n, mode=mode, gate_net=gn,
                            threshold=kw.get("threshold", 1.0))
        return CommGate(n, mode=mode,
                        threshold=kw.get("threshold", 1.0),
                        random_prob=kw.get("random_prob", 0.2))

    for method_name, mode, kw in methods:
        if mode == "periodic":
            tp.COMMUNICATION_INTERVAL = int(kw.get("interval", 1))
        for seed in args.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            exp_name = f"{args.group}_{method_name}_seed{seed}"
            logger = EvaluationLogger(exp_name, group=args.group)
            row_defaults = {
                "method_name": method_name, "seed": seed, "num_agents": n,
                "sensor_range": SENSOR_RANGE,
                "communication_interval": kw.get("interval", 1),
                "overlap_penalty_weight": 0.0, "max_episode_steps": MAX_EPISODE_STEP,
                "checkpoint_path": ckpt_path,
            }
            covs, comms = [], []
            for ep in range(args.num_maps):
                worker = TestWorker(0, net, ep, device=device, save_image=False, greedy=True)
                worker.comm_gate = make_gate(mode, kw)
                worker.run_episode()
                m = worker.perf_metrics
                m["map_id"] = ep  # episode->map is deterministic; use it for pairing
                row = build_episode_row(m, ep, row_defaults)
                logger.log_episode(row)
                cs = m.get("coverage_steps")
                if cs is not None:
                    save_coverage_curve(exp_name, seed, ep, cs, m.get("coverage_values", []))
                covs.append(m.get("final_coverage_ratio"))
                comms.append(m.get("total_communication_volume"))
            print(f"{method_name:14s} seed{seed}: "
                  f"cov~{np.mean(covs):.3f}  comm~{np.mean(comms):.0f}  ({args.num_maps} maps)")

    print(f"\n|#Raw CSVs -> {raw_dir}")
    print("|#Next: aggregate_results.py + analyze_setcomm.py on this group")


if __name__ == "__main__":
    main()
