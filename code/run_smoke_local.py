"""Ray-free local smoke runner: run ONE episode of TestWorker on CPU.

test_driver.py uses Ray for parallel meta-agents; this driver bypasses Ray so the
mechanism can be checked on a laptop. It does not write the experiment CSV (that is
test_driver's job); it just prints the episode's perf_metrics.

Example:
    python run_smoke_local.py --comm-mode spiking --threshold 0.8 --map-dir ../maps
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comm-mode", default="spiking",
                    choices=["periodic", "spiking", "spiking_learned", "random"])
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--refractory", type=int, default=0)
    ap.add_argument("--random-prob", type=float, default=0.2)
    ap.add_argument("--map-dir", default="../maps")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    # env vars MUST be set before importing test_parameter-dependent modules
    os.environ["BLMRE_COMM_MODE"] = args.comm_mode
    os.environ["BLMRE_SPIKE_THRESHOLD"] = str(args.threshold)
    os.environ["BLMRE_SPIKE_LEAK"] = str(args.leak)
    os.environ["BLMRE_SPIKE_REFRACTORY"] = str(args.refractory)
    os.environ["BLMRE_RANDOM_COMM_PROB"] = str(args.random_prob)
    os.environ["BLMRE_MAP_DIR"] = args.map_dir

    import torch
    import test_parameter as tp
    from model import PolicyNet
    from test_worker import TestWorker

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

    print(f"|#COMM_MODE={tp.COMM_MODE} METHOD={tp.METHOD_NAME} maps={args.map_dir}")
    keys = ["comm_mode", "final_coverage_ratio", "success_rate", "episode_steps",
            "communication_count", "spike_total", "spike_per_robot_mean",
            "mean_broadcast_interval", "total_communication_volume", "computation_time"]
    for ep in range(args.episodes):
        worker = TestWorker(0, net, ep, device=device, save_image=False, greedy=True)
        worker.run_episode()
        m = worker.perf_metrics
        print(f"[ep {ep}] " + "  ".join(
            f"{k}={m.get(k)}" for k in keys if k in m))


if __name__ == "__main__":
    main()
