"""
Lightweight REINFORCE trainer for the learned spiking communication gate (SET-Comm-L).

The pretrained exploration policy (backbone) is FROZEN; only the tiny GateNet that
decides WHEN each robot broadcasts is updated. Per-episode reward:

    R = final_coverage_ratio - beta * broadcast_fraction
    broadcast_fraction = total_broadcasts / (n_agents * steps)   in [0, 1]

This reuses the Env + Agent rollout directly (no Ray), so it runs on CPU or a single
GPU. Train and eval should use DISJOINT map folders (set --map-dir / BLMRE_MAP_DIR)
to keep the gate's generalization claim leakage-free.

Example:
    BLMRE_MAP_DIR=maps_gate_train python train_spiking_gate.py \
        --episodes 400 --beta 0.5 --lr 1e-3 --out model/spiking_gate.pth
"""

import argparse
import os
from copy import deepcopy

import numpy as np
import torch


def _send(robots, msg, sender_id):
    """Broadcast a message clone into every robot's mailbox (mirrors TestWorker.send_msg)."""
    for r in robots:
        if len(r.msgs[sender_id]) > 5:
            r.msgs[sender_id].pop(0)
        r.msgs[sender_id].append(msg.clone())


def rollout_episode(policy_net, gate, ep_idx, device, max_step, n_agents):
    """Run one training episode with the stochastic learned gate.

    Returns (logprobs, final_coverage, broadcast_fraction, steps).
    The backbone runs under no_grad; only gate logprobs carry gradient.
    """
    from env import Env
    from agent import Agent

    env = Env(ep_idx, plot=False, test=True)
    robots = [Agent(i, policy_net, deepcopy(env), None, device) for i in range(n_agents)]
    comm_loc = env.robot_locations.copy()  # last-broadcast positions (position-gated comms)

    def plocs(rid):
        l = comm_loc.copy()
        l[rid] = env.robot_locations[rid].copy()
        return l

    for r in robots:
        r.update_graph(env.belief_info, deepcopy(env.robot_locations[r.id]))
    for r in robots:
        r.update_planning_state(plocs(r.id))

    gate.reset_episode()
    gate.collect_logprobs = True
    total_broadcasts = 0
    steps = 0

    for i in range(max_step):
        steps = i + 1
        obs_list = []
        # --- decide + broadcast ---
        for r in robots:
            obs = r.get_no_padding_observation()
            obs_list.append(obs)
            lni, lnpm, lem, cli, cle, lepm = obs
            cc = torch.tensor(r.location, dtype=torch.float32, device=device).reshape(1, 1, 2)
            with torch.no_grad():
                _, csf = r.policy_net.get_current_state_feature(lni, lnpm, lem, cli, cc)
            if gate.should_broadcast(r.id, i, csf.detach(),
                                     coverage=float(env.explored_rate), utility=r.utility):
                _send(robots, csf.detach(), r.id)
                comm_loc[r.id] = env.robot_locations[r.id].copy()  # announce position
                total_broadcasts += 1
        # --- plan next waypoints ---
        selected = []
        dists = []
        for r in robots:
            stacked = r.get_stacked_msg()
            nxt, _, _ = r.select_next_waypoint(obs_list[r.id], stacked)
            selected.append(nxt)
            dists.append(np.linalg.norm(nxt - r.location))
        selected = np.array(selected).reshape(-1, 2)
        order = np.argsort(np.array(dists))
        seq = np.array(selected)[order]
        # --- resolve duplicate targets (mirror TestWorker conflict resolution) ---
        for j, loc in enumerate(seq):
            solved = seq[:j]
            while loc[0] + loc[1] * 1j in solved[:, 0] + solved[:, 1] * 1j:
                rid = order[j]
                nbrs = robots[rid].local_node_manager.local_nodes_dict.nearest_neighbors(loc.tolist(), 25)
                for node in nbrs:
                    coords = node.data.coords
                    if coords[0] + coords[1] * 1j in solved[:, 0] + solved[:, 1] * 1j:
                        continue
                    loc = coords
                    break
                seq[j] = loc
                selected[rid] = loc
        # --- advance the world ---
        for r, loc in zip(robots, selected):
            env.step(loc, r.id)
            r.env.step(loc, r.id)
            r.update_graph(r.env.belief_info, deepcopy(r.env.robot_locations[r.id]))
        gate.end_step(float(env.explored_rate))
        for r in robots:
            r.update_planning_state(plocs(r.id))
        if env.explored_rate >= 0.95:
            break

    coverage = float(env.explored_rate)
    bcast_frac = total_broadcasts / max(1, n_agents * steps)
    travel = float(np.sum([r.travel_dist for r in robots]))  # total team travel (matches eval metric scale)
    return gate.pop_logprobs(), coverage, bcast_frac, steps, travel


def main():
    ap = argparse.ArgumentParser(description="REINFORCE trainer for the SET-Comm learned gate.")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--beta", type=float, default=0.5, help="communication penalty weight in the reward")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4, help="episodes accumulated per gradient step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-dir", type=str, default=None, help="training map folder (sets BLMRE_MAP_DIR)")
    ap.add_argument("--out", type=str, default="model/spiking_gate.pth")
    # gate membrane hyperparameters (fixed during training; only firing head is learned)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--refractory", type=int, default=0)
    ap.add_argument("--w-drift", type=float, default=1.0)
    ap.add_argument("--w-cov", type=float, default=0.0)
    ap.add_argument("--travel-ref", type=float, default=1200.0, help="normalizer for travel in the efficiency reward")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    if args.map_dir:
        os.environ["BLMRE_MAP_DIR"] = args.map_dir
    os.environ.setdefault("BLMRE_COMM_MODE", "spiking_learned")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # import after env vars so test_parameter/parameter pick them up
    import test_parameter as tp
    from test_parameter import TEST_N_AGENTS, model_path, USE_GPU, NODE_INPUT_DIM, EMBEDDING_DIM
    from parameter import MAX_EPISODE_STEP
    from model import PolicyNet
    from spiking_comm import CommGate, GateNet

    device = torch.device("cuda") if (USE_GPU and torch.cuda.is_available()) else torch.device("cpu")

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "model", "clean", "checkpoint.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{model_path}/checkpoint.pth"
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy_net.load_state_dict(ckpt["policy_model"])
    policy_net.eval()
    for p in policy_net.parameters():
        p.requires_grad_(False)

    gate_net = GateNet().to(device)
    gate_net.train()
    opt = torch.optim.Adam(gate_net.parameters(), lr=args.lr)

    gate = CommGate(
        TEST_N_AGENTS, mode="spiking_learned", device=device,
        threshold=args.threshold, leak=args.leak, refractory=args.refractory,
        w_drift=args.w_drift, w_cov=args.w_cov, gate_net=gate_net,
        rng=np.random.default_rng(args.seed),
    )

    baseline = 0.0
    b_decay = 0.9
    batch_losses = []
    history = []

    for ep in range(args.episodes):
        logps, cov, frac, steps, travel = rollout_episode(
            policy_net, gate, ep, device, MAX_EPISODE_STEP, TEST_N_AGENTS
        )
        # coverage saturates under position-gating; reward LOW travel + LOW communication
        R = cov - travel / args.travel_ref - args.beta * frac
        baseline = b_decay * baseline + (1 - b_decay) * R
        history.append((cov, frac, R, travel))

        if logps:
            advantage = R - baseline
            ep_loss = -advantage * torch.stack(logps).sum()
            batch_losses.append(ep_loss)

        if batch_losses and (ep + 1) % args.batch == 0:
            opt.zero_grad()
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            opt.step()
            batch_losses = []

        if (ep + 1) % args.log_every == 0:
            recent = history[-args.log_every:]
            mc = np.mean([h[0] for h in recent])
            mf = np.mean([h[1] for h in recent])
            mr = np.mean([h[2] for h in recent])
            mt = np.mean([h[3] for h in recent])
            print(f"[ep {ep+1:4d}] coverage={mc:.3f} bcast_frac={mf:.3f} "
                  f"travel={mt:.0f} reward={mr:.3f} baseline={baseline:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(gate_net.state_dict(), args.out)
    print(f"|#Saved learned gate -> {args.out}")


if __name__ == "__main__":
    main()
