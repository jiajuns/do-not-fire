"""Clean exploration visualization: render one episode as a single-panel map with
robot trajectories, current positions, and frontiers; assemble a GIF. Overrides the
codebase's busy 4-panel plot. Comm config is taken from BLMRE_* env (set before running),
so the SAME map can be rendered under rich vs scarce communication for comparison.

Run (config set via env), e.g.:
  BLMRE_N_AGENTS=4 BLMRE_GATE_POSITIONS=1 BLMRE_COMM_MODE=periodic BLMRE_COMM_INTERVAL=1 \
  BLMRE_MAP_DIR=../maps_test VIZ_LABEL=richcomm VIZ_MAP=3 python viz_explore.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
torch.set_num_threads(2)

VIZ_MAP = int(os.environ.get("VIZ_MAP", "0"))
VIZ_LABEL = os.environ.get("VIZ_LABEL", "run")
VIZ_DIR = os.environ.get("VIZ_DIR", "results/viz")
os.makedirs(VIZ_DIR, exist_ok=True)

import test_parameter as tp
from model import PolicyNet
import test_worker as TW
from test_worker import TestWorker
from utils import get_cell_position_from_coords, get_frontier_in_map

_frames = []


def clean_render(self, step):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(self.env.robot_belief, cmap="gray")
    try:
        fr = get_frontier_in_map(self.env.belief_info)
        if fr is not None and len(fr):
            frc = get_cell_position_from_coords(np.array(fr), self.env.belief_info).reshape(-1, 2)
            ax.scatter(frc[:, 0], frc[:, 1], c="#ff7f0e", s=3, zorder=1, label="frontier")
    except Exception:
        pass
    cmap = plt.cm.tab10
    for robot in self.robot_list:
        col = cmap(robot.id % 10)
        tx = (np.array(robot.trajectory_x) - robot.global_map_info.map_origin_x) / robot.cell_size
        ty = (np.array(robot.trajectory_y) - robot.global_map_info.map_origin_y) / robot.cell_size
        ax.plot(tx, ty, "-", color=col, lw=2.2, zorder=2)
        rc = get_cell_position_from_coords(robot.location, robot.global_map_info)
        ax.plot(rc[0], rc[1], "o", color=col, ms=12, mec="white", mew=1.6, zorder=3)
    mt = max(r.travel_dist for r in self.robot_list)
    ax.set_title("%s   step %d   coverage %.1f%%   max-travel %.0f"
                 % (VIZ_LABEL, step, 100 * float(self.env.explored_rate), mt), fontsize=12)
    ax.axis("off"); fig.tight_layout()
    fn = "%s/%s_%03d.png" % (VIZ_DIR, VIZ_LABEL, step)
    fig.savefig(fn, dpi=110); plt.close(fig)
    _frames.append(fn); self.env.frame_files.append(fn)


# install clean renderer; disable the buggy windows-path make_gif (we assemble ourselves)
TestWorker.plot_local_env = clean_render
TW.make_gif = lambda *a, **k: None


def main():
    device = torch.device("cpu")
    net = PolicyNet(tp.NODE_INPUT_DIM, tp.EMBEDDING_DIM).to(device)
    ck = os.path.join(os.path.dirname(__file__), "..", "model", "clean", "checkpoint.pth")
    try:
        c = torch.load(ck, map_location=device)
    except Exception:
        c = torch.load(ck, map_location=device, weights_only=False)
    net.load_state_dict(c["policy_model"]); net.eval()
    np.random.seed(0); torch.manual_seed(0)
    w = TestWorker(0, net, VIZ_MAP, device=device, save_image=True, greedy=True)
    w.run_episode()
    gif = "%s/%s.gif" % (VIZ_DIR, VIZ_LABEL)
    with imageio.get_writer(gif, mode="I", duration=0.6) as wr:
        for f in _frames:
            wr.append_data(imageio.imread(f))
        for _ in range(4):  # hold last frame
            wr.append_data(imageio.imread(_frames[-1]))
    print("VIZ_DONE %s frames=%d coverage=%.3f steps=%s travel=%.0f"
          % (gif, len(_frames), float(w.env.explored_rate),
             w.perf_metrics.get("episode_steps"),
             max(r.travel_dist for r in w.robot_list)))


if __name__ == "__main__":
    main()
