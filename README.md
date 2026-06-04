**English** | [简体中文](README.zh-CN.md)

# Don't Fire Together — De-synchronised Communication Scheduling for Multi-Robot Exploration

A **test-time communication-scheduling layer** for bandwidth-limited multi-robot exploration on a
**frozen** learned (graph-attention) explorer. The finding: under scarce communication the controllable
cost is **synchronisation, not volume** — when robots broadcast on a common schedule their shared belief
goes stale together and they herd. De-synchronising the broadcast **phase** at the same per-robot rate
recovers efficiency for free; **OW-Desync** extends this to range-limited, heterogeneous channels.

Paper: *Don't Fire Together: De-synchronised Communication Scheduling for Bandwidth-Limited Multi-Robot
Exploration* (MDPI *Sensors*, under review).

## Repository layout
```
code/            de-synchronised schedulers (spiking_comm.py + BLMRE_* gating in
                 test_worker/test_parameter), evaluation runners (run_*.py),
                 analysis (analyze_*.py, owval_*.py), figure generators
                 (house_style.py, regen_*.py, gen_*.py), frozen-policy explorer
                 framework (model.py, env.py, ...). scripts/ = aggregation utils.
ros2_ws/         ROS 2 package (svoi_comm) for the real-robot scheduling layer
EXPERIMENT.md    real-robot validation protocol (4 robots + WiFi)
requirements.txt, LICENSE, .gitignore
```
> **Checkpoint.** The runners expect the frozen-policy weights at `model/clean/checkpoint.pth`
> (obtain from the upstream project; not redistributed here).
> **Built on a public upstream.** The frozen explorer is the publicly available bandwidth-limited
> multi-robot explorer of Ma et al., used **frozen and unmodified**. This repo releases only our
> scheduling layer, runners, and analysis.

## Quick start (simulation)
```bash
conda create -n desync python=3.10 && conda activate desync
pip install -r requirements.txt        # torch, numpy, pandas, scipy, matplotlib, imageio
# decisive phase-only control: synchronised vs even-staggered at matched rate, N=8
python code/run_sync.py
# generalisation across team size / packet loss / sensor range
python code/run_syncN.py ; python code/run_ploss.py ; python code/run_srange.py
# OW-Desync standalone validation (peak-AoI ratio vs round-robin)
python code/owval_master.py
# regenerate all paper figures
python code/regen_oldfigs.py ; python code/regen_owfigs.py ; python code/gen_meaningful_figs.py
```
Each scheduler is selected with `--comm-mode {periodic,staggered,owdesync,...}`; see
`code/run_eval_one.py` for the full flag list.

## Real-robot validation
4 robots + WiFi: the full protocol (which code to run, step-by-step operation, and exactly what data
and figures you get) is in **[EXPERIMENT.md](EXPERIMENT.md)**.

## Citation
```bibtex
@article{dontfiretogether2026, title={Don't Fire Together: De-synchronised Communication Scheduling
  for Bandwidth-Limited Multi-Robot Exploration}, journal={Sensors}, year={2026}, note={under review}}
```
Related collaborative dense-SLAM system: Que et al., *Sensors* **24**(22):7297, 2024,
doi:10.3390/s24227297.

## License
Our scheduling layer and analysis code: MIT. The upstream frozen policy keeps its original license.
