# Don't Fire Together — De-synchronised Communication Scheduling for Multi-Robot Exploration

A **test-time communication-scheduling layer** for bandwidth-limited multi-robot exploration on a
**frozen** learned (graph-attention) explorer. The finding: under scarce communication the controllable
cost is **synchronisation, not volume** — when robots broadcast on a common schedule their shared belief
goes stale together and they herd. De-synchronising the broadcast **phase** at the same per-robot rate
recovers efficiency for free; **OW-Desync** extends this to range-limited, heterogeneous channels.
Paper: *Don't Fire Together: De-synchronised Communication Scheduling for Bandwidth-Limited Multi-Robot
Exploration* (MDPI *Sensors*, under review).

一个面向带宽受限多机器人探索的**测试期通信调度层**,作用在**冻结**的图注意力探索策略之上。核心发现:通信稀缺时
可控的成本是**同步,而非通信量**——机器人按统一时刻广播会让共享信念同时变陈旧,导致扎堆;在相同每机广播频率下
**错开相位**即可免费恢复效率;**OW-Desync** 把它推广到限距、异构信道。

---

## Repository layout / 仓库结构
```
code/            de-synchronised schedulers (spiking_comm.py + BLMRE_* gating in
                 test_worker/test_parameter), evaluation runners (run_*.py),
                 analysis (analyze_*.py, owval_*.py), figure generators
                 (house_style.py, regen_*.py, gen_*.py), and the frozen-policy
                 explorer framework (model.py, env.py, ...).  scripts/ = aggregation utils.
ros2_ws/         ROS 2 package scaffold (svoi_comm) for the real-robot scheduling layer
EXPERIMENT.md    real-robot validation protocol (4 robots + WiFi)
requirements.txt, LICENSE, .gitignore
```
> **Checkpoint.** The runners expect the frozen-policy weights at `model/clean/checkpoint.pth`
> (obtain from the upstream project below; not redistributed here). / 运行需把冻结策略权重放到
> `model/clean/checkpoint.pth`(从下方上游获取,本仓不再分发)。

> **Built on a public upstream.** The frozen exploration policy is the publicly available
> bandwidth-limited multi-robot explorer of Ma et al. (used **frozen, unmodified**). This repo releases
> only **our scheduling layer**, runners, and analysis. / 本仓库仅发布**我们的调度层**+评测+分析;冻结策略来自
> 公开上游(Ma et al.),原样不改。

## Quick start (simulation) / 快速开始(仿真)
```bash
conda create -n desync python=3.10 && conda activate desync
pip install -r requirements.txt        # torch, numpy, pandas, scipy, matplotlib
# decisive phase-only control: synchronised vs even-staggered at matched rate, N=8
python code/run_sync.py                 # writes results/raw/setcomm_sync + summary
# generalisation across team size / planner backbone / packet loss / sensor range
python code/run_syncN.py                # N sweep
python code/run_ploss.py ; python code/run_srange.py
# OW-Desync standalone validation (peak-AoI ratio vs round-robin)
python code/owval_master.py
# regenerate all paper figures
python code/regen_oldfigs.py ; python code/regen_owfigs.py ; python code/gen_meaningful_figs.py
```
Each scheduler is selected with `--comm-mode {periodic,staggered,owdesync,...}`; see
`code/run_eval_one.py` for the full flag list. / 用 `--comm-mode` 选调度器,完整参数见 `run_eval_one.py`。

## Real-robot validation / 实机验证
4 robots + WiFi: see **[EXPERIMENT.md](EXPERIMENT.md)** for the full protocol — hardware, conditions,
what to record (rosbags), metrics, and the paired-Wilcoxon analysis. / 4 机 + WiFi 的完整协议(硬件、条件、
录什么、指标、统计)见 **[EXPERIMENT.md](EXPERIMENT.md)**。

## Reproducing the figures / 复现图
`code/regen_oldfigs.py` and `code/regen_owfigs.py` read the result CSVs and write `paper/.../figures`.
All figures share one house style (`code/house_style.py`). / 见上述脚本,统一风格在 `house_style.py`。

## Citation / 引用
```bibtex
@article{dontfiretogether2026, title={Don't Fire Together: De-synchronised Communication Scheduling
  for Bandwidth-Limited Multi-Robot Exploration}, journal={Sensors}, year={2026}, note={under review}}
```
Related collaborative dense-SLAM system: Que et al., *Sensors* **24**(22):7297, 2024,
doi:10.3390/s24227297.

## License / 许可
Our scheduling layer and analysis code: MIT (or as chosen). The upstream frozen policy keeps its
original license. / 本层 MIT;上游策略沿用其原许可。
