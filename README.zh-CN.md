[English](README.md) | **简体中文**

# Don't Fire Together —— 面向多机器人探索的去同步通信调度

一个作用在**冻结**的学习型(图注意力)探索策略之上的**测试期通信调度层**,用于带宽受限的多机器人探索。
核心发现:通信稀缺时,可控的成本是**同步,而不是通信量**——机器人按统一时刻广播,会让共享信念同时变陈旧,
从而扎堆。**在相同的每机广播频率下,只错开广播相位**,即可免费恢复效率;**OW-Desync** 把它推广到限距、异构信道。

论文:*Don't Fire Together: De-synchronised Communication Scheduling for Bandwidth-Limited Multi-Robot
Exploration*(MDPI *Sensors*,审稿中)。

## 仓库结构
```
code/            去同步调度器(spiking_comm.py + test_worker/test_parameter 里的 BLMRE_* 门控)、
                 评测运行脚本(run_*.py)、分析(analyze_*.py, owval_*.py)、绘图脚本
                 (house_style.py, regen_*.py, gen_*.py),以及冻结策略的探索框架
                 (model.py, env.py, ...)。scripts/ = 结果聚合工具。
ros2_ws/         真机调度层的 ROS 2 包(svoi_comm)
EXPERIMENT.md    真机验证协议(4 台机器人 + WiFi)
requirements.txt, LICENSE, .gitignore
```
> **权重(checkpoint)**:运行脚本默认从 `model/clean/checkpoint.pth` 读取冻结策略权重(从下方上游项目获取,
> 本仓不分发)。
> **基于公开上游**:冻结探索策略是 Ma 等人的带宽受限多机器人探索器,**原样冻结使用**。本仓只发布我们的调度层、
> 评测与分析。

## 快速开始(仿真)
```bash
conda create -n desync python=3.10 && conda activate desync
pip install -r requirements.txt        # torch, numpy, pandas, scipy, matplotlib, imageio
# 关键的"仅相位"对照:同 nominal rate 下 同步 vs 均匀错位,N=8
python code/run_sync.py
# 跨团队规模 / 丢包 / 传感半径 的泛化
python code/run_syncN.py ; python code/run_ploss.py ; python code/run_srange.py
# OW-Desync 独立验证(peak-AoI 相对 round-robin 的比值)
python code/owval_master.py
# 重新生成所有论文图
python code/regen_oldfigs.py ; python code/regen_owfigs.py ; python code/gen_meaningful_figs.py
```
调度器用 `--comm-mode {periodic,staggered,owdesync,...}` 选择,完整参数见 `code/run_eval_one.py`。

## 真机验证
4 台机器人 + WiFi:完整协议(**用哪些代码、怎么一步步操作、最终得到什么数据和图**)见
**[EXPERIMENT.zh-CN.md](EXPERIMENT.zh-CN.md)**。

## 引用
```bibtex
@article{dontfiretogether2026, title={Don't Fire Together: De-synchronised Communication Scheduling
  for Bandwidth-Limited Multi-Robot Exploration}, journal={Sensors}, year={2026}, note={under review}}
```
相关的协作式稠密 SLAM 系统:Que 等人,*Sensors* **24**(22):7297, 2024,doi:10.3390/s24227297。

## 许可
我们的调度层与分析代码:MIT;上游冻结策略沿用其原许可。
