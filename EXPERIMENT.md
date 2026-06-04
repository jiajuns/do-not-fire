# Real-Robot Validation Protocol / 实机验证协议

Goal: reproduce the **synchrony tax** on real hardware — at a **matched per-robot broadcast rate**,
de-synchronised broadcasting (even-staggered / OW-Desync) should reduce team travel and sensing overlap
versus synchronised broadcasting, on the **frozen** explorer, over a **real WiFi** channel.

目标:在真实硬件上复现 **synchrony tax**——在**每机广播频率相同**的前提下,去同步广播(even-staggered /
OW-Desync)相对同步广播,应在**冻结**策略 + **真实 WiFi** 信道上降低团队行进距离与感知重叠。

---

## 1. Hardware & software / 硬件与软件
- 4 wheeled robots, each: onboard compute (Jetson/NUC), Livox Mid-360 LiDAR, FAST-LIO2 odometry/mapping,
  ROS 2 + Nav2, WiFi (infrastructure AP or ad-hoc). Optional: motion-capture or a surveyed map for
  ground-truth coverage.
- The **frozen** pretrained explorer policy runs per robot; the de-synchronised **scheduling layer**
  gates *when* each robot publishes its `(pose, map embedding)` message on a shared ROS 2 topic. On a
  silent step, receivers keep the last received copy (stale).
- 4 台轮式机器人,每台:板载算力、Livox Mid-360、FAST-LIO2、ROS 2 + Nav2、WiFi;可选动捕/已测地图做真值。
  每台跑**冻结**策略;调度层只决定**何时**广播 `(位姿, 地图嵌入)`;静默步接收方沿用上次收到的(陈旧)副本。

## 2. Conditions / 实验条件
- **Independent variable = broadcast schedule** at the **same period T** (matched nominal rate):
  `synchronised` (all fire together) vs `even-staggered` (phase φ_k = round(kT/N)) vs `OW-Desync`.
- Sweep at least one period (e.g., T = 4, 8, 12 steps or seconds). Optionally degrade WiFi (distance,
  contention) to test under loss.
- **自变量 = 广播调度**(同一周期 T、同 nominal rate):同步 / 均匀错位 / OW-Desync;至少扫一个周期 T;可加
  距离/干扰制造丢包。

## 3. Procedure / 流程
1. Pick **K ≥ 3 indoor arenas/start layouts**. For each, run **each schedule**, **R ≥ 5 repeats**,
   pairing trials by (arena, start configuration) so sync vs de-sync are matched.
2. Run each trial until a coverage target (e.g., 95%) or a time cap.
3. Keep everything identical across schedules except the broadcast timing.
- 选 **K ≥ 3** 个室内场地/初始布局;每个场地跑**每种调度**、**R ≥ 5** 次重复;按(场地, 初始构型)**配对**。
  跑到覆盖率达标(如 95%)或超时;除广播时机外其余完全一致。

**Minimal viable demo / 最小可行版**: 1 arena, `synchronised` vs `even-staggered`, N = 4, fixed T,
~5 paired repeats. Even this moves the paper from "planned" to "demonstrated".

## 4. What to RECORD (one rosbag per trial) / 每次实验录制(每局一个 rosbag)
| Signal / 信号 | Use / 用途 |
|---|---|
| per-robot FAST-LIO2 pose / 每机位姿轨迹 | total travel distance / 行进距离 |
| per-robot occupancy map + explored cells / 占据栅格+已探格 | coverage, **sensing overlap** (redundant coverage) / 覆盖率、感知重叠 |
| broadcast events: `t, sender, payload_bytes` / 广播事件 | broadcast **count** + realised **volume (kB)**, schedule phase / 次数+实际字节、相位 |
| receive events: `t, receiver, sender` / 接收事件 | belief **staleness** age τ_i, **peak team staleness** maxₜ Σ τ_i / 陈旧度、峰值团队陈旧度 |
| WiFi stats: per-msg latency, delivery/loss, throughput / WiFi 时延/丢包/吞吐 | channel realism / 信道真实性 |
| coverage-over-time, wall-clock / 覆盖曲线、墙钟 | exploration time to target / 到达目标用时 |
| top-down video / 俯视视频 | qualitative figure (herd vs spread) / 定性图(扎堆 vs 散开) |

Record raw timestamps on **both** send and receive so staleness τ and realised volume are computed
offline (do not trust a single clock; sync clocks via chrony/PTP or log offsets).
收发两端都记原始时间戳,离线算 τ 和实际字节;用 chrony/PTP 对时或记录时钟偏移。

## 5. Metrics (mirror the simulation) / 指标(对齐仿真)
Primary: **total team travel**, **sensing overlap ratio**. Secondary: steps/time to target coverage,
realised comm **volume (kB)** and **broadcast count** (to show the *rate is matched, volume is not*),
**peak team staleness** (the mechanism), WiFi throughput/latency/loss.
主指标:团队行进距离、感知重叠;次要:到达覆盖用时、实际字节+广播次数(证明"频率相同、字节不同")、
峰值团队陈旧度(机制)、WiFi 吞吐/时延/丢包。

## 6. Analysis / 统计分析
Paired **Wilcoxon signed-rank** (synchronised vs even-staggered; synchronised vs OW-Desync) per metric
across (arena, repeat) pairs; report **%change, p, and bootstrap 95% CI** — identical pipeline to the
simulation tables. Expected: de-sync cuts travel/overlap; OW-Desync ≥ even-staggering on freshness and
does no harm on exploration; de-sync has a lower peak team staleness at equal broadcast count.
配对 Wilcoxon(同步 vs 均匀错位 / vs OW-Desync),报 %变化、p、bootstrap 95% CI,与仿真同一套分析。
预期:去同步降 travel/overlap;OW-Desync 在新鲜度上 ≥ 均匀错位且不损探索;同广播次数下去同步峰值陈旧度更低。

## 7. What the PAPER needs / 论文需要落地的产物
1. A table: schedule vs {travel %, overlap %, p, n} on real robots. / 真机结果表。
2. A timing/staleness plot (sync ticks aligned vs staggered; peak staleness). / 时序/陈旧度图。
3. A trajectory comparison figure (herd vs spread) from the video/bags. / 轨迹对比图。
4. WiFi throughput/latency/loss numbers (channel was real). / WiFi 链路实测数。
5. Released rosbags + analysis scripts for reproducibility. / 公开 rosbag + 分析脚本。
