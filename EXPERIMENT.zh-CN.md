[English](EXPERIMENT.md) | **简体中文**

# 真机验证协议

**要验证什么**:在真实硬件上复现 *synchrony tax*——在**每机广播频率相同**的前提下,去同步广播相对同步广播,
应在**冻结**探索策略 + **真实 WiFi** 信道上降低团队行进距离与感知重叠。

## 1. 硬件
4 台轮式机器人,每台:板载计算机、Livox Mid-360 + FAST-LIO2(里程计/建图)、ROS 2(Humble)+ Nav2、
WiFi(同一网络)。可选:动捕或已测地图用于真值覆盖率。

## 2. 你要运行的代码(本仓)
ROS 2 包 **`ros2_ws/src/svoi_comm`**:
- **`svoi_comm/svoi_core.py` —— `EventTriggeredGate`**:每台机器人的调度器,决定**何时**广播。模式与仿真一致:
  - `periodic`(固定 `interval`)—— **同相位**用作 **synchronised** 基线;每机**相位错开**用作 **even-staggered**;
  - `spiking` / `evtrig` / `random` —— 分布式、自去同步触发(无全局时钟)。
- **`svoi_comm/comm_node.py`**:每台机器人一个节点。订阅自身位姿(FAST-LIO2 的 `/robotI/Odometry`)和可选的
  64 维策略嵌入;运行门控;**仅在门控触发时**把自身状态 `[id, x, y, yaw, emb...]` 广播到共享 WiFi 话题
  `/svoi/state`;其余时刻沿用上次收到的队友(陈旧)状态;把聚合后的 `team_state` 转发给规划器;并记录
  `comm_stats = [fire_rate, bytes_sent, fires, ticks]`。
- **`launch/team.launch.py`**:单机测试(spawn N 个带命名空间的节点)。真机上每台机器人改为只跑**一个**
  `comm_node`,带自己的 `robot_id`。
- **冻结探索策略**在每台机器人上运行(来自上游框架;权重在 `model/clean/checkpoint.pth`),发布动作和
  64 维嵌入;规划器消费 `team_state`。

**要对比的调度条件**(用 `mode` / `interval` 参数设置,全部**同频率**):
1. **同步**:`mode:=periodic interval:=T`,所有机器人同相位(发包时刻一致);
2. **均匀错位(去同步)**:`mode:=periodic interval:=T`,机器人 *k* 用 balanced 的 `phase_offset` 错开
   (错开节点启动时刻,或加一个 `phase_offset` 参数);
3. *(可选)* **事件触发去同步**:`mode:=spiking` 或 `evtrig`(自去同步)。

OW-Desync 需要中心化的激活协调节点(待实现);最小真机演示用条件 **1 vs 2** 即可检验 synchrony tax。

## 3. 怎么一步步操作
1. 每台机器人**编译**:`cd ros2_ws && colcon build && source install/setup.bash`。
2. **每台机器人启动**:FAST-LIO2(发布 `/robotI/Odometry`)、冻结探索器 + Nav2,以及
   `ros2 run svoi_comm comm_node --ros-args -p robot_id:=I -p n_agents:=4 -p mode:=periodic -p interval:=T`。
3. **组网**:所有机器人在同一 WiFi、同一 `ROS_DOMAIN_ID`,确保 `/svoi/state` 能跨网;用 chrony/PTP 对时,
   或记录时钟偏移。
4. **试验**:对每个(场地, 初始布局),把**每种调度**(同步、均匀错位、[事件触发])各跑 **≥5 次**,**配对**。
   全程 `ros2 bag record -a`(或指定话题)。
5. 跑到覆盖率达标(如 95%)或超时;除广播时机外其余完全一致。

**最小可行版**:1 个场地、同步 vs 均匀错位、N=4、固定 T、~5 次配对重复。

## 4. 每次试验录制(每局一个 rosbag)
| 信号 | 用途 |
|---|---|
| 每机 FAST-LIO2 位姿 | 行进距离 |
| 每机占据栅格 + 已探格 | 覆盖率、**感知重叠**(冗余覆盖) |
| 广播事件 `t, sender, payload_bytes`(`comm_stats`) | 广播**次数** + 实际**字节(kB)** |
| 接收事件 `t, receiver, sender` | 信念**陈旧度** τ_i、**峰值团队陈旧度** maxₜ Σ τ_i |
| WiFi:逐消息时延、投递/丢包、吞吐 | 信道真实性 |
| 覆盖曲线、墙钟 | 到达目标覆盖的用时/步数 |
| 俯视视频 | 定性图(扎堆 vs 散开) |

**收发两端都记原始时间戳**,离线计算 τ 与实际字节。

## 5. 最终得到什么
对 rosbag 做离线解析(复用本仓的分析流水线),每种调度得到:团队行进距离、感知重叠、到达覆盖用时/步数;
实际通信**字节(kB)** + **广播次数**(证明"频率相同、字节不同");**峰值团队陈旧度**(机制);以及 WiFi
吞吐/时延/丢包。再做配对 **Wilcoxon**(同步 vs 去同步),即可产出论文素材:
1. **真机结果表**:调度 vs {travel %, overlap %, p, n};
2. **时序/陈旧度图**:对齐 vs 错开的发包时刻;去同步峰值陈旧度更低;
3. **轨迹对比图**(扎堆 vs 散开),来自视频/bag;
4. **WiFi 链路实测数**(证明信道是真实的);
5. 公开的 **rosbag + 解析脚本**,保证可复现。

**预期结果**:在广播次数相同的前提下,去同步降低行进与重叠、且峰值团队陈旧度更低——在硬件上证实 synchrony tax。
