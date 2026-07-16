[English](EXPERIMENT.md) | **简体中文**

# 真机验证协议

**要验证什么**:在真实硬件上复现 *synchrony tax*——在**每机地图/位姿广播频率相同**的前提下,
去同步广播相对同步广播,应在**固定的服务端 CURE 分配器** + **真实 WiFi** 信道上降低团队行进距离与感知重叠。

## 0. 修回必补清单 —— Sensors 大修要求的重复试验(2026-07)

审稿人要求**真机上的任务级证据**:重复配对试验,测行进距离和感知重叠,而不只是通信陈旧度。
下面第 1–5 节协议不变;本清单中,已有协议覆盖的打 `[x]`,**新增或必须修正**的为 `[ ]`:

- [x] 对照条件:同步 vs 均匀错位,`period_sec:=4`(第 2 节)
- [x] 每局记录:每机里程计(行进距离)、占据栅格/已探格(覆盖、重叠)、广播+接收事件及字节
      (`/svoi/comm_events`、服务端接收话题、pcap)、WiFi 统计、覆盖曲线、俯拍视频(第 4 节表格)
- [x] 同一场地、配对起始布局,跑到覆盖目标或超时(第 3 节)
- [ ] **至少 5 次配对试验**(上次只有 1 对;配对 Wilcoxon 检验最少 5 对,8 对以上更好)
- [ ] **开跑前修好 robot1 的录包**:前两次 run 它的 bag 里都没有 `/Odometry` 或 `/odom`
      (只有 costmap/rosout)。把它的里程计话题加进录制列表,先录 10 秒烟测,用
      `ros2 bag info` 确认后再正式开跑
- [ ] **每局确认四台车都在动**:上次同步局里 robot3 全程只走了 0.26 m(基本静止)。
      每局开始 30 秒内检查每台车都拿到目标并在移动,否则中止该局重跑
- [ ] 固定 `T = 4 s`(与已发表的演示一致);同步 = 共同相位,错位 = robot 0–3 相位
      `{0, 1, 2, 3}` s
- [ ] 条件顺序交替(sync、desync、desync、sync、……),抵消电量与场景漂移
- [ ] 每局单独一个 run ID 目录(`sync_<时间戳>` / `desync_<时间戳>`),并记录哪两局配成一对
- [ ] 可选:每局开始记录电池电量(有用的协变量)

**数据怎么用**:离线用 `code/parse_real_bags.py` 解析(依赖 `rosbags`,不需要安装 ROS);逐局行进距离来自
里程计,已知地图面积和门控事件来自 bag。跨机器人重叠只能在地图变换到经过验证的同一全局坐标系后计算,
不可把同名的局部 `map` frame 直接相叠。对各局指标做配对 Wilcoxon,结果写进论文真机小节。

## 1. 硬件
4 台轮式机器人,每台:板载计算机、Livox Mid-360 + FAST-LIO2(里程计/建图)、ROS 2(Humble)+ Nav2、
WiFi(同一网络)。可选:动捕或已测地图用于真值覆盖率。

## 2. 你要运行的代码(本仓)
ROS 2 包 **`ros2_ws/src/svoi_comm`**:
- **`PeriodicPhaseSchedule`**:以共同 Unix/ROS 系统时间为基准的秒级调度器;进程晚醒时丢弃错过的 slot,
  不补发突发包。
- **`map_gate_node`**:持续缓存本机 `/map`,仅在触发时发布 `/svoi/gated_map`、`/svoi/fire` 和
  `/svoi/comm_events`。配套补丁让上游 `robot_communication` 只上传门控地图和同 slot 的最新 TF。
- **`comm_node`**:保留作独立状态/AoI 机制测试,不作为本次 CURE 任务级实验的主数据链路。
- **服务端 CURE**(上游 `real_robots_explore`):接收 UDP 地图/TF,融合 `global_map`,检测 frontier,
  固定 greedy 分配器通过可靠 TCP 给各车发送 Nav2 目标。所有条件都使用完全相同的分配器参数。

**要对比的调度条件**(全部**每机每 4 秒一次**):
1. **同步**:`period_sec:=4 phase_sec:=0`,四车同相位;
2. **均匀错位**:`period_sec:=4`,实验索引 *k* 使用 `phase_sec:=k`,即 `{0,1,2,3}` 秒。

OW-Desync 需要中心化的激活协调节点(待实现);最小真机演示用条件 **1 vs 2** 即可检验 synchrony tax。

### 2.1 与现有 SenseBeetle 栈的实际接线

仅运行 `comm_node` 只会产生旁路 `/svoi/state`,不会限制现有 `robot_communication` 的 UDP 地图/TF 上传,
因此不能用于任务级行进/重叠对比。本仓新增 `map_gate_node`:它在本机导航域缓存 `/map`,按统一墙钟计划
发布 `/svoi/gated_map`。必须让 `robot_communication` 订阅门控后的地图:

```bash
# 将本仓部署到机器人 ~/communication/src/svoi_comm 后
cd ~/SenseBeetle
git apply --check ~/communication/src/svoi_comm/integration/robot_communication_map_gate.patch
git apply ~/communication/src/svoi_comm/integration/robot_communication_map_gate.patch
colcon build --packages-select robot_communication nav_bringup
```

正式试验统一使用秒参数:`period_sec:=4`;同步相位全为 `0`,去同步的实验索引 0–3 分别为
`0/1/2/3` 秒。按最新 `SUNNYsyy2005/SenseBettle`(基线 commit `58f89ee`)的一类车清单:
`{1@.48, 2@.43, 7@.141, 8@.145}` 分别映射到实验索引 `{0,1,2,3}`。四台机器必须传入
同一个未来 Unix 时间 `start_time`。例如 `192.168.8.43` 的导航 ID 是 2、实验索引是 1:

```bash
# 导航与地图门控必须在同一域
export ROS_DOMAIN_ID=2
ros2 launch nav_bringup bringup_real.launch.py \
  mode:=mapping lio:=fastlio lio_rviz:=False nav_rviz:=True \
  use_sim_time:=False use_foxglove_bridge:=False use_explore:=False \
  use_hik_camera:=False robot_id:=2 server_ip:=192.168.8.240

source ~/communication/install/setup.bash
ros2 launch svoi_comm map_gate.launch.py robot_id:=1 n_agents:=4 \
  period_sec:=4 phase_sec:=1 start_time:=<四车共同时间戳>
```

服务端使用上游 `real_robots_explore` 启动团队融合与分配(服务端 IP 以实际网卡地址为准):

```bash
ros2 launch real_cure_bringup real_explore.launch.py \
  robot_ids:=1,2,7,8 enable_exploration:=true
```

## 3. 怎么一步步操作

### 四车一键自主探索

现场清空且急停可用后，在本仓根目录运行：

```bash
./run_all.sh sync       # 四车同步通信、自主探索、自动录包
./run_all.sh stop       # 停车并安全保存 rosbag
./run_all.sh desync     # 四车按 0/1/2/3 秒错峰通信
```

启动时输入 `RUN` 作安全确认；已由外部流程确认安全时可添加 `--yes`。任一车的 SSH、串口、门控、
录包或导航启动失败，脚本会停止所有已启动组件。`status` 查看四车状态，`logs` 查看各车最近日志。

### 最简单的两车演示

车 1（`.48`）和车 2（`.43`）的串口、建图导航及服务端已启动后，只需在本仓根目录运行：

```bash
# 同步条件；自动设置共同开始时间、ROS 域、地图话题和相位
./run_real.sh sync

# 结束后停止，再以错位条件运行
./run_real.sh stop
./run_real.sh desync
```

上述命令也会在每台车上自动开始录包，`stop` 会安全关闭并保存。两车的门控输入统一使用
slam_toolbox 实际发布的 `/map`；`.48` 通信节点中名为 `map_postprocessed` 的输入已由补丁重映射到
`/svoi/gated_map`。用 `./run_real.sh status` 查看状态、
`./run_real.sh logs` 查看门控和录包日志。正式四车运行在命令末尾加 `all`；脚本会先检查
四车 SSH 和时钟，任一车辆离线或时钟偏差超过 2 秒都会中止。默认两车模式只能验证发包错峰和联调流程，
不能代替下述四车、至少 5 对的正式统计试验。

如果只要求先取得完整真机数据，可以不启动 CURE 服务端，让每台车的
`bringup_real.launch.py` 使用 `use_explore:=True`，由本机 `explore_lite` 自主给 Nav2 下目标。
`run_real.sh` 会记录 `/Odometry`、`/odom`、`/map`、`/map_postprocessed`、Nav2 动作状态和所有
`/svoi/*` 门控数据。此模式可以分析每车里程、覆盖与通信时序，但两车的探索决策彼此独立，
路线差异不能解释为同步/错峰通信造成的任务性能差异。

1. **递归获取上游**:`git clone --recurse-submodules git@github.com:SUNNYsyy2005/SenseBettle.git`,固定并记录 commit。
2. 每台机器人部署/编译 `svoi_comm`,应用 2.1 节补丁后重编 `robot_communication`。
3. 服务端启动 CURE;每台一类车在自己的 `ROS_DOMAIN_ID` 启动串口、FAST-LIO2、Nav2,
   `use_explore:=False`,并把 `server_ip` 指向 CURE 服务端。
4. 用 chrony/PTP 对时;从控制电脑生成一个至少 20 秒后的 `start_time`,四车分别运行
   `scripts/start_map_gate.sh <实验索引> sync|desync <start_time>`。
5. 每车运行 `scripts/record_trial.sh`,服务端运行 `scripts/record_server_trial.sh`;至少一端运行
   `scripts/capture_wifi.sh` 抓取 UDP 12130 和 TCP 12140 的真实流量。
6. 对每个场地/初始布局把同步和错位各跑 **≥5 次配对**,条件顺序交替;到覆盖阈值或统一超时即停止。

**最小可行版**:1 个场地、同步 vs 均匀错位、N=4、固定 T、~5 次配对重复。

## 4. 每次试验录制(每局一个 rosbag)
| 信号 | 用途 |
|---|---|
| 每机 FAST-LIO2 位姿 | 行进距离 |
| 每机占据栅格 + 已探格 | 覆盖率、**感知重叠**(冗余覆盖) |
| `/svoi/comm_events` + `/svoi/fire` | 计划发送时间、次数、ROS 序列化大小 |
| 服务端 `/robot_N/map` + `map_synced_tf` | 实际接收时间、地图陈旧度、全局融合输入 |
| pcap:UDP 12130 + TCP 12140 | 实际字节、包数、时延/丢包证据 |
| WiFi:逐消息时延、投递/丢包、吞吐 | 信道真实性 |
| 覆盖曲线、墙钟 | 到达目标覆盖的用时/步数 |
| 俯视视频 | 定性图(扎堆 vs 散开) |

**收发两端都记原始时间戳**;ROS 序列化大小不能冒充 WiFi 实际字节,后者以 pcap 为准。

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
