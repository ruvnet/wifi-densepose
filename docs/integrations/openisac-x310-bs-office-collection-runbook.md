# OpenISAC X310 BS 端办公室采集手册

> [!CAUTION]
> 本文的数据解释与采集命令基于 2026-07-16 对抗审查之前的协议，现已暂停执行。旧数据只能作为历史传输样本，不能支持目标、运动、人数、姿态或生命体征结论。恢复采集前请按 `x310-rf-direct.md` 的成对、版本化、仅回环契约重写实验方案。

版本日期：2026-07-14

本手册用于已经完成 Stage G' 的 BS 端（连接 NI USRP X310、UHD、OpenISAC
与 RuView 的主机）采集办公室场景数据。它不是从零部署手册，也不是让 agent
自动操控硬件的授权书。

当前已知事实：链路已经在单站 X310 上验证通过；OpenISAC 使用 3.1 GHz、50
Msps、`enable_backend_sensing_processing: true`，bridge 能持续将结果转为
`openisac-rd`。backend processing 后目标由远端 `range_bin=890` 转至近场
`range_bin=6-9`。本阶段的目标是建立可重复的办公室环境数据集，验证目标相对
变化，不是重做链路连通性验证。

## 1. 人与 agent 的分工

用户负责：

- 手动执行每一条命令，判断是否允许继续发射 RF。
- 保持 X310、天线、线缆、朝向、频点与增益不变，除非明确开始新的对照实验。
- 记录房间内其他人的明显走动、开门、移动椅子等不可控事件。
- 将每阶段的命令输出和观察结果贴回 BS 端 agent。

BS 端 agent 负责：

- 每次只给一个阶段的一组命令，等待用户回传后再继续。
- 仅收集、检查、汇总和解释数据；不得自行修改 OpenISAC 配置、TX gain、频点、
 采样率或启动新的 RF 发射。
- 不要执行 `kill`、`pkill`、重启 X310、修改网卡或使用 sudo，除非用户明确授权。
- 遇到 underrun、overrun、decode error、端口冲突或 RF 安全不明确时停止并收集证据。

办公室无法清空不是阻塞条件。静态办公室背景就是本实验的基线；关键是硬件几何
保持不变、动作单一、标签完整。

## 2. 给 BS 端 agent 的启动提示词

将以下文本直接贴给 BS 端 agent：

```text
你正在协助我在这台连接 NI USRP X310 的 BS 主机上做办公室环境单站采集。
Stage G' 已通过：OpenISAC backend processing 已开启，3.1 GHz / 50 Msps，
RuView rf-direct 已持续收到 openisac-rd；不要重新做部署或链路验证。

请严格使用仓库中的：
docs/integrations/openisac-x310-bs-office-collection-runbook.md

你只能给我手动执行的命令并等待我贴回结果；不能假设可以自行操控硬件或终止进程。
不得主动修改 OpenISAC 配置、TX gain、频点、采样率、天线或网络配置；不得在我
明确确认 RF 安全前触发、重启或扩大 RF 发射。

按 Stage J0 开始。每次只推进一个阶段，固定按以下格式回复：
阶段：
目的：
请手动执行：
预期通过标准：
失败时请贴回：
是否涉及新的 RF 发射：是/否
```

## 3. 固定实验边界

本轮所有采集默认保持下列条件不变：

| 项目 | 固定值或原则 |
| --- | --- |
| 体制 | 单站 monostatic sensing |
| 中心频率 / 采样率 | 沿用当前 3.1 GHz / 50 Msps 配置，不在本轮更改 |
| backend processing | 保持 `enable_backend_sensing_processing: true` |
| 目标输出 | `openisac-rd`，`frame_format=2`，近场 targets 通常在 range bin 6-9 |
| 硬件几何 | X310、天线、线缆、桌面设备、朝向不移动 |
| 数据位置 | raw、metadata、JSONL 仅保留在 BS 主机本地 |
| 比较标准 | 同一几何下的相对变化；不要把 range bin 直接当作未经标定的米数 |

此前 BS 端验证使用 RuView HTTP `3002`、WebSocket `3001`、RF-direct UDP
`5020`。以当前进程实际监听端口为准；原先手册中的 HTTP `3000` 不应被盲目
覆盖。

## 4. 数据量与命名

在约 10 Hz 下，一帧约为 801 KB raw + 76 KB metadata。每分钟约 0.5 GB；先做
60 秒短采集，确认剩余空间后再延长。不要为了节省空间删除异常样本，应该将其
标记为异常。

每个 run 建独立目录：

```bash
export RUVIEW_DIR=/path/to/ruview
export RUVIEW_HTTP_PORT=3002
export OPENISAC_PORT=8888
export OPENISAC_CONTROL_PORT=9999
export RUVIEW_RF_PORT=5020
export RUN_ID="$(date +%Y%m%d_%H%M%S)_office_<scenario>"
export RUN_DIR="$RUVIEW_DIR/data/office-collection/$RUN_ID"
mkdir -p "$RUN_DIR/raw"
df -h "$RUVIEW_DIR/data"
```

将 `<scenario>` 替换成以下之一：

- `quasi_static`：无人刻意活动，但无需清空办公室。
- `occupied_still`：一名受试者在指定位置保持静止。
- `small_motion`：一名受试者只做小幅手臂或躯干动作。
- `walk_approach`：一名受试者沿预定直线接近后退回。

建立标签文件，所有未知项写 `unknown`，不要编造：

```bash
printf '%s\n' \
  "run_id=$RUN_ID" \
  "scenario=<scenario>" \
  "operator=<initials-or-unknown>" \
  "occupancy=<0-or-1>" \
  "subject_location=<fixed-position-description>" \
  "planned_action=<action-and-time-window>" \
  "uncontrolled_events=<none-or-description>" \
  "config=3.1GHz,50Msps,backend_processing=true,tx_gain=unchanged" \
  > "$RUN_DIR/manifest.txt"
```

## 5. Stage J0：只读预检

目的：确认目前运行的进程和端口，不重启、不改配置。

```bash
hostname
date -Is
df -h "$RUVIEW_DIR/data"
ss -ltnup | grep -E ":($RUVIEW_HTTP_PORT|3001|$RUVIEW_RF_PORT|$OPENISAC_PORT|$OPENISAC_CONTROL_PORT)\\b" || true
curl -s "http://127.0.0.1:$RUVIEW_HTTP_PORT/health"
curl -s "http://127.0.0.1:$RUVIEW_HTTP_PORT/api/v1/sensing/latest" > "$RUN_DIR/latest-before.json"
```

通过标准：RuView health 可访问；OpenISAC、RuView server 若已运行则保持运行；没有
其他进程占用 `8888` 导致新 bridge 无法绑定。若 API 有 token，由用户在当前 shell
中设置后再在 curl 中带 Authorization header，agent 不应索要或回显 token。

若 OpenISAC 或 RuView server 未运行，agent 必须停止，报告缺失进程和端口状态，等待
用户决定是否按既有手册启动。不要在 Stage J 中自行重启它们。

若 `8888/udp` 已由之前的常驻 bridge 占用，agent 必须先报告进程信息，而不是再启动
一个 bridge。用户确认该进程确为旧 bridge 后，可以在旧 bridge 所在终端手动按
`Ctrl+C`；OpenISAC 和 RuView server 保持运行。然后再启动本手册中带 `--duration 60`
的短采集 bridge。不得使用 `kill`、`pkill` 或猜测性地终止进程。

## 6. Stage J1：采集静态办公室基线

目的：在无法清空办公室的条件下，得到“正常背景”样本。此处的 `quasi_static` 不是
理想空场；期间只要求不安排人为动作。

1. 按第 4 节创建 `quasi_static` run 目录和 manifest。
2. 让环境自然保持 60 秒。若有人走动、开门或移动椅子，保留该数据并写入
   `uncontrolled_events`。
3. 在独立终端运行以下 bridge；这只接收当前 OpenISAC 已经发出的 UDP 数据，不改变
   发射参数：

```bash
cd "$RUVIEW_DIR"
python3 scripts/openisac_to_ruview_bridge.py \
  --openisac-host 0.0.0.0 \
  --openisac-port "$OPENISAC_PORT" \
  --control-port "$OPENISAC_CONTROL_PORT" \
  --ruview-host 127.0.0.1 \
  --ruview-port "$RUVIEW_RF_PORT" \
  --center-freq-hz 3100000000 \
  --sample-rate-hz 50000000 \
  --feature-rate-hz 10 \
  --duration 60 \
  --record-jsonl "$RUN_DIR/frames.jsonl" \
  --record-raw-dir "$RUN_DIR/raw" \
  --verbose | tee "$RUN_DIR/bridge.log"
```

4. bridge 自然退出后，保存结束快照：

```bash
curl -s "http://127.0.0.1:$RUVIEW_HTTP_PORT/api/v1/sensing/latest" > "$RUN_DIR/latest-after.json"
find "$RUN_DIR/raw" -maxdepth 1 -type f -printf '%f %s bytes\n' | tail -n 12
wc -l "$RUN_DIR/frames.jsonl"
tail -n 2 "$RUN_DIR/frames.jsonl"
```

通过标准：bridge 输出连续 `seq=... motion=... targets=...`；JSONL 持续写入；raw 和
metadata payload 都出现；无 `decode error`、`overrun`、`underrun`。目标数量不要求为
零，办公室固定反射是基线的一部分。

## 7. Stage J2：受试者静止对照

目的：在固定背景上添加一个已知占用状态，观察 targets、strength 和 motion 是否有
可重复的差异。受试者站/坐在一个固定位置，持续 60 秒；避免手势、转身和其他人员
穿行。

重复 Stage J1 的建目录、bridge 命令和结束检查，但将场景改为 `occupied_still`，并在
manifest 中记录受试者位置与朝向。不要用这段数据测试呼吸或人数精度；目前只验证
“固定占用与办公室背景不同”。

## 8. Stage J3：单一小幅动作

目的：验证 `motion_energy` 与 target strength 对单一已知动作有响应。

重复 Stage J1，场景命名为 `small_motion`，时长 60 秒，并在 `planned_action` 中写清：

```text
0-15 s 静止；15-35 s 小幅挥动一只手臂；35-60 s 静止
```

只允许一名受试者执行动作。若动作窗内的 `motion_energy` 没有明显区别，不立刻改
CFAR 或 gain；先保留数据，检查日志、范围 bin 和办公室中的非计划事件。

## 9. Stage J4：接近与后退

目的：验证近场 range bin 是否在受试者接近、后退时呈相对一致的变化。它不是距离
绝对标定实验。

重复 Stage J1，场景命名为 `walk_approach`，建议时序：

```text
0-10 s 静止；10-25 s 沿固定路线慢速接近；25-35 s 静止；
35-50 s 沿原路线后退；50-60 s 静止
```

记录走动路线和起止相对位置。只比较同一运行内或硬件完全未动的运行间趋势；不要
把 `range_bin=6-9` 换算成绝对距离，除非后续另做距离标定。

## 10. 每个 run 的最小回传包

BS 端 agent 在每个采集结束后，要求用户贴回：

```text
1. manifest.txt 内容（隐私信息可匿名化）
2. bridge.log 最后 20 行
3. wc -l frames.jsonl 的结果
4. raw 目录最后 12 个文件名和大小
5. frames.jsonl 最后 2 行（可移除时间戳或身份信息）
6. latest-before.json 与 latest-after.json 的 enhanced_motion 摘要
7. 本次实际发生的非计划事件
```

agent 必须按以下格式解释，不应凭一两帧下结论：

```text
阶段：J1 / J2 / J3 / J4
状态：PASS / DEGRADED / BLOCKED
数据完整性：frames、raw、metadata、错误计数
环境标签：是否完整；是否存在非计划动作
相对观察：target range bin、strength、motion_energy 的趋势
结论边界：本 run 能证明什么；还不能证明什么
下一步：只给一个建议的采集场景，或停止并回报问题
```

## 11. 判读与停止规则

| 现象 | 初步判读 | BS 端 agent 的动作 |
| --- | --- | --- |
| J1 有 targets | 正常，可能是固定多径或环境活动 | 记录，继续 J2，不要把它称为误报 |
| J2 与 J1 的 targets 有可重复差异 | 占用状态可能可观测 | 继续 J3，积累重复样本 |
| J3 在动作窗内 motion 上升，前后回落 | 动态变化可观测 | 继续 J4 或重复 J3 |
| J4 随接近/后退出现相对 bin 变化 | 近场相对范围趋势可观测 | 记录趋势，不做米级宣称 |
| 无论场景都完全相同 | 几何、阈值或标签可能有问题 | 保存数据，停止自动推进，要求检查现场与日志 |
| decode error / raw 缺失 | parser 或 UDP 重组问题 | 停止，保留 payload，转 Stage H 离线回放 |
| overrun / underrun | 数据链路或 X310 主机性能问题 | 停止本轮，保留日志，不调高增益 |

采集完成后，至少保留 `quasi_static`、`occupied_still`、`small_motion`、
`walk_approach` 各一组，再决定是否进入阈值调优、距离标定或离线算法分析。

## 12. 与已有手册的关系

- 首次部署、进程缺失、端口与 parser 问题：参见
  `docs/integrations/openisac-x310-agent-runbook.md`。
- 本手册只覆盖已经通过 Stage G' 后的办公室采集。
- 不要因为顶层 `cfar`、`micro_doppler` 或 `aggregate_metadata_channels` 为 null 就判定
  失败；当前 bridge 的有效 backend 目标可能位于
  `enhanced_motion.openisac.aggregate_channels[].targets`。
