# RuView + OpenISAC/X310 对抗性审查报告

日期：2026-07-16

## 结论

当前系统已经证明的能力是：NI USRP X310 上的 OpenISAC 数据能够经 UDP bridge
进入 RuView `rf-direct`，并被记录、解析和离线回放。

当前系统**尚未证明**人体 presence、人数、姿态、生命体征或绝对距离。任何把
`targets`、`motion_energy`、`estimated_persons=1` 或 UI 骨架解释为“已检测到一个
人”的结论，都超出了现有证据。

审查范围是当前 `D:/ruview` 工作树，包括尚未提交的 X310/OpenISAC 集成修改；不对
OpenISAC 的 RF 算法正确性、人体标签或真实办公室数据做未经提供的推断。

## 已验证事实

- X310/OpenISAC -> UDP 8888 -> Python bridge -> UDP 5020 -> RuView `rf-direct`
  的传输链路已经在真实硬件上运行。
- backend processing 启用后，raw payload 与 metadata sidecar 均可生成和保存。
- Python bridge 单元测试共 8 个，通过定向运行：

  ```bash
  .venv/Scripts/python.exe -m pytest tests/test_openisac_to_ruview_bridge.py -q --no-cov -p no:cacheprovider
  ```

- Rust 定向测试未完成：本机 Cargo 缺少 crates.io 依赖，下载时发生 Windows TLS 凭据
  错误（`SEC_E_NO_CREDENTIALS`）。这不是通过，也不是代码正确性的证据。

## 第一性原理判断

一个 RF 峰值可由直达泄漏、静态多径、家具、墙面、设备、电缆或人体产生。只有在
独立真值、可复现实验和预先定义指标下，才可把观测量提升为“人体”语义。

因此系统应分为两个层次：

```text
RfObservation
  RD / metadata / CFAR clusters / micro-Doppler / SNR / frame status
       |
       | 仅由经过验证、带版本和置信度的算法提升
       v
SensingInference
  motion / presence / range track / count / pose / vitals
```

在 `SensingInference` 尚未验证前，UI 和 API 只能展示观测层，不能生成或发布人体
语义。

## 阻断级问题

### 1. 静态反射会被标为运动

`summarize_range_doppler()` 将非 DC 占比与 `SNR / 80` 相加作为
`motion_energy`。见 `scripts/openisac_to_ruview_bridge.py` 的
`motion_energy` 计算。

对一个只有零多普勒静态峰的合成 RD 帧，实测输出为：

```text
motion_energy = 0.8400818913658418
snr_db = 59.999999587442396
top_target = { range_bin: 8, doppler_bin: 50, strength_db: 0.0 }
```

随后 RuView 以极低阈值将该值转换为 presence/motion。因此当前 `motion_energy`
不是经验证的运动量，不能用于办公室运动或人体 presence 结论。

### 2. raw RD 中的 `targets` 并不是 CFAR 目标

`_top_targets()` 只是从整张 RD 图选择最强的八个非零 bin。它没有噪声估计、局部极值
约束、CFAR 门限、聚类或时间一致性。当前 raw 路径不应把这些点称为 `targets`。

真正的 CFAR points / target clusters 位于 metadata 分支的
`metadata_to_ruview_frame()`。因此此前的“range bin 从 890 到 6-9”只能说明最大峰
的位置变化，不能证明近场人体目标已确认。

### 3. raw 与 metadata 未按 frame ID 配对

bridge 在每个 completed payload 到达后立即转发。raw RD 和 metadata sidecar 同一
frame ID 会变成两条独立 RuView 更新，`/api/v1/sensing/latest` 的内容取决于最后
到达者。

OpenISAC 自带的 `backend_sensing_viewer.py` 会先按 frame ID 缓存 raw 和 metadata，
再显示一份合并结果。当前 bridge 没有这个语义配对，因此出现 metadata 文件已生成、
但 RuView latest 只有 raw `aggregate_channels`、`cfar=null` 的现象并不意外。

### 4. RuView 将观测量升级为人数和程序化 pose

`rf_direct_udp_receiver_task()` 中，presence 为真时直接设置
`estimated_persons = 1`。这不是人数估计。

`/api/v1/pose/current` 在没有模型结果时调用 `derive_pose_from_sensing()`；后者用固定
的 COCO-17 骨架偏移、正弦呼吸、步态摆动和伪噪声生成骨架。该 API 不能作为当前
RF 测量的姿态输出。

## 高风险问题

### 1. 数据中断后仍被报告为活动

`effective_source()` 只为 ESP32 实现离线检测；RF-direct 没有最后帧时间或 timeout。
`broadcast_tick_task()` 会重复广播 `latest_update`，而 health/readiness 仍可返回
ready/healthy。停止 bridge 或 OpenISAC 后，旧 presence、targets 和 UI 状态可能继续
存在。

### 2. UDP 接收面默认暴露且没有完整性边界

RuView RF-direct receiver 绑定 `0.0.0.0:<port>`，不校验发送者、帧版本、序列单调性
或消息认证。Docker Compose 也发布了 `5010/udp` 和 `5020/udp`。局域网任意主机均可
发送 JSON 并触发传感语义及 MQTT 下游。

OpenISAC bridge 同样默认监听 `0.0.0.0:8888`。

### 3. bridge 可被不完整或恶意 UDP chunk 耗尽内存

`FrameAssembler` 直接按包头声明的 `total_chunks` 创建 `[None] * total_chunks`，没有
上限；partial frames 也没有 TTL、容量上限或发送者隔离。一个超大声明值可导致巨大
内存分配；持续丢包或大量伪造 frame ID 会积累未完成帧。

### 4. 错误 source 会静默进入模拟数据

CLI `source` 是任意字符串；启动任务的 match 对未知 source 使用
`simulated_data_task()`。例如拼错 `--source rf-direct` 的值时，不会 fail fast，而会
进入模拟路径。这与仓库“模拟数据必须显式开启”的安全意图相冲突。

### 5. 多个仓库级 API 指标是硬编码或与能力无关

- `/api/v1/info` 总是声明 `pose_estimation: true`。
- `/api/v1/pose/stats` 固定返回 `average_confidence: 0.87`。
- `/api/v1/stream/status` 固定 `active: true`，且在处理过两帧后报告 10 FPS。
- `/health/health` 返回固定 CPU、内存和磁盘百分比。

这些输出会让操作者把演示状态误解为实时测量状态。

## 已达成的设计决策

1. 当前阶段只声称真实 RF 传输、记录、解析和回放闭环。
2. `rf-direct` 默认处于实验隔离模式；未验证的人体语义、pose、vitals、人数与 MQTT
   自动化必须关闭。
3. metadata 中的 CFAR cluster 是当前唯一可提升为候选目标的来源；raw RD 最强 bin
   仅用于诊断与可视化。
4. raw 与 metadata 必须按 frame ID 配对；metadata 缺失/超时采用 fail-closed，不能
   用 raw top-k 回退为“目标”。
5. RF UDP 默认仅限 loopback；远程发送需显式启用来源白名单和完整性保护。
6. RF 帧必须具有协议版本、源时间/单调序列、接收时间、配置哈希和 freshness 状态；
   失联后清空检测状态并暴露 `offline/stale`。
7. 采用独立真值：人工时序、测距位置或临时视频只用于离线标注，不进入部署输出。
8. 能力按以下顺序独立解锁：传输 -> CFAR 观测 -> motion -> presence -> 距离/跟踪 ->
   人数 -> 姿态/生命体征。

## P0 整改门槛

在开始解释办公室数据前，至少完成：

- 实现 versioned `RfObservation`；将 generic `SensingInference` 从 RF-direct 默认路径
  移除。
- raw/metadata frame pairing、超时、丢包/乱序/重复帧计数及可观测指标。
- 为 FrameAssembler 添加最大 chunk 数、最大 payload、partial-frame TTL、总帧数上限。
- RF/bridge 默认 loopback，Docker 默认不暴露 RF UDP；未知 source 直接拒绝启动。
- 删除或明确标记程序化 pose、硬编码健康状态和假 FPS；按 source capability manifest
  向 UI/API 声明可用能力。
- 增加 Rust 与 Python 端到端测试：静态零多普勒不得产生 motion、metadata 配对、失联
  失效、非法 UDP 拒绝、source typo 拒绝、重复/乱序/缺 chunk 行为。

## 研究验证门槛

P0 完成后，先做预注册的办公室数据集：每个 run 保存 raw、metadata、合并 observation、
配置快照、环境标签和真值时间轴。先评估 CFAR 观测与单一动作的可重复性，再分别报告
motion 和 presence 的误报率、漏报率、延迟及跨时段复测。没有这些结果前，不启用
人数、pose 或 vitals。

## 当前工作树说明

RF-direct/OpenISAC 集成目前是本地工作树中的未提交修改，主要涉及
`main.rs`、Docker、README、bridge 脚本和 Python 测试。它不应被视为上游 RuView
已经具备、已经审查或已经发布的稳定能力。
