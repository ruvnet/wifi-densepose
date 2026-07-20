# OpenISAC X310 Hardware Handoff Checklist

> [!CAUTION]
> 本文基于 2026-07-16 对抗审查之前的协议，已经停止作为执行清单使用。不要按本文开放 `0.0.0.0` UDP、把峰值解释为目标/运动，或单独回放 raw/metadata。当前安全契约与命令以 `x310-rf-direct.md` 为准。

日期：2026-06-13

这份清单用于从“无硬件软件验证”切换到 `isac` 主机上的 X310/OpenISAC 实流验证。
在执行主动 RF 发射前，请先确认频段、衰减器、线缆、屏蔽箱或合法 OTA 实验条件。

如果由另一台主机上的 agent 执行，请优先使用更严格的 agent runbook：
`D:/ruview/docs/integrations/openisac-x310-agent-runbook.md`。

## 目标拓扑

```text
NI USRP X310
   |
   | 10GbE / SFP+，由 UHD 管理
   v
isac 主机
   - UHD
   - OpenISAC OFDMModulator
   - RuView sensing-server --source rf-direct
   - scripts/openisac_to_ruview_bridge.py
   |
   | HTTP/WebSocket/SSH
   v
笔记本浏览器
```

原则：raw IQ 和 OpenISAC 高吞吐 UDP 留在 `isac` 主机本地；笔记本只做 UI 和控制。

## 上机前准备

1. 同步 RuView repo 到 `isac` 主机。
2. 确认 `isac` 主机能运行：

```bash
uhd_find_devices
uhd_usrp_probe
python -c "import numpy; print(numpy.__version__)"
```

3. 套用 UHD socket buffer 建议：

```bash
sudo sysctl -w net.core.rmem_max=24912805
sudo sysctl -w net.core.wmem_max=24912805
```

4. 如果 NIC/X310 链路支持，按 OpenISAC 的方式调优网卡：

```bash
cd /path/to/OpenISAC
sudo OPENISAC_TARGET_IFACES=<x310_nic_name> ./scripts/set_performance.bash
```

5. RF 安全检查：

- 优先用 cabled loopback + 足够衰减器。
- 或使用屏蔽箱。
- OTA 前确认频段、功率、天线和当地法规。
- 不要直接照抄 OpenISAC 样例里的高增益设置。

## Step 1：先跑 RuView RF-Direct Server

在 `isac` 主机：

```bash
cd /path/to/ruview/v2
cargo run -p wifi-densepose-sensing-server -- \
  --source rf-direct \
  --rf-udp-port 5020 \
  --http-port 3000 \
  --ws-port 3001 \
  --bind-addr 0.0.0.0
```

笔记本打开：

```text
http://<isac-host>:3000/ui/index.html
```

如果 `--bind-addr 0.0.0.0` 暴露在非可信网络，设置 `RUVIEW_API_TOKEN`。

## Step 2：无 OpenISAC Demo 验证桥接链路

在 `isac` 主机另一个 shell：

```bash
cd /path/to/ruview
python scripts/openisac_to_ruview_bridge.py \
  --demo \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --record-jsonl data/openisac-demo.jsonl \
  --duration 10 \
  --verbose
```

通过标准：

- bridge 输出 `demo seq=... targets=...`。
- RuView latest sensing API 中 source 为 `rf-direct`，`enhanced_motion.source` 为 `openisac-rd-demo`。
- `data/openisac-demo.jsonl` 有 JSONL frame。

## Step 3：启动 OpenISAC

在 OpenISAC build 目录使用 X310 config。第一轮建议：

- 使用低风险频段/链路。
- 降低 TX gain。
- 如可行，先 cabled attenuation。
- 确认 `mono_sensing_ip` 指向 `127.0.0.1` 或 `isac` 本机地址。
- 确认 `mono_sensing_port: 8888`。
- 确认 `control_port: 9999`。
- 如果启用 backend sensing processing，则 bridge 能解析 `SMD1`/`ASM1` metadata。

示例：

```bash
cd /path/to/OpenISAC/build
cp ../config/Modulator_X310.yaml ./Modulator.yaml
./OFDMModulator
```

具体 OpenISAC 启动命令以本地 build 和配置为准。

## Step 4：启动 OpenISAC -> RuView Bridge

在 `isac` 主机：

```bash
cd /path/to/ruview
python scripts/openisac_to_ruview_bridge.py \
  --openisac-host 0.0.0.0 \
  --openisac-port 8888 \
  --control-port 9999 \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --center-freq-hz 3100000000 \
  --sample-rate-hz 50000000 \
  --feature-rate-hz 10 \
  --record-jsonl data/openisac-bridge.jsonl \
  --record-raw-dir data/openisac-raw \
  --verbose
```

通过标准：

- bridge 输出 `seq=... motion=... targets=...`。
- `data/openisac-bridge.jsonl` 持续写入。
- `data/openisac-raw/` 中出现 `frame_*_raw.bin` 或 `frame_*_metadata.bin`。
- RuView `enhanced_motion.source` 为 `openisac-rd`。
- 如果是 aggregate stream，`enhanced_motion.openisac.aggregate_channels` 或
  `enhanced_motion.openisac.aggregate_metadata_channels` 有每通道摘要。

## Step 5：如果 Parser 报错

保留 `data/openisac-raw/`，不要只看实时日志。

离线回放单个 payload：

```bash
python scripts/openisac_to_ruview_bridge.py \
  --replay-payload data/openisac-raw/frame_000000001_raw.bin \
  --wire-rows 100 \
  --wire-cols 1024 \
  --range-fft-size 1024 \
  --doppler-fft-size 100 \
  --record-jsonl data/openisac-replay.jsonl \
  --verbose
```

如果是 metadata 文件：

```bash
python scripts/openisac_to_ruview_bridge.py \
  --replay-payload data/openisac-raw/frame_000000001_metadata.bin \
  --record-jsonl data/openisac-replay.jsonl \
  --verbose
```

常见排查方向：

- OpenISAC runtime params 没收到，导致 `wire_rows/wire_cols` 与真实 payload 不一致。
- OpenISAC 使用 compact raw，但 bridge 仍按 dense RD 解析。
- OpenISAC 开启 aggregate stream，需要查看 `ASG1`/`ASM1` 路径。
- UDP 丢 chunk，导致重组 payload 不完整。
- NIC buffer/MTU/IRQ 未调优。

## 何时算软件阶段结束

在没有真实 X310/OpenISAC 实流之前，软件阶段已经覆盖：

- single dense RD
- single metadata `SMD1`
- aggregate RD `ASG1`
- aggregate metadata `ASM1`
- chunk reassembly
- JSONL/raw recording
- raw payload replay
- Rust `rf-direct` 扩展字段透传

下一步必须依赖 `isac` 主机和 X310/OpenISAC 实流。若真实 payload 与当前解析器不一致，
应优先保存 raw payload，再用 `--replay-payload` 在本地快速修 parser。
