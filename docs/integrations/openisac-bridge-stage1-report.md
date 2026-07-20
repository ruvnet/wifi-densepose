# OpenISAC -> RuView Bridge Stage 1 开发报告

> [!NOTE]
> 这是审查前的历史实现报告，不是当前接口规范。其中 raw/metadata 独立转发、`target`/`motion` 映射和旧 JSON 形状已被废弃；当前契约见 `x310-rf-direct.md`。

报告日期：2026-06-13

## 本阶段目标

按照前一份 OpenISAC 借鉴报告，第一阶段不移植 OpenISAC 的完整 C++ OFDM 后端，
而是实现一个可以本地验证的桥接器：

```text
OpenISAC sensing UDP 8888
        -> scripts/openisac_to_ruview_bridge.py
        -> RuView rf-direct JSON 5020
```

这个阶段的边界是：在没有 X310、没有 OpenISAC 实时后端的笔记本上，也能验证协议解析、
range-Doppler 摘要、metadata target 映射和 RuView JSON 输出形状。真实 OTA 或 cabled
RF 实验留到 USRP 主机上进行。

## 已实现内容

新增脚本：

- `D:/ruview/scripts/openisac_to_ruview_bridge.py`

新增/扩展测试：

- `D:/ruview/tests/test_openisac_to_ruview_bridge.py`

新增实施计划：

- `D:/ruview/docs/superpowers/plans/2026-06-13-openisac-bridge.md`

新增硬件交接清单：

- `D:/ruview/docs/integrations/openisac-x310-hardware-handoff.md`
- `D:/ruview/docs/integrations/openisac-x310-agent-runbook.md`

桥接器当前能力：

- 支持 OpenISAC 12 字节 UDP chunk header 重组。
- 支持 metadata chunk flag，即 `total_chunks` 高位为 metadata 标记。
- 支持解析 OpenISAC `CTRL/PARM` runtime params。
- 支持收到 OpenISAC `CTRL/RDY ` 后向 `9999/udp` 请求 params。
- 支持 dense complex64/complex float16 payload 基础解码。
- 支持 metadata sidecar `SMD1` 的 CFAR points、target clusters、micro-Doppler 摘要解析。
- 支持 aggregate metadata sidecar `ASM1` 的多通道解析，并把每个通道摘要保存在
  `openisac.aggregate_metadata_channels`。
- 支持把 range-Doppler 矩阵压缩成 RuView `rf-direct` JSON。
- 支持把 metadata clusters 压缩成 RuView `rf-direct` JSON。
- 支持 OpenISAC aggregate payload `ASG1` 的多通道解码，并把每个通道摘要保存在 `openisac.aggregate_channels`。
- 支持 `--demo` 模式，直接向 RuView 发送合成 `openisac-rd-demo` frame。
- 支持 `--record-jsonl` 保存转发给 RuView 的 JSON frame。
- 支持 `--record-raw-dir` 保存已重组的 OpenISAC 原始 payload，便于上 USRP 主机后做回放和 parser 调试。
- 支持 `--replay-payload` 对已保存的 raw/metadata payload 做离线回放和 JSONL 生成。
- Rust `rf-direct` 接收器已增加可选 `targets`、`cfar`、`micro_doppler`、`openisac` 字段，并会把它们透传到 `enhanced_motion`。

## 使用方式

先启动 RuView server：

```bash
cd D:/ruview/v2
cargo run -p wifi-densepose-sensing-server -- \
  --source rf-direct \
  --rf-udp-port 5020 \
  --http-port 3000 \
  --ws-port 3001 \
  --bind-addr 0.0.0.0
```

无硬件 demo：

```bash
cd D:/ruview
python scripts/openisac_to_ruview_bridge.py \
  --demo \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --record-jsonl data/openisac-demo.jsonl \
  --duration 10 \
  --verbose
```

OpenISAC live bridge：

```bash
cd D:/ruview
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

离线回放录制 payload：

```bash
cd D:/ruview
python scripts/openisac_to_ruview_bridge.py \
  --replay-payload data/openisac-raw/frame_000000001_raw.bin \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --wire-rows 100 \
  --wire-cols 1024 \
  --record-jsonl data/openisac-replay.jsonl \
  --verbose
```

## 当前数据契约

桥接器发送到 RuView 的 JSON 会保留 RuView 已支持字段：

- `source`
- `node_id`
- `sequence`
- `center_freq_hz`
- `sample_rate_hz`
- `feature_rate_hz`
- `motion_energy`
- `amplitude`
- `snr_db`
- `confidence`
- `range_bins`

并附带 RuView 会透传到 `enhanced_motion`、后续 UI 可以提升使用的字段：

- `targets`
- `cfar`
- `micro_doppler`
- `openisac`

这样做的好处是：当前 Rust server 可以继续消费原有 `rf-direct` 字段，同时保留
OpenISAC target/CFAR/micro-Doppler 细节。未来要做 RF 面板时，这些字段已经形成了稳定语义。

## 本阶段验证范围

无需硬件可以验证：

- range-Doppler 峰值会生成 target list。
- `motion_energy`、`snr_db`、`range_bins` 有稳定数值。
- metadata cluster 会映射成 `targets`。
- micro-Doppler metadata 会映射成摘要。
- chunked UDP payload 可以乱序重组。
- aggregate `ASG1` payload 可以拆成多通道 frame，并选择最强通道作为 RuView 主摘要。
- aggregate metadata `ASM1` payload 可以拆成多通道 metadata，并选择最强通道作为 RuView 主摘要。
- 已录制 payload 可以离线回放成 RuView JSONL。
- Python 文件可以编译。
- Rust sensing-server 可以编译，并且 `RfDirectFrame` 的扩展字段不会破坏现有 rf-direct 路径。

必须等 USRP/OpenISAC 环境才能验证：

- `OFDMModulator` 实际 8888/udp payload 与本地 parser 的完全兼容性。
- `CTRL/RDY`、`CTRL/PARM` 在真实 OpenISAC runtime 下的握手时序。
- dense RD、compact raw、aggregate stream 在高吞吐下的丢包表现。
- X310 主机 CPU/NIC 调优后的实时稳定性。
- RF 发射安全、频段、衰减器或屏蔽环境配置。

## 下一阶段建议

下一阶段应进入 USRP 主机前的最后准备：

1. 把更新后的 RuView repo 同步到 `isac` 主机。
2. 在 `isac` 主机先跑 `--demo`，确认 RuView server 能收到 `openisac-rd-demo`。
3. 启动 OpenISAC `OFDMModulator`，让桥接器监听 `8888/udp`，同时打开 `--record-jsonl` 和 `--record-raw-dir`。
4. 若 parser 报错，优先用 raw payload 回放定位 OpenISAC 实际 frame format。
5. 如果 OpenISAC 配置启用了双 RX 或 aggregate stream，检查 `enhanced_motion.openisac.aggregate_channels` 是否包含每路通道摘要。
6. 如果 OpenISAC 配置启用了 backend metadata sidecar，检查
   `enhanced_motion.openisac.aggregate_metadata_channels` 是否包含每路通道摘要。

完成这些准备后，软件侧就基本进入硬件接入边界：需要在 `isac` 主机启动 OpenISAC
`OFDMModulator`，让桥接器监听 `8888/udp`，再观察 RuView 是否稳定收到 `openisac-rd`
frame。
