# OpenISAC 对 RuView X310 RF-Direct 路线的借鉴报告

报告日期：2026-06-13

本文评估 `D:/isac/OpenISAC_new/` 对当前 RuView USRP X310
RF-direct sensing 路线的参考价值。结论很直接：先借鉴它的接口、运行时纪律、
X310 配置方式和 sensing 输出表达，不要现在就把 OpenISAC 的完整 C++ OFDM
栈并入 RuView。

## 摘要

OpenISAC 是一个基于 UHD/USRP 的实时 OFDM 通感一体化实验平台。它自己的定位是
“idea -> OTA experiment”，也就是让研究想法快速跑到真实空口实验上，而不是做
WiFi 或 5G 标准兼容栈。

这和我们当前 RuView 的 X310 路线高度一致：把 NI X310 当成原生 RF sensing 仪器，
而不是把它伪装成 ESP32 或 WiFi CSI 数据源。

我建议的下一步是做一个 OpenISAC-to-RuView 桥接器：

```text
X310 + OpenISAC OFDMModulator
        |
        | UDP sensing stream，通常是 8888
        v
openisac_to_ruview_bridge.py
        |
        | RuView rf-direct JSON，5020/udp
        v
RuView sensing-server --source rf-direct
```

这样 OpenISAC 保持为 USRP 主机上的高吞吐 OFDM/UHD worker，RuView 保持为更轻的
UI、API、语义层和集成层。这个路径能避免在 RF 实验尚未证明哪些观测量最有价值之前，
就把 AFF3CT、FFTW、OpenMP、yaml-cpp、CMake 和一大套 C++ 实时运行时拖进 RuView。

## OpenISAC 是什么

重点查看过的文件：

- `D:/isac/OpenISAC_new/README.md`
- `D:/isac/OpenISAC_new/CMakeLists.txt`
- `D:/isac/OpenISAC_new/src/OFDMModulator.cpp`
- `D:/isac/OpenISAC_new/src/OFDMDemodulator.cpp`
- `D:/isac/OpenISAC_new/include/OFDMCore.hpp`
- `D:/isac/OpenISAC_new/scripts/backend_sensing_viewer.py`
- `D:/isac/OpenISAC_new/scripts/sensing_runtime_protocol.py`

OpenISAC 主要有两个实时后端：

- `OFDMModulator`：基站侧，发送 OFDM 帧，接收 monostatic sensing 回波，并输出 sensing 数据。
- `OFDMDemodulator`：UE 侧，接收并解码 OFDM 帧，也支持 bistatic sensing。

项目结构大致如下：

- `src/`、`include/`：C++ PHY、sensing、线程和运行时逻辑。
- `config/`：X310/B210 等设备的 YAML 配置样例。
- `scripts/`：Python viewer、运行时协议解析、benchmark、配置 Web 编辑器、主机性能调优脚本。
- `capture/`：离线 sensing 结果绘图。
- `docs/`、`site/`：文档站点。

OpenISAC 的许可证是 BSD 2-Clause，允许选择性借鉴和改造，但要保留对应 attribution。
依赖侧比较重：UHD、Boost、OpenMP、FFTW3f、yaml-cpp、AFF3CT，以及 Python 绘图依赖。

## 最值得借鉴的部分

### 1. X310 运行时配置模型

相关文件：

- `D:/isac/OpenISAC_new/config/Modulator_X310.yaml`
- `D:/isac/OpenISAC_new/config/Modulator_X310RTxRx.yaml`

值得借鉴的点：

- 用 YAML 管理 RF profile：中心频率、采样率、带宽、clock source、time source、
  wire format、增益、通道和每路 RX alignment。
- 把共享 `device_args`、`tx_device_args`、`rx_device_args`、per-channel override 分开。
- 显式建模 `sensing_rx_channels`，而不是把单一路径写死。
- 保留 X310/UHD 关键参数，例如 `master_clock_rate`、收发 frame 数、`sc16` wire format。

OpenISAC X310 样例中的典型参数：

- `fft_size: 1024`
- `cp_length: 128`
- `num_symbols: 100`
- `sensing_symbol_num: 100`
- `range_fft_size: 1024`
- `doppler_fft_size: 100`
- `sample_rate: 50000000`
- `bandwidth: 50000000`
- `center_freq: 3100000000`
- `mono_sensing_port: 8888`
- `control_port: 9999`

RuView 不应该直接照抄这些数值，但应该借鉴配置结构。当前
`scripts/x310_cw_worker.py` 用 CLI 参数做第一轮 CW 测试是合理的；一旦实验分叉为
CW、multi-tone、stepped-frequency、OpenISAC bridge 等模式，YAML profile 会更清楚。

### 2. USRP 主机性能调优

相关文件：

- `D:/isac/OpenISAC_new/scripts/set_performance.bash`
- `D:/isac/OpenISAC_new/scripts/isolate_cpus.bash`

值得借鉴的点：

- 只调优高速 NIC，支持按速度或 allowlist 选择网卡。
- 在 X310/NIC 链路支持时启用 jumbo MTU。
- 增大 NIC ring size，OpenISAC 默认是 `4096`。
- 有隔离 CPU 时，把 NIC IRQ pin 到专用核心。
- 必要时停止 `irqbalance`。
- 用 CPU affinity 启动实时 workload，而不是完全依赖 Linux 调度。

这和你的物理拓扑直接相关：笔记本没有 SFP/SFP+ 网卡，USRP 接在另一台主机上，所以
UHD、DSP worker、RuView server 都应尽量跑在 USRP 主机。笔记本只做浏览器、SSH 和控制面。

### 3. OFDM sensing DSP 算法

相关文件：

- `D:/isac/OpenISAC_new/include/OFDMCore.hpp`

最有价值的模块：

- `SensingProcessor`：channel estimation、FFT shift、MTI、range IFFT、
  Doppler FFT、Hamming window、phase compensation。
- `MicroDopplerState`：对固定 range bin 做滑窗 micro-Doppler spectrum。
- `run_ca_cfar_2d_full`
- `run_os_cfar_2d_full`
- `cluster_detected_targets`

这些是很强的算法参考，但不应该作为第一步直接搬进 RuView。更好的顺序是：先让
OpenISAC 自己产生 range-Doppler 或 metadata frame，RuView 通过桥接脚本消费摘要结果。
等真实实验表明哪些观测量稳定、有用，再选择性重写或移植。

### 4. UDP 运行时协议和 viewer 模式

相关文件：

- `D:/isac/OpenISAC_new/scripts/sensing_runtime_protocol.py`
- `D:/isac/OpenISAC_new/scripts/backend_sensing_viewer.py`

值得借鉴的协议设计：

- viewer 先请求 runtime params，再按返回参数解码 frame。
- 支持多种 frame format：
  - dense channel buffer
  - compact raw
  - dense range-Doppler
  - compact sparse
- flags 能表达 MTI、compact mask、backend sensing processing、aggregated stream、
  metadata sidecar 等状态。
- 能解析 CFAR target clusters 和可选 micro-Doppler metadata。

重要常量：

- compact magic：`CSM1` / `0x43534D31`
- aggregate magic：`ASG1` / `0x41534731`
- metadata sidecar magic：`SMD1`
- aggregate metadata sidecar：`ASM1`
- monostatic sensing 默认端口：`8888`
- runtime control 默认端口：`9999`

这是当前最值得借的部分。RuView 已经有 `5020/udp` 的 RF-direct JSON 接口；
桥接器只需要把 OpenISAC 二进制/metadata frame 解码成低速 JSON feature frame。

### 5. 运行时控制词汇

相关文件：

- `D:/isac/OpenISAC_new/src/OFDMModulator.cpp`

值得借鉴的命令：

- `STRD`：sensing symbol stride
- `SKIP`：跳过 sensing FFT
- `MTI `：启停 MTI
- `CFEN`、`CFTD`、`CFTR`、`CFGD`、`CFGR`、`CFAL`、`CFMR`、`CFDC`、`CFMP`、
  `CFRK`、`CFSD`、`CFSR`：CFAR 控制
- `MDEN`、`MDRB`：micro-Doppler 控制
- `TXGN`、`RXGN`：TX/RX gain 控制
- `MRST`：measurement/runtime reset

RuView 应该借鉴这个“实时控制闭环”的交互模式，不一定要完全继承命令名。后续 RF 面板可以暴露
stride、MTI、CFAR threshold/rank、gain、micro-Doppler bin 等控件，再把控制包发到
OpenISAC 的 `9999/udp`。

## 暂时不建议借鉴的部分

现在不建议把 OpenISAC 整个后端并入 RuView。

原因：

- 架构职责不同：OpenISAC 是 OFDM PHY 实验引擎；RuView 是 sensing server、UI、
  semantic API 和集成面。
- 依赖过重：AFF3CT、FFTW、OpenMP、yaml-cpp、Boost、CMake 会让 RuView RF 路线更难安装和调试。
- 实时风险更高：在第一轮 RF-direct bridge 还没跑通前改 C++ UHD 主循环，会放大排错面。
- 实验速度会下降：最快有价值的实验，是把 raw/high-rate RF 处理留在 UHD 旁边，只把低速 feature 发给 RuView。

暂时不要移植 LDPC/FEC、packet payload ingest、完整 OFDM modulation、完整 PyQt viewer。
除非后续实验明确证明 RuView 必须直接拥有这些层。

## 对当前 RuView 的直接适配关系

当前 RuView 已有 RF-direct 相关文件：

- `D:/ruview/scripts/x310_cw_worker.py`
- `D:/ruview/docs/integrations/x310-rf-direct.md`
- `D:/ruview/v2/crates/wifi-densepose-sensing-server/src/main.rs`

RuView 当前 `RfDirectFrame` 支持：

- `source`
- `node_id`
- `sequence`
- `center_freq_hz`
- `sample_rate_hz`
- `feature_rate_hz`
- `motion_energy`
- `breathing_bpm`
- `breathing_confidence`
- `heart_rate_bpm`
- `heartbeat_confidence`
- `phase_track_rad`
- `phase_delta_rad`
- `amplitude`
- `snr_db`
- `confidence`
- `range_bins`

这个 contract 刻意保持很小。桥接脚本第一版也应保持小，只输出 RuView 当前能消费的核心字段。
更丰富的 `targets`、`cfar_clusters`、`micro_doppler` 可以先作为附加 JSON 字段发出；当前 Rust
接收器会忽略未知字段，后续再把它们提升到 `enhanced_motion`。

建议第一版 OpenISAC 映射：

```json
{
  "source": "openisac-rd",
  "node_id": 1,
  "sequence": 123,
  "center_freq_hz": 3100000000,
  "sample_rate_hz": 50000000,
  "feature_rate_hz": 10,
  "motion_energy": 0.42,
  "amplitude": 0.8,
  "snr_db": 31.0,
  "confidence": 0.74,
  "range_bins": [0.02, 0.05, 0.19, 0.44],
  "targets": [
    {
      "range_bin": 63,
      "doppler_bin": 8,
      "strength_db": 18.4
    }
  ]
}
```

## 推荐集成路线

### Phase 0：保留 CW baseline

继续保留 `scripts/x310_cw_worker.py` 作为最低复杂度 sanity test。它能验证 X310 链路、
UHD Python binding、RF 安全设置、相位稳定性，以及 RuView `5020/udp` ingestion。

### Phase 1：实现 OpenISAC bridge

新增脚本：

- `D:/ruview/scripts/openisac_to_ruview_bridge.py`

职责：

- 监听 OpenISAC monostatic sensing UDP，默认 `8888`。
- 复用或移植 `sensing_runtime_protocol.py` 中必要的解码逻辑。
- 在可用时请求/解析 OpenISAC runtime params。
- 把 dense RD、compact sparse、metadata sidecar frame 转成 RuView RF-direct JSON。
- 发到 `127.0.0.1:5020` 或用户指定的 host/port。
- 提供 `--demo` 或 `--from-file`，方便没有 X310 的笔记本做本地验证。

这是最直接的下一步。

### Phase 2：增加 RF experiment profiles

bridge 跑通后，再增加 profile：

- `cw-baseline`
- `openisac-x310-mono`
- `openisac-x310-dual-rx`
- `gpsdo-clocked`
- `low-rate-safe-lab`

profile 格式可以借鉴 OpenISAC YAML，但要比 OpenISAC 完整配置更小。

### Phase 3：增加 RF 面板和控制闭环

等真实 bridge frame 流起来后，再做 RuView RF/OpenISAC 面板。第一版控件：

- source mode
- center frequency display
- sample rate display
- SNR/quality
- motion energy
- range-Doppler peak list
- MTI toggle
- CFAR enable/threshold/rank
- TX/RX gain

后续这些控件可以把控制包发到 OpenISAC `9999/udp`。

### Phase 4：选择性移植算法

只有当 OpenISAC bridge 实验证明稳定有价值后，再选择性重写或移植：

- OS-CFAR cluster extraction
- micro-Doppler summary
- compact sparse target serialization
- YAML profile validation

优先用 Python/Rust 处理已经降维过的数据，不要太早引入 C++ 后端依赖。

## 风险和约束

- 主动 RF 发射可能受法规限制。OTA 前应使用衰减器、线缆、屏蔽环境，或明确合法的实验频段。
- OpenISAC X310 样例使用 50 MS/s 和较高带宽。RuView smoke test 不应一上来就跑满，除非 USRP 主机和 NIC 已调优。
- `Modulator_X310RTxRx.yaml` 里有 `tx_gain: 90`，这和你 X310 UBX-160 的实际增益范围不一致。
  样例 gain 不能当成安全硬件默认值。
- OpenISAC 默认面向 Linux 实时实验主机。笔记本没有 SFP/SFP+ 路径，不适合作为 raw IQ 中转。
- bridge 必须能处理 UDP chunk 丢失、runtime params 改变、metadata 缺失等情况。

## 最具体的下一步

已开始实现 `scripts/openisac_to_ruview_bridge.py`。第一阶段目标收窄到：

1. 从 UDP 8888 解码 OpenISAC dense range-Doppler 或 metadata sidecar frame。
2. 计算紧凑摘要：
   - peak power / peak strength
   - motion energy
   - top range bins
   - top Doppler bins
   - metadata 存在时附带 CFAR clusters
3. 发送 RuView-compatible RF-direct JSON 到 UDP 5020。
4. 先用合成 frame 或回放 frame 在无 X310 环境验证。
5. 再在 USRP 主机上用以下拓扑验证：

```bash
cargo run -p wifi-densepose-sensing-server -- \
  --source rf-direct \
  --rf-udp-port 5020 \
  --http-port 3000 \
  --ws-port 3001 \
  --bind-addr 0.0.0.0

python scripts/openisac_to_ruview_bridge.py \
  --openisac-host 0.0.0.0 \
  --openisac-port 8888 \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020
```

如果这一步跑通，RuView 就获得了从真实 OFDM-ISAC worker 到现有 UI/API 的路径，同时仍然保留
更简单的 CW baseline，方便排错和对照实验。

## Stage 1 当前用法

无硬件 demo：

```bash
python scripts/openisac_to_ruview_bridge.py \
  --demo \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --duration 10 \
  --verbose
```

OpenISAC live bridge：

```bash
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

配套 RuView server：

```bash
cd D:/ruview/v2
cargo run -p wifi-densepose-sensing-server -- \
  --source rf-direct \
  --rf-udp-port 5020 \
  --http-port 3000 \
  --ws-port 3001 \
  --bind-addr 0.0.0.0
```

Stage 1/2 的本地验证不需要 X310：测试会验证 range-Doppler 摘要、metadata cluster 映射、
OpenISAC chunk reassembly、aggregate `ASG1` 多通道 RD 解码、aggregate `ASM1` 多通道 metadata 解码、
payload 离线回放、JSONL/raw 录制，
以及 Python 编译；Rust `rf-direct` 接收器也会
保留 `targets`、`cfar`、`micro_doppler`、`openisac` 扩展字段到 `enhanced_motion`。
真正的停止点是 USRP 主机上 OpenISAC `OFDMModulator` 输出 8888/udp 数据后的 live 联调。

硬件交接清单见 `D:/ruview/docs/integrations/openisac-x310-hardware-handoff.md`。
