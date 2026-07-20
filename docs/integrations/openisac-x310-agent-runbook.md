# OpenISAC X310 Claude Code 手动操作 Runbook

> [!CAUTION]
> 本文基于 2026-07-16 对抗审查之前的协议，已经停止作为执行手册使用。不要按本文开放 `0.0.0.0` UDP、把峰值解释为目标/运动，或单独回放 raw/metadata。当前安全契约与命令以 `x310-rf-direct.md` 为准。

版本日期：2026-06-15

本手册给部署在 USRP 主机 `isac` 上的 Claude Code 使用，但**不是自动执行手册**。
实际命令由用户手动复制、执行、观察硬件状态并把输出贴回给 Claude Code。
Claude Code 的职责是：按阶段给出下一条命令、解释输出、判断是否继续、记录证据、在
RF 或硬件风险不明确时要求停止。

目标是在单站 X310 上验证：

```text
OpenISAC OFDMModulator -> UDP 8888 -> RuView bridge -> UDP 5020 -> RuView rf-direct
```

这不是双站流程。第一轮只做单站 monostatic sensing：X310 本机 TX + RX。

## 0. 角色分工

用户负责：

- 手动复制并执行命令。
- 确认线缆、衰减器、屏蔽箱、天线、频段和功率等 RF 安全条件。
- 决定是否允许进入任何会导致 X310 发射的步骤。
- 把命令输出、日志、错误和观察结果贴回 Claude Code。

Claude Code 负责：

- 一次只给一个小阶段的命令，不要假设用户已经执行。
- 等用户贴回输出后再判断下一步。
- 不要自行声称硬件已连接、RF 安全或 OpenISAC 已正常运行；必须从用户贴回的证据判断。
- 遇到 RF 风险、权限不足、设备未发现、parser 报错时，停止推进并要求用户确认。

## 1. Claude Code 启动提示词

用户可以把下面这段直接贴给 `isac` 主机上的 Claude Code：

```text
你正在辅助我在这台连通 NI USRP X310 的主机上手动验证 OpenISAC -> RuView rf-direct 单站链路。
你不能假设自己可以自动操作硬件；每一步请给我需要手动执行的命令，并等待我贴回输出。
不要主动推进到 RF 发射步骤，除非我明确确认线缆/衰减器/屏蔽箱或合法 OTA 条件已经准备好。

请使用仓库中的 runbook：
docs/integrations/openisac-x310-agent-runbook.md

目标链路是：
OpenISAC OFDMModulator -> UDP 8888 -> scripts/openisac_to_ruview_bridge.py -> UDP 5020 -> RuView --source rf-direct

请从 Stage A 环境盘点开始。每一步输出：
1. 当前阶段
2. 我需要手动执行的命令
3. 你期望看到的通过标准
4. 如果失败要收集什么证据
然后等待我贴回结果。
```

## 2. 操作原则

1. Claude Code 不要要求用户主动发射 RF，除非用户已经明确确认以下任一条件：
   - cabled loopback + 足够衰减器；
   - 屏蔽箱；
   - 合法 OTA 实验频段、功率、天线、位置已经确认。
2. 不要照抄 OpenISAC 样例中的高 TX gain。第一轮 TX gain 应从低值开始。
3. raw IQ 和 OpenISAC 高吞吐数据只留在 `isac` 本机，不要转发到笔记本。
4. 每一步都保存证据：命令、退出码、关键日志、生成文件路径。
5. 如果 parser 报错，优先保留 `data/openisac-raw/`，不要只看实时日志。
6. 如果遇到未知硬件/RF 风险，停止在当前步骤，回报风险，不要试错发射。
7. 对长时间运行的命令，Claude Code 应提示用户“保持这个终端运行，再开一个新终端执行下一步”。
8. 如果命令需要 `sudo`，Claude Code 只给出命令和风险说明，由用户决定是否执行。

## 3. 每阶段输出格式

Claude Code 每次给用户下一步时，使用：

```text
阶段:
目的:
请手动执行:
预期通过标准:
失败时请贴回:
是否涉及 RF 发射: 是/否
```

用户贴回输出后，Claude Code 再按下面格式总结：

```text
阶段:
状态: PASS / FAIL / BLOCKED
主机:
工作目录:
命令:
关键输出:
生成文件:
下一步:
```

最终回报必须包含：

- `uhd_usrp_probe` 是否成功；
- RuView server 是否启动；
- bridge demo 是否能进入 RuView；
- OpenISAC 是否启动；
- bridge live 是否收到 `seq=... motion=... targets=...`；
- 是否生成 `data/openisac-bridge.jsonl`；
- 是否生成 `data/openisac-raw/frame_*`；
- RuView `/api/v1/sensing/latest` 中 `enhanced_motion` 的内容摘要。

## 4. 约定路径变量

这些变量由用户手动设置。路径不确定时先搜索，不要猜死：

```bash
export RUVIEW_DIR=/path/to/ruview
export OPENISAC_DIR=/path/to/OpenISAC
```

如果不知道路径：

```bash
find "$HOME" /opt /data -maxdepth 4 -type f -name openisac_to_ruview_bridge.py 2>/dev/null
find "$HOME" /opt /data -maxdepth 4 -type f -name OFDMModulator 2>/dev/null
```

找到后再设置：

```bash
cd "$RUVIEW_DIR"
pwd
git status --short
```

## 5. Stage A：只读硬件/环境盘点

目的：确认 X310、UHD、Python、Rust/Cargo、OpenISAC 二进制是否可用。

执行：

```bash
hostname
date -Is
which uhd_find_devices || true
which uhd_usrp_probe || true
uhd_find_devices
uhd_usrp_probe
python3 --version || python --version
python3 -c "import numpy; print(numpy.__version__)" || python -c "import numpy; print(numpy.__version__)"
cargo --version || true
```

通过标准：

- `uhd_find_devices` 找到 X310；
- `uhd_usrp_probe` 成功，能看到 X310、UBX-160、地址、clock/time source；
- Python 能 import numpy；
- 如果要从源码跑 RuView，cargo 可用。

失败处理：

- UHD 不存在：停止，回报“UHD missing”。
- X310 找不到：停止，回报网卡/IP/供电/线缆状态。
- numpy 不存在：安装或让用户授权安装后再继续。

## 6. Stage B：主机网络与 socket buffer 准备

先查看网卡，不直接改：

```bash
ip -br addr
ip route
for i in /sys/class/net/*; do
  iface=$(basename "$i")
  [ -r "$i/speed" ] && echo "$iface speed=$(cat "$i/speed" 2>/dev/null)"
done
sysctl net.core.rmem_max net.core.wmem_max
```

如果 `rmem_max/wmem_max` 低于 UHD 建议值，执行：

```bash
sudo sysctl -w net.core.rmem_max=24912805
sudo sysctl -w net.core.wmem_max=24912805
```

如果知道 X310 所在高速 NIC 名称，例如 `enp5s0f0`，可 dry-run OpenISAC 调优：

```bash
cd "$OPENISAC_DIR"
sudo OPENISAC_TARGET_IFACES=<x310_nic_name> ./scripts/set_performance.bash --dry-run
```

只有 dry-run 看起来正确时，才执行真实调优：

```bash
sudo OPENISAC_TARGET_IFACES=<x310_nic_name> ./scripts/set_performance.bash
```

失败处理：

- 不确定 NIC 名称：不要运行真实调优，只回报候选 NIC。
- `sudo` 不可用：跳过调优，回报限制。

## 7. Stage C：RuView server 启动

在 `isac` 主机启动 RuView server：

```bash
cd "$RUVIEW_DIR/v2"
cargo run -p wifi-densepose-sensing-server -- \
  --source rf-direct \
  --rf-udp-port 5020 \
  --http-port 3000 \
  --ws-port 3001 \
  --ui-path ../ui \
  --bind-addr 0.0.0.0
```

保持该进程运行。另开 shell 验证：

```bash
curl -s http://127.0.0.1:3000/health
curl -s http://127.0.0.1:3000/api/v1/sensing/latest | python3 -m json.tool || true
```

通过标准：

- `/health` 返回 JSON；
- server 日志显示 `rf-direct` 或 RF-direct UDP listener；
- `/api/v1/sensing/latest` 在 demo 前可以没有最新帧，但 endpoint 应可访问。

如果设置了 `RUVIEW_API_TOKEN`，curl 要使用：

```bash
curl -s -H "Authorization: Bearer $RUVIEW_API_TOKEN" http://127.0.0.1:3000/api/v1/sensing/latest
```

失败处理：

- cargo build 失败：保存完整日志；
- 端口被占用：记录占用进程 `ss -ltnup | grep -E '3000|3001|5020'`，不要随意 kill，除非用户授权。

## 8. Stage D：无硬件 bridge demo

目的：先验证 RuView `5020/udp` ingest 和 bridge JSON shape。

执行：

```bash
cd "$RUVIEW_DIR"
mkdir -p data
python3 scripts/openisac_to_ruview_bridge.py \
  --demo \
  --ruview-host 127.0.0.1 \
  --ruview-port 5020 \
  --record-jsonl data/openisac-demo.jsonl \
  --duration 10 \
  --verbose
```

验证：

```bash
tail -n 3 data/openisac-demo.jsonl
curl -s http://127.0.0.1:3000/api/v1/sensing/latest | python3 -m json.tool | head -n 120
```

通过标准：

- bridge 输出 `demo seq=... motion=... targets=...`；
- `data/openisac-demo.jsonl` 存在并有 JSON；
- RuView latest 中：
  - `source` 为 `rf-direct`；
  - `enhanced_motion.source` 为 `openisac-rd-demo`；
  - `enhanced_motion.targets` 存在。

失败处理：

- JSONL 有内容但 RuView latest 没变化：检查 RuView 是否监听 5020/udp。
- bridge 报 Python import 错：安装缺失依赖，至少需要 numpy。

## 9. Stage E：OpenISAC 配置检查

在启动 OpenISAC 前，检查 config。

```bash
cd "$OPENISAC_DIR"
grep -nE 'center_freq|sample_rate|tx_gain|mono_sensing_ip|mono_sensing_port|control_port|enable_backend_sensing_processing|sensing_output_mode|sensing_rx_channel_count|sensing_rx_channels' config/Modulator_X310*.yaml
```

第一轮建议：

- `mono_sensing_ip` 指向 `127.0.0.1` 或 `isac` 本机 IP；
- `mono_sensing_port: 8888`；
- `control_port: 9999`；
- TX gain 从低值开始；
- 如果要让 bridge 看到 metadata，启用 backend sensing processing；
- 如果先求稳，dense RD 比 compact raw 更容易排错。

停止条件：

- RF 安全条件未确认；
- TX gain 看起来异常高；
- OpenISAC 目标频点/带宽不明确；
- 不知道 X310 连接到哪张 NIC。

## 10. Stage F：启动 OpenISAC OFDMModulator

在 OpenISAC build 目录：

```bash
cd "$OPENISAC_DIR/build"
cp ../config/Modulator_X310.yaml ./Modulator.yaml
./OFDMModulator
```

保持该进程运行。记录日志中：

- UHD device args；
- center freq；
- sample rate；
- TX/RX gain；
- sensing output mode；
- UDP sensing destination；
- control port；
- underrun/overrun/error。

如果 `./OFDMModulator` 不存在：

```bash
cd "$OPENISAC_DIR"
find . -maxdepth 3 -type f -name OFDMModulator -o -name 'OFDMModulator*'
```

不要在未确认依赖时盲目 rebuild；如果需要 rebuild，先回报缺失依赖。

## 11. Stage G：启动 OpenISAC -> RuView live bridge

另开 shell：

```bash
cd "$RUVIEW_DIR"
mkdir -p data/openisac-raw
python3 scripts/openisac_to_ruview_bridge.py \
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

如果实际 OpenISAC config 使用不同频率/采样率，把 `--center-freq-hz` 和
`--sample-rate-hz` 改成 config 中的值。

通过标准：

- bridge 输出 `seq=... motion=... targets=...`；
- `data/openisac-bridge.jsonl` 持续增长；
- `data/openisac-raw/frame_*_raw.bin` 或 `frame_*_metadata.bin` 出现；
- RuView `/api/v1/sensing/latest` 中 `enhanced_motion.source` 为 `openisac-rd`；
- 如果是 aggregate RD，出现 `enhanced_motion.openisac.aggregate_channels`；
- 如果是 aggregate metadata，出现 `enhanced_motion.openisac.aggregate_metadata_channels`。

验证命令：

```bash
ls -lh data/openisac-bridge.jsonl data/openisac-raw | tail -n 20
tail -n 2 data/openisac-bridge.jsonl
curl -s http://127.0.0.1:3000/api/v1/sensing/latest | python3 -m json.tool | head -n 160
```

## 12. Stage H：parser 失败时的离线回放

如果 bridge 报 decode error，不要删除 raw payload。先列文件：

```bash
ls -lh data/openisac-raw | tail -n 20
```

回放 raw RD payload：

```bash
python3 scripts/openisac_to_ruview_bridge.py \
  --replay-payload data/openisac-raw/frame_000000001_raw.bin \
  --wire-rows 100 \
  --wire-cols 1024 \
  --range-fft-size 1024 \
  --doppler-fft-size 100 \
  --record-jsonl data/openisac-replay.jsonl \
  --verbose
```

回放 metadata payload：

```bash
python3 scripts/openisac_to_ruview_bridge.py \
  --replay-payload data/openisac-raw/frame_000000001_metadata.bin \
  --record-jsonl data/openisac-replay.jsonl \
  --verbose
```

如果回放 raw 失败，尝试用 OpenISAC runtime params 对齐：

- `wire_rows`
- `wire_cols`
- `frame_format`
- `range_fft_size`
- `doppler_fft_size`
- `wire_data_format`

必须回报：

```bash
python3 scripts/openisac_to_ruview_bridge.py --help
ls -lh data/openisac-raw | tail -n 20
tail -n 20 data/openisac-bridge.jsonl 2>/dev/null || true
```

## 13. Stage I：可选 CW baseline

如果 OpenISAC 实流不稳定，但 UHD/Python 可用，可以用 RuView 的 CW worker 单独验证
X310 RF-direct path。仅在 RF 安全条件满足时执行：

```bash
cd "$RUVIEW_DIR"
python3 scripts/x310_cw_worker.py \
  --device-args addr=192.168.10.2 \
  --center-freq 2450000000 \
  --rate 1000000 \
  --feature-rate-hz 20 \
  --tone-hz 25000 \
  --tx-chan 0 \
  --rx-chan 1 \
  --tx-ant TX/RX \
  --rx-ant RX2 \
  --tx-gain 0 \
  --rx-gain 10 \
  --tx-amplitude 0.05 \
  --host 127.0.0.1 \
  --port 5020 \
  --verbose
```

不要把 CW baseline 当成 OpenISAC 成功；它只证明 RuView rf-direct 和 UHD 基础路径可用。

## 14. Claude Code 最终交付包

完成或失败都要回传以下内容：

```text
1. uhd_find_devices 摘要
2. uhd_usrp_probe 摘要：X310 serial、FPGA、FW、daughterboard、IP
3. RuView server 启动命令和关键日志
4. bridge demo 命令、输出、openisac-demo.jsonl 前 2 行
5. OpenISAC config 关键字段
6. OpenISAC 启动日志关键行
7. live bridge 输出
8. data/openisac-bridge.jsonl 前/后各 2 行
9. data/openisac-raw 文件列表
10. /api/v1/sensing/latest 中 enhanced_motion 摘要
11. 是否出现 underrun/overrun/decode error
12. 下一步建议：继续调参 / 修 parser / 检查 RF 链路 / 停止等待用户
```

## 15. 成功定义

最低成功：

- RuView server 运行；
- bridge demo 进入 RuView；
- OpenISAC 或 raw payload 至少产生一帧可被 bridge 转为 `rf-direct` JSON。

完整成功：

- OpenISAC `OFDMModulator` 在 X310 单站配置下持续输出；
- bridge live 连续转发；
- RuView `/api/v1/sensing/latest` 能看到 `openisac-rd`；
- `targets`、`cfar`、`micro_doppler` 或 `openisac.aggregate_*` 至少一种真实字段出现；
- JSONL/raw payload 均被保存，便于后续分析。

停止并回报：

- RF 安全条件不明确；
- X310/UHD 不可用；
- OpenISAC 无法启动；
- bridge 收不到 UDP 8888；
- parser 报错但 raw payload 已保存；
- RuView server 无法启动或端口冲突无法处理。
