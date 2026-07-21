# RuView / OpenISAC 加固后的补充审计发现

日期：2026-07-20
范围：在修复 `ruview-openisac-adversarial-review.md` 所列问题时，对相邻 SDR 入口、历史生产者和取证路径做的二次审计。

## 结论

本轮加固后的 `rf-direct` 主路径已经改为仅回环、版本化、成对且默认不产生人体语义。但仓库中仍保留一条旧 `usrp` 推理入口、一个已经与新协议不兼容的 X310 生产者，以及两个会削弱配置一致性或原始证据可追溯性的桥接器缺口。它们不应被理解为已由本轮 `rf-direct` 修复覆盖。

## 2026-07-21 修复状态

本报告保留原始发现和证据，以下状态记录后续整改结果：

| 发现 | 状态 | 整改 |
|---|---|---|
| F-01 | 已修复 | 删除 `usrp` 规范化 source、UDP 接收任务、旧 JSON 解析器、CLI 端口和 Docker 环境变量；`--source usrp` 现在启动失败。 |
| F-02 | 已修复 | `scripts/x310_cw_worker.py` 改为失败关闭的迁移提示；旧实现移入 `archive/experiments/x310_cw_unvalidated_experiment.py` 并要求显式风险确认。 |
| F-03 | 已修复 | 参数更新会原子清空 assembler 未完成分片和 pairer 未完成配对，同时递增 `config_epoch`，但保留同一生产者实例的序列高水位。 |
| F-04 | 已修复 | 原始证据按 run/config epoch/sender 分目录，使用独占创建和冲突后缀，并写入包含 SHA-256、配置哈希及发送方的 manifest。 |
| F-05 | 已修复 | `ruview.rf_observation` 升级为 v2，加入随机 `source_instance_id` 与 `config_epoch`；Rust 接受新实例的序列归零，并拒绝已退役实例的后续重放；退役历史满时失败关闭而不淘汰重放记录。 |

信任边界仍然是本机 loopback。此次整改没有引入远程认证，也没有把 RF
诊断观测升级为运动、presence、人数、姿态或生命体征证据。

## F-01 — 高：旧 `usrp` 入口仍可把未认证 UDP 特征升级为人体推断

位置：`v2/crates/wifi-densepose-sensing-server/src/main.rs` 的 `usrp_udp_receiver_task`。

证据：

- 监听地址仍由 `format!("0.0.0.0:{udp_port}")` 生成；
- 任意可达发送方只要通过旧 JSON 解析，就会进入 CSI 特征提取、平滑分类、人数估计和生命体征计算；
- 产出的 `SensingUpdate` 仍包含 `classification`、`vital_signs`、`estimated_persons`、`signal_field` 和 `enhanced_motion`；
- `--source usrp` 仍是公开、可选的规范化数据源。

影响：即使 `rf-direct` 已经失败关闭，操作者仍可能选择旧 `usrp` 模式，在没有来源认证、版本契约或能力证据的情况下重新得到同类人体推断；原生运行时还会暴露到所有网卡。Docker 已不再发布 5010/5020 UDP，因此容器默认暴露面较小，但原生部署风险仍存在。

建议：将 `usrp` 标为 legacy 并默认禁用；若必须保留，至少复用 `rf-direct` 的回环绑定、版本/序列/新鲜度校验、能力清单和“观测值不升级为人体语义”策略。完成前不要把 `usrp` 作为受支持的 X310/OpenISAC 替代入口。

## F-02 — 中：`x310_cw_worker.py` 仍发送已废弃的推理形状

位置：`scripts/x310_cw_worker.py`。

证据：脚本仍直接构造并发送 `motion_energy`、`breathing_bpm`、`breathing_confidence`、`confidence` 和 `range_bins` 等无版本 JSON。新 `rf-direct` 接收器要求 `ruview.rf_observation` v1，因此会正确拒绝这些帧。

影响：脚本名称和历史文档会让操作者以为它仍是可用的 X310 直连生产者；实际结果是无提示数据中断，或有人为了恢复功能而重新打开旧的宽松解析器。

建议：把脚本明确重命名/移动为历史实验，或改成只生产版本化诊断观测。任何呼吸、运动等算法应先进入带标签的离线验证流程，不应再次作为在线可信字段发送。

## F-03 — 中：参数更新没有清空正在重组的 UDP 分片

位置：`scripts/openisac_to_ruview_bridge.py` 的实时控制包处理。

证据：收到新的 OpenISAC 参数后，代码会替换 `params` 并新建 `FramePairer`，但保留原 `FrameAssembler`。参数切换前开始、切换后完成的分片载荷会按新参数解码，并获得新配置哈希。

影响：配置边界附近可能出现维度误解、错误配对或不可复现的诊断记录；在恶意或异常控制包序列下，配置哈希不能完整证明载荷是在该配置下产生的。

建议：参数更新时原子地清空 assembler 与 pairer，或为每个分片保存配置 epoch，并只允许同一 epoch 的 raw/metadata 配对。

## F-04 — 低：原始录制文件名可能覆盖不同发送方或不同运行的证据

位置：`scripts/openisac_to_ruview_bridge.py` 的 `FrameRecorder.record_payload`。

证据：文件名只有 `frame_<frame_id>_<raw|metadata>.bin`，没有发送方、配置哈希、运行 ID 或防覆盖标记；`Path.write_bytes` 会覆盖同名文件。

影响：发送方重启后 frame ID 归零、同一目录被重复使用，或多个本地发送方使用相同 frame ID 时，原始证据可能被静默替换，JSONL 与二进制载荷也可能失去一一对应关系。

建议：按运行 ID/配置 epoch 建目录，将 sender 与配置哈希纳入文件名或 manifest，并使用独占创建/显式冲突记录。

## F-05 — 中：没有受信任的生产者 epoch，重启后的序列归零会被永久拒绝

位置：Python `FramePairer` 的 sender 高水位和 Rust `last_rf_sequence` 的全局高水位。

证据：两端都只接受严格递增的 `sequence`，但协议没有 `source_instance_id`、启动 epoch 或经过认证的 reset 握手。即使 Python 因发送端源端口变化而把重启视为新 sender，Rust 仍会将归零后的序列与重启前高水位比较。

影响：生产者正常重启、frame ID 回绕或设备重新初始化后，接收器可能持续计为 out-of-order 并拒绝所有新观测，直到 RuView 服务也重启。简单地允许较小序列会重新打开重放窗口，因此不能用宽松比较修复。

建议：在下一版协议加入随机的 `source_instance_id`/启动 epoch，并把它纳入完整性边界；只有受信任的新 epoch 才能原子地重置配对、序列和配置状态。另一个保守方案是由本地监督器通过受保护控制通道显式重置接收状态。

## 优先级建议

1. 先关闭或迁移旧 `usrp` 入口（F-01）。
2. 同步处置旧 X310 worker，防止为了兼容而回退安全边界（F-02）。
3. 在下一次真实硬件采集前修复配置/生产者 epoch 与录制覆盖问题（F-03、F-04、F-05）。

以上发现最初来自静态代码审计；2026-07-21 的状态表记录代码整改，不把没有
真实 X310 硬件的传输测试写成算法有效性证据。
