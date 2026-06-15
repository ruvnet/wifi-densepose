# ADR-171: Drone Swarm Benchmarking & Evaluation Methodology — Metrics, Leaderboards, and Statistical Rigor

| Field      | Value                                                                                   |
|------------|-----------------------------------------------------------------------------------------|
| Status     | Accepted (peer-reviewed 2026-05-30)                                                     |
| Date       | 2026-05-30                                                                              |
| Deciders   | ruv                                                                                     |
| Relates to | ADR-148 (ruview-swarm), ADR-147 (OccWorld), ADR-146 (RF encoder), ADR-028 (witness)    |

> Companion to ADR-148. ADR-148 shipped the swarm and 5 criterion micro-benchmarks
> plus a `SotaComparison` against Wi2SAR. This ADR defines **how we evaluate the swarm
> rigorously** — what metrics, what statistics, what baselines, and an honest account
> of which external leaderboards do and do not apply.

---

## 1. Context

ADR-148's `ruview-swarm` reports performance via five `criterion` micro-benchmarks and a
single `SotaComparison` (localization 1.732 m vs Wi2SAR 5 m; coverage ~223 s vs 810 s).
These numbers are **internally valid but insufficient as scientific claims**:

- The criterion figures (3.3 µs MARL inference, 43 µs RRT-APF, 54 ns fusion, 248 µs PPO
  step) measure **wall-clock latency**, not policy quality or coverage/localization quality.
- The 1.732 m localization comes from a **single synthetic geometry** (3 drones at 120°
  around a known point), not a distribution of victim positions under realistic noise.
- The 223 s coverage is an **analytic estimate** (`estimate_coverage_time_secs()`), not an
  episode rollout.
- All numbers are **single-run point estimates**. The MARL reproducibility literature
  (Henderson 2018; Agarwal 2021; Gorsane 2022) shows single/few-seed point estimates
  routinely flip algorithm rankings and overstate gains.

We need a defined, reproducible evaluation methodology before any "beats SOTA" claim can
survive external review, and an honest position on external leaderboards.

---

## 2. Decision

Adopt a two-tier evaluation methodology:

1. **Micro-benchmarks (criterion)** — keep for compute-latency regression gating only.
   Explicitly labeled as latency, never as quality.
2. **Domain evaluation harness** — a seeded, multi-run, statistically-reported harness
   producing SAR metrics (localization CEP, coverage, detection rate) and MARL metrics
   (IQM return, probability-of-improvement) over **≥10 seeds with 95% stratified-bootstrap
   confidence intervals**, against **≥3 baselines**, following the Agarwal/Gorsane standard.

Do **not** claim leaderboard standing — no public leaderboard accepts drone-swarm CSI-SAR
submissions. Comparisons to Wi2SAR are **paper-to-paper**, labeled as such, acknowledging
the sensing-modality difference (RSS bearing vs CSI multi-view fusion).

---

## 3. External Leaderboard Landscape — Honest Assessment

**There is no public, externally-administered leaderboard that accepts a drone-swarm,
CSI-based, multi-view SAR system.** This is a research niche; comparison is paper-to-paper.
The adjacent options and their fit:

| Benchmark / Leaderboard | Domain | Live submission? | Fit for ruview-swarm |
|-------------------------|--------|------------------|----------------------|
| **Wi2SAR** (arxiv 2604.09115) | Drone WiFi SAR | No (paper) | **Direct baseline** — paper-to-paper only; RSS bearing ≠ CSI fusion |
| **MARL4DRP** (Springer 2023) | Drone routing MARL | No | Closest drone-MARL benchmark; would need a routing→coverage adapter |
| **CSI-Bench** (NeurIPS 2025) | Static WiFi sensing | Splits + paper baselines | Adjacent (localization task) but no moving-sensor/multi-view fusion |
| **SMAC / SMACv2** | StarCraft cooperative MARL | No live LB | Structural analogy (CTDE) only; combat task, not coverage |
| **PettingZoo MPE** (Simple Spread) | 2D cooperative particles | No | Cheap MARL **correctness check**, no physics/CSI |
| **Melting Pot** | Social-dynamics MARL | Closed (NeurIPS '24) | Not applicable |
| **MAMuJoCo / Hanabi / GRF / Overcooked** | Various cooperative MARL | No live LB | Not applicable |
| **OmniDrones / gym-pybullet-drones / Pegasus** | Drone-control sim platforms | No (platforms) | **Training infrastructure**, not leaderboards; no CSI layer |

**Conclusion:** We will (a) keep Wi2SAR as the cited paper baseline, (b) optionally build a
MARL4DRP/MPE adapter to post a recognized cooperative-MARL number (tangential to SAR), and
(c) **not** represent any internal number as a leaderboard placement.

---

## 4. Evaluation Metrics

### 4.1 SAR Domain Metrics (primary — comparable to Wi2SAR)

| Metric | Definition | Reporting |
|--------|-----------|-----------|
| Localization CEP50 | Median horizontal error, fused victim position vs ground truth | m, 95% CI |
| Localization CEP95 | 95th-percentile horizontal error | m |
| **GDOP** | Geometric Dilution of Precision of the contributing-drone constellation at detection time | dimensionless (tracked per detection) |
| Coverage rate @ T | Fraction of area scanned ≥1× within T=240 s | %, 95% CI |
| Coverage time to 95% | Time to scan 95% of bounded area | s, mean ± CI |
| Time-to-first-detection | Mission start → first confident detection (conf > 0.85) | s, 95% CI |
| Detection rate | P(detected \| victim present) per mission | %, 95% CI |
| False-alarm rate | P(confident detection \| no victim) | %, 95% CI |
| Collision rate | Collisions (d < 1.5 m) per mission | count/mission |
| Overlap ratio | Fraction of path re-covering scanned cells | % |

### 4.2 MARL Policy-Quality Metrics

| Metric | Definition |
|--------|-----------|
| IQM episodic return | Interquartile mean over 10 seeds × 50 eval episodes (Agarwal 2021) |
| Probability of improvement | P(MAPPO return > IPPO return) on a random episode |
| Optimality gap | Expected gap to a defined reference performance |
| Performance profile | Fraction of (seed, episode) with localization error < τ, plotted vs τ ∈ [0,10] m |
| Sample efficiency | Return vs training steps (curve, not point) |

### 4.3 Micro-benchmarks (criterion — latency only)

Retained from ADR-148, **labeled as compute latency, not quality**:
`marl_actor_inference` 3.3 µs · `rrt_apf_100iter` 43 µs · `multiview_fusion_3drones` 54 ns ·
`demo_coverage_estimate` 100 ps · `ppo_update_64transitions` 248 µs. Purpose: prove the
control loop has no compute bottleneck (all ≪ the 10 ms / 100 Hz budget) and gate
performance regressions. They are **not** evidence of policy or localization quality.

---

## 5. Statistical Protocol (Agarwal 2021 / Gorsane 2022)

| Requirement | Standard adopted |
|-------------|------------------|
| Seeds per condition | **≥10** training runs from distinct seeds |
| Evaluation episodes | 50 fixed, versioned episodes per trained policy (10 victim layouts × 5 CSI-noise levels) |
| Aggregate metric | **IQM** (not mean, not median) + performance profiles |
| Confidence intervals | **95% stratified bootstrap**, 1,000 resamples |
| Baselines (≥3) | Random walk (lower bound), Boustrophedon+manual-triangulation (heuristic), IPPO (no shared critic) |
| Reproducibility | Versioned YAML config (drone count, area, victims, CSI σ amplitude / κ phase, wind, packet loss) + all seeds committed with results |

Rationale: Henderson et al. (2018) found ≤5-seed point estimates flip rankings; Agarwal et
al. (2021, NeurIPS Outstanding Paper) show IQM needs ~10 runs for the statistical power that
the median needs ~200 runs for; Gorsane et al. (2022) made ≥10 seeds + IQM + stratified CIs
the cooperative-MARL standard. `rliable` (google-research/rliable) is the reference impl.

---

## 6. Reproducibility Harness (`evals/`)

A new evaluation harness (separate from criterion micro-benchmarks):

1. **Seeded episodes** — every episode, noise perturbation, and training run seeded from a
   versioned config; seeds committed with results (no `Date.now()`/unseeded RNG).
2. **Per-episode logging** — coverage %, localization error, GDOP, time-to-first-detection,
   collisions, detection binary → JSONL (reuses the ADR-148 telemetry schema).
3. **Aggregation** — IQM ± 95% stratified-bootstrap CI across the 10-seed × 50-episode matrix.
4. **Baseline sweep** — random / boustrophedon-heuristic / IPPO / MAPPO, so
   probability-of-improvement and performance profiles are computable.
5. **Output** — committed `evals/RESULTS.md`: a reproducible internal leaderboard ranking
   our 6 flight patterns × learning patterns on the SAR metrics, plus the Wi2SAR paper row.

This `RESULTS.md` is the **real, defensible "leaderboard" for this system** — patterns ranked
against each other and the cited baseline, reproducibly, with CIs.

### 6.1 Dual-stage pipeline (compute-cost mitigation)

The full matrix is **10 seeds × 50 episodes × ≥4 conditions = ≥2,000 rollouts per policy**.
Running each rollout against the OccWorld 3D prior (ADR-147, ~375 ms/inference) would melt
the L4 / RTX 5080 budget. Split evaluation into two stages:

- **Stage 1 — Kinematic (fast, full matrix).** Stripped vector environment; OccWorld paths
  pre-cached or treated as static analytical volumes. Produces episodic **return, IQM,
  sample-efficiency curves, coverage %, GDOP, localization error** over the full 10-seed matrix.
- **Stage 2 — High-fidelity physics (sub-sampled).** Take the **3 median seeds** (by Stage-1
  IQM) into Gazebo + PX4 SITL with full CSI phase/amplitude noise. Extracts **false-alarm
  rate** and **collision rate** under realistic dynamics (heading-rate limits, APF repulsion,
  motor response) that the kinematic sim omits.

Stage 1 is CI-runnable today; Stage 2 requires the Gazebo/PX4 SITL bring-up (follow-on).

### 6.2 Noise sweep (coherence-gate threshold)

The config generator systematically varies the two CSI noise parameters:
- **σ** — Gaussian amplitude noise (CSI magnitude)
- **κ** — von Mises phase concentration (lower κ = noisier phase)

Sweeping (σ, κ) isolates the exact environmental threshold where `CrossViewpointAttention`
(ADR-016) drops out of its coherence gate (`coherence_gate.rs` Accept → PredictOnly/Reject,
ADR-135). This finds the operating envelope, not just a single-point accuracy.

### 6.3 GDOP tracking

Localization accuracy is meaningless without the constellation geometry that produced it.
The harness records **GDOP** per detection: 3 drones in a ~120° constellation give the
√3 ≈ 1.73× CRLB improvement; 3 **collinear** drones degrade toward the single-view
Cramer-Rao limit (~2.9 m). Reporting localization error **stratified by GDOP band** prevents
the headline number from being a best-case geometric artifact.

---

## 7. Evidence Grading of Current ADR-148 Numbers

| Claim | Grade | Why |
|-------|-------|-----|
| criterion latencies (3.3 µs / 43 µs / 54 ns / 248 µs) | **High** | Deterministic compute, hardware-specific, reproducible |
| Wi2SAR baseline (5 m, 160k m²/13.5 min) | **High** | Published field trial, open source |
| 1.732 m 3-view localization | **Low–Medium** | Single synthetic geometry; no noise distribution; CRLB predicts ~2.9 m for N=3 |
| 223 s 4-drone coverage | **Low** | Analytic estimate, not an episode rollout |
| "beats SOTA" | **Directional only** | Valid as paper-to-paper direction; not leaderboard, not multi-seed |

The √N multi-view scaling claim is theoretically sound (CRLB: σ ∝ 1/√(N·SNR); N=3 → √3 ≈
1.73× improvement), but the measured 1.732 m must be reproduced over a victim-position and
noise distribution before it is defensible.

---

## 8. Consequences

### Positive
- Converts scattered numbers into a reproducible, statistically-honest evaluation.
- The `RESULTS.md` internal leaderboard ranks the 6 flight × 4 learning patterns fairly.
- Aligns with the recognized MARL evaluation standard (IQM + stratified CIs + ≥10 seeds).
- Honest external-leaderboard position avoids overclaiming.

### Costs / Risks
- ≥10 seeds × 50 episodes × N patterns × N baselines is a real compute cost — this is where
  the ADR-148 GCP L4 / local RTX 5080 training budget is actually spent.
- Requires the MARL policy to be **trained to convergence** first (the ADR-148 5-episode CPU
  run shows decreasing value_loss, not convergence).
- Coverage/localization must move from analytic estimate / synthetic geometry to **episode
  rollouts under realistic CSI noise** before headline numbers are republished.

### Open issues → follow-on work
1. Train MAPPO/IPPO to convergence (M4 follow-on) before running the eval harness.
2. Build the seeded `evals/` harness + `RESULTS.md` generator.
3. Optional: MARL4DRP or MPE Simple-Spread adapter for a recognized cooperative-MARL number.
4. Re-state ADR-148 §14 headline numbers with CIs once the harness has run.

---

## 9. Research Notes & References

Compiled by `ruflo-goals:deep-researcher` (2026-05-30). Full landscape in the agent record.

**MARL evaluation rigor**
- Henderson et al., "Deep RL That Matters", arxiv 1709.06560 — ≤5-seed estimates flip rankings
- Agarwal et al., "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021, arxiv 2108.13264 — IQM, performance profiles, stratified bootstrap; `rliable`
- Gorsane et al., "Standardised Evaluation Protocol for Cooperative MARL", NeurIPS 2022, arxiv 2209.10485 — ≥10 seeds + IQM standard
- BenchMARL, arxiv 2312.01472 — operationalizes the above

**Cooperative-MARL benchmarks**
- SMACv2, arxiv 2212.07489 · PettingZoo MPE (Farama) · Melting Pot (DeepMind, NeurIPS 2024 contest) · MAMuJoCo (Gymnasium-Robotics) · MARL4DRP, Springer 2023 (closest drone-MARL)

**Drone-sim platforms**
- gym-pybullet-drones, arxiv 2103.02142 · OmniDrones, IEEE RA-L 2024 · Pegasus, arxiv 2307.05263 · Flightmare (IROS 2021) · AirSim (discontinued 2022) · Crazyswarm2

**SAR / coverage / CSI sensing**
- Wi2SAR, arxiv 2604.09115 (direct baseline: 5 m, 160k m²/13.5 min, 18.4° median AoA)
- CSI-Bench, NeurIPS 2025, arxiv 2505.21866 (461 h WiFi sensing, localization task)
- Coverage path planning, PMC9571681 (boustrophedon ~5% faster than spiral)
- Bio-inspired SAR, Nature s41598-025-33223-z (PSO > Levy/ACO on exploration score)
- CRLB for CSI localization, IEEE 8110647 (σ ∝ 1/√(N·SNR))

**Tooling**
- criterion.rs known limitations — wall-clock only, not algorithmic quality
- rliable, github.com/google-research/rliable

---

*ADR authored with research support from `ruflo-goals:deep-researcher` (2026-05-30).
 Companion to ADR-148. Defines the evaluation methodology that the ADR-148 headline
 numbers must satisfy before being republished as defensible claims.*
