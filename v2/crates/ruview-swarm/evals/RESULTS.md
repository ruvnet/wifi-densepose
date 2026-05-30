# ruview-swarm Evaluation Results (ADR-149 Stage 1, kinematic)

Statistically-rigorous evaluation harness: seeded multi-run rollouts with IQM + 95% stratified-bootstrap confidence intervals (Agarwal et al., NeurIPS 2021).

## Run configuration

- **Stage**: 1 (kinematic, self-contained, deterministic per seed)
- **Episodes per pattern**: 100 (seed × episode matrix)
- **CI method**: 95% stratified bootstrap of the IQM, stratified by seed
- **GDOP**: 2-D geometric dilution of precision at first detection

> **Stage 2 pending**: high-fidelity Gazebo/PX4 SITL evaluation (false-alarm rate, real collision rate on the median seeds) is a follow-on — see ADR-149 §6.1. The collision figures below are a kinematic min-separation proxy, not SITL physics.

## Flight-pattern leaderboard

| Flight pattern | Coverage IQM [95% CI] | Localization (m) IQM [95% CI] | Detection rate | Mean GDOP |
|----------------|-----------------------|-------------------------------|----------------|-----------|
| partitioned_lawnmower | 1.000 [1.000, 1.000] | 7.022 [5.669, 8.379] | 100.0% | 0.000 |
| pheromone | 0.662 [0.652, 0.671] | 4.110 [3.346, 5.141] | 95.0% | 1.598 |
| levy_flight | 0.490 [0.489, 0.491] | 3.523 [2.897, 4.160] | 100.0% | 0.000 |
| boustrophedon | 0.370 [0.370, 0.370] | 2.740 [2.357, 3.207] | 100.0% | 0.000 |
| spiral | 0.336 [0.336, 0.336] | 3.082 [2.678, 3.568] | 100.0% | 0.000 |
| potential_field | 0.254 [0.252, 0.256] | 4.343 [3.489, 5.265] | 100.0% | 0.000 |
| _Wi2SAR (paper baseline)_ | _n/a_ | _5.0 (paper)_ | _n/a_ | _n/a_ |

_Wi2SAR row is the published single-drone localization figure (arxiv 2604.09115), shown paper-to-paper for reference only — it was not re-run through this kinematic harness._
