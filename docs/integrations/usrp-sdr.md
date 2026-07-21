# USRP / SDR Integration (Retired Compatibility Path)

The legacy `--source usrp` JSON feature bridge was removed on 2026-07-21.
It accepted unauthenticated UDP on every interface and upgraded arbitrary
CSI-shaped amplitude/phase vectors into motion, presence, person count, pose,
and vital-sign outputs without a versioned contract or matched validation.

The sensing server now rejects `--source usrp`. `USRP_UDP_PORT` and
`--usrp-udp-port` are no longer deployment options. Do not restore the old
parser to make historical producers work.

For X310/OpenISAC transport, use [USRP X310 RF-Direct Integration](x310-rf-direct.md).
The supported path is deliberately narrower:

- OpenISAC and the Python bridge communicate over loopback;
- raw range-Doppler and metadata halves must share a frame ID and configuration epoch;
- the bridge emits `ruview.rf_observation` protocol v2 with a random producer instance ID;
- RuView exposes diagnostics and CFAR candidate clusters without creating human semantics;
- producer restarts rotate the instance ID while replay from retired instances is rejected;
- remote unauthenticated RF UDP remains unsupported.

The former CW producer is also retired. Its unvalidated implementation is kept
only under `archive/experiments/x310_cw_unvalidated_experiment.py` for historical
audit reproduction; `scripts/x310_cw_worker.py` now fails closed with migration
guidance.

Raw IQ, amplitude/phase vectors, and diagnostic peaks are observations—not
evidence of a person, motion, breathing, pose, or vital signs. Those capabilities
require separate labelled offline validation before they can enter a deployed
inference contract.
