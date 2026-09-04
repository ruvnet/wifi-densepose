# ADR 355: Local empty room bootstrap prior

Status: Accepted

Date: 2026-09-04

## Context

Most installations are occupied when RuView first starts. An uncalibrated
signal heuristic can interpret HVAC, clock error, neighboring motion, or a
short covariance window as a person. It can also surface plausible numeric
vitals before a room model establishes occupant attribution.

One empty home is not representative training data for all homes and contains
no positive human examples. It is useful only as a recent negative prior for
the same installation. It cannot define a universal human signal model.

The field model was memory only. The server also used a relative `data/`
directory, which is unreliable for embedded application hosts.

## Decision

RuView may persist a completed empty room field model as a local bootstrap
prior only after two server controlled stages:

1. Normal calibration reaches both 12,000 accepted frames and 600 seconds.
2. Promotion observes 12 new one second samples. At least 10 must resolve
   empty, all ticks must be fresh, and none may publish numeric vital signs.

The stored image contains aggregate means, environmental modes, mode energies,
noise statistics, configuration, contributing node IDs, and a server generated
model ID. It excludes raw CSI, waveforms, device addresses, room names, pose,
identity, credentials, and positive human labels.

The image is bound to a SHA 256 digest of the stable installation ID, carries
an integrity digest, is capped at 1 MiB, and inherits the field model expiry.
Files with an invalid schema, dimensions, values, node set, installation
binding, digest, size, or lifetime fail closed.

Restoring an image creates a field model in `bootstrap_only` mode. It may
reduce an uncalibrated false person count, but it cannot authorize calibrated
evidence or numeric heart and breathing rates. Explicit calibration replaces
it with fresh statistics. Cancelling an unfinished capture restores a valid
prior. Administrator scoped reset removes it.

The server accepts `--data-dir` or `RUVIEW_DATA_DIR`, allowing an application
host to provide a protected state directory.

## Consequences

Startup can use recent measured background structure instead of only generic
heuristics. No accuracy lift is claimed until occupied and empty held out data
from multiple installations is recorded.

The largest failure mode is overfitting to a temporarily quiet home and
suppressing a weak occupant. Controls are low authority, 24 hour expiry,
installation binding, vital abstention, explicit recalibration, and reset.

A human signal model still requires positive, consented, leakage separated
occupied sequences. Empty data is never relabeled as human training.

## Implementation

1. `wifi-densepose-signal::ruvsense::field_model` exports and validates
   `FieldModelSnapshotV1`.
2. `wifi-densepose-sensing-server::bootstrap_baseline` owns schema, integrity,
   storage, lifetime, and the held out gate.
3. `POST /api/v1/calibration/bootstrap/promote` measures the validation window
   and stores the image only on pass.
4. `GET /api/v1/calibration/status` reports bootstrap authority and explicitly
   denies calibrated evidence and numeric vital authority.
5. Numeric vitals fail closed unless an explicit fresh calibration resolves
   exactly one occupant at sufficient signal quality and confidence.

## Acceptance test

With a physically empty room, complete calibration on stable nodes, promote
the model, and restart the server with the same installation ID and data
directory. Status must report `binding_mode=bootstrap_only`, zero collection
frames, and false authority for calibrated evidence and numeric vitals. Twelve
fresh empty samples must contain no numeric vital signs. Starting a new room
calibration must replace the prior and begin at frame zero.
