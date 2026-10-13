# ADR 426: Selective public CSI intake and native format diagnostics

Status: accepted for draft research implementation, no model promotion.

## Context

ADR 424 rejected the predictive encoder. ADR 425 checks supplied metadata but does
not authenticate captures or enforce a training gate. A concrete next gap is
access to fresh public complex CSI without downloading a multi-gigabyte archive
or silently forcing its axes into the existing amplitude model.

## Decision

Add a source-specific bounded HTTP Range ZIP reader for Zenodo 18924941. Select
only three authored beacon capture1 members and their configuration files. Use
flat local names, exact range validation, byte/request budgets, ZIP CRC and local
SHA256. A missing immutable server validator remains an explicit limitation;
partial reads never claim verification of the full publisher archive digest.

Keep the MATLAB v7.3 HDF diagnostic optional and separate from training. Check
native metadata and hash before inspecting the first 32 frames. Preserve antenna,
subcarrier and frame axes; report finite/zero amplitude diagnostics without
predicting occupancy. Disable all promotion in receipts. Synthetic injected
inputs remain labeled synthetic.

## Consequences and alternatives

The verified transfer is about 5.67 MB instead of downloading the 16.1 GB archive.
This is measured transfer volume, not an inference speedup. Native HDF introduces
an optional dependency and parser trust boundary; enforce outer process limits
and reject external storage, links, virtual datasets and unexpected filters.
Full downloading would improve whole-archive digest verification at a much
higher transfer cost and is deferred. Blind conversion to the existing 52 channel
model is rejected because the source has 256 subcarriers and 4 antennas.

The catalog/README license discrepancy remains unresolved. Selected samples stay
private and are exposed exploratory data, never independent confirmation.
Identity aliases and physical room/participant/device provenance require review
before scientific splitting. See [protocol, tests and rollback](../zenodo-csi-intake.md).
