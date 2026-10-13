# ADR 427: Enforced independent-confirmation training mode

Status: accepted for draft research implementation; no model promotion.

## Context

ADR 425 added a strict metadata preflight, but it remained advisory and was not
bound to the dataset manifest consumed by a trainer. A passing metadata document
could therefore be paired with different dataset bytes. The predictive paths
were rejected by ADR 424, while the temporal encoder measures only a proxy
embedding objective; neither is an appropriate confirmation entrypoint.

## Decision

Add an explicit, opt-in confirmation mode only to the supervised amplitude
trainer. The mode requires a complete metadata plan, exposure ledger, frozen
protocol SHA-256 and frozen dataset-manifest SHA-256. Before importing the
dataset loader, decoding data, creating output, or training, the gate validates
bounded regular files, leakage declarations, exposure history, coverage, and
both frozen hashes. It also requires the plan and dataset manifest to have the
same retained signal-digest set, canonical dataset/revision, partition mapping
(`train`, `validation`, `test` to `train`, `development`, `confirmation`) and
numeric activity IDs. Room and participant declarations bind to the loader's
numeric source identities using canonical `room:N` and `participant:N` tokens.
Session and, when claimed, device identifiers must also exist in the dataset
manifest and match; missing provenance blocks the gate. Trial number is not
treated as a physical session. This binds supplied declarations to the exact
records the supervised loader will consume without assigning activity semantics. Invalid
input exits 1; valid but blocked evidence exits 2.
The dataset loader parses and hashes the manifest through one checked descriptor,
then compares that digest before resolving or decoding the referenced dataset
bundle. In confirmation mode, the NPZ final component is also opened once as a
bounded regular file without following symlinks. Its bytes are copied while
hashed into a private temporary snapshot; only a matching, exact-length snapshot
is inspected and decoded. The snapshot is removed when loading ends. Its
returned digest is checked again before training.
Confirmation mode cannot optimize an existing bundle and every receipt remains
`promotionAuthorized: false`.

Legacy research operation remains available without implying confirmation.
Current ESP-Fi is in the known-exposure set, so it cannot pass as independent
confirmation. Rollback removes the four gate flags and helper without changing
the legacy training path.

## Consequences

This closes accidental gate bypass within the selected mode and binds the
declared review to exact manifest bytes and retained record identities. It does not authenticate captures,
consent, licenses, aliases, stable physical identities, or exposure-ledger
completeness. The inputs are unsigned. The manifest hash binds declarations and
referenced payload hashes, not provenance truth. Parent directories remain a
trusted deployment boundary. Snapshot creation requires temporary storage up to
200 MB and can fail closed when that storage is unavailable. This cap covers the
largest fixed supervised schema at temporal pool one plus conservative DEFLATE,
NPY and ZIP overhead; native-resolution intake retains the unchanged 600 MB
shared bound.
Path replacement or same-inode mutation cannot change the verified snapshot
after it is copied; changes observed during copying fail the pinned digest or
exact-length checks. Passing software tests makes no sensing-accuracy claim.

The current ESP-Fi adapter pins the canonical dataset name and source revision,
but its upstream records do not establish physical session or device identity.
Together with its prior exposure, this makes it correctly ineligible for an
independent-confirmation run even when its file hashes are intact.

