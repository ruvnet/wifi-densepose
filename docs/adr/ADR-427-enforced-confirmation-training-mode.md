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
both frozen hashes. Invalid input exits 1; valid but blocked evidence exits 2.
The dataset loader's returned manifest digest is checked again before training.
Confirmation mode cannot optimize an existing bundle and every receipt remains
`promotionAuthorized: false`.

Legacy research operation remains available without implying confirmation.
Current ESP-Fi is in the known-exposure set, so it cannot pass as independent
confirmation. Rollback removes the four gate flags and helper without changing
the legacy training path.

## Consequences

This closes accidental gate bypass within the selected mode and binds the
declared review to exact manifest bytes. It does not authenticate captures,
consent, licenses, aliases, stable physical identities, or exposure-ledger
completeness. The inputs are unsigned. The manifest hash binds declarations and
referenced payload hashes, not provenance truth. A concurrent manifest change
may be decoded before the loader's digest mismatch aborts, although it cannot
reach training. Passing software tests makes no sensing-accuracy claim.

