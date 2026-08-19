# ADR-326: Tenant-scoped RuVector spatial memory and anomaly explanations

- **Status**: Accepted — implementation complete; repository-wide and deployment gates pending
- **Date**: 2026-08-19
- **Decision owners**: RuView maintainers
- **Extends**: ADR-312, ADR-319, ADR-325
- **Implements**: ruvnet/RuView#1640
- **Tags**: cognitum-spaces, ruvector, memory, tenant-isolation, explanation, privacy

## Context

ADR-325 requires anomaly explanations grounded in tenant-local spatial history,
but the deployed client only returns a current list. A global vector index would
be unsafe: filtering nearest-neighbor results after the search can reveal that a
different tenant has a close match, even when identifiers are removed. A memory
record can also launder returned RuView-derived state into a second independent
observation, reset freshness, or form circular evidence.

Spatial memory must be useful without storing OAuth/API credentials, raw CSI/CIR,
RF tensors, pose frames, vital waveforms, recordings, identity observations, or
unbounded agent transcripts. Persistence also needs explicit retention,
deletion, provenance, and key-rotation behavior.

## Decision

### 1. Partition before similarity

`ruview-spatial-memory` owns a `SpatialMemory` map keyed by the exact authenticated
`(tenant_id, workspace_id)` pair. Each partition owns its own RuVector HNSW index.
Ingest and search resolve the partition first; no global ANN query exists. Site,
space, schema version, and time-window constraints narrow within the selected
partition before results are returned.

### 2. Bounded semantic records

An accepted record contains:

- tenant/workspace/site/space and stable record identity;
- source ID, message ID, record ID, monotonic event sequence, schema version;
- original `observed_at`/`expires_at` and a retention deadline;
- a bounded finite semantic feature vector, uncertainty, and evidence label;
- provenance and witness digests, plus bounded derivation references;
- explicit observation/inference classification.

Credentials and P0/P1 fields have no representation in the type. Strings,
features, references, record counts, and query `k` are bounded. Non-finite
features and uncertainty fail closed.

### 3. Lineage and replay

The partition rejects:

- changed reuse of `(source_id, message_id)`;
- a non-increasing sequence for the same source;
- duplicate derivation references;
- self-reference, missing/forward parents, and therefore every cycle;
- expired input or a provenance/witness substitution.

A recollection keeps its original lineage, timestamp, uncertainty, and evidence
label. It cannot increment corroborating-source count or become independent
support for its own ancestor.

### 4. Persistent encrypted storage

Snapshots are encrypted with XChaCha20-Poly1305 under a caller-supplied 256-bit
key and a non-secret key ID. The authenticated associated data binds the storage
format and key ID. The envelope is bounded and versioned; plaintext spatial
records are never written to disk. Loading requires a keyring containing the
named key. Rotation decrypts with the old key, atomically creates a new
generation under the new key ID, reload-verifies that generation, and leaves
the source intact. Snapshots never overwrite an existing path implicitly.

Deletion supports a tenant/workspace partition, a record, and retention cutoff.
Every deletion rebuilds that partition's HNSW index so removed records cannot be
returned from stale graph nodes.

### 5. Explanations

`explain` compares a bounded query vector with nearest tenant-local history and
returns the exact authenticated partition, generation time, ordered record IDs,
RuVector distances, original uncertainty/evidence labels, and provenance/witness
digests. Its basis explicitly says that similarity is not causation. The API
does not expose the vectors or invent a causal explanation.

History provides context, not authority. An explanation cannot authorize an
action, increase certificate class, or replace a policy decision.

## Consequences

### Positive

- Cross-tenant ANN leakage is structurally unavailable.
- Explanations cite the exact tenant-local records used.
- Replay/cycle/provenance substitution are rejected before indexing.
- Encrypted persistence has explicit key IDs and rotation behavior.

### Costs and limitations

- Partition-local HNSW uses more indexes than a global graph.
- Deletes and key rotation rebuild indexes.
- No detection-quality or latency claim is made; tests are `SYNTHETIC` unless a
  reproducer explicitly marks a measurement.
- Cloud Cognitum does not receive the local encrypted memory file.

## Validation

- cross-tenant and cross-workspace nearest-neighbor denial;
- duplicate record/message, stale-sequence, self/duplicate/missing-parent, and
  provenance-substitution tests;
- expiry, retention deletion, whole-partition deletion, sealed round-trip,
  tamper rejection, wrong-key rejection, and key-rotation tests;
- explanation citations and retained evidence/provenance labels;
- no forbidden raw-field or credential representation;
- the focused `ruview-spatial-memory` crate suite passes with `SYNTHETIC`
  evidence on 2026-08-19;
- the whole-workspace Windows gate was non-terminal (compiler crash in parallel,
  timeout when serialized), so Linux CI, a RustSec advisory scan, and package
  review remain release gates.

## Alternatives considered

**One global HNSW followed by filtering.** Rejected: ranking itself crosses the
tenant boundary.

**Cloud vector memory.** Rejected as the default: it expands the privacy and
credential boundary without being needed for local explanations.

**Plain JSONL persistence.** Rejected because tenant spatial history is sensitive
even when raw sensing is excluded.

## References

- ADR-312: Long-term spatial memory
- ADR-319: Witness chain
- ADR-325: Cognitum Spaces activation and governed exchange
- Cognitum API ADR-101
- ruvnet/RuView#1640
