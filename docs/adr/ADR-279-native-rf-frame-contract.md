# ADR-279: Native RF frame contract — `RfFrameV2` is authoritative, the canonical tensor is a derived view

| Field | Value |
|-------|-------|
| **Status** | Accepted — **implemented** (`ruview-unified/src/frame.rs`; 5 invariant tests) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273 (amends ADR-274 §2) |
| **Relates to** | ADR-136 (`CanonicalFrame` — extended, not replaced), ADR-262 (provenance discipline), ADR-282 (evidence ladder policy) |

## 0. PROOF discipline

Grades per ADR-273 §0. This ADR is a **correction** to ADR-274 §2, adopted before any measured-data debt accumulates.

## 1. Context — the architectural correction

ADR-274 made the 56-bin × 8-snapshot canonical `RfTensor` the adapter output, which is right for *compatibility* but wrong as the *authoritative* format: resampling every device into one fixed tensor discards bandwidth (a 320 MHz 802.11bk capture and a 20 MHz 802.11n capture become indistinguishable), antenna structure, phase state, and hardware-specific information a foundation encoder should learn from (the WiLLM lesson: lightweight per-device adapters into a shared *latent*, not a shared *tensor*). RuView's own history proves the cost of premature canonicalization — MERIDIAN's normalizer is useful precisely because the native data was still around.

## 2. Decision — `RfFrameV2`

The authoritative record preserves the native capture. Fields per the implementation: schema version, frame id, timestamp, modality (now including `WifiCir`, `WifiBfReport`, `FmcwRangeAzimuth`, `FmcwDopplerAzimuth` alongside CSI/SRS/FMCW/UWB/BLE-CS), **declared native axes** (`FieldAxis`: time/frequency/delay/Doppler/range/azimuth/elevation/antenna/polarization), centre frequency, bandwidth, sample rate, arbitrary-rank `native_shape` + `native_iq` + explicit `valid_mask`, TX/RX `Pose3` in one building frame, `AntennaElement` geometry, `sample_age_ns`, `CalibrationState` with a **declared `PhaseState`** (`Raw | Sanitized | Calibrated | Unavailable`), `SignalQuality`, and `FrameProvenance`.

Seven required invariants, each enforced in the validated constructor or proven by a test:

1. **Native samples are never overwritten** — `to_canonical(&self)` is read-only; `canonical_view_is_derived_and_native_is_untouched` asserts byte-identical native IQ + mask after derivation.
2. Subcarrier/antenna masks are explicit (`valid_mask`, arity-checked).
3. Phase declares its state — consumers branch on `PhaseState` instead of guessing whether detrending happened.
4. TX/RX geometry uses one building coordinate system (`Pose3`).
5. Results retain source identity via `receipt_id` (consumed by the Gaussian memory's `source_receipts` lineage, ADR-275).
6. **Synthetic and measured frames can never share a provenance class**, strengthened to an evidence rule: `Synthetic ⇒ exactly L0Simulation`, `Measured ⇒ ≥ L1CapturedReplay` — both directions rejected at construction (`synthetic_and_measured_provenance_can_never_alias`).
7. Sample age is carried through the whole path (frame → tensor → age gate → `BoundedEvent`).

## 3. The canonical tensor is demoted to a compatibility view

`RfFrameV2::to_canonical()` derives the ADR-274 tensor **through the exact same normalization code path as every adapter** (`adapters::normalize_grid` — one normalization, many entry points), after mask-aware gap-filling (invalid bins interpolated from nearest valid neighbors on the complex plane). Rank ≠ 3 frames have no canonical projection and say so with a typed error. The existing ESP32/Intel/Atheros 114→56 projections stay as-is; they simply stop being the storage format.

## 4. The mandatory split manifest

The brief's leakage rule is now code: `PartitionKey` gains a `session` dimension (packet-session leakage is as real as room leakage) and `eval::SplitManifest` certifies per-dimension disjointness across **all seven** dimensions (room/day/person/chipset/firmware/layout/session):

```text
train_rooms ∩ test_rooms = ∅ … train_sessions ∩ test_sessions = ∅
```

`fully_disjoint()` is the bar for reporting a result as leakage-resistant; a room-holdout split that still shares people *says so* in its manifest instead of masquerading (test `split_manifest_certifies_per_dimension_disjointness`). The hidden real-world test set requirement (never accessible to synthetic generation/calibration) is process, recorded in ADR-282 §4.

## 5. Consequences

- New hardware (PicoScenes, Intel, Atheros, Realtek radar, 320 MHz 802.11bk) lands as an `RfFrameV2` producer + latent adapter; nothing is lost at ingest. Vendor conformance receipt = the constructor's invariants (native shape preserved, phase state declared, timestamps monotonic, geometry present, loss measured, synthetic flag correct).
- The encoder input contract (ADR-274) is unchanged *today* (it consumes the derived view); migrating the tokenizer to native-resolution tokens is the flagged follow-up once real multi-bandwidth data exists (P2).
- Storage cost rises (native + derived); accepted — the derived view can always be recomputed, the native never can be.

## 6. Verification

`cargo test -p ruview-unified frame::` — 5 tests: provenance aliasing, shape/mask/axes arity, derived-view purity + gap-filling, rank/geometry rejection, P3162 import-profile validation (`SyntheticApertureSoundingDataset`, ADR-281 §5). All MEASURED-CODE.
