//! RaBitQ-style binary sketch — cheap similarity sensor for CSI/pose embeddings.
//!
//! Implements **Pass 1** of [ADR-084](../../../../../docs/adr/ADR-084-rabitq-similarity-sensor.md):
//! a thin RuView-flavored API over `ruvector_core::quantization::BinaryQuantized`.
//!
//! # Why a sketch
//!
//! Every "have I seen something like this before?" comparison in the RuView
//! pipeline (AETHER re-ID, room fingerprinting, mincut prefilter, novelty
//! detection, mesh-exchange compression, privacy event log) shares the same
//! shape: dense float embedding → similarity score → top-K candidates.
//! The full-precision compare is expensive — `O(d)` float operations per pair,
//! cache-unfriendly because every dimension is a 4-byte load.
//!
//! A 1-bit sketch (one bit per embedding dimension, packed into bytes) collapses
//! the compare to a hardware-accelerated POPCNT/NEON-vcnt over ~32× less
//! memory. The published *RaBitQ* algorithm (Gao & Long, SIGMOD 2024) wraps
//! this with a randomized rotation for theoretical error bounds; we ship the
//! pure sign-quantization variant first and add the rotation later if
//! benchmark-measured top-K coverage drops below the ADR-084 acceptance
//! threshold of 90%.
//!
//! # Acceptance criteria (ADR-084 §"Acceptance test")
//!
//! - Sketch compare cost reduction: **8×–30×** vs full-float compare.
//! - Top-K coverage: **≥ 90%** agreement with full-float top-K.
//! - End-to-end accuracy regression: **< 1 percentage point**.
//!
//! Pass 1 establishes the API and the unit-test foundation. Pass 2+ wires it
//! into specific pipeline sites and measures the criteria there.
//!
//! # Use sites (ADR-084)
//!
//! 1. AETHER re-ID hot-cache filter (`signal::ruvsense::pose_tracker`)
//! 2. Cluster-Pi novelty sensor (`sensing-server` `SketchBank`)
//! 3. Mesh-exchange compression (ADR-066 swarm bridge)
//! 4. Privacy-preserving event log (cluster Pi)
//! 5. Mincut prefilter (`ruvector::signal::subcarrier`)
//!
//! All sites take a `&Sketch` instead of an `&[f32]`; the bridge to dense
//! embeddings is `Sketch::from_embedding`.

use crate::rotation::Rotation;
use ruvector_core::quantization::{BinaryQuantized, QuantizedVector};
use std::collections::BinaryHeap;

/// Errors raised by the sketch API.
#[derive(Debug, thiserror::Error)]
pub enum SketchError {
    /// The sketch's `sketch_version` does not match the `SketchBank`'s.
    /// This guards against silently comparing sketches produced by different
    /// embedding-model generations.
    #[error("sketch_version mismatch: bank={bank}, query={query}")]
    SketchVersionMismatch {
        /// Version stored in the bank.
        bank: u16,
        /// Version on the incoming sketch.
        query: u16,
    },

    /// The sketch's embedding dimension does not match the bank's.
    /// Two sketches of different dimensions cannot be compared.
    #[error("embedding_dim mismatch: bank={bank}, query={query}")]
    EmbeddingDimMismatch {
        /// Dimension stored in the bank.
        bank: u16,
        /// Dimension on the incoming sketch.
        query: u16,
    },

    /// Embedding dimension exceeds `u16::MAX` (65,535).
    ///
    /// Returned by [`Sketch::try_from_embedding`] to surface what
    /// `from_embedding`'s `debug_assert!` would have hidden in release
    /// builds — silently truncating the dimension count would otherwise
    /// let two different-length embeddings compare as if they were the
    /// same length. See ADR-084 §"Versioning" and the security-review
    /// finding L2 on PR #435 for context.
    #[error("embedding dimension {got} exceeds u16::MAX ({max})")]
    EmbeddingDimOverflow {
        /// Actual length of the input embedding.
        got: usize,
        /// Maximum supported dimension (`u16::MAX`).
        max: usize,
    },
}

/// A 1-bit binary sketch of a dense embedding vector.
///
/// 32× smaller than the source `[f32]` and compared via SIMD-accelerated
/// hamming distance (NEON `vcnt` on aarch64, POPCNT on x86_64). Use as a
/// cheap pre-filter before full-precision comparison.
///
/// # Versioning
///
/// `sketch_version` distinguishes sketches produced by different embedding
/// generations. Bumping the embedding model invalidates all stored sketches;
/// the `SketchBank` rejects mismatched versions at compare time so callers
/// never silently compare incompatible sketches.
///
/// `embedding_dim` is the source vector's length (not the byte-packed size);
/// kept as a check that two sketches are actually comparable.
#[derive(Debug, Clone)]
pub struct Sketch {
    /// 1-bit-per-dimension packed bytes.
    inner: BinaryQuantized,
    /// Source-embedding dimension (e.g., 128 for AETHER).
    embedding_dim: u16,
    /// Schema version of the producing embedding model.
    sketch_version: u16,
}

impl Sketch {
    /// Construct a sketch from a dense f32 embedding.
    ///
    /// Each dimension contributes one bit: `1` if the value is `> 0.0`,
    /// `0` otherwise. This is the standard sign-quantization step.
    ///
    /// `sketch_version` must be supplied by the caller and bumped whenever
    /// the embedding model that produced the input changes meaningfully
    /// (e.g., a re-trained AETHER head). Two sketches with different
    /// `sketch_version`s are not comparable.
    pub fn from_embedding(embedding: &[f32], sketch_version: u16) -> Self {
        // L2 hardening (PR #435 security review): in release builds the
        // previous `debug_assert!` was compiled out, allowing silent
        // u16-truncation when `embedding.len() > u16::MAX`. Saturate to
        // u16::MAX rather than truncate so two over-long embeddings
        // compare as same-dimensional rather than as accidentally-short.
        // Callers that need a hard error should use `try_from_embedding`.
        let embedding_dim = embedding.len().min(u16::MAX as usize) as u16;
        Self {
            inner: BinaryQuantized::quantize(embedding),
            embedding_dim,
            sketch_version,
        }
    }

    /// Fallible constructor that rejects embeddings longer than
    /// `u16::MAX` (65,535) instead of saturating, raising
    /// [`SketchError::EmbeddingDimOverflow`]. Use this when an
    /// over-long input should fail loudly rather than silently
    /// produce a sketch that disagrees with its source on
    /// `embedding_dim`.
    pub fn try_from_embedding(embedding: &[f32], sketch_version: u16) -> Result<Self, SketchError> {
        if embedding.len() > u16::MAX as usize {
            return Err(SketchError::EmbeddingDimOverflow {
                got: embedding.len(),
                max: u16::MAX as usize,
            });
        }
        Ok(Self::from_embedding(embedding, sketch_version))
    }

    /// Construct a sketch from a dense f32 embedding **with RaBitQ Pass 2
    /// rotation** ([ADR-156 §8](../../../../../docs/adr/ADR-156-ruvector-fusion-beyond-sota.md)).
    ///
    /// Applies the deterministic randomized orthogonal rotation `R = H·D`
    /// (Fast Hadamard Transform + seeded ±1 sign flips, see [`Rotation`]) to
    /// the embedding *before* sign-quantization. The rotation decorrelates
    /// coordinates so each sign bit carries more independent information,
    /// improving top-K recall on anisotropic / correlated embedding
    /// distributions — the published RaBitQ construction.
    ///
    /// The resulting sketch has the **same `embedding_dim`, packed-byte
    /// length, and `sketch_version`** as a Pass-1 sketch of the same input, so
    /// it is fully interchangeable in [`SketchBank`] and [`WireSketch`]. The
    /// *only* requirement is that the index and the query use the **same
    /// [`Rotation`]** (same seed + dim) — otherwise their sign bits live in
    /// different rotated frames and the hamming distance is meaningless.
    ///
    /// Pass-1 (`from_embedding`) and Pass-2 sketches must **not** be mixed in
    /// one bank. Use [`SketchBank::with_rotation`] to make a bank that rotates
    /// every insert and query consistently.
    pub fn from_embedding_rotated(
        embedding: &[f32],
        sketch_version: u16,
        rotation: &Rotation,
    ) -> Self {
        let rotated = rotation.apply(embedding);
        // Preserve the *source* embedding_dim semantics of Pass 1 (saturating
        // to u16::MAX) so banks/wire framing are byte-identical to Pass 1.
        let embedding_dim = embedding.len().min(u16::MAX as usize) as u16;
        Self {
            inner: BinaryQuantized::quantize(&rotated),
            embedding_dim,
            sketch_version,
        }
    }

    /// Hamming distance to another sketch in `[0, embedding_dim]`.
    ///
    /// Returns `None` if the two sketches have different `embedding_dim` or
    /// `sketch_version` — comparing them would be semantically meaningless.
    /// Use [`Sketch::distance_unchecked`] when the caller has already
    /// validated the sketches come from the same producer.
    pub fn distance(&self, other: &Self) -> Result<u32, SketchError> {
        if self.embedding_dim != other.embedding_dim {
            return Err(SketchError::EmbeddingDimMismatch {
                bank: self.embedding_dim,
                query: other.embedding_dim,
            });
        }
        if self.sketch_version != other.sketch_version {
            return Err(SketchError::SketchVersionMismatch {
                bank: self.sketch_version,
                query: other.sketch_version,
            });
        }
        Ok(self.inner.distance(&other.inner) as u32)
    }

    /// Hamming distance without compatibility checks.
    ///
    /// Faster than [`Sketch::distance`] (no version/dim check) but the
    /// caller is responsible for guaranteeing both sketches come from the
    /// same embedding model and dimension. Use only on sketches retrieved
    /// from the same `SketchBank`.
    #[inline]
    pub fn distance_unchecked(&self, other: &Self) -> u32 {
        self.inner.distance(&other.inner) as u32
    }

    /// Source-embedding dimension (number of dimensions in the original
    /// `[f32]`, not the packed byte length).
    #[inline]
    pub fn embedding_dim(&self) -> u16 {
        self.embedding_dim
    }

    /// Schema version of the producing embedding model.
    #[inline]
    pub fn sketch_version(&self) -> u16 {
        self.sketch_version
    }

    /// Borrow the inner ruvector-core `BinaryQuantized` for advanced use
    /// (e.g., serialisation through ruvector's existing infrastructure).
    /// Most callers should use [`Sketch::distance`] or [`SketchBank`].
    #[inline]
    pub fn as_inner(&self) -> &BinaryQuantized {
        &self.inner
    }

    /// Borrow the packed sketch bytes (1 bit per source-embedding
    /// dimension, ceil-divided into bytes). Used by [`WireSketch`] to
    /// produce a wire-format payload without re-quantizing. Length is
    /// `(embedding_dim + 7) / 8` bytes.
    #[inline]
    pub fn packed_bytes(&self) -> &[u8] {
        &self.inner.bits
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ADR-084 Pass 4 — wire-format primitive (cluster-channel-agnostic)
// ─────────────────────────────────────────────────────────────────────────────

/// Magic bytes for ADR-084 sketch wire frames. Receivers reject any
/// payload that doesn't start with these four bytes — the same shape
/// of magic-prefix check ADR-018's CSI binary frame uses (e.g.
/// `0xC5110001`). Picked to be distinct from any existing RuView magic.
pub const WIRE_SKETCH_MAGIC: u32 = 0xC511_0084;

/// On-the-wire schema version. Bump on any field reordering or addition.
/// `Sketch::sketch_version` (the *embedding model* version) is a
/// separate concept and travels in the payload.
pub const WIRE_SKETCH_FORMAT_VERSION: u16 = 1;

/// Maximum wire-payload size the deserializer will accept. Guards
/// against a malicious sender claiming `embedding_dim = u16::MAX`
/// (would imply 8 KiB of packed bits) and exhausting receiver memory.
/// 8 KiB matches the largest reasonable production embedding (post-
/// rotation 65,535-d sign-quantized) plus a few bytes of header.
pub const WIRE_SKETCH_MAX_BYTES: usize = 9 * 1024;

/// Errors raised by [`WireSketch::deserialize`].
#[derive(Debug, thiserror::Error)]
pub enum WireSketchError {
    /// Payload shorter than the fixed header (12 bytes).
    #[error("wire payload too short: got {got} bytes, header needs {needed}")]
    TooShort {
        /// Bytes received.
        got: usize,
        /// Minimum bytes required (12).
        needed: usize,
    },
    /// Payload larger than [`WIRE_SKETCH_MAX_BYTES`].
    #[error("wire payload exceeds max ({got} > {max})")]
    TooLarge {
        /// Bytes received.
        got: usize,
        /// Maximum bytes accepted.
        max: usize,
    },
    /// Magic bytes do not match [`WIRE_SKETCH_MAGIC`].
    #[error("wire magic mismatch: got 0x{got:08X}, expected 0x{expected:08X}")]
    MagicMismatch {
        /// Magic value received.
        got: u32,
        /// Magic value expected.
        expected: u32,
    },
    /// Format version is newer than the receiver knows how to parse.
    #[error("wire format_version {got} > supported {max}")]
    UnsupportedVersion {
        /// Version received.
        got: u16,
        /// Highest version this build understands.
        max: u16,
    },
    /// `embedding_dim` and the byte payload disagree on size.
    #[error("payload byte count mismatch: header dim={dim} → expected {expected_bytes}, got {got_bytes}")]
    PayloadSizeMismatch {
        /// Embedding dimension in the header.
        dim: u16,
        /// Bytes the header implies.
        expected_bytes: usize,
        /// Bytes actually present.
        got_bytes: usize,
    },
}

/// Serialize / deserialize a `Sketch` plus its novelty score for
/// transmission over any channel — cluster↔cluster mesh, sensor→Pi UDP,
/// gateway→cloud QUIC, etc.
///
/// # Wire layout (little-endian, packed)
///
/// | Offset | Field              | Width | Notes                                      |
/// |--------|--------------------|-------|--------------------------------------------|
/// | 0      | `magic`            | u32   | [`WIRE_SKETCH_MAGIC`]                      |
/// | 4      | `format_version`   | u16   | [`WIRE_SKETCH_FORMAT_VERSION`]             |
/// | 6      | `sketch_version`   | u16   | embedding-model schema version             |
/// | 8      | `embedding_dim`    | u16   | source-embedding dimensions                |
/// | 10     | `novelty_q15`      | u16   | novelty in `[0,1]` × 32_767 (saturated)    |
/// | 12     | `bits[]`           | var   | `(embedding_dim + 7) / 8` bytes            |
///
/// Header is exactly **12 bytes**; payload is `ceil(embedding_dim/8)`
/// bytes. Total for a 128-d AETHER sketch is 12 + 16 = **28 bytes**.
///
/// # Why the receiver is paranoid
///
/// All deserialization paths validate magic, format_version,
/// embedding_dim → payload-bytes consistency, and total size before
/// touching `BinaryQuantized`. A malformed UDP packet from a
/// non-RuView sender will produce a typed `WireSketchError`, never a
/// panic. Caps via [`WIRE_SKETCH_MAX_BYTES`] guard against memory-
/// exhaustion attacks.
pub struct WireSketch;

impl WireSketch {
    /// Header size (magic + format_version + sketch_version + dim + novelty).
    pub const HEADER_BYTES: usize = 12;

    /// Encode a sketch + novelty score for transmission. `novelty` is
    /// clamped to `[0.0, 1.0]` and quantized to a `u16` (q15 fixed-
    /// point) so the wire payload is fixed-size. Encoding never
    /// allocates more than `Self::HEADER_BYTES + sketch.packed_bytes().len()`.
    pub fn serialize(sketch: &Sketch, novelty: f32) -> Vec<u8> {
        let bits = sketch.packed_bytes();
        let total = Self::HEADER_BYTES + bits.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&WIRE_SKETCH_MAGIC.to_le_bytes());
        out.extend_from_slice(&WIRE_SKETCH_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&sketch.sketch_version.to_le_bytes());
        out.extend_from_slice(&sketch.embedding_dim.to_le_bytes());
        let nov_q15: u16 = (novelty.clamp(0.0, 1.0) * 32_767.0).round() as u16;
        out.extend_from_slice(&nov_q15.to_le_bytes());
        out.extend_from_slice(bits);
        out
    }

    /// Decode a sketch + novelty score from an untrusted byte buffer.
    /// Returns the parsed `(Sketch, novelty)` tuple, or a typed error.
    pub fn deserialize(buf: &[u8]) -> Result<(Sketch, f32), WireSketchError> {
        // Length floor: must contain at least the header.
        if buf.len() < Self::HEADER_BYTES {
            return Err(WireSketchError::TooShort {
                got: buf.len(),
                needed: Self::HEADER_BYTES,
            });
        }
        // Length ceiling: defend against memory-exhaustion attacks via
        // claimed-but-impossible large dims.
        if buf.len() > WIRE_SKETCH_MAX_BYTES {
            return Err(WireSketchError::TooLarge {
                got: buf.len(),
                max: WIRE_SKETCH_MAX_BYTES,
            });
        }

        let magic = u32::from_le_bytes(buf[0..4].try_into().expect("4-byte slice"));
        if magic != WIRE_SKETCH_MAGIC {
            return Err(WireSketchError::MagicMismatch {
                got: magic,
                expected: WIRE_SKETCH_MAGIC,
            });
        }

        let format_version = u16::from_le_bytes(buf[4..6].try_into().expect("2-byte slice"));
        if format_version > WIRE_SKETCH_FORMAT_VERSION {
            return Err(WireSketchError::UnsupportedVersion {
                got: format_version,
                max: WIRE_SKETCH_FORMAT_VERSION,
            });
        }

        let sketch_version = u16::from_le_bytes(buf[6..8].try_into().expect("2-byte slice"));
        let embedding_dim = u16::from_le_bytes(buf[8..10].try_into().expect("2-byte slice"));
        let nov_q15 = u16::from_le_bytes(buf[10..12].try_into().expect("2-byte slice"));

        let expected_bits = (embedding_dim as usize).div_ceil(8);
        let got_bits = buf.len() - Self::HEADER_BYTES;
        if expected_bits != got_bits {
            return Err(WireSketchError::PayloadSizeMismatch {
                dim: embedding_dim,
                expected_bytes: expected_bits,
                got_bytes: got_bits,
            });
        }

        let bits = buf[Self::HEADER_BYTES..].to_vec();
        let sketch = Sketch {
            inner: BinaryQuantized {
                bits,
                dimensions: embedding_dim as usize,
            },
            embedding_dim,
            sketch_version,
        };
        let novelty = (nov_q15 as f32) / 32_767.0;
        Ok((sketch, novelty))
    }
}

/// A bank of sketches with stable IDs, queried for top-K nearest neighbours
/// by hamming distance.
///
/// Used at every "have I seen this before" site in the pipeline. The bank
/// enforces `sketch_version` and `embedding_dim` consistency at insertion
/// time, so `topk` queries never need to re-check.
///
/// # Invariants
///
/// - All sketches in a bank share the same `embedding_dim` and `sketch_version`.
/// - Bank IDs (`u32`) are caller-assigned and stable across `topk` calls;
///   the bank does not renumber on insertion or removal.
#[derive(Debug, Clone)]
pub struct SketchBank {
    /// (id, sketch) pairs in insertion order.
    entries: Vec<(u32, Sketch)>,
    /// Locked at first insertion; all subsequent inserts must match.
    embedding_dim: Option<u16>,
    /// Locked at first insertion; all subsequent inserts must match.
    sketch_version: Option<u16>,
    /// Optional RaBitQ Pass-2 rotation ([ADR-156 §8]). When `Some`, the
    /// embedding-taking helpers ([`SketchBank::insert_embedding`],
    /// [`SketchBank::topk_embedding`], [`SketchBank::novelty_embedding`])
    /// rotate every embedding through this exact rotation before sketching, so
    /// index-time and query-time sketches always share one rotated frame. The
    /// raw [`SketchBank::insert`] / [`SketchBank::topk`] paths are unchanged —
    /// callers using pre-built sketches are responsible for having rotated them
    /// with the same `Rotation`.
    rotation: Option<Rotation>,
}

impl SketchBank {
    /// Create an empty bank. Dimension and version are locked at the first
    /// `insert` call. No Pass-2 rotation (pure Pass-1, default behaviour).
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            embedding_dim: None,
            sketch_version: None,
            rotation: None,
        }
    }

    /// Create a bank with a pre-locked `embedding_dim` and `sketch_version`.
    /// Use when the bank's expected schema is known at construction.
    /// No Pass-2 rotation (pure Pass-1).
    pub fn with_schema(embedding_dim: u16, sketch_version: u16) -> Self {
        Self {
            entries: Vec::new(),
            embedding_dim: Some(embedding_dim),
            sketch_version: Some(sketch_version),
            rotation: None,
        }
    }

    /// Create a **RaBitQ Pass-2** bank that rotates every embedding through
    /// `rotation` before sketching ([ADR-156 §8]).
    ///
    /// Use the embedding-taking helpers ([`SketchBank::insert_embedding`],
    /// [`SketchBank::topk_embedding`], [`SketchBank::novelty_embedding`]) with
    /// this bank so the index and queries share the same rotated frame. The
    /// `embedding_dim` / `sketch_version` schema is still locked at first
    /// insert exactly as for a Pass-1 bank — a Pass-2 sketch is byte-identical
    /// in shape to a Pass-1 sketch, only its bits differ.
    pub fn with_rotation(rotation: Rotation) -> Self {
        Self {
            entries: Vec::new(),
            embedding_dim: None,
            sketch_version: None,
            rotation: Some(rotation),
        }
    }

    /// The Pass-2 rotation this bank applies to embeddings, if any.
    #[inline]
    pub fn rotation(&self) -> Option<&Rotation> {
        self.rotation.as_ref()
    }

    /// Sketch a raw embedding using this bank's rotation policy: Pass-2
    /// (`from_embedding_rotated`) if the bank has a rotation, else Pass-1
    /// (`from_embedding`). The single place index-time and query-time sketching
    /// agree on the rotated frame.
    fn sketch_embedding(&self, embedding: &[f32], sketch_version: u16) -> Sketch {
        match &self.rotation {
            Some(r) => Sketch::from_embedding_rotated(embedding, sketch_version, r),
            None => Sketch::from_embedding(embedding, sketch_version),
        }
    }

    /// Insert a raw embedding, sketching it through the bank's rotation policy.
    /// Convenience wrapper over [`SketchBank::insert`] that guarantees the
    /// stored sketch used the same (Pass-1 or Pass-2) frame the queries will.
    pub fn insert_embedding(
        &mut self,
        id: u32,
        embedding: &[f32],
        sketch_version: u16,
    ) -> Result<(), SketchError> {
        let sketch = self.sketch_embedding(embedding, sketch_version);
        self.insert(id, sketch)
    }

    /// Top-K over a raw query embedding, sketched through the bank's rotation
    /// policy. Equivalent to `bank.topk(&bank.sketch(query), k)` but cannot get
    /// the rotation frame wrong.
    pub fn topk_embedding(
        &self,
        query: &[f32],
        sketch_version: u16,
        k: usize,
    ) -> Result<Vec<(u32, u32)>, SketchError> {
        let q = self.sketch_embedding(query, sketch_version);
        self.topk(&q, k)
    }

    /// Novelty of a raw query embedding, sketched through the bank's rotation
    /// policy. See [`SketchBank::novelty`].
    pub fn novelty_embedding(
        &self,
        query: &[f32],
        sketch_version: u16,
    ) -> Result<f32, SketchError> {
        let q = self.sketch_embedding(query, sketch_version);
        self.novelty(&q)
    }

    /// Number of sketches in the bank.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the bank has no sketches.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Locked embedding dimension, or `None` if the bank is empty and
    /// no schema was pre-supplied.
    #[inline]
    pub fn embedding_dim(&self) -> Option<u16> {
        self.embedding_dim
    }

    /// Locked sketch version, or `None` if the bank is empty and
    /// no schema was pre-supplied.
    #[inline]
    pub fn sketch_version(&self) -> Option<u16> {
        self.sketch_version
    }

    /// Insert a sketch with caller-assigned ID. Locks the bank's schema on
    /// first insertion; rejects subsequent inserts that mismatch.
    pub fn insert(&mut self, id: u32, sketch: Sketch) -> Result<(), SketchError> {
        match self.embedding_dim {
            None => self.embedding_dim = Some(sketch.embedding_dim),
            Some(d) if d != sketch.embedding_dim => {
                return Err(SketchError::EmbeddingDimMismatch {
                    bank: d,
                    query: sketch.embedding_dim,
                });
            }
            _ => {}
        }
        match self.sketch_version {
            None => self.sketch_version = Some(sketch.sketch_version),
            Some(v) if v != sketch.sketch_version => {
                return Err(SketchError::SketchVersionMismatch {
                    bank: v,
                    query: sketch.sketch_version,
                });
            }
            _ => {}
        }
        self.entries.push((id, sketch));
        Ok(())
    }

    /// Top-K nearest neighbours by hamming distance, ascending.
    ///
    /// Returns up to `k` `(id, distance)` pairs sorted by distance. If the
    /// bank has fewer than `k` entries, returns all of them. If `k == 0`,
    /// returns empty.
    ///
    /// Returns `Err` if the query's `embedding_dim` or `sketch_version`
    /// disagrees with the bank's locked schema. (Cannot return `Err` if the
    /// bank is empty *and* no schema was pre-supplied — there's nothing to
    /// disagree with.)
    pub fn topk(&self, query: &Sketch, k: usize) -> Result<Vec<(u32, u32)>, SketchError> {
        if k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(d) = self.embedding_dim {
            if d != query.embedding_dim {
                return Err(SketchError::EmbeddingDimMismatch {
                    bank: d,
                    query: query.embedding_dim,
                });
            }
        }
        if let Some(v) = self.sketch_version {
            if v != query.sketch_version {
                return Err(SketchError::SketchVersionMismatch {
                    bank: v,
                    query: query.sketch_version,
                });
            }
        }
        // Partial top-K via a fixed-size **max-heap** of `(distance, id)`.
        // `BinaryHeap` is a max-heap, so `peek()` is the *largest* distance
        // currently held — the worst of the running best-k. Each candidate is
        // O(1)-compared against that worst; only a *smaller* distance triggers
        // an O(log k) pop+push, evicting the current worst. The heap therefore
        // retains the k *smallest* distances. Total O(n log k), touching the
        // long tail only with a single comparison each.
        //
        // BUG FIX (ADR-156 §8 Pass-2 work): this loop previously used
        // `BinaryHeap<Reverse<(d, id)>>` and called the peek "the largest".
        // `Reverse` turns the max-heap into a **min-heap**, so `peek()` was the
        // *smallest* distance; evicting on `d < worst` then kept the k
        // *farthest* neighbours and returned them as "nearest". The pre-existing
        // unit tests only exercised the `n <= k` fast path (≤ 3 entries), so the
        // inversion went unnoticed until the Pass-2 coverage harness measured
        // near-random top-K on n > k. Pinned by `topk_heap_path_returns_nearest`.
        //
        // Fast path: when n ≤ k there is nothing to discard, so a plain
        // collect + sort is faster than building a heap.
        let n = self.entries.len();
        if n <= k {
            let mut scored: Vec<(u32, u32)> = self
                .entries
                .iter()
                .map(|(id, sk)| (*id, sk.distance_unchecked(query)))
                .collect();
            scored.sort_by_key(|&(_, d)| d);
            return Ok(scored);
        }

        let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
        for (id, sk) in &self.entries {
            let d = sk.distance_unchecked(query);
            if heap.len() < k {
                heap.push((d, *id));
            } else if let Some(&(worst, _)) = heap.peek() {
                // `peek()` is the largest distance in the best-k (max-heap).
                // The `if let` is defensive: when `heap.len() == k > 0` the
                // heap is non-empty, so this never takes the `else`. Same
                // hot-path cost (one bounds check), zero panic risk.
                if d < worst {
                    heap.pop();
                    heap.push((d, *id));
                }
            }
        }
        // Drain the max-heap and sort ascending-by-distance per the public
        // contract (heap drain order is unspecified beyond the root).
        let mut scored: Vec<(u32, u32)> = heap.into_iter().map(|(d, id)| (id, d)).collect();
        scored.sort_by_key(|&(_, d)| d);
        Ok(scored)
    }

    /// Compute the novelty score of a query against the bank in `[0.0, 1.0]`.
    ///
    /// Defined as `min_distance / embedding_dim`, so 0.0 means "exact bit
    /// match exists in the bank" and 1.0 means "every bit differs from the
    /// nearest stored sketch." Returns 1.0 (max novelty) on an empty bank.
    /// Returns `Err` on schema mismatch.
    pub fn novelty(&self, query: &Sketch) -> Result<f32, SketchError> {
        if self.entries.is_empty() {
            return Ok(1.0);
        }
        let topk = self.topk(query, 1)?;
        let min_distance = topk.first().map(|&(_, d)| d).unwrap_or(u32::MAX);
        Ok(min_distance as f32 / query.embedding_dim as f32)
    }
}

impl Default for SketchBank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_embedding_packs_one_bit_per_dim() {
        let v = vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5];
        let s = Sketch::from_embedding(&v, 1);
        assert_eq!(s.embedding_dim(), 8);
        assert_eq!(s.sketch_version(), 1);
        // Distance to self is 0
        assert_eq!(s.distance_unchecked(&s), 0);
    }

    #[test]
    fn distance_is_hamming_count() {
        let a = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1);
        let b = Sketch::from_embedding(&[-0.5, -0.5, -0.5, -0.5], 1);
        // All 4 dims flipped sign → 4 bit differences.
        assert_eq!(a.distance(&b).unwrap(), 4);
    }

    #[test]
    fn distance_rejects_mismatched_dims() {
        let a = Sketch::from_embedding(&[0.5, 0.5], 1);
        let b = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1);
        let err = a.distance(&b).unwrap_err();
        assert!(matches!(err, SketchError::EmbeddingDimMismatch { .. }));
    }

    #[test]
    fn distance_rejects_mismatched_versions() {
        let a = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1);
        let b = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 2);
        let err = a.distance(&b).unwrap_err();
        assert!(matches!(err, SketchError::SketchVersionMismatch { .. }));
    }

    #[test]
    fn bank_topk_returns_sorted_by_distance() {
        let mut bank = SketchBank::new();
        // id 10: identical
        bank.insert(10, Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1))
            .unwrap();
        // id 20: 1 bit different (last dim flipped)
        bank.insert(20, Sketch::from_embedding(&[0.5, 0.5, 0.5, -0.5], 1))
            .unwrap();
        // id 30: 2 bits different
        bank.insert(30, Sketch::from_embedding(&[-0.5, 0.5, -0.5, 0.5], 1))
            .unwrap();

        let query = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1);
        let topk = bank.topk(&query, 3).unwrap();

        assert_eq!(topk.len(), 3);
        assert_eq!(topk[0].0, 10); // 0 distance
        assert_eq!(topk[1].0, 20); // 1 distance
        assert_eq!(topk[2].0, 30); // 2 distance
        assert!(topk[0].1 <= topk[1].1);
        assert!(topk[1].1 <= topk[2].1);
    }

    #[test]
    fn topk_heap_path_returns_nearest() {
        // Regression for the heap-inversion bug found during ADR-156 §8 Pass-2
        // work: with n > k the topk used a min-heap (`Reverse`) but treated its
        // peek as the max, so it returned the k *farthest* sketches. Build a
        // bank where the answer is unambiguous and assert the genuine nearest
        // come back. The OLD code returns the farthest here and fails.
        let dim = 64;
        let k = 4;
        // Query is all-positive (every bit 1).
        let query = Sketch::from_embedding(&vec![1.0f32; dim], 1);
        let mut bank = SketchBank::new();
        // id j has its first `j` dims flipped negative → hamming j to the
        // all-positive query. So nearest-4 are ids 0,1,2,3 (hamming 0,1,2,3);
        // farthest are 5..8. n = 9 > k = 4 → exercises the heap path.
        //
        // CRITICAL ordering: insert FARTHEST-FIRST (id 8 down to 0). This fills
        // the heap's first k slots with far entries, so the nearest entries
        // arrive only after the heap is full and MUST trigger eviction of the
        // current worst. The old `Reverse` (min-heap-as-max) bug peeked the
        // smallest distance and never evicted, so it kept the first-seen
        // (farthest) k and this assertion fails on the old code. Inserting
        // nearest-first would mask the bug (the heap fills with the right
        // answer by luck), so the order here is load-bearing.
        for j in (0..=8u32).rev() {
            let mut v = vec![1.0f32; dim];
            for d in v.iter_mut().take(j as usize) {
                *d = -1.0;
            }
            bank.insert(j, Sketch::from_embedding(&v, 1)).unwrap();
        }
        let top = bank.topk(&query, k).unwrap();
        assert_eq!(top.len(), k);
        let ids: Vec<u32> = top.iter().map(|&(id, _)| id).collect();
        let dists: Vec<u32> = top.iter().map(|&(_, d)| d).collect();
        assert_eq!(ids, vec![0, 1, 2, 3], "topk must return the NEAREST k, got {ids:?}");
        assert_eq!(dists, vec![0, 1, 2, 3], "distances must be the smallest k");
    }

    #[test]
    fn bank_topk_zero_returns_empty() {
        let mut bank = SketchBank::new();
        bank.insert(1, Sketch::from_embedding(&[0.5, 0.5], 1))
            .unwrap();
        let q = Sketch::from_embedding(&[0.5, 0.5], 1);
        assert_eq!(bank.topk(&q, 0).unwrap().len(), 0);
    }

    #[test]
    fn bank_topk_more_than_size_returns_all() {
        let mut bank = SketchBank::new();
        bank.insert(1, Sketch::from_embedding(&[0.5, 0.5], 1))
            .unwrap();
        bank.insert(2, Sketch::from_embedding(&[-0.5, 0.5], 1))
            .unwrap();
        let q = Sketch::from_embedding(&[0.5, 0.5], 1);
        assert_eq!(bank.topk(&q, 100).unwrap().len(), 2);
    }

    #[test]
    fn bank_locks_schema_on_first_insert() {
        let mut bank = SketchBank::new();
        bank.insert(1, Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1))
            .unwrap();
        // Different version → reject
        let err = bank
            .insert(2, Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 2))
            .unwrap_err();
        assert!(matches!(err, SketchError::SketchVersionMismatch { .. }));
        // Different dim → reject
        let err = bank
            .insert(3, Sketch::from_embedding(&[0.5, 0.5], 1))
            .unwrap_err();
        assert!(matches!(err, SketchError::EmbeddingDimMismatch { .. }));
    }

    #[test]
    fn bank_with_schema_rejects_first_mismatching_insert() {
        let mut bank = SketchBank::with_schema(4, 7);
        let err = bank
            .insert(1, Sketch::from_embedding(&[0.5, 0.5], 7))
            .unwrap_err();
        assert!(matches!(err, SketchError::EmbeddingDimMismatch { .. }));
    }

    #[test]
    fn novelty_zero_for_exact_match_one_for_empty() {
        let bank_empty = SketchBank::new();
        let q = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1);
        assert_eq!(bank_empty.novelty(&q).unwrap(), 1.0);

        let mut bank = SketchBank::new();
        bank.insert(1, q.clone()).unwrap();
        assert_eq!(bank.novelty(&q).unwrap(), 0.0);
    }

    #[test]
    fn novelty_is_proportional_to_min_distance() {
        let mut bank = SketchBank::new();
        // Bank has one sketch with all 8 dims positive.
        bank.insert(1, Sketch::from_embedding(&[0.5; 8], 1))
            .unwrap();
        // Query flips half the dims → 4 bit difference / 8 dims = 0.5.
        let query = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5], 1);
        let novelty = bank.novelty(&query).unwrap();
        assert!((novelty - 0.5).abs() < 1e-6);
    }

    #[test]
    fn try_from_embedding_rejects_over_long_input() {
        // L2 security-review finding (PR #435): the infallible
        // `from_embedding` saturates to u16::MAX; the fallible
        // `try_from_embedding` must surface the overflow so callers can
        // detect the misuse. We can't actually allocate a 65,536-f32
        // vector in unit tests cheaply (that's 256 KiB, fine), but we
        // can fabricate a `Vec` with `len() > u16::MAX` and check the
        // error path.
        let too_long: Vec<f32> = vec![0.5; (u16::MAX as usize) + 1];
        let err = Sketch::try_from_embedding(&too_long, 1).unwrap_err();
        match err {
            SketchError::EmbeddingDimOverflow { got, max } => {
                assert_eq!(got, (u16::MAX as usize) + 1);
                assert_eq!(max, u16::MAX as usize);
            }
            _ => panic!("expected EmbeddingDimOverflow, got {err:?}"),
        }

        // The infallible path should *saturate* to u16::MAX rather
        // than panic in release. Verify the saturation is observable
        // on `embedding_dim()`.
        let s = Sketch::from_embedding(&too_long, 1);
        assert_eq!(s.embedding_dim(), u16::MAX);
    }

    // ─── ADR-084 Pass 4 wire-format tests ────────────────────────────────────

    #[test]
    fn wire_serialize_round_trip() {
        let v = vec![0.5_f32, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5];
        let sketch = Sketch::from_embedding(&v, 7);
        let bytes = WireSketch::serialize(&sketch, 0.42);

        // Header (12) + 1 byte (8 dims / 8) = 13 bytes total.
        assert_eq!(bytes.len(), WireSketch::HEADER_BYTES + 1);

        let (decoded, novelty) = WireSketch::deserialize(&bytes).expect("round-trip");
        assert_eq!(decoded.embedding_dim(), 8);
        assert_eq!(decoded.sketch_version(), 7);
        assert_eq!(decoded.distance_unchecked(&sketch), 0);
        // q15 quantization round-trips with bounded error.
        assert!((novelty - 0.42).abs() < 1.0 / 32_767.0 * 2.0);
    }

    #[test]
    fn wire_rejects_short_buffer() {
        let err = WireSketch::deserialize(&[0u8; 5]).unwrap_err();
        match err {
            WireSketchError::TooShort { got: 5, needed } => {
                assert_eq!(needed, WireSketch::HEADER_BYTES);
            }
            _ => panic!("expected TooShort, got {err:?}"),
        }
    }

    #[test]
    fn wire_rejects_oversized_buffer() {
        let big = vec![0u8; WIRE_SKETCH_MAX_BYTES + 1];
        let err = WireSketch::deserialize(&big).unwrap_err();
        assert!(matches!(err, WireSketchError::TooLarge { .. }));
    }

    #[test]
    fn wire_rejects_bad_magic() {
        let mut bytes = WireSketch::serialize(&Sketch::from_embedding(&[0.5; 16], 1), 0.0);
        bytes[0..4].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
        let err = WireSketch::deserialize(&bytes).unwrap_err();
        assert!(matches!(err, WireSketchError::MagicMismatch { .. }));
    }

    #[test]
    fn wire_rejects_unsupported_format_version() {
        let mut bytes = WireSketch::serialize(&Sketch::from_embedding(&[0.5; 16], 1), 0.0);
        // Bump format_version to 99 — beyond what this build supports.
        bytes[4..6].copy_from_slice(&99_u16.to_le_bytes());
        let err = WireSketch::deserialize(&bytes).unwrap_err();
        assert!(matches!(
            err,
            WireSketchError::UnsupportedVersion { got: 99, .. }
        ));
    }

    #[test]
    fn wire_rejects_payload_size_mismatch() {
        // Build a valid 16-d sketch (2 bytes), then claim dim=24 in the
        // header (would need 3 bytes). Payload-size check must fire.
        let mut bytes = WireSketch::serialize(&Sketch::from_embedding(&[0.5; 16], 1), 0.0);
        bytes[8..10].copy_from_slice(&24_u16.to_le_bytes());
        let err = WireSketch::deserialize(&bytes).unwrap_err();
        match err {
            WireSketchError::PayloadSizeMismatch {
                dim: 24,
                expected_bytes: 3,
                got_bytes: 2,
            } => {}
            _ => panic!("expected PayloadSizeMismatch, got {err:?}"),
        }
    }

    #[test]
    fn wire_envelope_size_for_aether_128d() {
        // Documented size sanity: a 128-d AETHER sketch should fit in
        // 12-byte header + 16-byte payload = 28 bytes total.
        let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let sketch = Sketch::from_embedding(&v, 1);
        let bytes = WireSketch::serialize(&sketch, 0.5);
        assert_eq!(
            bytes.len(),
            28,
            "AETHER 128-d must wire to exactly 28 bytes"
        );
    }

    #[test]
    fn topk_rejects_query_with_wrong_schema() {
        let mut bank = SketchBank::with_schema(4, 1);
        bank.insert(1, Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 1))
            .unwrap();
        let bad_dim = Sketch::from_embedding(&[0.5, 0.5], 1);
        assert!(matches!(
            bank.topk(&bad_dim, 1).unwrap_err(),
            SketchError::EmbeddingDimMismatch { .. }
        ));
        let bad_ver = Sketch::from_embedding(&[0.5, 0.5, 0.5, 0.5], 99);
        assert!(matches!(
            bank.topk(&bad_ver, 1).unwrap_err(),
            SketchError::SketchVersionMismatch { .. }
        ));
    }

    // ─── ADR-156 §8 / ADR-084 Pass 2 — randomized rotation ───────────────────

    #[test]
    fn rotated_sketch_has_same_shape_as_pass1() {
        // A Pass-2 sketch must be byte-shape-identical to a Pass-1 sketch of
        // the same input: same embedding_dim, same packed-byte length, same
        // sketch_version. Only the bits differ. This is what lets Pass-2
        // sketches travel through the unchanged WireSketch / SketchBank schema.
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.21).sin()).collect();
        let rot = Rotation::new(0xA5A5_A5A5, 128);
        let p1 = Sketch::from_embedding(&v, 3);
        let p2 = Sketch::from_embedding_rotated(&v, 3, &rot);
        assert_eq!(p1.embedding_dim(), p2.embedding_dim());
        assert_eq!(p1.sketch_version(), p2.sketch_version());
        assert_eq!(p1.packed_bytes().len(), p2.packed_bytes().len());
        // The rotation actually changed the bits (else it would be a no-op on
        // this correlated input).
        assert_ne!(
            p1.packed_bytes(),
            p2.packed_bytes(),
            "rotation should change the sign bits on correlated input"
        );
    }

    #[test]
    fn rotated_sketch_is_deterministic_for_seed() {
        // Same (seed, dim) rotation → identical sketch bits across constructions
        // (the index-time == query-time contract, at the sketch layer).
        let v: Vec<f32> = (0..96).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.3).collect();
        let s1 = Sketch::from_embedding_rotated(&v, 1, &Rotation::new(7, 96));
        let s2 = Sketch::from_embedding_rotated(&v, 1, &Rotation::new(7, 96));
        assert_eq!(s1.distance_unchecked(&s2), 0, "same seed must agree exactly");
    }

    #[test]
    fn rotated_bank_self_match_is_zero_distance() {
        // A rotated bank queried with the same embedding it stored must return
        // that id at distance 0 — proves the bank rotates index and query in
        // the same frame.
        let rot = Rotation::new(0xBEEF, 64);
        let mut bank = SketchBank::with_rotation(rot);
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).cos()).collect();
        bank.insert_embedding(42, &v, 1).unwrap();
        let top = bank.topk_embedding(&v, 1, 1).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 42);
        assert_eq!(top[0].1, 0, "self-query in a rotated bank must be distance 0");
    }

    #[test]
    fn pass2_coverage_not_worse_than_pass1() {
        // The core regression: on a small fixed anisotropic fixture, Pass-2
        // (rotation) coverage must be >= Pass-1 coverage. Rotation must not
        // *hurt* recall. (We do not assert a hard >= 90% here — that is the
        // measurement reported in the ADR, not a unit-test invariant — but we
        // do pin that rotation is not a regression.)
        use crate::coverage::{measure_pass1, measure_pass2, CoverageParams};
        let p = CoverageParams {
            n: 512,
            n_queries: 32,
            n_clusters: 32,
            ..CoverageParams::aether_default(0x00C0_FFEE)
        };
        let c1 = measure_pass1(p).coverage;
        let c2 = measure_pass2(p, 0x1234_5678_9ABC_DEF0).coverage;
        assert!(
            c2 + 1e-9 >= c1,
            "Pass-2 coverage {c2:.4} regressed below Pass-1 {c1:.4}"
        );
    }

    /// Deterministic, test-runnable coverage measurement that PRINTS the
    /// numbers quoted in ADR-084 / ADR-156 §8. Run with `--nocapture` to see:
    ///   cargo test -p wifi-densepose-ruvector --no-default-features \
    ///     pass2_coverage_report -- --nocapture
    #[test]
    fn pass2_coverage_report() {
        use crate::coverage::{measure_pass1, measure_pass2, CoverageParams};
        let base = CoverageParams::aether_default(0xAD00_0084);
        let rot_seed = 0x5EED_C0DE_1234_5678u64;
        println!(
            "\n=== ADR-156 §8 RaBitQ Pass-2 coverage report (anisotropic synthetic) ==="
        );
        println!(
            "dim={} N={} K={} queries={} master_seed=0x{:X} rotation_seed=0x{:X}",
            base.dim, base.n, base.k, base.n_queries, base.seed, rot_seed
        );
        // Strict bar: candidate_k == K.
        let p1 = measure_pass1(base).coverage;
        let p2 = measure_pass2(base, rot_seed).coverage;
        println!(
            "candidate_k=K={:<2}  Pass1={:6.2}%  Pass2={:6.2}%  bar=90%  {}",
            base.k,
            p1 * 100.0,
            p2 * 100.0,
            if p2 >= 0.90 { "PASS" } else { "BELOW-BAR" }
        );
        // Over-fetch curve (models fetch C >= K candidates, refine to K).
        for &c in &[16usize, 24, 32, 64] {
            let pc = CoverageParams {
                candidate_k: c,
                ..base
            };
            let cp1 = measure_pass1(pc).coverage;
            let cp2 = measure_pass2(pc, rot_seed).coverage;
            println!(
                "candidate_k={:<3}     Pass1={:6.2}%  Pass2={:6.2}%",
                c,
                cp1 * 100.0,
                cp2 * 100.0
            );
        }
        println!("========================================================================\n");
        // Always-true sanity so the test asserts something.
        assert!((0.0..=1.0).contains(&p1));
        assert!((0.0..=1.0).contains(&p2));
    }
}
