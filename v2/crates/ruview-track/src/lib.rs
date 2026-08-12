//! # `ruview-track` — persistent, privacy-preserving probabilistic tracking (ADR-307)
//!
//! Builds **track continuity without civil identity**. A [`TrackManager`]
//! ingests per-frame [`Detection`]s (a container + 2-D position + a coarse,
//! non-reversible [`CoarseFeature`] + an injected timestamp) and maintains
//! persistent [`Track`](ruview_ontology::Track) entities, each bound to a
//! pseudonymous [`Person`](ruview_ontology::Person) such as `person_7`. It
//! answers "person_7 moved kitchen → hallway → bedroom" via per-entity
//! [histories](TrackManager::history) — across zones, rooms, and modalities.
//!
//! This crate produces and updates the **canonical ADR-306 ontology types**
//! (`Track`, `Person`, `Container`, `EvidenceLevel`, `SemanticProvenance`) from
//! [`ruview_ontology`]; it invents no per-crate identity shape (ADR-300 rule 3).
//!
//! ## The four privacy invariants (ADR-307 §3), enforced by construction
//!
//! 1. **No civil-identity binding.** The pseudonym is a synthetic id with no
//!    field or join key to a name, account, MAC, or phone — the ontology
//!    `Person`/`Track` schema carries no such field, so a binding is impossible.
//! 2. **Coarse, non-reversible features.** [`CoarseFeature`] quantizes to a few
//!    buckets and offers no de-quantizer; no long-term biometric template is
//!    persisted.
//! 3. **Opaque, rotatable ids.** Pseudonyms are `person_N` strings and can be
//!    rotated with [`TrackManager::rotate_pseudonym`].
//! 4. **UNKNOWN is first-class** (ADR-300 rule 1). An unmatched or ambiguous
//!    detection spawns a *tentative* track and returns an
//!    [`Association::Unknown`] outcome — it never forces a wrong join and never
//!    errors. Under-linking (a fresh pseudonym when unsure) is the privacy-safe
//!    failure mode.
//!
//! ## Evidence discipline
//!
//! This crate asserts **no accuracy number** (ADR-307 §Validation). Emitted
//! nodes carry the caller-supplied [`EvidenceLevel`](ruview_ontology::EvidenceLevel)
//! (default `L1`, heuristic/synthetic) and a pseudonymous
//! [`SemanticProvenance`](ruview_ontology::SemanticProvenance); tentative tracks
//! are floored to `L0`. In-crate tests use synthetic in-code fixtures only.
//!
//! ## Example
//!
//! ```
//! use ruview_track::*;
//! use ruview_ontology::{Container, SpaceId};
//!
//! let kitchen = Container::Space { id: SpaceId::new("kitchen")? };
//! let hallway = Container::Space { id: SpaceId::new("hallway")? };
//!
//! let mut topo = Topology::new();
//! topo.connect(&kitchen, &hallway); // a doorway between them
//!
//! let mut mgr = TrackManager::new(TrackerConfig::default(), topo);
//! let feat = CoarseFeature::quantize(&[0.2, 0.7, 0.4])?;
//!
//! let a = mgr.ingest(Detection::new(kitchen, [1.0, 1.0], feat.clone(), 1_000)?)?;
//! let b = mgr.ingest(Detection::new(hallway, [1.4, 1.1], feat, 1_500)?)?;
//!
//! // Same persistent pseudonym followed across the doorway.
//! assert_eq!(a.person, b.person);
//! assert!(matches!(b.association, Association::Matched { .. }));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod config;
mod error;
mod feature;
mod manager;
mod topology;

pub use config::TrackerConfig;
pub use error::TrackError;
pub use feature::{CoarseFeature, COARSE_LEVELS, MAX_FEATURE_DIM};
pub use manager::{
    Association, Detection, IngestOutcome, TrackManager, TrackState, UnknownReason, Waypoint,
};
pub use topology::Topology;
