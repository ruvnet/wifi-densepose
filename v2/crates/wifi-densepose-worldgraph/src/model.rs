//! ADR-139 §2.1 — typed node/edge model.
//!
//! Nodes and edges are `serde` enums (NOT boxed trait objects) for
//! deterministic, schema-versioned, RVF-friendly persistence. Cross-ADR
//! references (ADR-137 evidence, ADR-141 privacy decision) are carried as
//! opaque content-address `String` handles so the WorldGraph compiles and
//! persists independently of those crates (§2.1, §2.3).

use serde::{Deserialize, Serialize};

/// Stable, monotonic identity for a world entity. Distinct from petgraph's
/// `NodeIndex` (graph-internal handle); `WorldId` survives RVF round-trips and
/// node removal. `WorldId(0)` is the "assign me one" sentinel for `upsert_node`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldId(pub u64);

impl WorldId {
    /// The "allocate a fresh id" sentinel.
    pub const UNASSIGNED: WorldId = WorldId(0);

    /// Whether this id is the unassigned sentinel.
    #[must_use]
    pub fn is_unassigned(&self) -> bool {
        self.0 == 0
    }
}

/// Local ENU coordinate in metres relative to the installation origin (ADR-044).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnuPoint {
    /// East offset (m).
    pub east_m: f64,
    /// North offset (m).
    pub north_m: f64,
    /// Up offset (m).
    pub up_m: f64,
}

/// MAT `ZoneBounds` reprojected into the installation ENU frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ZoneBoundsEnu {
    /// Axis-aligned rectangle.
    Rectangle {
        /// Minimum east (m).
        min_e: f64,
        /// Minimum north (m).
        min_n: f64,
        /// Maximum east (m).
        max_e: f64,
        /// Maximum north (m).
        max_n: f64,
    },
    /// Circle.
    Circle {
        /// Centre east (m).
        center_e: f64,
        /// Centre north (m).
        center_n: f64,
        /// Radius (m).
        radius_m: f64,
    },
    /// Polygon (east, north) vertices.
    Polygon {
        /// (east, north) vertices.
        vertices: Vec<(f64, f64)>,
    },
}

impl ZoneBoundsEnu {
    /// Whether an ENU point lies within these bounds (up ignored).
    #[must_use]
    pub fn contains(&self, p: &EnuPoint) -> bool {
        match self {
            Self::Rectangle { min_e, min_n, max_e, max_n } => {
                p.east_m >= *min_e && p.east_m <= *max_e && p.north_m >= *min_n && p.north_m <= *max_n
            }
            Self::Circle { center_e, center_n, radius_m } => {
                let de = p.east_m - center_e;
                let dn = p.north_m - center_n;
                (de * de + dn * dn).sqrt() <= *radius_m
            }
            Self::Polygon { vertices } => point_in_polygon(p.east_m, p.north_m, vertices),
        }
    }
}

fn point_in_polygon(px: f64, py: f64, verts: &[(f64, f64)]) -> bool {
    if verts.len() < 3 {
        return false;
    }
    // Ray-casting parity test.
    let mut inside = false;
    let mut j = verts.len() - 1;
    for i in 0..verts.len() {
        let (xi, yi) = verts[i];
        let (xj, yj) = verts[j];
        let intersect = ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Sensing modality of a physical device placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorModality {
    /// WiFi CSI sensing node (ESP32-S3/C6).
    WifiCsi,
    /// 60 GHz mmWave FMCW radar.
    MmWave,
    /// Ultra-wideband ranging beacon (ADR-144).
    Uwb,
    /// Coarse presence sensor.
    Presence,
}

/// Kind of persistent static anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorKind {
    /// A persistent RF reflector (ADR-143 RF SLAM).
    Reflector,
    /// A piece of furniture inferred from reflector clustering.
    Furniture,
    /// A surveyed UWB beacon (ADR-144).
    UwbBeacon,
}

/// Mandatory provenance for every [`WorldNode::SemanticState`] (house rule):
/// every semantic belief traces to signal evidence + model + calibration +
/// privacy decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticProvenance {
    /// ADR-137 `EvidenceRef` content-address handle(s).
    pub evidence: Vec<String>,
    /// Model version (ADR-136 `model_id`/`model_version`) that produced this.
    pub model_version: String,
    /// Calibration version (ADR-135 baseline id) in effect.
    pub calibration_version: String,
    /// Privacy decision (ADR-141 mode + action) it was derived under.
    pub privacy_decision: String,
}

/// A typed world node (ADR-139 §2.1). Persistence-deterministic serde enum.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorldNode {
    /// A bounded interior space, linked to a HomeCore `area_id` (ADR-127).
    Room {
        /// Stable id (or `UNASSIGNED` to allocate).
        id: WorldId,
        /// HomeCore registry area_id — the entity-linkage join key.
        area_id: Option<String>,
        /// Human name.
        name: String,
        /// Room footprint in local ENU.
        bounds_enu: ZoneBoundsEnu,
        /// Floor index.
        floor: i16,
    },
    /// A sub-region of a room targeted for sensing (MAT ScanZone analogue).
    Zone {
        /// Stable id.
        id: WorldId,
        /// Containing room.
        parent_room: WorldId,
        /// Human name.
        name: String,
        /// Zone footprint.
        bounds_enu: ZoneBoundsEnu,
    },
    /// A wall segment (coarse 2D topological element in ENU).
    Wall {
        /// Stable id.
        id: WorldId,
        /// Segment start.
        a: EnuPoint,
        /// Segment end.
        b: EnuPoint,
        /// Coarse RF attenuation (dB): drywall ≈ 3, brick ≈ 12.
        rf_attenuation_db: f32,
    },
    /// A passable opening between two rooms.
    Doorway {
        /// Stable id.
        id: WorldId,
        /// Centre point.
        center: EnuPoint,
        /// Opening width (m).
        width_m: f32,
    },
    /// A physical sensing device placement (ADR-113 placement target).
    Sensor {
        /// Stable id.
        id: WorldId,
        /// Matches HomeCore `EntityEntry.device_id`.
        device_id: String,
        /// Placement in local ENU.
        position: EnuPoint,
        /// Sensing modality.
        modality: SensorModality,
    },
    /// A directed RF propagation channel between two sensors (ADR-138 LinkGroup member).
    RfLink {
        /// Stable id.
        id: WorldId,
        /// Transmit sensor node.
        tx: WorldId,
        /// Receive sensor node.
        rx: WorldId,
        /// ADR-138 MLO LinkGroup id.
        link_group_id: Option<String>,
        /// Centre frequency (MHz).
        center_freq_mhz: u32,
    },
    /// A tracked person (Kalman track id from ruvsense `pose_tracker`).
    PersonTrack {
        /// Stable id.
        id: WorldId,
        /// Tracker track id.
        track_id: u64,
        /// Last known ENU position.
        last_position: EnuPoint,
        /// AETHER re-ID embedding handle.
        reid_embedding_ref: Option<String>,
    },
    /// A persistent static reflector / object (ADR-143 / ADR-144 anchor).
    ObjectAnchor {
        /// Stable id.
        id: WorldId,
        /// ENU position.
        position: EnuPoint,
        /// Anchor classification.
        anchor_kind: AnchorKind,
        /// Confidence in [0, 1].
        confidence: f32,
    },
    /// A discrete detected event (fall, entry, gesture) at a point in time.
    Event {
        /// Stable id.
        id: WorldId,
        /// Event type tag.
        event_type: String,
        /// Wall-clock time (Unix ms).
        at_unix_ms: i64,
        /// Containing room/zone.
        located_in: Option<WorldId>,
    },
    /// A fused semantic belief about the world (the ADR-140 record's graph anchor).
    SemanticState {
        /// Stable id.
        id: WorldId,
        /// Human-readable belief statement.
        statement: String,
        /// Confidence in [0, 1].
        confidence: f32,
        /// Mandatory provenance (house rule).
        provenance: SemanticProvenance,
        /// Belief validity start (Unix ms).
        valid_from_unix_ms: i64,
    },
}

impl WorldNode {
    /// The embedded stable id of this node.
    #[must_use]
    pub fn id(&self) -> WorldId {
        match self {
            Self::Room { id, .. }
            | Self::Zone { id, .. }
            | Self::Wall { id, .. }
            | Self::Doorway { id, .. }
            | Self::Sensor { id, .. }
            | Self::RfLink { id, .. }
            | Self::PersonTrack { id, .. }
            | Self::ObjectAnchor { id, .. }
            | Self::Event { id, .. }
            | Self::SemanticState { id, .. } => *id,
        }
    }

    /// Overwrite the embedded id (used by `upsert_node` when allocating one).
    pub(crate) fn set_id(&mut self, new: WorldId) {
        match self {
            Self::Room { id, .. }
            | Self::Zone { id, .. }
            | Self::Wall { id, .. }
            | Self::Doorway { id, .. }
            | Self::Sensor { id, .. }
            | Self::RfLink { id, .. }
            | Self::PersonTrack { id, .. }
            | Self::ObjectAnchor { id, .. }
            | Self::Event { id, .. }
            | Self::SemanticState { id, .. } => *id = new,
        }
    }

    /// Static kind tag for diagnostics/queries.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Room { .. } => "room",
            Self::Zone { .. } => "zone",
            Self::Wall { .. } => "wall",
            Self::Doorway { .. } => "doorway",
            Self::Sensor { .. } => "sensor",
            Self::RfLink { .. } => "rf_link",
            Self::PersonTrack { .. } => "person_track",
            Self::ObjectAnchor { .. } => "object_anchor",
            Self::Event { .. } => "event",
            Self::SemanticState { .. } => "semantic_state",
        }
    }
}

/// A typed edge between two [`WorldNode`]s (ADR-139 §2.1). Stored as the
/// petgraph edge weight; metadata is structurally per-relation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "rel", rename_all = "snake_case")]
pub enum WorldEdge {
    /// sensor/rf_link → observable node. Weight is field-of-regard quality.
    Observes {
        /// Field-of-regard quality in [0, 1].
        quality: f32,
        /// Last observation time (Unix ms).
        last_seen_unix_ms: i64,
    },
    /// person/object/event → room/zone containment.
    LocatedIn {
        /// Containment start (Unix ms).
        since_unix_ms: i64,
    },
    /// room ↔ room through a doorway (undirected pair stored as two edges).
    AdjacentTo {
        /// The connecting doorway node.
        via_doorway: WorldId,
    },
    /// sensor/rf_link → sensor/rf_link physical/clock support (ADR-138).
    Supports {
        /// Support strength in [0, 1].
        strength: f32,
    },
    /// evidence/state → evidence/state: sources disagree (ADR-137).
    Contradicts {
        /// Disagreement magnitude.
        magnitude: f32,
        /// ADR-137 contradiction-flag content-address handle.
        flag: String,
    },
    /// semantic_state → prior state/evidence provenance chain (ADR-137).
    DerivedFrom {
        /// ADR-137 evidence content-address handle.
        evidence: String,
    },
    /// sensor → node: observation constrained by a privacy mode (ADR-141).
    PrivacyLimitedBy {
        /// Limiting privacy mode name.
        mode: String,
        /// Action evaluated.
        action: String,
        /// Whether observation is allowed under the current mode.
        allowed: bool,
    },
}

impl WorldEdge {
    /// Static relation tag.
    #[must_use]
    pub fn rel(&self) -> &'static str {
        match self {
            Self::Observes { .. } => "observes",
            Self::LocatedIn { .. } => "located_in",
            Self::AdjacentTo { .. } => "adjacent_to",
            Self::Supports { .. } => "supports",
            Self::Contradicts { .. } => "contradicts",
            Self::DerivedFrom { .. } => "derived_from",
            Self::PrivacyLimitedBy { .. } => "privacy_limited_by",
        }
    }
}
