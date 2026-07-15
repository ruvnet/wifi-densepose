//! Disaster event aggregate root.

use chrono::{DateTime, Utc};
use geo::Point;
use uuid::Uuid;

use super::{Coordinates3D, ScanZone, ScanZoneId, Survivor, SurvivorId, VitalSignsReading};
use crate::MatError;

/// Unique identifier for a disaster event
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DisasterEventId(Uuid);

impl DisasterEventId {
    /// Create a new random event ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for DisasterEventId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DisasterEventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Types of disaster events
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DisasterType {
    /// Building collapse (explosion, structural failure)
    BuildingCollapse,
    /// Earthquake
    Earthquake,
    /// Landslide or mudslide
    Landslide,
    /// Avalanche (snow)
    Avalanche,
    /// Flood
    Flood,
    /// Mine collapse
    MineCollapse,
    /// Industrial accident
    Industrial,
    /// Tunnel collapse
    TunnelCollapse,
    /// Unknown or other
    Unknown,
}

impl DisasterType {
    /// Get typical debris profile for this disaster type
    pub fn typical_debris_profile(&self) -> super::DebrisProfile {
        use super::{DebrisMaterial, DebrisProfile, MetalContent, MoistureLevel};

        match self {
            DisasterType::BuildingCollapse => DebrisProfile {
                primary_material: DebrisMaterial::Mixed,
                void_fraction: 0.25,
                moisture_content: MoistureLevel::Dry,
                metal_content: MetalContent::Moderate,
            },
            DisasterType::Earthquake => DebrisProfile {
                primary_material: DebrisMaterial::HeavyConcrete,
                void_fraction: 0.2,
                moisture_content: MoistureLevel::Dry,
                metal_content: MetalContent::Moderate,
            },
            DisasterType::Avalanche => DebrisProfile {
                primary_material: DebrisMaterial::Snow,
                void_fraction: 0.4,
                moisture_content: MoistureLevel::Wet,
                metal_content: MetalContent::None,
            },
            DisasterType::Landslide => DebrisProfile {
                primary_material: DebrisMaterial::Soil,
                void_fraction: 0.15,
                moisture_content: MoistureLevel::Wet,
                metal_content: MetalContent::None,
            },
            DisasterType::Flood => DebrisProfile {
                primary_material: DebrisMaterial::Mixed,
                void_fraction: 0.3,
                moisture_content: MoistureLevel::Saturated,
                metal_content: MetalContent::Low,
            },
            DisasterType::MineCollapse | DisasterType::TunnelCollapse => DebrisProfile {
                primary_material: DebrisMaterial::Soil,
                void_fraction: 0.2,
                moisture_content: MoistureLevel::Damp,
                metal_content: MetalContent::Low,
            },
            DisasterType::Industrial => DebrisProfile {
                primary_material: DebrisMaterial::Metal,
                void_fraction: 0.35,
                moisture_content: MoistureLevel::Dry,
                metal_content: MetalContent::High,
            },
            DisasterType::Unknown => DebrisProfile::default(),
        }
    }

    /// Get expected maximum survival time (hours)
    pub fn expected_survival_hours(&self) -> u32 {
        match self {
            DisasterType::Avalanche => 2,     // Limited air, hypothermia
            DisasterType::Flood => 6,         // Drowning risk
            DisasterType::MineCollapse => 72, // Air supply critical
            DisasterType::BuildingCollapse => 96,
            DisasterType::Earthquake => 120,
            DisasterType::Landslide => 48,
            DisasterType::TunnelCollapse => 72,
            DisasterType::Industrial => 72,
            DisasterType::Unknown => 72,
        }
    }
}

impl Default for DisasterType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Current status of the disaster event
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EventStatus {
    /// Event just reported, setting up
    Initializing,
    /// Active search and rescue
    Active,
    /// Search suspended (weather, safety)
    Suspended,
    /// Primary rescue complete, secondary search
    SecondarySearch,
    /// Event closed
    Closed,
}

/// Aggregate root for a disaster event
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DisasterEvent {
    id: DisasterEventId,
    event_type: DisasterType,
    start_time: DateTime<Utc>,
    location: Point<f64>,
    description: String,
    scan_zones: Vec<ScanZone>,
    survivors: Vec<Survivor>,
    status: EventStatus,
    metadata: EventMetadata,
}

/// Additional metadata for a disaster event
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EventMetadata {
    /// Estimated number of people in area at time of disaster
    pub estimated_occupancy: Option<u32>,
    /// Known survivors (already rescued)
    pub confirmed_rescued: u32,
    /// Known fatalities
    pub confirmed_deceased: u32,
    /// Weather conditions
    pub weather: Option<String>,
    /// Lead agency
    pub lead_agency: Option<String>,
    /// Notes
    pub notes: Vec<String>,
}

impl DisasterEvent {
    /// Create a new disaster event
    pub fn new(event_type: DisasterType, location: Point<f64>, description: &str) -> Self {
        Self {
            id: DisasterEventId::new(),
            event_type,
            start_time: Utc::now(),
            location,
            description: description.to_string(),
            scan_zones: Vec::new(),
            survivors: Vec::new(),
            status: EventStatus::Initializing,
            metadata: EventMetadata::default(),
        }
    }

    /// Get the event ID
    pub fn id(&self) -> &DisasterEventId {
        &self.id
    }

    /// Get the event type
    pub fn event_type(&self) -> &DisasterType {
        &self.event_type
    }

    /// Get the start time
    pub fn start_time(&self) -> &DateTime<Utc> {
        &self.start_time
    }

    /// Get the location
    pub fn location(&self) -> &Point<f64> {
        &self.location
    }

    /// Get the description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the scan zones
    pub fn zones(&self) -> &[ScanZone] {
        &self.scan_zones
    }

    /// Get mutable scan zones
    pub fn zones_mut(&mut self) -> &mut [ScanZone] {
        &mut self.scan_zones
    }

    /// Get the survivors
    pub fn survivors(&self) -> Vec<&Survivor> {
        self.survivors.iter().collect()
    }

    /// Get mutable survivors
    pub fn survivors_mut(&mut self) -> &mut [Survivor] {
        &mut self.survivors
    }

    /// Get the current status
    pub fn status(&self) -> &EventStatus {
        &self.status
    }

    /// Get metadata
    pub fn metadata(&self) -> &EventMetadata {
        &self.metadata
    }

    /// Get mutable metadata
    pub fn metadata_mut(&mut self) -> &mut EventMetadata {
        &mut self.metadata
    }

    /// Add a scan zone
    pub fn add_zone(&mut self, zone: ScanZone) {
        self.scan_zones.push(zone);

        // Activate event if first zone
        if self.status == EventStatus::Initializing {
            self.status = EventStatus::Active;
        }
    }

    /// Remove a scan zone
    pub fn remove_zone(&mut self, zone_id: &ScanZoneId) {
        self.scan_zones.retain(|z| z.id() != zone_id);
    }

    /// Record a new detection.
    ///
    /// Deduplication is two-tiered so that the same trapped person re-detected
    /// across successive scan cycles is updated in place rather than counted as a
    /// new survivor (which would fabricate a mass-casualty event):
    ///
    /// 1. **Spatial** — if the detection has a real `location`, match an existing
    ///    survivor within `LOCATION_DEDUP_RADIUS_M`.
    /// 2. **Zone + vitals-signature** — if there is NO usable location (no
    ///    multi-node geometry / RSSI available, which is the common edge case
    ///    for a single-node deployment), match an existing *active* survivor in
    ///    the SAME zone whose most recent vital-sign signature is compatible
    ///    (same breathing presence and rate band, same heartbeat presence, same
    ///    movement class). Without this, every scan cycle would push a brand new
    ///    survivor for the one person actually present.
    ///
    /// This is conservative on the safety side: two genuinely distinct survivors
    /// in the same zone with materially different vitals (e.g. different
    /// breathing-rate bands, or one with a pulse and one without) are kept
    /// separate; only readings that are plausibly the same person collapse.
    pub fn record_detection(
        &mut self,
        zone_id: ScanZoneId,
        vitals: VitalSignsReading,
        location: Option<Coordinates3D>,
    ) -> Result<&Survivor, MatError> {
        // Tier 1: spatial dedup when a real location is available.
        let existing_id = if let Some(loc) = &location {
            self.find_nearby_survivor(loc, Self::LOCATION_DEDUP_RADIUS_M)
                .cloned()
        } else {
            // Tier 2: zone + vitals-signature dedup when location is unavailable.
            self.find_matching_survivor_by_signature(&zone_id, &vitals)
                .cloned()
        };

        if let Some(existing) = existing_id {
            // Update existing survivor
            let survivor = self
                .survivors
                .iter_mut()
                .find(|s| s.id() == &existing)
                .ok_or_else(|| MatError::Domain("Survivor not found".into()))?;
            survivor.update_vitals(vitals);
            if let Some(l) = location {
                survivor.update_location(l);
            }
            return Ok(survivor);
        }

        // Create new survivor
        let survivor = Survivor::new(zone_id, vitals, location);
        self.survivors.push(survivor);
        // Safe: we just pushed, so last() is always Some
        Ok(self
            .survivors
            .last()
            .expect("survivors is non-empty after push"))
    }

    /// Radius (metres) within which a located detection is treated as the same
    /// survivor for spatial deduplication.
    const LOCATION_DEDUP_RADIUS_M: f64 = 2.0;

    /// Find a survivor near a location
    fn find_nearby_survivor(&self, location: &Coordinates3D, radius: f64) -> Option<&SurvivorId> {
        for survivor in &self.survivors {
            if let Some(loc) = survivor.location() {
                if loc.distance_to(location) < radius {
                    return Some(survivor.id());
                }
            }
        }
        None
    }

    /// Find an existing *active*, *un-located* survivor in the same zone whose
    /// most-recent vital signature is compatible with `vitals`.
    ///
    /// Only survivors without a fixed location participate: a survivor that has
    /// a known position is handled by spatial dedup, and collapsing a located
    /// survivor into an un-located reading would lose information. Returns the
    /// first compatible match (there is normally at most one un-located survivor
    /// per zone precisely because this dedup keeps it from multiplying).
    fn find_matching_survivor_by_signature(
        &self,
        zone_id: &ScanZoneId,
        vitals: &VitalSignsReading,
    ) -> Option<&SurvivorId> {
        for survivor in &self.survivors {
            if survivor.zone_id() != zone_id {
                continue;
            }
            if survivor.location().is_some() {
                continue;
            }
            if !matches!(
                survivor.status(),
                super::survivor::SurvivorStatus::Active | super::survivor::SurvivorStatus::Lost
            ) {
                continue;
            }
            if let Some(latest) = survivor.vital_signs().latest() {
                if Self::vitals_signature_matches(latest, vitals) {
                    return Some(survivor.id());
                }
            }
        }
        None
    }

    /// Decide whether two vital-sign readings are plausibly the same person.
    ///
    /// Matches on coarse, detection-stable features rather than exact values
    /// (CSI-derived rates jitter cycle-to-cycle): breathing presence + rate band,
    /// heartbeat presence, and movement class. Breathing rate is bucketed into
    /// START-relevant bands (<10, 10–30, >30 bpm) with a small tolerance so a
    /// breath rate hovering near a band edge does not split one person in two.
    fn vitals_signature_matches(a: &VitalSignsReading, b: &VitalSignsReading) -> bool {
        // Breathing presence must agree.
        if a.breathing.is_some() != b.breathing.is_some() {
            return false;
        }
        if let (Some(ba), Some(bb)) = (&a.breathing, &b.breathing) {
            // Same START rate band, with a 1.5 bpm tolerance at band edges.
            const EDGE_TOL: f32 = 1.5;
            let band = |r: f32| -> i8 {
                if r < 10.0 - EDGE_TOL {
                    0
                } else if r > 30.0 + EDGE_TOL {
                    2
                } else {
                    1
                }
            };
            if band(ba.rate_bpm) != band(bb.rate_bpm) {
                return false;
            }
        }

        // Heartbeat presence must agree.
        if a.heartbeat.is_some() != b.heartbeat.is_some() {
            return false;
        }

        // Movement class must agree.
        a.movement.movement_type == b.movement.movement_type
    }

    /// Get survivor by ID
    pub fn get_survivor(&self, id: &SurvivorId) -> Option<&Survivor> {
        self.survivors.iter().find(|s| s.id() == id)
    }

    /// Get mutable survivor by ID
    pub fn get_survivor_mut(&mut self, id: &SurvivorId) -> Option<&mut Survivor> {
        self.survivors.iter_mut().find(|s| s.id() == id)
    }

    /// Get zone by ID
    pub fn get_zone(&self, id: &ScanZoneId) -> Option<&ScanZone> {
        self.scan_zones.iter().find(|z| z.id() == id)
    }

    /// Set event status
    pub fn set_status(&mut self, status: EventStatus) {
        self.status = status;
    }

    /// Suspend operations
    pub fn suspend(&mut self, reason: &str) {
        self.status = EventStatus::Suspended;
        self.metadata.notes.push(format!(
            "[{}] Suspended: {}",
            Utc::now().format("%Y-%m-%d %H:%M:%S"),
            reason
        ));
    }

    /// Resume operations
    pub fn resume(&mut self) {
        if self.status == EventStatus::Suspended {
            self.status = EventStatus::Active;
            self.metadata.notes.push(format!(
                "[{}] Resumed operations",
                Utc::now().format("%Y-%m-%d %H:%M:%S")
            ));
        }
    }

    /// Close the event
    pub fn close(&mut self) {
        self.status = EventStatus::Closed;
    }

    /// Get time since event started
    pub fn elapsed_time(&self) -> chrono::Duration {
        Utc::now() - self.start_time
    }

    /// Get count of survivors by triage status
    pub fn triage_counts(&self) -> TriageCounts {
        use super::TriageStatus;

        let mut counts = TriageCounts::default();
        for survivor in &self.survivors {
            match survivor.triage_status() {
                TriageStatus::Immediate => counts.immediate += 1,
                TriageStatus::Delayed => counts.delayed += 1,
                TriageStatus::Minor => counts.minor += 1,
                TriageStatus::Deceased => counts.deceased += 1,
                TriageStatus::Unknown => counts.unknown += 1,
            }
        }
        counts
    }
}

/// Triage status counts
#[derive(Debug, Clone, Default)]
pub struct TriageCounts {
    /// Immediate (Red)
    pub immediate: u32,
    /// Delayed (Yellow)
    pub delayed: u32,
    /// Minor (Green)
    pub minor: u32,
    /// Deceased (Black)
    pub deceased: u32,
    /// Unknown
    pub unknown: u32,
}

impl TriageCounts {
    /// Total count
    pub fn total(&self) -> u32 {
        self.immediate + self.delayed + self.minor + self.deceased + self.unknown
    }

    /// Count of living survivors
    pub fn living(&self) -> u32 {
        self.immediate + self.delayed + self.minor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BreathingPattern, BreathingType, ConfidenceScore, ZoneBounds};

    fn create_test_vitals() -> VitalSignsReading {
        VitalSignsReading {
            breathing: Some(BreathingPattern {
                rate_bpm: 16.0,
                amplitude: 0.8,
                regularity: 0.9,
                pattern_type: BreathingType::Normal,
            }),
            heartbeat: None,
            movement: Default::default(),
            timestamp: Utc::now(),
            confidence: ConfidenceScore::new(0.8),
        }
    }

    #[test]
    fn test_event_creation() {
        let event = DisasterEvent::new(
            DisasterType::Earthquake,
            Point::new(-122.4194, 37.7749),
            "Test earthquake event",
        );

        assert!(matches!(event.event_type(), DisasterType::Earthquake));
        assert_eq!(event.status(), &EventStatus::Initializing);
    }

    #[test]
    fn test_add_zone_activates_event() {
        let mut event =
            DisasterEvent::new(DisasterType::BuildingCollapse, Point::new(0.0, 0.0), "Test");

        assert_eq!(event.status(), &EventStatus::Initializing);

        let zone = ScanZone::new("Zone A", ZoneBounds::rectangle(0.0, 0.0, 10.0, 10.0));
        event.add_zone(zone);

        assert_eq!(event.status(), &EventStatus::Active);
    }

    #[test]
    fn test_record_detection() {
        let mut event = DisasterEvent::new(DisasterType::Earthquake, Point::new(0.0, 0.0), "Test");

        let zone = ScanZone::new("Zone A", ZoneBounds::rectangle(0.0, 0.0, 10.0, 10.0));
        let zone_id = zone.id().clone();
        event.add_zone(zone);

        let vitals = create_test_vitals();
        event.record_detection(zone_id, vitals, None).unwrap();

        assert_eq!(event.survivors().len(), 1);
    }

    #[test]
    fn test_disaster_type_survival_hours() {
        assert!(
            DisasterType::Avalanche.expected_survival_hours()
                < DisasterType::Earthquake.expected_survival_hours()
        );
    }

    /// Count-inflation regression (FAILS on the old code, which returned 3).
    ///
    /// Three detections of the SAME person (identical vitals, no usable location
    /// because no multi-node geometry is available) must collapse to a single
    /// survivor. Previously, `record_detection` only deduplicated when a location
    /// was present, so an un-located trapped person re-detected every scan cycle
    /// produced N survivors — a fabricated mass-casualty count.
    #[test]
    fn test_identical_vitals_no_location_dedup_to_one() {
        let mut event = DisasterEvent::new(DisasterType::Earthquake, Point::new(0.0, 0.0), "Test");
        let zone = ScanZone::new("Zone A", ZoneBounds::rectangle(0.0, 0.0, 10.0, 10.0));
        let zone_id = zone.id().clone();
        event.add_zone(zone);

        for _ in 0..3 {
            event
                .record_detection(zone_id.clone(), create_test_vitals(), None)
                .unwrap();
        }

        assert_eq!(
            event.survivors().len(),
            1,
            "same un-located person detected 3x must be ONE survivor, not three"
        );
    }

    /// Counterpart: two genuinely DIFFERENT survivors in the same zone (different
    /// breathing-rate bands) must remain separate — dedup must not under-count.
    #[test]
    fn test_distinct_vitals_no_location_stay_separate() {
        let mut event = DisasterEvent::new(DisasterType::Earthquake, Point::new(0.0, 0.0), "Test");
        let zone = ScanZone::new("Zone A", ZoneBounds::rectangle(0.0, 0.0, 10.0, 10.0));
        let zone_id = zone.id().clone();
        event.add_zone(zone);

        // Person 1: normal breathing (16 bpm band 1).
        event
            .record_detection(zone_id.clone(), create_test_vitals(), None)
            .unwrap();

        // Person 2: tachypneic breathing (38 bpm band 2) — distinct survivor.
        let fast = VitalSignsReading {
            breathing: Some(BreathingPattern {
                rate_bpm: 38.0,
                amplitude: 0.8,
                regularity: 0.5,
                pattern_type: BreathingType::Labored,
            }),
            heartbeat: None,
            movement: Default::default(),
            timestamp: Utc::now(),
            confidence: ConfidenceScore::new(0.8),
        };
        event.record_detection(zone_id, fast, None).unwrap();

        assert_eq!(event.survivors().len(), 2);
    }
}
