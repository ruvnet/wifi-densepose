//! Conversion from tracked COCO-17 skeletons to semantic joint observations.

use crate::error::{Error, Result};
use crate::model::{ActorObservation, JointTarget, SceneFrame, Vec3};

const NUM_KEYPOINTS: usize = 17;
const LEFT_SHOULDER: usize = 5;
const RIGHT_SHOULDER: usize = 6;
const LEFT_ELBOW: usize = 7;
const RIGHT_ELBOW: usize = 8;
const LEFT_WRIST: usize = 9;
const RIGHT_WRIST: usize = 10;
const LEFT_HIP: usize = 11;
const RIGHT_HIP: usize = 12;
const LEFT_KNEE: usize = 13;
const RIGHT_KNEE: usize = 14;
const LEFT_ANKLE: usize = 15;
const RIGHT_ANKLE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tentative,
    Active,
    Lost,
    Terminated,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KeypointObservation {
    pub position: Vec3,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackObservation {
    pub id: u64,
    pub state: TrackState,
    pub keypoints: [KeypointObservation; NUM_KEYPOINTS],
    pub time_since_update: u64,
}

#[cfg(feature = "ruview")]
impl From<&wifi_densepose_signal::ruvsense::PoseTrack> for TrackObservation {
    fn from(track: &wifi_densepose_signal::ruvsense::PoseTrack) -> Self {
        use wifi_densepose_signal::ruvsense::TrackLifecycleState;
        Self {
            id: track.id.0,
            state: match track.lifecycle {
                TrackLifecycleState::Tentative => TrackState::Tentative,
                TrackLifecycleState::Active => TrackState::Active,
                TrackLifecycleState::Lost => TrackState::Lost,
                TrackLifecycleState::Terminated => TrackState::Terminated,
            },
            keypoints: std::array::from_fn(|i| {
                let point = track.keypoints[i].position();
                KeypointObservation {
                    position: Vec3::new(point[0], point[1], point[2]),
                    confidence: track.keypoints[i].confidence,
                }
            }),
            time_since_update: track.time_since_update,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Coordinate system up vector after room calibration.
    pub world_up: Vec3,
    /// Coordinate system forward vector after room calibration.
    pub world_forward: Vec3,
    /// Ignore tracks below this aggregate confidence.
    pub min_confidence: f32,
    /// Confidence used until an inference backend populates keypoint confidence.
    pub unscored_confidence: f32,
    /// Emit lost tracks as predicted observations.
    pub include_lost: bool,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            world_up: Vec3::new(0.0, 1.0, 0.0),
            world_forward: Vec3::new(0.0, 0.0, 1.0),
            min_confidence: 0.2,
            unscored_confidence: 0.5,
            include_lost: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuViewAdapter {
    config: AdapterConfig,
}

impl RuViewAdapter {
    pub fn new(config: AdapterConfig) -> Result<Self> {
        if !config.world_up.is_finite() || !config.world_forward.is_finite() {
            return Err(Error::Observation("coordinate axes must be finite".into()));
        }
        if !(0.0..=1.0).contains(&config.min_confidence)
            || !(0.0..=1.0).contains(&config.unscored_confidence)
        {
            return Err(Error::Observation("confidence must be in [0, 1]".into()));
        }
        if norm(config.world_up) < 1e-6 || norm(config.world_forward) < 1e-6 {
            return Err(Error::Observation(
                "coordinate axes must be non-zero".into(),
            ));
        }
        let alignment = dot(normalize(config.world_up), normalize(config.world_forward)).abs();
        if alignment > 0.98 {
            return Err(Error::Observation(
                "world up and forward axes must not be parallel".into(),
            ));
        }
        Ok(Self { config })
    }

    /// Convert all visible tracks into one synchronized observation frame.
    pub fn frame(&self, timestamp_ms: u64, tracks: &[TrackObservation]) -> SceneFrame {
        let actors = tracks
            .iter()
            .filter_map(|track| self.actor(track))
            .collect();
        SceneFrame {
            timestamp_ms,
            actors,
        }
    }

    /// Direct bridge for the RuView tracker aggregate. The feature is optional
    /// so protocol-only consumers do not pull the signal and RuVector graph.
    #[cfg(feature = "ruview")]
    pub fn frame_from_pose_tracks(
        &self,
        timestamp_ms: u64,
        tracks: &[&wifi_densepose_signal::ruvsense::PoseTrack],
    ) -> SceneFrame {
        let observations: Vec<TrackObservation> =
            tracks.iter().map(|track| (*track).into()).collect();
        self.frame(timestamp_ms, &observations)
    }

    pub fn actor(&self, track: &TrackObservation) -> Option<ActorObservation> {
        if track.state == TrackState::Terminated
            || (!self.config.include_lost && track.state == TrackState::Lost)
        {
            return None;
        }

        let confidence = track_confidence(track, self.config.unscored_confidence);
        if confidence < self.config.min_confidence {
            return None;
        }

        let p = |index: usize| track.keypoints[index].position;
        if track.keypoints.iter().any(|kp| !kp.position.is_finite()) {
            return None;
        }

        let left_hip = p(LEFT_HIP);
        let right_hip = p(RIGHT_HIP);
        let position = midpoint(left_hip, right_hip);
        let up = normalize(self.config.world_up);
        let forward = normalize(sub(
            self.config.world_forward,
            scale(up, dot(self.config.world_forward, up)),
        ));

        let mut joints = Vec::with_capacity(10);
        add_flexion(
            &mut joints,
            "elbow_left",
            p(LEFT_SHOULDER),
            p(LEFT_ELBOW),
            p(LEFT_WRIST),
            joint_confidence(track, &[LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], confidence),
        );
        add_flexion(
            &mut joints,
            "elbow_right",
            p(RIGHT_SHOULDER),
            p(RIGHT_ELBOW),
            p(RIGHT_WRIST),
            joint_confidence(
                track,
                &[RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
                confidence,
            ),
        );
        add_flexion(
            &mut joints,
            "knee_left",
            left_hip,
            p(LEFT_KNEE),
            p(LEFT_ANKLE),
            joint_confidence(track, &[LEFT_HIP, LEFT_KNEE, LEFT_ANKLE], confidence),
        );
        add_flexion(
            &mut joints,
            "knee_right",
            right_hip,
            p(RIGHT_KNEE),
            p(RIGHT_ANKLE),
            joint_confidence(track, &[RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE], confidence),
        );

        add_signed_flex(
            &mut joints,
            "hip_left",
            sub(p(LEFT_KNEE), left_hip),
            up,
            forward,
            joint_confidence(track, &[LEFT_HIP, LEFT_KNEE], confidence),
        );
        add_signed_flex(
            &mut joints,
            "hip_right",
            sub(p(RIGHT_KNEE), right_hip),
            up,
            forward,
            joint_confidence(track, &[RIGHT_HIP, RIGHT_KNEE], confidence),
        );
        add_signed_flex(
            &mut joints,
            "shoulder_left",
            sub(p(LEFT_ELBOW), p(LEFT_SHOULDER)),
            up,
            forward,
            joint_confidence(track, &[LEFT_SHOULDER, LEFT_ELBOW], confidence),
        );
        add_signed_flex(
            &mut joints,
            "shoulder_right",
            sub(p(RIGHT_ELBOW), p(RIGHT_SHOULDER)),
            up,
            forward,
            joint_confidence(track, &[RIGHT_SHOULDER, RIGHT_ELBOW], confidence),
        );

        let shoulder_mid = midpoint(p(LEFT_SHOULDER), p(RIGHT_SHOULDER));
        let torso = sub(shoulder_mid, position);
        let hinge = angle_deg(torso, up).clamp(0.0, 180.0);
        joints.push(JointTarget {
            joint: "pelvis".into(),
            action: "hinge".into(),
            degrees: hinge,
            confidence: joint_confidence(
                track,
                &[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
                confidence,
            ),
        });

        Some(ActorObservation {
            actor_id: format!("person_{}", track.id),
            track_id: track.id,
            confidence,
            position,
            joints,
        })
    }
}

fn track_confidence(track: &TrackObservation, fallback: f32) -> f32 {
    let scored: Vec<f32> = track
        .keypoints
        .iter()
        .map(|kp| kp.confidence)
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    let measured = if scored.is_empty() {
        fallback
    } else {
        scored.iter().sum::<f32>() / scored.len() as f32
    };
    let freshness = 1.0 / (1.0 + track.time_since_update as f32 * 0.2);
    (measured * freshness).clamp(0.0, 1.0)
}

fn joint_confidence(track: &TrackObservation, indices: &[usize], fallback: f32) -> f32 {
    indices.iter().fold(1.0_f32, |confidence, &index| {
        let measured = track.keypoints[index].confidence;
        confidence.min(if measured.is_finite() && measured > 0.0 {
            measured
        } else {
            fallback
        })
    })
}

fn add_flexion(out: &mut Vec<JointTarget>, name: &str, a: Vec3, b: Vec3, c: Vec3, confidence: f32) {
    let inner = angle_deg(sub(a, b), sub(c, b));
    if inner.is_finite() {
        out.push(JointTarget {
            joint: name.into(),
            action: "flex".into(),
            degrees: (180.0 - inner).clamp(0.0, 180.0),
            confidence,
        });
    }
}

fn add_signed_flex(
    out: &mut Vec<JointTarget>,
    name: &str,
    limb: Vec3,
    up: Vec3,
    forward: Vec3,
    confidence: f32,
) {
    let down = scale(up, -1.0);
    let degrees = dot(limb, forward).atan2(dot(limb, down)).to_degrees();
    if degrees.is_finite() {
        out.push(JointTarget {
            joint: name.into(),
            action: if degrees >= 0.0 { "flex" } else { "extend" }.into(),
            degrees: degrees.abs().clamp(0.0, 180.0),
            confidence,
        });
    }
}

fn midpoint(a: Vec3, b: Vec3) -> Vec3 {
    scale(add(a, b), 0.5)
}
fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}
fn scale(v: Vec3, s: f32) -> Vec3 {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}
fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
fn norm(v: Vec3) -> f32 {
    dot(v, v).sqrt()
}
fn normalize(v: Vec3) -> Vec3 {
    scale(v, 1.0 / norm(v).max(1e-9))
}
fn angle_deg(a: Vec3, b: Vec3) -> f32 {
    let denom = norm(a) * norm(b);
    if denom < 1e-9 {
        return 0.0;
    }
    (dot(a, b) / denom).clamp(-1.0, 1.0).acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standing_track() -> TrackObservation {
        let mut keypoints = [KeypointObservation::default(); NUM_KEYPOINTS];
        let mut set = |index, x, y, z| {
            keypoints[index] = KeypointObservation {
                position: Vec3::new(x, y, z),
                confidence: 0.9,
            };
        };
        set(LEFT_SHOULDER, -0.2, 1.5, 0.0);
        set(RIGHT_SHOULDER, 0.2, 1.5, 0.0);
        set(LEFT_ELBOW, -0.25, 1.1, 0.0);
        set(RIGHT_ELBOW, 0.25, 1.1, 0.0);
        set(LEFT_WRIST, -0.25, 0.75, 0.0);
        set(RIGHT_WRIST, 0.25, 0.75, 0.0);
        set(LEFT_HIP, -0.12, 0.9, 0.0);
        set(RIGHT_HIP, 0.12, 0.9, 0.0);
        set(LEFT_KNEE, -0.12, 0.5, 0.0);
        set(RIGHT_KNEE, 0.12, 0.5, 0.0);
        set(LEFT_ANKLE, -0.12, 0.05, 0.0);
        set(RIGHT_ANKLE, 0.12, 0.05, 0.0);
        TrackObservation {
            id: 7,
            state: TrackState::Active,
            keypoints,
            time_since_update: 0,
        }
    }

    #[test]
    fn converts_track_to_named_actor() {
        let adapter = RuViewAdapter::new(AdapterConfig::default()).unwrap();
        let actor = adapter.actor(&standing_track()).unwrap();
        assert_eq!(actor.actor_id, "person_7");
        assert_eq!(actor.track_id, 7);
        assert!(actor.joints.len() >= 9);
        assert!(actor.joints.iter().all(|j| j.degrees.is_finite()));
    }

    #[test]
    fn rejects_invalid_axes() {
        let config = AdapterConfig {
            world_up: Vec3::default(),
            ..Default::default()
        };
        assert!(RuViewAdapter::new(config).is_err());
    }

    #[test]
    fn rejects_parallel_coordinate_axes() {
        let config = AdapterConfig {
            world_forward: Vec3::new(0.0, 2.0, 0.0),
            ..Default::default()
        };
        assert!(RuViewAdapter::new(config).is_err());
    }

    #[test]
    fn filters_stale_low_confidence_tracks() {
        let mut track = standing_track();
        track.time_since_update = 100;
        let adapter = RuViewAdapter::new(AdapterConfig::default()).unwrap();
        assert!(adapter.actor(&track).is_none());
    }

    #[cfg(feature = "ruview")]
    #[test]
    fn converts_native_ruview_pose_track() {
        use wifi_densepose_signal::ruvsense::{PoseTrack, TrackId, TrackLifecycleState};

        let source = standing_track();
        let positions = std::array::from_fn(|i| {
            let p = source.keypoints[i].position;
            [p.x, p.y, p.z]
        });
        let mut native = PoseTrack::new(TrackId(23), &positions, 0, 128);
        native.lifecycle = TrackLifecycleState::Active;
        let observed = TrackObservation::from(&native);
        assert_eq!(observed.id, 23);
        assert_eq!(observed.state, TrackState::Active);
        assert_eq!(
            observed.keypoints[LEFT_HIP].position,
            source.keypoints[LEFT_HIP].position
        );
    }
}
