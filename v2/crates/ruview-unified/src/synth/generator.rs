//! Domain-randomized synthetic dataset generator (ADR-276 §4).
//!
//! Randomizes *physics parameters*, not textures: room geometry, wall
//! permittivity/conductivity, antenna placement, person kinematics and RCS,
//! plus the hardware nuisances that break naive models in the field —
//! chipset gain and phase offsets, carrier-frequency-offset drift, phase
//! noise, packet loss (snapshot duplication), and interference bursts.
//! Every window carries a full [`PartitionKey`] so the ADR-273 strict
//! anti-leakage splits (held-out rooms / days / people / chipsets /
//! firmware / layouts) are possible by construction.

use ndarray::Array3;
use num_complex::Complex64;
use rand::Rng;

use crate::eval::PartitionKey;
use crate::math::seeded_rng;
use crate::synth::raytrace::synthesize_csi;
use crate::synth::room::{Material, PersonSpec, RoomSpec};
use crate::tensor::{
    CalibrationMeta, LinkGeometry, RfModality, RfTensor, CANONICAL_BINS, CANONICAL_SNAPSHOTS,
};

/// Generator configuration.
#[derive(Debug, Clone, Copy)]
pub struct SynthConfig {
    /// Master seed (same seed ⇒ byte-identical corpus).
    pub seed: u64,
    /// Number of distinct rooms.
    pub n_rooms: usize,
    /// Windows per room (half with a person, half empty, interleaved).
    pub windows_per_room: usize,
    /// Links (TX→RX pairs) per room.
    pub links: usize,
    /// Snapshot spacing in seconds.
    pub snapshot_dt_s: f64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self { seed: 0xC0FFEE, n_rooms: 8, windows_per_room: 24, links: 3, snapshot_dt_s: 0.05 }
    }
}

/// One labeled synthetic window.
#[derive(Debug, Clone)]
pub struct LabeledWindow {
    /// Canonical tensor (modality [`RfModality::Synthetic`]).
    pub tensor: RfTensor,
    /// Whether a person is present in the room during this window.
    pub presence: bool,
    /// Person position at the window's mid-time, when present.
    pub person_pos: Option<[f64; 3]>,
    /// Full provenance key for strict splits.
    pub key: PartitionKey,
}

/// Per-room randomized nuisance profile (the "chipset").
#[derive(Debug, Clone)]
struct HardwareProfile {
    chipset: String,
    firmware: String,
    layout: String,
    gain: f64,
    phase_offset: f64,
    cfo_rad_per_snap: f64,
    noise_sigma: f64,
}

/// The generator.
pub struct SynthGenerator {
    cfg: SynthConfig,
}

impl SynthGenerator {
    /// New generator.
    #[must_use]
    pub fn new(cfg: SynthConfig) -> Self {
        Self { cfg }
    }

    /// 56 subcarrier frequencies over 20 MHz around 2.437 GHz.
    #[must_use]
    pub fn subcarrier_freqs() -> Vec<f64> {
        (0..CANONICAL_BINS)
            .map(|k| 2.437e9 - 10e6 + 20e6 * k as f64 / (CANONICAL_BINS - 1) as f64)
            .collect()
    }

    /// Generates the full labeled corpus, deterministically from the seed.
    ///
    /// # Panics
    /// Only on internal invariant violation (tensor construction from
    /// generated finite values cannot fail).
    #[must_use]
    pub fn generate(&self) -> Vec<LabeledWindow> {
        let mut rng = seeded_rng(self.cfg.seed);
        let freqs = Self::subcarrier_freqs();
        let mut out = Vec::with_capacity(self.cfg.n_rooms * self.cfg.windows_per_room);

        for room_idx in 0..self.cfg.n_rooms {
            // --- Randomized physics for this room ---
            let size = [
                rng.gen_range(4.0..10.0),
                rng.gen_range(3.0..8.0),
                rng.gen_range(2.4..3.2),
            ];
            let material = Material {
                rel_permittivity: rng.gen_range(2.0..7.0),
                conductivity_s_m: rng.gen_range(0.002..0.1),
            };
            let links: Vec<LinkGeometry> = (0..self.cfg.links)
                .map(|_| LinkGeometry {
                    tx_pos: [
                        rng.gen_range(0.3..size[0] - 0.3),
                        rng.gen_range(0.3..size[1] - 0.3),
                        rng.gen_range(1.0..2.0),
                    ],
                    rx_pos: [
                        rng.gen_range(0.3..size[0] - 0.3),
                        rng.gen_range(0.3..size[1] - 0.3),
                        rng.gen_range(1.0..2.0),
                    ],
                })
                .collect();
            let hw = HardwareProfile {
                chipset: format!("chip-{}", room_idx % 3),
                firmware: format!("fw-{}", room_idx % 2),
                layout: format!("layout-{}", (room_idx / 2) % 2),
                gain: rng.gen_range(0.5..2.0),
                phase_offset: rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI),
                cfo_rad_per_snap: rng.gen_range(-0.3..0.3),
                noise_sigma: rng.gen_range(0.01..0.05),
            };
            let person_id = format!("p{}", room_idx % 4);
            // Person kinematics randomized per room; the person walks a
            // straight segment that stays inside the room for the corpus
            // duration (velocity kept small relative to room size).
            let person = PersonSpec {
                start: [
                    rng.gen_range(size[0] * 0.25..size[0] * 0.75),
                    rng.gen_range(size[1] * 0.25..size[1] * 0.75),
                    rng.gen_range(1.0..1.6),
                ],
                velocity: {
                    let speed = rng.gen_range(0.3..1.0);
                    let ang: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
                    [speed * ang.cos() * 0.2, speed * ang.sin() * 0.2, 0.0]
                },
                rcs_m2: rng.gen_range(0.3..0.8),
            };

            let occupied = RoomSpec::new(size, material, vec![person]).expect("generated in range");
            let empty = RoomSpec::new(size, material, vec![]).expect("generated in range");

            for w in 0..self.cfg.windows_per_room {
                let presence = w % 2 == 0;
                let room = if presence { &occupied } else { &empty };
                // Window start times cycle so the person oscillates within
                // the room instead of walking out of it.
                let t0 = (w % 6) as f64 * CANONICAL_SNAPSHOTS as f64 * self.cfg.snapshot_dt_s;

                let mut data =
                    Array3::zeros((self.cfg.links, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
                for (l, link) in links.iter().enumerate() {
                    let mut prev: Option<Vec<Complex64>> = None;
                    for s in 0..CANONICAL_SNAPSHOTS {
                        let t = t0 + s as f64 * self.cfg.snapshot_dt_s;
                        // Packet loss: 5 % of snapshots re-deliver the
                        // previous frame instead of a fresh capture.
                        let lost = prev.is_some() && rng.gen_bool(0.05);
                        let h: Vec<Complex64> = if lost {
                            prev.clone().expect("guarded by prev.is_some()")
                        } else {
                            synthesize_csi(room, link.tx_pos, link.rx_pos, &freqs, t)
                        };
                        // Chipset gain + static phase + CFO drift.
                        let rot = Complex64::from_polar(
                            hw.gain,
                            hw.phase_offset + hw.cfo_rad_per_snap * s as f64,
                        );
                        // Interference burst: 3 % of snapshots take a strong
                        // wideband hit; otherwise thermal noise only.
                        let burst = if rng.gen_bool(0.03) { 10.0 } else { 1.0 };
                        for (b, hv) in h.iter().enumerate() {
                            let noise = Complex64::new(
                                rng.gen_range(-1.0..1.0) * hw.noise_sigma * burst * 1e-4,
                                rng.gen_range(-1.0..1.0) * hw.noise_sigma * burst * 1e-4,
                            );
                            data[[l, b, s]] = hv * rot + noise;
                        }
                        prev = Some(h);
                    }
                }

                let mid_t = t0 + 0.5 * CANONICAL_SNAPSHOTS as f64 * self.cfg.snapshot_dt_s;
                let tensor = RfTensor::new(
                    RfModality::Synthetic,
                    2.437e9,
                    20e6,
                    data,
                    links.clone(),
                    rng.gen_range(0.0..0.2),
                    (room_idx as u64) << 32 | w as u64,
                    format!("synth-{}", hw.chipset),
                    rng.gen_range(0.6..0.95),
                    (hw.noise_sigma / 0.05).clamp(0.0, 1.0) * 0.5,
                    CalibrationMeta::default(),
                )
                .expect("generated tensor is finite and in range");

                out.push(LabeledWindow {
                    tensor,
                    presence,
                    person_pos: presence.then(|| occupied.people[0].position_at(mid_t)),
                    key: PartitionKey {
                        room: format!("room-{room_idx}"),
                        day: format!("day-{}", w / (self.cfg.windows_per_room / 2).max(1)),
                        person: if presence { person_id.clone() } else { "none".into() },
                        chipset: hw.chipset.clone(),
                        firmware: hw.firmware.clone(),
                        layout: hw.layout.clone(),
                        // Windows sharing a start-time slot within a room
                        // form one capture session.
                        session: format!("room-{room_idx}-s{}", w % 6),
                    },
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(seed: u64) -> SynthConfig {
        SynthConfig { seed, n_rooms: 3, windows_per_room: 6, links: 2, snapshot_dt_s: 0.05 }
    }

    #[test]
    fn corpus_is_byte_deterministic_per_seed() {
        let a = SynthGenerator::new(small_cfg(7)).generate();
        let b = SynthGenerator::new(small_cfg(7)).generate();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.presence, y.presence);
            assert_eq!(x.key, y.key);
            for (u, v) in x.tensor.data.iter().zip(y.tensor.data.iter()) {
                assert!(u == v, "same seed must give identical complex samples");
            }
        }
        // And a different seed gives a different corpus.
        let c = SynthGenerator::new(small_cfg(8)).generate();
        assert!(a
            .iter()
            .zip(&c)
            .any(|(x, y)| x.tensor.data.iter().zip(y.tensor.data.iter()).any(|(u, v)| u != v)));
    }

    #[test]
    fn presence_windows_carry_more_temporal_energy() {
        let corpus = SynthGenerator::new(small_cfg(42)).generate();
        let temporal_energy = |w: &LabeledWindow| {
            // Mean per-bin variance across snapshots.
            let (links, bins, snaps) = w.tensor.dims();
            let mut acc = 0.0;
            for l in 0..links {
                for b in 0..bins {
                    let vals: Vec<f64> =
                        (0..snaps).map(|s| w.tensor.data[[l, b, s]].norm()).collect();
                    let m = vals.iter().sum::<f64>() / snaps as f64;
                    acc += vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / snaps as f64;
                }
            }
            acc / (links * bins) as f64
        };
        let present: Vec<f64> =
            corpus.iter().filter(|w| w.presence).map(temporal_energy).collect();
        let absent: Vec<f64> =
            corpus.iter().filter(|w| !w.presence).map(temporal_energy).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            mean(&present) > 5.0 * mean(&absent),
            "a moving person must dominate temporal variance: present {} vs absent {}",
            mean(&present),
            mean(&absent)
        );
    }

    #[test]
    fn labels_and_partition_keys_are_complete() {
        let corpus = SynthGenerator::new(small_cfg(1)).generate();
        assert_eq!(corpus.len(), 18);
        for w in &corpus {
            assert_eq!(w.tensor.modality, RfModality::Synthetic, "honest labeling");
            assert_eq!(w.presence, w.person_pos.is_some());
            assert!(!w.key.room.is_empty());
            assert!(!w.key.chipset.is_empty());
            if let Some(p) = w.person_pos {
                assert!(p.iter().all(|v| v.is_finite()));
            }
        }
        // Multiple rooms exist so strict room-holdout splits are possible.
        let rooms: std::collections::BTreeSet<&str> =
            corpus.iter().map(|w| w.key.room.as_str()).collect();
        assert_eq!(rooms.len(), 3);
    }
}
