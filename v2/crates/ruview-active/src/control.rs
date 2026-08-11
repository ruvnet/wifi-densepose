//! Controllable degrees of freedom of an RF measurement (ADR-306 §1).
//!
//! **SYNTHETIC / L0 model scaffold.** These types describe *what a controller
//! could ask hardware to configure*; constructing one drives **no** radio and
//! emits **no** RF. Every axis is a typed enum / validated range so a malformed
//! configuration is rejected at the boundary rather than reaching an actuator.
//!
//! Each axis is optional and capability-gated: a deployment advertises the
//! values it can actually set through [`ControlCapability`]. A commodity ESP32
//! that can only vary its sounding cadence exposes a capability whose only
//! non-empty axis is [`ControlCapability::cadences`]; an all-empty capability
//! means nothing is controllable and the controller degrades to the passive
//! planner (ADR-306 §2, ADR-280).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum number of distinct values accepted per control axis. Bounds
/// allocation when a capability set is built from untrusted input.
pub const MAX_AXIS_VALUES: usize = 64;

/// Maximum number of antenna chains a synthetic aperture may model.
pub const MAX_CHAINS: u8 = 16;

/// Smallest modelled sounding interval, in milliseconds (fastest cadence).
pub const MIN_CADENCE_MS: u32 = 1;

/// Largest modelled sounding interval, in milliseconds (slowest cadence).
pub const MAX_CADENCE_MS: u32 = 60_000;

/// Reasons a control value or capability set is rejected at the boundary.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ControlError {
    /// A channel number is not a valid channel for its band.
    #[error("channel {number} is not valid in band {band:?}")]
    InvalidChannel {
        /// The rejected band.
        band: Band,
        /// The rejected channel number.
        number: u16,
    },
    /// A bandwidth value (in MHz) is not a recognised channel width.
    #[error("bandwidth {mhz} MHz is not a recognised channel width")]
    InvalidBandwidth {
        /// The rejected width in MHz.
        mhz: u16,
    },
    /// A sounding interval is outside the modelled `[MIN, MAX]` cadence range.
    #[error("cadence interval {interval_ms} ms is outside [{min}, {max}] ms")]
    InvalidCadence {
        /// The rejected interval in milliseconds.
        interval_ms: u32,
        /// The accepted minimum.
        min: u32,
        /// The accepted maximum.
        max: u32,
    },
    /// An antenna selection is empty (no chains active).
    #[error("antenna selection must activate at least one chain")]
    EmptyAntennaSelection,
    /// An antenna chain index is out of range for the declared aperture.
    #[error("antenna chain index {index} is out of range for {num_chains} chains (max {max})")]
    AntennaChainOutOfRange {
        /// The offending chain index.
        index: u8,
        /// The declared number of chains.
        num_chains: u8,
        /// The largest permitted chain count.
        max: u8,
    },
    /// A capability axis listed more than [`MAX_AXIS_VALUES`] values.
    #[error("control axis lists {len} values, exceeding the maximum {max}")]
    AxisTooLarge {
        /// Actual number of values supplied.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
}

/// The RF band a channel belongs to. Determines which channel numbers are
/// valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    /// 2.4 GHz band.
    Ghz24,
    /// 5 GHz band.
    Ghz5,
    /// 6 GHz band (Wi-Fi 6E).
    Ghz6,
}

/// The standard 5 GHz channel numbers RuView may model probing.
const GHZ5_CHANNELS: &[u16] = &[
    36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
    149, 153, 157, 161, 165,
];

/// A validated Wi-Fi channel: a band plus a channel number known to that band.
///
/// This is a *choice of which spectrum to probe*, not an instruction to any
/// radio. Construction validates the number against its band so an invalid
/// channel can never enter a [`ControlAction`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Channel {
    band: Band,
    number: u16,
}

impl Channel {
    /// Construct a validated channel, rejecting a number that is not valid in
    /// its band.
    pub fn new(band: Band, number: u16) -> Result<Self, ControlError> {
        let valid = match band {
            Band::Ghz24 => (1..=14).contains(&number),
            Band::Ghz5 => GHZ5_CHANNELS.contains(&number),
            // Wi-Fi 6E channels are the odd numbers 1..=233.
            Band::Ghz6 => (1..=233).contains(&number) && number % 2 == 1,
        };
        if valid {
            Ok(Self { band, number })
        } else {
            Err(ControlError::InvalidChannel { band, number })
        }
    }

    /// The band this channel is in.
    #[must_use]
    pub fn band(&self) -> Band {
        self.band
    }

    /// The channel number.
    #[must_use]
    pub fn number(&self) -> u16 {
        self.number
    }
}

/// A validated channel width. Wider widths probe more spectrum per sounding and
/// are treated as *more exploratory* by the controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bandwidth {
    /// 20 MHz.
    Bw20,
    /// 40 MHz.
    Bw40,
    /// 80 MHz.
    Bw80,
    /// 160 MHz.
    Bw160,
    /// 320 MHz (Wi-Fi 7).
    Bw320,
}

impl Bandwidth {
    /// Construct a bandwidth from a width in MHz, rejecting unrecognised widths.
    pub fn from_mhz(mhz: u16) -> Result<Self, ControlError> {
        Ok(match mhz {
            20 => Self::Bw20,
            40 => Self::Bw40,
            80 => Self::Bw80,
            160 => Self::Bw160,
            320 => Self::Bw320,
            other => return Err(ControlError::InvalidBandwidth { mhz: other }),
        })
    }

    /// The width in MHz. Also the exploration-ordering key (wider = more
    /// exploratory).
    #[must_use]
    pub fn mhz(&self) -> u16 {
        match self {
            Self::Bw20 => 20,
            Self::Bw40 => 40,
            Self::Bw80 => 80,
            Self::Bw160 => 160,
            Self::Bw320 => 320,
        }
    }
}

impl PartialOrd for Bandwidth {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bandwidth {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.mhz().cmp(&other.mhz())
    }
}

/// A validated sounding cadence: the interval between solicited soundings, in
/// milliseconds. A *shorter* interval is a faster cadence and is treated as
/// *more exploratory* (more measurements per unit time).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Cadence {
    interval_ms: u32,
}

impl Cadence {
    /// Construct a cadence from a sounding interval, rejecting an interval
    /// outside `[MIN_CADENCE_MS, MAX_CADENCE_MS]`.
    pub fn from_interval_ms(interval_ms: u32) -> Result<Self, ControlError> {
        if (MIN_CADENCE_MS..=MAX_CADENCE_MS).contains(&interval_ms) {
            Ok(Self { interval_ms })
        } else {
            Err(ControlError::InvalidCadence {
                interval_ms,
                min: MIN_CADENCE_MS,
                max: MAX_CADENCE_MS,
            })
        }
    }

    /// The sounding interval in milliseconds.
    #[must_use]
    pub fn interval_ms(&self) -> u32 {
        self.interval_ms
    }
}

/// A validated subset of a distributed aperture's antenna chains. Activating
/// *more* chains widens the aperture and is treated as *more exploratory*.
///
/// The selection is bounded by the ADR-280 `CoherentSensorGroup` compatibility
/// proof in a fielded system; here it is a validated, deterministic set of
/// chain indices with no coherence claim.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AntennaSelection {
    num_chains: u8,
    /// Active chain indices, sorted ascending and deduplicated.
    active: Vec<u8>,
}

impl AntennaSelection {
    /// Construct a validated antenna selection over an aperture of
    /// `num_chains` chains, rejecting an empty selection or any index at or
    /// beyond `num_chains` / [`MAX_CHAINS`]. Indices are sorted and
    /// deduplicated so the selection is canonical.
    pub fn new(active: impl IntoIterator<Item = u8>, num_chains: u8) -> Result<Self, ControlError> {
        if num_chains == 0 || num_chains > MAX_CHAINS {
            return Err(ControlError::AntennaChainOutOfRange {
                index: 0,
                num_chains,
                max: MAX_CHAINS,
            });
        }
        let mut chains: Vec<u8> = active.into_iter().collect();
        chains.sort_unstable();
        chains.dedup();
        if chains.is_empty() {
            return Err(ControlError::EmptyAntennaSelection);
        }
        if let Some(&idx) = chains.iter().find(|&&i| i >= num_chains) {
            return Err(ControlError::AntennaChainOutOfRange {
                index: idx,
                num_chains,
                max: MAX_CHAINS,
            });
        }
        Ok(Self {
            num_chains,
            active: chains,
        })
    }

    /// The declared aperture size (total chains).
    #[must_use]
    pub fn num_chains(&self) -> u8 {
        self.num_chains
    }

    /// The active chain indices (sorted, deduplicated).
    #[must_use]
    pub fn active(&self) -> &[u8] {
        &self.active
    }

    /// The number of active chains. Also the exploration-ordering key (more
    /// active chains = wider aperture = more exploratory).
    #[must_use]
    pub fn chain_count(&self) -> usize {
        self.active.len()
    }
}

/// A proposed measurement configuration: the controllable axes a controller
/// asks to set for the next sounding. Each axis is `None` when the deployment
/// cannot control it (it is left at the hardware default); a `Some` value is a
/// validated choice.
///
/// **This is a plan, never an emission.** Nothing here drives a radio, changes
/// pairing state, or transmits — an [`ControlAction`] is data describing what a
/// governed actuation *would* request through the ADR-280 fail-closed surface.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlAction {
    /// Which channel to probe, if channel is controllable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<Channel>,
    /// Which channel width to probe, if bandwidth is controllable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bandwidth: Option<Bandwidth>,
    /// How often to solicit a sounding, if cadence is controllable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cadence: Option<Cadence>,
    /// Which antenna chains to activate, if antenna selection is controllable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub antenna: Option<AntennaSelection>,
}

impl ControlAction {
    /// True when no axis is set — the action configures nothing.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.channel.is_none()
            && self.bandwidth.is_none()
            && self.cadence.is_none()
            && self.antenna.is_none()
    }
}

/// The set of control values a deployment can actually set, per axis
/// (capability-gated by the ADR-317 HAL in a fielded system). An axis with no
/// values is not controllable on this deployment; an all-empty capability is
/// the ESP32-style passive fallback trigger.
///
/// Values are validated, deduplicated, and sorted into
/// *least-exploratory-first* order at construction, so the controller can map a
/// scalar exploration level onto a value deterministically.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlCapability {
    /// Controllable channels (probe order preserved as supplied, deduplicated).
    pub channels: Vec<Channel>,
    /// Controllable bandwidths, sorted ascending (narrowest first).
    pub bandwidths: Vec<Bandwidth>,
    /// Controllable cadences, sorted least-exploratory-first (slowest first).
    pub cadences: Vec<Cadence>,
    /// Controllable antenna selections, sorted least-exploratory-first
    /// (fewest chains first).
    pub antennas: Vec<AntennaSelection>,
}

impl ControlCapability {
    /// An empty capability: nothing is controllable. The controller degrades to
    /// the passive planner for any zone under this capability.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Build a validated, canonicalised capability set. Each axis is
    /// deduplicated, bounded to [`MAX_AXIS_VALUES`], and sorted into
    /// least-exploratory-first order so exploration mapping is deterministic.
    ///
    /// Channels keep caller order (deduplicated) because band/number has no
    /// intrinsic exploration ranking — the controller sweeps them by cycle.
    pub fn new(
        channels: Vec<Channel>,
        mut bandwidths: Vec<Bandwidth>,
        mut cadences: Vec<Cadence>,
        mut antennas: Vec<AntennaSelection>,
    ) -> Result<Self, ControlError> {
        // Dedup channels while preserving first-seen order.
        let mut seen = Vec::new();
        let mut channels_dedup = Vec::new();
        for c in channels {
            if !seen.contains(&c) {
                seen.push(c);
                channels_dedup.push(c);
            }
        }
        check_len(channels_dedup.len())?;

        bandwidths.sort_unstable();
        bandwidths.dedup();
        check_len(bandwidths.len())?;

        // Least exploratory first = slowest (largest interval) first.
        cadences.sort_unstable_by(|a, b| b.interval_ms().cmp(&a.interval_ms()));
        cadences.dedup();
        check_len(cadences.len())?;

        // Least exploratory first = fewest chains first; tie-break by indices.
        antennas.sort_by(|a, b| {
            a.chain_count()
                .cmp(&b.chain_count())
                .then_with(|| a.active().cmp(b.active()))
        });
        antennas.dedup();
        check_len(antennas.len())?;

        Ok(Self {
            channels: channels_dedup,
            bandwidths,
            cadences,
            antennas,
        })
    }

    /// True when no axis has any controllable value.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
            && self.bandwidths.is_empty()
            && self.cadences.is_empty()
            && self.antennas.is_empty()
    }
}

fn check_len(len: usize) -> Result<(), ControlError> {
    if len > MAX_AXIS_VALUES {
        Err(ControlError::AxisTooLarge {
            len,
            max: MAX_AXIS_VALUES,
        })
    } else {
        Ok(())
    }
}
