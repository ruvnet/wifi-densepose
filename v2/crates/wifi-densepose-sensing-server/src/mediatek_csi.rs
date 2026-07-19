//! Bounded summaries for ADR-267 MediaTek MIMO CSI frames.

use serde::Serialize;
use wifi_densepose_hardware::mediatek_csi::{CsiFlags, CsiFrame, CsiPayload, ReportKind};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct MediatekCsiSnapshot {
    pub event_type: &'static str,
    pub source: &'static str,
    pub report_kind: &'static str,
    pub sequence: u32,
    pub timestamp_us: u64,
    pub device_id: String,
    pub chipset: &'static str,
    pub center_freq_khz: u32,
    pub bandwidth_mhz: u16,
    pub tx_count: u8,
    pub rx_count: u8,
    pub subcarrier_count: u16,
    pub element_count: usize,
    pub ppdu_type: String,
    pub rssi_dbm: Vec<i8>,
    pub noise_floor_dbm: i8,
    pub calibrated: bool,
    pub synthetic: bool,
    pub saturated: bool,
    pub time_synchronized: bool,
    pub dropped_predecessor: bool,
    pub calibration_id: u32,
    pub subcarrier_spacing_hz: f32,
    pub mean_amplitude: Option<f32>,
    pub peak_amplitude: Option<f32>,
}

impl MediatekCsiSnapshot {
    pub(crate) fn from_frame(frame: &CsiFrame) -> Self {
        let synthetic = frame.flags.contains(CsiFlags::SYNTHETIC);
        let (mean_amplitude, peak_amplitude) = amplitude_summary(frame);
        Self {
            event_type: "mediatek_csi",
            source: if synthetic {
                "mediatek:simulated"
            } else {
                "mediatek"
            },
            report_kind: match frame.report_kind {
                ReportKind::Csi => "csi",
                ReportKind::Capabilities => "capabilities",
            },
            sequence: frame.sequence,
            timestamp_us: frame.timestamp_us,
            device_id: format!("{:016x}", frame.device_id),
            chipset: frame.chipset.name(),
            center_freq_khz: frame.center_freq_khz,
            bandwidth_mhz: frame.bandwidth_mhz,
            tx_count: frame.tx_count,
            rx_count: frame.rx_count,
            subcarrier_count: frame.subcarrier_count,
            element_count: frame.payload.len(),
            ppdu_type: format!("{:?}", frame.ppdu_type).to_ascii_lowercase(),
            rssi_dbm: frame.payload.rssi_dbm().to_vec(),
            noise_floor_dbm: frame.noise_floor_dbm,
            calibrated: frame.flags.contains(CsiFlags::CALIBRATED),
            synthetic,
            saturated: frame.flags.contains(CsiFlags::SATURATED),
            time_synchronized: frame.flags.contains(CsiFlags::TIME_SYNCHRONIZED),
            dropped_predecessor: frame.flags.contains(CsiFlags::DROPPED_PREDECESSOR),
            calibration_id: frame.calibration_id,
            subcarrier_spacing_hz: frame.subcarrier_spacing_hz,
            mean_amplitude,
            peak_amplitude,
        }
    }
}

fn amplitude_summary(frame: &CsiFrame) -> (Option<f32>, Option<f32>) {
    let amplitudes: Vec<f32> = match &frame.payload {
        CsiPayload::ComplexI16 { values, .. } => values
            .iter()
            .map(|[i, q]| (*i as f32).hypot(*q as f32) * frame.scale)
            .collect(),
        CsiPayload::ComplexF32 { values, .. } => values
            .iter()
            .map(|[i, q]| i.hypot(*q) * frame.scale)
            .collect(),
        CsiPayload::Bytes(_) => return (None, None),
    };
    if amplitudes.is_empty() {
        return (None, None);
    }
    let mean = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;
    let peak = amplitudes.into_iter().max_by(f32::total_cmp);
    (Some(mean), peak)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wifi_densepose_hardware::mediatek_csi::simulator::{MediatekCsiSimulator, SimulatorConfig};

    #[test]
    fn simulator_summary_preserves_dimensions_and_provenance() {
        let mut sim = MediatekCsiSimulator::new(SimulatorConfig::default()).unwrap();
        let snapshot = MediatekCsiSnapshot::from_frame(&sim.next_frame());
        assert_eq!(snapshot.source, "mediatek:simulated");
        assert_eq!(
            (
                snapshot.tx_count,
                snapshot.rx_count,
                snapshot.subcarrier_count
            ),
            (2, 3, 256)
        );
        assert_eq!(snapshot.element_count, 1536);
        assert!(snapshot.mean_amplitude.unwrap() > 0.0);
        assert!(snapshot.peak_amplitude.unwrap() >= snapshot.mean_amplitude.unwrap());
    }

    #[test]
    fn capability_summary_does_not_invent_signal_statistics() {
        let sim = MediatekCsiSimulator::new(SimulatorConfig::default()).unwrap();
        let snapshot = MediatekCsiSnapshot::from_frame(&sim.capabilities_frame());
        assert_eq!(snapshot.report_kind, "capabilities");
        assert_eq!(snapshot.mean_amplitude, None);
        assert!(snapshot.rssi_dbm.is_empty());
    }
}
