//! Bounded, privacy-conscious summaries of RTL8720F radar transport frames.

use serde::Serialize;
use wifi_densepose_hardware::rtl8720f::{RadarFlags, RadarFrame, RadarPayload, ReportType};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct RealtekRadarSnapshot {
    pub event_type: &'static str,
    pub source: &'static str,
    pub report_type: &'static str,
    pub sequence: u32,
    pub timestamp_us: u64,
    pub device_id: String,
    pub center_freq_khz: u32,
    pub bandwidth_mhz: u16,
    pub antenna_count: u8,
    pub element_count: usize,
    pub calibrated: bool,
    pub synthetic: bool,
    pub interference_detected: bool,
    pub saturated: bool,
    pub time_synchronized: bool,
    pub calibration_id: u32,
    pub bin_spacing: f32,
    pub peak_range_m: Option<f32>,
    pub peak_power: Option<f32>,
    pub mean_cfr_amplitude: Option<f32>,
}

impl RealtekRadarSnapshot {
    pub(crate) fn from_frame(frame: &RadarFrame) -> Self {
        let synthetic = frame.flags.contains(RadarFlags::SYNTHETIC);
        let (peak_range_m, peak_power) = range_peak(frame);
        Self {
            event_type: "realtek_radar",
            source: if synthetic {
                "realtek:simulated"
            } else {
                "realtek"
            },
            report_type: report_type_name(frame.report_type),
            sequence: frame.sequence,
            timestamp_us: frame.timestamp_us,
            device_id: format!("{:016x}", frame.device_id),
            center_freq_khz: frame.center_freq_khz,
            bandwidth_mhz: frame.bandwidth_mhz,
            antenna_count: frame.antenna_count,
            element_count: frame.payload.len(),
            calibrated: frame.flags.contains(RadarFlags::CALIBRATED),
            synthetic,
            interference_detected: frame.flags.contains(RadarFlags::INTERFERENCE_DETECTED),
            saturated: frame.flags.contains(RadarFlags::SATURATED),
            time_synchronized: frame.flags.contains(RadarFlags::TIME_SYNCHRONIZED),
            calibration_id: frame.calibration_id,
            bin_spacing: frame.bin_spacing,
            peak_range_m,
            peak_power,
            mean_cfr_amplitude: mean_cfr_amplitude(frame),
        }
    }
}

fn report_type_name(report_type: ReportType) -> &'static str {
    match report_type {
        ReportType::Cfr => "cfr",
        ReportType::RangeNear => "range_near",
        ReportType::RangeFar => "range_far",
        ReportType::Interference => "interference",
        ReportType::Capabilities => "capabilities",
    }
}

fn range_peak(frame: &RadarFrame) -> (Option<f32>, Option<f32>) {
    let max = match &frame.payload {
        RadarPayload::PowerU16(values) => values
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| *value)
            .map(|(index, value)| (index, *value as f32 * frame.scale)),
        RadarPayload::PowerF32(values) => values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, value)| (index, *value * frame.scale)),
        _ => None,
    };
    max.map_or((None, None), |(index, power)| {
        (Some(index as f32 * frame.bin_spacing), Some(power))
    })
}

fn mean_cfr_amplitude(frame: &RadarFrame) -> Option<f32> {
    let (sum, count) = match &frame.payload {
        RadarPayload::ComplexI16(values) => (
            values
                .iter()
                .map(|[i, q]| ((*i as f32).hypot(*q as f32)) * frame.scale)
                .sum::<f32>(),
            values.len(),
        ),
        RadarPayload::ComplexF32(values) => (
            values
                .iter()
                .map(|[i, q]| i.hypot(*q) * frame.scale)
                .sum::<f32>(),
            values.len(),
        ),
        _ => return None,
    };
    (count != 0).then_some(sum / count as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wifi_densepose_hardware::rtl8720f::simulator::{Rtl8720fSimulator, SimulatorConfig};

    #[test]
    fn synthetic_range_summary_has_peak_and_provenance() {
        let mut simulator = Rtl8720fSimulator::new(SimulatorConfig::default()).unwrap();
        let snapshot =
            RealtekRadarSnapshot::from_frame(&simulator.next_frame(ReportType::RangeNear));
        assert_eq!(snapshot.source, "realtek:simulated");
        assert!(snapshot.synthetic);
        assert!(snapshot.peak_range_m.is_some());
        assert!(snapshot.peak_power.unwrap() > 0.0);
        assert_eq!(snapshot.mean_cfr_amplitude, None);
    }

    #[test]
    fn synthetic_cfr_summary_exposes_only_aggregate_amplitude() {
        let mut simulator = Rtl8720fSimulator::new(SimulatorConfig::default()).unwrap();
        let snapshot = RealtekRadarSnapshot::from_frame(&simulator.next_frame(ReportType::Cfr));
        assert!(snapshot.mean_cfr_amplitude.unwrap() > 0.0);
        assert_eq!(snapshot.peak_power, None);
    }
}
