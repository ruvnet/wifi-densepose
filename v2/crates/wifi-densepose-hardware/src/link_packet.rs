//! Link-status packet decoder (issue #1542 ask 1: BSSID telemetry).
//!
//! The associated AP *is* the sensing-link geometry: when a station roams,
//! every downstream consumer of that node's CSI is silently looking at a
//! different Fresnel volume. Field data on #1542 shows published RSSI cannot
//! reveal the change (−47.7 vs −48.2 dBm across two completely different
//! link geometries), so the BSSID itself is the only observable.
//!
//! Emitted by the firmware on the same UDP socket as ADR-018 CSI frames and
//! ADR-110 sync packets, distinguished by leading magic `0xC511_A111` (next
//! value in the ADR-110 auxiliary-packet family). Low rate: every
//! `CONFIG_LINK_STATUS_EVERY_N_FRAMES` CSI callbacks (default 600 ≈ 30 s at
//! 20 Hz) plus one immediate emission whenever the BSSID observed via
//! `esp_wifi_sta_get_ap_info()` differs from the previously reported one —
//! so a roam is visible within one CSI callback, not one cadence period.
//!
//! Wire format (32 bytes, little-endian, mirrors the sync-packet layout):
//! ```text
//! [0..3]   magic 0xC511A111 (LE u32)
//! [4]      node_id
//! [5]      proto_ver (currently 0x01)
//! [6]      flags: bit 0 = ap_info_valid (esp_wifi_sta_get_ap_info() == ESP_OK)
//! [7]      primary channel (0 when ap_info_valid = 0)
//! [8..13]  bssid[6] (all-zero when ap_info_valid = 0)
//! [14]     AP RSSI as seen by the station, dBm (i8; 0 when invalid)
//! [15]     reserved
//! [16..19] reassoc_count (LE u32) — WIFI_EVENT_STA_CONNECTED events since boot
//! [20..31] reserved
//! ```

use serde::{Deserialize, Serialize};

use crate::error::ParseError;

/// Magic constant in the first 4 little-endian bytes of every link-status packet.
pub const LINK_PACKET_MAGIC: u32 = 0xC511_A111;
/// Total wire size of a link-status packet.
pub const LINK_PACKET_SIZE: usize = 32;
/// Wire protocol version currently emitted by firmware.
pub const LINK_PACKET_PROTO_VER: u8 = 0x01;

/// Decoded #1542 link-status packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkStatusPacket {
    pub node_id: u8,
    pub proto_ver: u8,
    /// False when the firmware's `esp_wifi_sta_get_ap_info()` call failed
    /// (not associated); `bssid`/`channel`/`ap_rssi_dbm` are zero then.
    pub ap_info_valid: bool,
    /// Primary channel of the associated AP.
    pub channel: u8,
    /// BSSID of the associated AP — the sensing link's far endpoint identity.
    pub bssid: [u8; 6],
    /// RSSI of the AP as measured by the station (dBm). Complements the
    /// frame-level RSSI already in ADR-018 headers; comes from the same
    /// `wifi_ap_record_t` read as the BSSID.
    pub ap_rssi_dbm: i8,
    /// Count of `WIFI_EVENT_STA_CONNECTED` events since boot. Monotonic per
    /// boot; a host observing an increment without a reboot marker knows a
    /// re-association happened even if the BSSID ended up unchanged.
    pub reassoc_count: u32,
}

impl LinkStatusPacket {
    /// Decode a 32-byte link-status packet. Host should dispatch on the
    /// leading magic before calling (same convention as `SyncPacket`).
    pub fn from_bytes(buf: &[u8]) -> Result<Self, ParseError> {
        if buf.len() < LINK_PACKET_SIZE {
            return Err(ParseError::InsufficientData {
                needed: LINK_PACKET_SIZE,
                got: buf.len(),
            });
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != LINK_PACKET_MAGIC {
            return Err(ParseError::InvalidMagic { expected: LINK_PACKET_MAGIC, got: magic });
        }
        let node_id = buf[4];
        let proto_ver = buf[5];
        let ap_info_valid = (buf[6] & 0x01) != 0;
        let channel = buf[7];
        let mut bssid = [0u8; 6];
        bssid.copy_from_slice(&buf[8..14]);
        let ap_rssi_dbm = buf[14] as i8;
        // buf[15] reserved
        let reassoc_count = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        // buf[20..32] reserved
        Ok(Self {
            node_id,
            proto_ver,
            ap_info_valid,
            channel,
            bssid,
            ap_rssi_dbm,
            reassoc_count,
        })
    }

    /// Serialize back to wire bytes (32 bytes, little-endian).
    pub fn to_bytes(&self) -> [u8; LINK_PACKET_SIZE] {
        let mut out = [0u8; LINK_PACKET_SIZE];
        out[0..4].copy_from_slice(&LINK_PACKET_MAGIC.to_le_bytes());
        out[4] = self.node_id;
        out[5] = self.proto_ver;
        out[6] = if self.ap_info_valid { 0x01 } else { 0x00 };
        out[7] = self.channel;
        out[8..14].copy_from_slice(&self.bssid);
        out[14] = self.ap_rssi_dbm as u8;
        // out[15] reserved zero
        out[16..20].copy_from_slice(&self.reassoc_count.to_le_bytes());
        // out[20..32] reserved zero
        out
    }

    /// Canonical lowercase colon-separated BSSID string (`aa:bb:cc:dd:ee:ff`)
    /// for JSON/MQTT surfaces.
    pub fn bssid_string(&self) -> String {
        let b = &self.bssid;
        format!(
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> LinkStatusPacket {
        LinkStatusPacket {
            node_id: 3,
            proto_ver: 1,
            ap_info_valid: true,
            channel: 6,
            bssid: [0x6c, 0xae, 0xf6, 0xb6, 0x52, 0xc7],
            ap_rssi_dbm: -48,
            reassoc_count: 2,
        }
    }

    #[test]
    fn typical_packet_roundtrips() {
        let pkt = typical();
        let wire = pkt.to_bytes();
        let decoded = LinkStatusPacket::from_bytes(&wire).unwrap();
        assert_eq!(decoded, pkt);
        assert_eq!(decoded.bssid_string(), "6c:ae:f6:b6:52:c7");
    }

    #[test]
    fn unassociated_packet_roundtrips_with_zero_fields() {
        let pkt = LinkStatusPacket {
            node_id: 5,
            proto_ver: 1,
            ap_info_valid: false,
            channel: 0,
            bssid: [0; 6],
            ap_rssi_dbm: 0,
            reassoc_count: 7,
        };
        let decoded = LinkStatusPacket::from_bytes(&pkt.to_bytes()).unwrap();
        assert_eq!(decoded, pkt);
        assert!(!decoded.ap_info_valid);
        assert_eq!(decoded.bssid_string(), "00:00:00:00:00:00");
    }

    #[test]
    fn magic_mismatch_is_typed_error() {
        let mut wire = typical().to_bytes();
        wire[0] = 0x01;
        match LinkStatusPacket::from_bytes(&wire).unwrap_err() {
            ParseError::InvalidMagic { got, .. } => assert_ne!(got, LINK_PACKET_MAGIC),
            other => panic!("expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn short_packet_is_typed_error() {
        let wire = [0u8; 16];
        match LinkStatusPacket::from_bytes(&wire).unwrap_err() {
            ParseError::InsufficientData { needed, got } => {
                assert_eq!(needed, LINK_PACKET_SIZE);
                assert_eq!(got, 16);
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    /// Hosts dispatch CSI vs sync vs link purely on the leading u32; the
    /// three magics must never collide.
    #[test]
    fn link_magic_is_distinct_from_sync_and_csi() {
        assert_ne!(LINK_PACKET_MAGIC, crate::sync_packet::SYNC_PACKET_MAGIC);
        assert_ne!(LINK_PACKET_MAGIC, crate::esp32_parser::ESP32_CSI_MAGIC);
    }

    /// Canonical wire pin (same convention as the sync packet's
    /// `canonical_wire_bytes_match_python_decoder`): if this hex stops
    /// matching, a decoder drifted from the wire.
    #[test]
    fn canonical_wire_bytes_pin() {
        let canonical: [u8; 32] = [
            0x11, 0xa1, 0x11, 0xc5, // magic 0xC511A111 (LE u32)
            0x03,                   // node_id = 3
            0x01,                   // proto_ver = 1
            0x01,                   // flags: ap_info_valid
            0x06,                   // channel 6
            0x6c, 0xae, 0xf6, 0xb6, 0x52, 0xc7, // bssid
            0xd0,                   // ap_rssi = -48 dBm (i8)
            0x00,                   // reserved
            0x02, 0x00, 0x00, 0x00, // reassoc_count = 2
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let decoded = LinkStatusPacket::from_bytes(&canonical).unwrap();
        assert_eq!(decoded, typical());
        assert_eq!(decoded.to_bytes(), canonical,
                   "to_bytes drifted from the canonical pin");
    }
}
