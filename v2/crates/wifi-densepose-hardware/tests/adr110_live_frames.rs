//! ADR-110 / issue #1005: real ESP32-C6 HE-LTF CSI frames captured live.
//!
//! Both fixtures below are verbatim UDP payloads captured on 2026-06-11 from
//! an ESP32-C6 (node_id 12, IDF v5.5 build) streaming to UDP :5005 — the
//! same node, same link, seconds apart. The 532-byte frame is an HE-SU
//! capture (256 subcarrier bins = 242 active HE20 tones); the 148-byte frame
//! is the HT fallback grid (64 bins) the same firmware emits for non-HE
//! traffic. They are the canonical regression fixtures for the non-fixed
//! subcarrier count introduced by HE-LTF.

use wifi_densepose_hardware::{Bandwidth, Esp32CsiParser, PpduType};

/// 532-byte HE-SU frame: header + 256 subcarrier I/Q pairs.
/// magic=0xC5110001 node=12 ant=1 nsub=256 freq=2432 seq=11610
/// rssi=-40 noise=-87 byte18=0x01 (HE-SU) byte19=0x10 (15.4-sync valid)
const HE_FRAME_HEX: &str = "010011c50c010001800900005a2d0000d8a9011000000000000000000000f70ef70ef50cf30bf209f108f006ef03ee02ee00eefdeffbeff8f0f7f1f4f2f3f4f1f5f0f7eef8edfaecfdecffeb01ea03ea05e908ea0aeb0deb0fec11ee13f015f216f318f519f71afa1bfd1bff1c021c051b071b0a1a0c190f1811161315161218101a0e1b0c1c091d071e041f0120ff20fc20f91ff71ff41ef11def1cec1be919e717e615e413e311e10edf0cde09dd06dc04dc01dcffdcfbdcf9ddf6def3dff0e0ede2eae4e8e6e6e8e4eae2ebe0eedef1dcf4dbf7dafad9fdd900d903d806d909d90cda0fdc12dc14dd17df1ae11ce31ee520e722e924ed25f127f328f629f929fd2900290329062809270c260e26122516061a00001c201c1f1a211722142411250e260c27082804280129fe29fb28f927f627f426f125ef23ec22ea20e81eea20e81e891b53a82951565d4ffafbfebe9abddb10222aa47b3b371fd2c0860cd4d86ea2f35faccd46b0b66f6ff0050f2da27d1c92f7f8e1017cb545afd3e3fe60db6f478dc85a33b3454cf6df9061194a0a0fc3e0eedf76f1d292cb25c8f541dfcc4109f9f1a34955520ad8ffa3694ac395cbf6c19073a4aefb1ebf47c76730458431805d9f18ff2e81955e8752b29757f66e289f72f8e35309a737547c040444cbda1a81d221d950037ec38fd9d1dd0f56c3dc707a7bbfe66ca5a97ab7cc17d68d38ba43a1806f91f5911a5967e2c9f7f07186";

/// 148-byte HT frame from the same node: header + 64 subcarrier I/Q pairs.
/// magic=0xC5110001 node=12 ant=1 nsub=64 freq=2432 seq=11622
/// rssi=-79 noise=-87 byte18=0x00 (HT/legacy) byte19=0x10
const HT_FRAME_HEX: &str = "010011c50c01400080090000662d0000b1a900100000000000000000fcfaf909f013f112f213f212f311f410f511f510f610f510f411f410f411f312f213f214f214f212f313f513f512f611f610f80ef90df90c0000010eff11fe13ff11fe1300000000ff01000001010002000200020204000301040103000400040002ff03ff03fe02fe02fe01fd00edfc03fa000000000000";

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn live_he_su_frame_532_bytes_parses_with_256_subcarriers() {
    let data = unhex(HE_FRAME_HEX);
    assert_eq!(data.len(), 532);

    let (frame, consumed) = Esp32CsiParser::parse_frame(&data).expect("HE frame must parse");
    assert_eq!(consumed, 532);
    assert_eq!(frame.metadata.node_id, 12);
    assert_eq!(frame.metadata.n_antennas, 1);
    assert_eq!(frame.metadata.n_subcarriers, 256);
    assert_eq!(frame.subcarrier_count(), 256);
    assert_eq!(frame.metadata.channel_freq_mhz, 2432);
    assert_eq!(frame.metadata.sequence, 11610);
    assert_eq!(frame.metadata.rssi_dbm, -40);
    assert_eq!(frame.metadata.noise_floor_dbm, -87);
    // ADR-110 byte 18: HE-SU PPDU. Byte 19 bit 4: ESP-NOW time-sync valid.
    assert_eq!(frame.metadata.ppdu_type, PpduType::HeSu);
    assert!(frame.metadata.ppdu_type.is_he());
    assert!(frame.metadata.adr018_flags.ieee802154_sync_valid);
    assert!(!frame.metadata.adr018_flags.bw40);
    // 256-FFT HE-LTF on a 20 MHz channel — NOT 160 MHz.
    assert_eq!(frame.metadata.bandwidth, Bandwidth::Bw20);
    assert!(frame.is_valid());
}

#[test]
fn live_ht_frame_148_bytes_parses_with_64_subcarriers() {
    let data = unhex(HT_FRAME_HEX);
    assert_eq!(data.len(), 148);

    let (frame, consumed) = Esp32CsiParser::parse_frame(&data).expect("HT frame must parse");
    assert_eq!(consumed, 148);
    assert_eq!(frame.metadata.node_id, 12);
    assert_eq!(frame.metadata.n_subcarriers, 64);
    assert_eq!(frame.metadata.channel_freq_mhz, 2432);
    assert_eq!(frame.metadata.sequence, 11622);
    assert_eq!(frame.metadata.rssi_dbm, -79);
    assert_eq!(frame.metadata.noise_floor_dbm, -87);
    assert_eq!(frame.metadata.ppdu_type, PpduType::HtLegacy);
    assert!(!frame.metadata.ppdu_type.is_he());
    // 64-bin full HT20 FFT grid on a 20 MHz channel — NOT 40 MHz.
    assert_eq!(frame.metadata.bandwidth, Bandwidth::Bw20);
    assert!(frame.is_valid());
}

#[test]
fn live_interleaved_stream_parses_both_grids() {
    // The live node interleaves HE (84%) and HT (16%) frames on one socket.
    let mut stream = unhex(HE_FRAME_HEX);
    stream.extend_from_slice(&unhex(HT_FRAME_HEX));
    stream.extend_from_slice(&unhex(HE_FRAME_HEX));

    let (frames, consumed) = Esp32CsiParser::parse_stream(&stream);
    assert_eq!(frames.len(), 3);
    assert_eq!(consumed, 532 + 148 + 532);
    assert_eq!(frames[0].metadata.n_subcarriers, 256);
    assert_eq!(frames[1].metadata.n_subcarriers, 64);
    assert_eq!(frames[2].metadata.n_subcarriers, 256);
    assert_eq!(frames[0].metadata.ppdu_type, PpduType::HeSu);
    assert_eq!(frames[1].metadata.ppdu_type, PpduType::HtLegacy);
}
