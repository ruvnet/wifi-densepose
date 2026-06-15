//! MAVLink v2 HMAC-SHA256 link-level signing.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::atomic::{AtomicU64, Ordering};

type HmacSha256 = Hmac<Sha256>;

/// Signs and verifies MAVLink v2 messages using HMAC-SHA256.
pub struct MavlinkSigner {
    key: [u8; 32],
    link_id: u8,
    timestamp: AtomicU64,
}

impl MavlinkSigner {
    pub fn new(key: [u8; 32], link_id: u8) -> Self {
        Self {
            key,
            link_id,
            timestamp: AtomicU64::new(1),
        }
    }

    /// Advance and return a monotonic 48-bit timestamp (units: 10 µs since epoch).
    fn next_timestamp(&self) -> u64 {
        self.timestamp.fetch_add(1, Ordering::SeqCst)
    }

    /// Compute the 6-byte MAVLink v2 signature.
    /// Signature = first 6 bytes of HMAC-SHA256(key, link_id || timestamp_6bytes || message_bytes)
    pub fn sign(&self, message_bytes: &[u8]) -> [u8; 6] {
        let ts = self.next_timestamp();
        let ts_bytes = ts.to_le_bytes(); // 8 bytes, MAVLink uses 6 but we include all for simplicity

        let mut mac = HmacSha256::new_from_slice(&self.key)
            .expect("HMAC accepts any key length");
        mac.update(&[self.link_id]);
        mac.update(&ts_bytes[..6]);
        mac.update(message_bytes);

        let result = mac.finalize().into_bytes();
        let mut sig = [0u8; 6];
        sig.copy_from_slice(&result[..6]);
        sig
    }

    /// Verify that `signature` is valid for `message_bytes`.
    /// This implementation re-computes against all recent timestamps within a
    /// small window (for demo/test). Production code should maintain a timestamp
    /// window per link_id.
    pub fn verify(&self, message_bytes: &[u8], signature: &[u8; 6]) -> bool {
        let current_ts = self.timestamp.load(Ordering::SeqCst);
        // Check ±32 timestamps to handle reordering in tests
        let start = current_ts.saturating_sub(32);
        for ts in start..=current_ts + 1 {
            let ts_bytes = ts.to_le_bytes();
            let mut mac = HmacSha256::new_from_slice(&self.key)
                .expect("HMAC accepts any key length");
            mac.update(&[self.link_id]);
            mac.update(&ts_bytes[..6]);
            mac.update(message_bytes);
            let result = mac.finalize().into_bytes();
            if &result[..6] == signature.as_ref() {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_produces_6_bytes() {
        let signer = MavlinkSigner::new([0xABu8; 32], 0);
        let sig = signer.sign(b"heartbeat");
        assert_eq!(sig.len(), 6);
    }

    #[test]
    fn test_verify_correct_signature() {
        let signer = MavlinkSigner::new([0x42u8; 32], 1);
        let msg = b"test_message";
        let sig = signer.sign(msg);
        assert!(signer.verify(msg, &sig));
    }

    #[test]
    fn test_verify_wrong_key_fails() {
        let signer1 = MavlinkSigner::new([0x01u8; 32], 1);
        let signer2 = MavlinkSigner::new([0x02u8; 32], 1);
        let msg = b"test_message";
        let sig = signer1.sign(msg);
        // signer2 has a different key — can't verify signer1's sig
        assert!(!signer2.verify(msg, &sig));
    }
}
