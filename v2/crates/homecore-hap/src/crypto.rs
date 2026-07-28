//! HAP cryptographic composition over RustCrypto primitives.
#![cfg_attr(not(feature = "hap-server"), allow(dead_code))]

use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce};
use hkdf::Hkdf;
use sha2::Sha512;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::HapError;

pub(crate) const MAX_RECORD_PLAINTEXT: usize = 1024;
pub(crate) const RECORD_TAG_BYTES: usize = 16;

pub(crate) fn hkdf_sha512(
    salt: &[u8],
    input_key: &[u8],
    info: &[u8],
) -> Result<[u8; 32], HapError> {
    let mut output = [0u8; 32];
    Hkdf::<Sha512>::new(Some(salt), input_key)
        .expand(info, &mut output)
        .map_err(|_| HapError::Protocol("HKDF output length is invalid".into()))?;
    Ok(output)
}

fn label_nonce(label: &[u8; 8]) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[4..].copy_from_slice(label);
    nonce
}

pub(crate) fn seal_labeled(
    key: &[u8; 32],
    label: &[u8; 8],
    plaintext: &[u8],
) -> Result<Vec<u8>, HapError> {
    seal(key, &label_nonce(label), plaintext, &[])
}

pub(crate) fn open_labeled(
    key: &[u8; 32],
    label: &[u8; 8],
    ciphertext_and_tag: &[u8],
) -> Result<Vec<u8>, HapError> {
    open(key, &label_nonce(label), ciphertext_and_tag, &[])
}

fn seal(
    key: &[u8; 32],
    nonce: &[u8; 12],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, HapError> {
    ChaCha20Poly1305::new(key.into())
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| HapError::Protocol("ChaCha20-Poly1305 encryption failed".into()))
}

fn open(
    key: &[u8; 32],
    nonce: &[u8; 12],
    ciphertext_and_tag: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, HapError> {
    ChaCha20Poly1305::new(key.into())
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext_and_tag,
                aad,
            },
        )
        .map_err(|_| HapError::Protocol("ChaCha20-Poly1305 authentication failed".into()))
}

#[derive(Zeroize, ZeroizeOnDrop)]
pub(crate) struct SessionKeys {
    accessory_to_controller: [u8; 32],
    controller_to_accessory: [u8; 32],
}

impl SessionKeys {
    pub(crate) fn derive(shared_secret: &[u8; 32]) -> Result<Self, HapError> {
        Ok(Self {
            accessory_to_controller: hkdf_sha512(
                b"Control-Salt",
                shared_secret,
                b"Control-Read-Encryption-Key",
            )?,
            controller_to_accessory: hkdf_sha512(
                b"Control-Salt",
                shared_secret,
                b"Control-Write-Encryption-Key",
            )?,
        })
    }

    #[cfg(test)]
    pub(crate) fn controller_view(&self) -> Self {
        Self {
            accessory_to_controller: self.controller_to_accessory,
            controller_to_accessory: self.accessory_to_controller,
        }
    }
}

/// Stateful HAP IP record protection. A failed decryption is terminal: callers
/// must close the connection and must never retry with the same counter.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(crate) struct RecordLayer {
    read_key: [u8; 32],
    write_key: [u8; 32],
    read_counter: u64,
    write_counter: u64,
}

impl RecordLayer {
    pub(crate) fn accessory(keys: SessionKeys) -> Self {
        Self {
            read_key: keys.controller_to_accessory,
            write_key: keys.accessory_to_controller,
            read_counter: 0,
            write_counter: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn controller(keys: SessionKeys) -> Self {
        Self {
            read_key: keys.controller_to_accessory,
            write_key: keys.accessory_to_controller,
            read_counter: 0,
            write_counter: 0,
        }
    }

    pub(crate) fn encrypt(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, HapError> {
        let mut output = Vec::with_capacity(
            plaintext.len() + plaintext.len().div_ceil(MAX_RECORD_PLAINTEXT) * 18,
        );
        for chunk in plaintext.chunks(MAX_RECORD_PLAINTEXT) {
            let length = u16::try_from(chunk.len())
                .expect("HAP record chunks never exceed the u16 range")
                .to_le_bytes();
            let nonce = record_nonce(self.write_counter);
            let encrypted = seal(&self.write_key, &nonce, chunk, &length)?;
            self.write_counter = self
                .write_counter
                .checked_add(1)
                .ok_or_else(|| HapError::Protocol("HAP write nonce exhausted".into()))?;
            output.extend_from_slice(&length);
            output.extend_from_slice(&encrypted);
        }
        Ok(output)
    }

    pub(crate) fn decrypt(
        &mut self,
        length_bytes: [u8; 2],
        ciphertext_and_tag: &[u8],
    ) -> Result<Vec<u8>, HapError> {
        let length = u16::from_le_bytes(length_bytes) as usize;
        if length > MAX_RECORD_PLAINTEXT {
            return Err(HapError::Protocol(
                "encrypted HAP record exceeds 1024 bytes".into(),
            ));
        }
        if ciphertext_and_tag.len() != length + RECORD_TAG_BYTES {
            return Err(HapError::Protocol(
                "encrypted HAP record length does not match framing".into(),
            ));
        }
        let nonce = record_nonce(self.read_counter);
        let plaintext = open(&self.read_key, &nonce, ciphertext_and_tag, &length_bytes)?;
        self.read_counter = self
            .read_counter
            .checked_add(1)
            .ok_or_else(|| HapError::Protocol("HAP read nonce exhausted".into()))?;
        Ok(plaintext)
    }
}

fn record_nonce(counter: u64) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[4..].copy_from_slice(&counter.to_le_bytes());
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_record_vector_and_multiframe_roundtrip() {
        let shared = [0x42; 32];
        let keys = SessionKeys::derive(&shared).unwrap();
        let mut vector_accessory = RecordLayer::accessory(keys);
        let vector = vector_accessory.encrypt(b"HAP").unwrap();
        assert_eq!(
            vector,
            [
                0x03, 0x00, 0xa2, 0x37, 0x30, 0x29, 0xba, 0xe2, 0xa9, 0xa6, 0xbb, 0x5b, 0xff, 0xed,
                0x6a, 0x29, 0x74, 0x12, 0xd1, 0x6d, 0x7a,
            ]
        );
        let mut accessory = RecordLayer::accessory(SessionKeys::derive(&shared).unwrap());
        let controller_keys = SessionKeys::derive(&shared).unwrap().controller_view();
        let mut controller = RecordLayer::controller(controller_keys);
        let plaintext = vec![0x5a; 2050];
        let encrypted = accessory.encrypt(&plaintext).unwrap();
        assert_eq!(&encrypted[..2], &[0, 4]);

        let mut offset = 0;
        let mut decrypted = Vec::new();
        while offset < encrypted.len() {
            let length_bytes: [u8; 2] = encrypted[offset..offset + 2].try_into().unwrap();
            let length = u16::from_le_bytes(length_bytes) as usize;
            let end = offset + 2 + length + RECORD_TAG_BYTES;
            decrypted.extend(
                controller
                    .decrypt(length_bytes, &encrypted[offset + 2..end])
                    .unwrap(),
            );
            offset = end;
        }
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn replay_tamper_and_oversize_fail_closed() {
        let shared = [7; 32];
        let mut sender = RecordLayer::accessory(SessionKeys::derive(&shared).unwrap());
        let frame = sender.encrypt(b"authenticated").unwrap();
        let length: [u8; 2] = frame[..2].try_into().unwrap();
        let mut receiver =
            RecordLayer::controller(SessionKeys::derive(&shared).unwrap().controller_view());
        assert_eq!(
            receiver.decrypt(length, &frame[2..]).unwrap(),
            b"authenticated"
        );
        assert!(receiver.decrypt(length, &frame[2..]).is_err());
        assert!(receiver
            .decrypt(1025u16.to_le_bytes(), &vec![0; 1025 + RECORD_TAG_BYTES])
            .is_err());
    }
}
