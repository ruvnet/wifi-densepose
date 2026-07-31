//! Server-side HAP Pair-Verify M1-M4.
#![cfg_attr(not(feature = "hap-server"), allow(dead_code))]

use std::sync::Arc;

use ed25519_dalek::{Signature, Signer, VerifyingKey};
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::{Zeroize, Zeroizing};

use crate::crypto::{hkdf_sha512, open_labeled, seal_labeled, SessionKeys};
use crate::error::HapError;
use crate::pairing::PairingStore;
use crate::protocol::{
    encode_items, error_response, Tlv8, TLV_ENCRYPTED_DATA, TLV_ERROR_AUTHENTICATION,
    TLV_IDENTIFIER, TLV_PUBLIC_KEY, TLV_SIGNATURE, TLV_STATE,
};

struct AwaitM3 {
    controller_public: [u8; 32],
    accessory_public: [u8; 32],
    shared_secret: Zeroizing<[u8; 32]>,
    session_key: Zeroizing<[u8; 32]>,
}

enum Phase {
    Idle,
    AwaitM3(AwaitM3),
}

pub(crate) struct AuthenticatedSession {
    pub(crate) controller_id: String,
    pub(crate) admin: bool,
    pub(crate) keys: SessionKeys,
}

pub(crate) struct PairVerifyResponse {
    pub(crate) body: Vec<u8>,
    pub(crate) authenticated: Option<AuthenticatedSession>,
    pub(crate) terminal: bool,
}

pub(crate) struct PairVerify {
    store: Arc<PairingStore>,
    phase: Phase,
}

impl PairVerify {
    pub(crate) fn new(store: Arc<PairingStore>) -> Self {
        Self {
            store,
            phase: Phase::Idle,
        }
    }

    pub(crate) fn handle(&mut self, request: &[u8]) -> Result<PairVerifyResponse, HapError> {
        let tlv = Tlv8::parse(request)?;
        let state = tlv
            .byte(TLV_STATE)
            .ok_or_else(|| HapError::Protocol("Pair-Verify requires one-byte State".into()))?;
        match state {
            1 => self.m1(&tlv),
            3 => self.m3(&tlv),
            _ => Err(HapError::Protocol(
                "Pair-Verify state is out of sequence".into(),
            )),
        }
    }

    fn m1(&mut self, tlv: &Tlv8) -> Result<PairVerifyResponse, HapError> {
        if !matches!(self.phase, Phase::Idle) {
            return Err(HapError::Protocol("Pair-Verify M1 was replayed".into()));
        }
        if !self.store.is_paired()? {
            return Ok(PairVerifyResponse {
                body: error_response(2, TLV_ERROR_AUTHENTICATION),
                authenticated: None,
                terminal: true,
            });
        }
        let controller_public: [u8; 32] =
            required_exact(tlv, TLV_PUBLIC_KEY, 32, "controller Curve25519 public key")?
                .try_into()
                .expect("length checked");
        let mut secret_bytes = Zeroizing::new([0u8; 32]);
        getrandom::getrandom(secret_bytes.as_mut())
            .map_err(|error| HapError::Protocol(format!("generate X25519 secret: {error}")))?;
        let secret = StaticSecret::from(*secret_bytes);
        let accessory_public = PublicKey::from(&secret).to_bytes();
        let shared = secret.diffie_hellman(&PublicKey::from(controller_public));
        if !shared.was_contributory() {
            return Ok(PairVerifyResponse {
                body: error_response(2, TLV_ERROR_AUTHENTICATION),
                authenticated: None,
                terminal: true,
            });
        }
        let shared_secret = Zeroizing::new(shared.to_bytes());
        let accessory_id = self.store.accessory_id()?;
        let signing_key = self.store.signing_key()?;
        let mut accessory_info = Zeroizing::new(Vec::with_capacity(32 + accessory_id.len() + 32));
        accessory_info.extend_from_slice(&accessory_public);
        accessory_info.extend_from_slice(accessory_id.as_bytes());
        accessory_info.extend_from_slice(&controller_public);
        let signature = signing_key.sign(&accessory_info).to_bytes();
        let plaintext = Zeroizing::new(encode_items([
            (TLV_IDENTIFIER, accessory_id.as_bytes()),
            (TLV_SIGNATURE, signature.as_slice()),
        ]));
        let session_key = Zeroizing::new(hkdf_sha512(
            b"Pair-Verify-Encrypt-Salt",
            shared_secret.as_ref(),
            b"Pair-Verify-Encrypt-Info",
        )?);
        let encrypted = seal_labeled(&session_key, b"PV-Msg02", &plaintext)?;
        self.phase = Phase::AwaitM3(AwaitM3 {
            controller_public,
            accessory_public,
            shared_secret,
            session_key,
        });
        Ok(PairVerifyResponse {
            body: encode_items([
                (TLV_STATE, [2].as_slice()),
                (TLV_PUBLIC_KEY, accessory_public.as_slice()),
                (TLV_ENCRYPTED_DATA, encrypted.as_slice()),
            ]),
            authenticated: None,
            terminal: false,
        })
    }

    fn m3(&mut self, tlv: &Tlv8) -> Result<PairVerifyResponse, HapError> {
        let Phase::AwaitM3(state) = std::mem::replace(&mut self.phase, Phase::Idle) else {
            return Err(HapError::Protocol(
                "Pair-Verify M3 arrived without M1".into(),
            ));
        };
        match self.finish_m3(tlv, state) {
            Ok(authenticated) => Ok(PairVerifyResponse {
                body: encode_items([(TLV_STATE, [4].as_slice())]),
                authenticated: Some(authenticated),
                terminal: true,
            }),
            Err(_) => Ok(PairVerifyResponse {
                body: error_response(4, TLV_ERROR_AUTHENTICATION),
                authenticated: None,
                terminal: true,
            }),
        }
    }

    fn finish_m3(&self, tlv: &Tlv8, state: AwaitM3) -> Result<AuthenticatedSession, HapError> {
        let encrypted = tlv
            .get(TLV_ENCRYPTED_DATA)
            .filter(|value| (17..=4096).contains(&value.len()))
            .ok_or_else(|| HapError::Protocol("invalid Pair-Verify encrypted data".into()))?;
        let mut plaintext =
            Zeroizing::new(open_labeled(&state.session_key, b"PV-Msg03", encrypted)?);
        let sub_tlv = Tlv8::parse(&plaintext)?;
        let controller_id_bytes = sub_tlv
            .get(TLV_IDENTIFIER)
            .filter(|value| !value.is_empty() && value.len() <= 64)
            .ok_or_else(|| HapError::Protocol("invalid controller identifier".into()))?;
        let controller_id = std::str::from_utf8(controller_id_bytes)
            .map_err(|_| HapError::Protocol("controller identifier is not UTF-8".into()))?
            .to_owned();
        let signature_bytes: [u8; 64] =
            required_exact(&sub_tlv, TLV_SIGNATURE, 64, "controller signature")?
                .try_into()
                .expect("length checked");
        let pairing = self
            .store
            .get(&controller_id)?
            .ok_or_else(|| HapError::Protocol("unknown controller pairing".into()))?;
        let key = VerifyingKey::from_bytes(&pairing.public_key)
            .map_err(|_| HapError::Protocol("persisted controller key is invalid".into()))?;
        let mut controller_info =
            Zeroizing::new(Vec::with_capacity(32 + controller_id_bytes.len() + 32));
        controller_info.extend_from_slice(&state.controller_public);
        controller_info.extend_from_slice(controller_id_bytes);
        controller_info.extend_from_slice(&state.accessory_public);
        key.verify_strict(&controller_info, &Signature::from_bytes(&signature_bytes))
            .map_err(|_| HapError::Protocol("controller transcript signature rejected".into()))?;
        let keys = SessionKeys::derive(&state.shared_secret)?;
        plaintext.zeroize();
        Ok(AuthenticatedSession {
            controller_id,
            admin: pairing.admin,
            keys,
        })
    }
}

fn required_exact<'a>(
    tlv: &'a Tlv8,
    kind: u8,
    length: usize,
    name: &str,
) -> Result<&'a [u8], HapError> {
    tlv.get(kind)
        .filter(|value| value.len() == length)
        .ok_or_else(|| HapError::Protocol(format!("{name} must contain {length} bytes")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::RecordLayer;
    use crate::pairing::{ControllerPairing, SetupCode};
    use ed25519_dalek::SigningKey;

    fn setup() -> (tempfile::TempDir, Arc<PairingStore>, SigningKey) {
        let directory = tempfile::tempdir().unwrap();
        let store = PairingStore::create(
            directory.path().join("pairings.json"),
            SetupCode::parse("518-26-003").unwrap(),
            Some("AA:BB:CC:DD:EE:FF".into()),
        )
        .unwrap();
        let controller = SigningKey::from_bytes(&[0x44; 32]);
        store
            .add_initial(ControllerPairing {
                controller_id: "controller-1".into(),
                public_key: controller.verifying_key().to_bytes(),
                admin: true,
            })
            .unwrap();
        (directory, Arc::new(store), controller)
    }

    #[test]
    fn full_m1_through_m4_authenticates_transcript_and_derives_record_keys() {
        let (_directory, store, controller_signing) = setup();
        let mut server = PairVerify::new(store.clone());
        let controller_secret = StaticSecret::from([0x33; 32]);
        let controller_public = PublicKey::from(&controller_secret).to_bytes();
        let m1 = encode_items([
            (TLV_STATE, [1].as_slice()),
            (TLV_PUBLIC_KEY, controller_public.as_slice()),
        ]);
        let m2 = Tlv8::parse(&server.handle(&m1).unwrap().body).unwrap();
        let accessory_public: [u8; 32] = m2.get(TLV_PUBLIC_KEY).unwrap().try_into().unwrap();
        let shared = controller_secret.diffie_hellman(&PublicKey::from(accessory_public));
        let session_key = hkdf_sha512(
            b"Pair-Verify-Encrypt-Salt",
            shared.as_bytes(),
            b"Pair-Verify-Encrypt-Info",
        )
        .unwrap();
        let accessory_sub_tlv = Tlv8::parse(
            &open_labeled(
                &session_key,
                b"PV-Msg02",
                m2.get(TLV_ENCRYPTED_DATA).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let accessory_id = accessory_sub_tlv.get(TLV_IDENTIFIER).unwrap();
        let accessory_signature: [u8; 64] = accessory_sub_tlv
            .get(TLV_SIGNATURE)
            .unwrap()
            .try_into()
            .unwrap();
        let mut accessory_info = Vec::new();
        accessory_info.extend_from_slice(&accessory_public);
        accessory_info.extend_from_slice(accessory_id);
        accessory_info.extend_from_slice(&controller_public);
        VerifyingKey::from_bytes(&store.accessory_public_key().unwrap())
            .unwrap()
            .verify_strict(
                &accessory_info,
                &Signature::from_bytes(&accessory_signature),
            )
            .unwrap();

        let controller_id = b"controller-1";
        let mut controller_info = Vec::new();
        controller_info.extend_from_slice(&controller_public);
        controller_info.extend_from_slice(controller_id);
        controller_info.extend_from_slice(&accessory_public);
        let signature = controller_signing.sign(&controller_info).to_bytes();
        let sub_tlv = encode_items([
            (TLV_IDENTIFIER, controller_id.as_slice()),
            (TLV_SIGNATURE, signature.as_slice()),
        ]);
        let encrypted = seal_labeled(&session_key, b"PV-Msg03", &sub_tlv).unwrap();
        let m3 = encode_items([
            (TLV_STATE, [3].as_slice()),
            (TLV_ENCRYPTED_DATA, encrypted.as_slice()),
        ]);
        let result = server.handle(&m3).unwrap();
        assert_eq!(Tlv8::parse(&result.body).unwrap().byte(TLV_STATE), Some(4));
        let authenticated = result.authenticated.unwrap();
        assert_eq!(authenticated.controller_id, "controller-1");
        let mut accessory_records = RecordLayer::accessory(authenticated.keys);
        let encrypted_record = accessory_records.encrypt(b"response").unwrap();
        assert!(!encrypted_record
            .windows(8)
            .any(|window| window == b"response"));
    }

    #[test]
    fn all_zero_key_replay_and_tamper_fail_closed() {
        let (_directory, store, _) = setup();
        let mut server = PairVerify::new(store);
        let zero_key = encode_items([
            (TLV_STATE, [1].as_slice()),
            (TLV_PUBLIC_KEY, [0u8; 32].as_slice()),
        ]);
        let response = Tlv8::parse(&server.handle(&zero_key).unwrap().body).unwrap();
        assert_eq!(
            response.byte(crate::protocol::TLV_ERROR),
            Some(TLV_ERROR_AUTHENTICATION)
        );

        let controller_secret = StaticSecret::from([9; 32]);
        let controller_public = PublicKey::from(&controller_secret).to_bytes();
        let m1 = encode_items([
            (TLV_STATE, [1].as_slice()),
            (TLV_PUBLIC_KEY, controller_public.as_slice()),
        ]);
        let _ = server.handle(&m1).unwrap();
        assert!(server.handle(&m1).is_err());
        let tampered = encode_items([
            (TLV_STATE, [3].as_slice()),
            (TLV_ENCRYPTED_DATA, [0u8; 17].as_slice()),
        ]);
        let response = server.handle(&tampered).unwrap();
        assert!(response.authenticated.is_none());
    }
}
