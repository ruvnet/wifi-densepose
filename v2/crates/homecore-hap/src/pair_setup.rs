//! Server-side HAP Pair-Setup M1-M6.
#![cfg_attr(not(feature = "hap-server"), allow(dead_code))]

use std::sync::Arc;

use ed25519_dalek::{Signature, Signer, VerifyingKey};
use sha2_11::Sha512;
use srp::ServerG3072;
use zeroize::{Zeroize, Zeroizing};

use crate::crypto::{hkdf_sha512, open_labeled, seal_labeled};
use crate::error::HapError;
use crate::pairing::{ControllerPairing, PairingStore};
use crate::protocol::{
    encode_items, error_response, Tlv8, TLV_ENCRYPTED_DATA, TLV_ERROR_AUTHENTICATION,
    TLV_ERROR_BUSY, TLV_ERROR_MAX_TRIES, TLV_ERROR_UNAVAILABLE, TLV_FLAGS, TLV_IDENTIFIER,
    TLV_METHOD, TLV_PROOF, TLV_PUBLIC_KEY, TLV_SALT, TLV_SIGNATURE, TLV_STATE,
};

const SRP_USERNAME: &[u8] = b"Pair-Setup";
const PAIR_SETUP_METHOD: u8 = 0;

enum Phase {
    Idle,
    AwaitM3 {
        salt: [u8; 16],
        verifier: Vec<u8>,
        server_secret: Zeroizing<Vec<u8>>,
    },
    AwaitM5 {
        session_key: Zeroizing<Vec<u8>>,
    },
}

pub(crate) struct PairSetup {
    store: Arc<PairingStore>,
    phase: Phase,
    owns_global_slot: bool,
}

pub(crate) struct PairSetupResponse {
    pub(crate) body: Vec<u8>,
    pub(crate) paired: bool,
    pub(crate) terminal: bool,
}

impl PairSetup {
    pub(crate) fn new(store: Arc<PairingStore>) -> Self {
        Self {
            store,
            phase: Phase::Idle,
            owns_global_slot: false,
        }
    }

    pub(crate) fn handle(&mut self, request: &[u8]) -> Result<PairSetupResponse, HapError> {
        let tlv = Tlv8::parse(request)?;
        let state = tlv
            .byte(TLV_STATE)
            .ok_or_else(|| HapError::Protocol("Pair-Setup requires one-byte State".into()))?;
        match state {
            1 => self.m1(&tlv),
            3 => self.m3(&tlv),
            5 => self.m5(&tlv),
            _ => Err(HapError::Protocol(
                "Pair-Setup state is out of sequence".into(),
            )),
        }
    }

    fn m1(&mut self, tlv: &Tlv8) -> Result<PairSetupResponse, HapError> {
        if !matches!(self.phase, Phase::Idle) {
            return Err(HapError::Protocol("Pair-Setup M1 was replayed".into()));
        }
        if tlv.byte(TLV_METHOD) != Some(PAIR_SETUP_METHOD) {
            return Ok(response(
                error_response(2, TLV_ERROR_UNAVAILABLE),
                false,
                true,
            ));
        }
        if tlv.get(TLV_FLAGS).is_some() {
            // Transient/split setup changes the session-key lifecycle and is
            // deliberately rejected instead of being partially implemented.
            return Ok(response(
                error_response(2, TLV_ERROR_UNAVAILABLE),
                false,
                true,
            ));
        }
        if self.store.is_paired()? {
            return Ok(response(
                error_response(2, TLV_ERROR_UNAVAILABLE),
                false,
                true,
            ));
        }
        if self.store.pair_setup_locked_out() {
            return Ok(response(
                error_response(2, TLV_ERROR_MAX_TRIES),
                false,
                true,
            ));
        }
        if !self.store.try_begin_pair_setup() {
            return Ok(response(error_response(2, TLV_ERROR_BUSY), false, true));
        }
        self.owns_global_slot = true;

        let (salt, verifier) = self.store.setup_record()?;
        let mut server_secret = Zeroizing::new(vec![0u8; 64]);
        getrandom::getrandom(server_secret.as_mut_slice())
            .map_err(|error| HapError::Protocol(format!("generate SRP secret: {error}")))?;
        let server = ServerG3072::<Sha512>::new_with_options(false);
        let public_key = server.compute_public_ephemeral(&server_secret, &verifier);
        self.phase = Phase::AwaitM3 {
            salt,
            verifier,
            server_secret,
        };
        Ok(response(
            encode_items([
                (TLV_STATE, [2].as_slice()),
                (TLV_PUBLIC_KEY, public_key.as_slice()),
                (TLV_SALT, salt.as_slice()),
            ]),
            false,
            false,
        ))
    }

    fn m3(&mut self, tlv: &Tlv8) -> Result<PairSetupResponse, HapError> {
        let Phase::AwaitM3 {
            salt,
            verifier,
            server_secret,
        } = std::mem::replace(&mut self.phase, Phase::Idle)
        else {
            return Err(HapError::Protocol(
                "Pair-Setup M3 arrived without M1".into(),
            ));
        };
        let result = (|| {
            let client_public = required_bounded(tlv, TLV_PUBLIC_KEY, 384, 384, "SRP public key")?;
            let client_proof = required_bounded(tlv, TLV_PROOF, 64, 64, "SRP proof")?;
            let server = ServerG3072::<Sha512>::new_with_options(false);
            let verifier_state = server
                .process_reply(
                    SRP_USERNAME,
                    &salt,
                    &server_secret,
                    &verifier,
                    client_public,
                )
                .map_err(|_| HapError::Protocol("invalid SRP public key".into()))?;
            let session_key = verifier_state
                .verify_client(client_proof)
                .map_err(|_| HapError::Protocol("SRP proof rejected".into()))?
                .to_vec();
            let proof = verifier_state.proof().to_vec();
            Ok::<_, HapError>((session_key, proof))
        })();
        match result {
            Ok((session_key, proof)) => {
                self.phase = Phase::AwaitM5 {
                    session_key: Zeroizing::new(session_key),
                };
                Ok(response(
                    encode_items([(TLV_STATE, [4].as_slice()), (TLV_PROOF, proof.as_slice())]),
                    false,
                    false,
                ))
            }
            Err(_) => {
                self.store.record_pair_setup_failure();
                self.release_slot();
                Ok(response(
                    error_response(4, TLV_ERROR_AUTHENTICATION),
                    false,
                    true,
                ))
            }
        }
    }

    fn m5(&mut self, tlv: &Tlv8) -> Result<PairSetupResponse, HapError> {
        let Phase::AwaitM5 { session_key } = std::mem::replace(&mut self.phase, Phase::Idle) else {
            return Err(HapError::Protocol(
                "Pair-Setup M5 arrived without authenticated M3".into(),
            ));
        };
        let result = self.finish_m5(tlv, &session_key);
        if result.is_err() {
            self.store.record_pair_setup_failure();
        }
        self.release_slot();
        match result {
            Ok(body) => Ok(response(body, true, true)),
            Err(_) => Ok(response(
                error_response(6, TLV_ERROR_AUTHENTICATION),
                false,
                true,
            )),
        }
    }

    fn finish_m5(&self, tlv: &Tlv8, session_key: &[u8]) -> Result<Vec<u8>, HapError> {
        let encrypted = required_bounded(
            tlv,
            TLV_ENCRYPTED_DATA,
            17,
            4096,
            "Pair-Setup encrypted data",
        )?;
        let encryption_key = hkdf_sha512(
            b"Pair-Setup-Encrypt-Salt",
            session_key,
            b"Pair-Setup-Encrypt-Info",
        )?;
        let mut plaintext = Zeroizing::new(open_labeled(&encryption_key, b"PS-Msg05", encrypted)?);
        let sub_tlv = Tlv8::parse(&plaintext)?;
        let controller_id_bytes =
            required_bounded(&sub_tlv, TLV_IDENTIFIER, 1, 64, "controller identifier")?;
        let controller_id = std::str::from_utf8(controller_id_bytes)
            .map_err(|_| HapError::Protocol("controller identifier is not UTF-8".into()))?
            .to_owned();
        let controller_key: [u8; 32] =
            required_bounded(&sub_tlv, TLV_PUBLIC_KEY, 32, 32, "controller LTPK")?
                .try_into()
                .expect("length checked");
        let signature_bytes: [u8; 64] =
            required_bounded(&sub_tlv, TLV_SIGNATURE, 64, 64, "controller signature")?
                .try_into()
                .expect("length checked");
        let verifying_key = VerifyingKey::from_bytes(&controller_key)
            .map_err(|_| HapError::Protocol("controller LTPK is invalid".into()))?;
        let controller_x = hkdf_sha512(
            b"Pair-Setup-Controller-Sign-Salt",
            session_key,
            b"Pair-Setup-Controller-Sign-Info",
        )?;
        let mut controller_info =
            Zeroizing::new(Vec::with_capacity(32 + controller_id_bytes.len() + 32));
        controller_info.extend_from_slice(&controller_x);
        controller_info.extend_from_slice(controller_id_bytes);
        controller_info.extend_from_slice(&controller_key);
        verifying_key
            .verify_strict(&controller_info, &Signature::from_bytes(&signature_bytes))
            .map_err(|_| HapError::Protocol("controller signature rejected".into()))?;

        let signing_key = self.store.signing_key()?;
        let accessory_id = self.store.accessory_id()?;
        let accessory_public = signing_key.verifying_key().to_bytes();
        let accessory_x = hkdf_sha512(
            b"Pair-Setup-Accessory-Sign-Salt",
            session_key,
            b"Pair-Setup-Accessory-Sign-Info",
        )?;
        let mut accessory_info = Zeroizing::new(Vec::with_capacity(32 + accessory_id.len() + 32));
        accessory_info.extend_from_slice(&accessory_x);
        accessory_info.extend_from_slice(accessory_id.as_bytes());
        accessory_info.extend_from_slice(&accessory_public);
        let accessory_signature = signing_key.sign(&accessory_info).to_bytes();
        let mut response_plaintext = Zeroizing::new(encode_items([
            (TLV_IDENTIFIER, accessory_id.as_bytes()),
            (TLV_PUBLIC_KEY, accessory_public.as_slice()),
            (TLV_SIGNATURE, accessory_signature.as_slice()),
        ]));
        let response_encrypted = seal_labeled(&encryption_key, b"PS-Msg06", &response_plaintext)?;

        // Commit only after every authentication and response construction
        // step has succeeded, and before emitting success-shaped M6.
        self.store.add_initial(ControllerPairing {
            controller_id,
            public_key: controller_key,
            admin: true,
        })?;
        plaintext.zeroize();
        response_plaintext.zeroize();
        Ok(encode_items([
            (TLV_STATE, [6].as_slice()),
            (TLV_ENCRYPTED_DATA, response_encrypted.as_slice()),
        ]))
    }

    fn release_slot(&mut self) {
        if self.owns_global_slot {
            self.store.end_pair_setup();
            self.owns_global_slot = false;
        }
    }
}

impl Drop for PairSetup {
    fn drop(&mut self) {
        self.release_slot();
    }
}

fn response(body: Vec<u8>, paired: bool, terminal: bool) -> PairSetupResponse {
    PairSetupResponse {
        body,
        paired,
        terminal,
    }
}

fn required_bounded<'a>(
    tlv: &'a Tlv8,
    kind: u8,
    min: usize,
    max: usize,
    name: &str,
) -> Result<&'a [u8], HapError> {
    let value = tlv
        .get(kind)
        .ok_or_else(|| HapError::Protocol(format!("missing {name}")))?;
    if !(min..=max).contains(&value.len()) {
        return Err(HapError::Protocol(format!(
            "{name} must contain {min}..={max} bytes"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use srp::ClientG3072;

    fn setup() -> (tempfile::TempDir, Arc<PairingStore>) {
        let directory = tempfile::tempdir().unwrap();
        let store = PairingStore::create(
            directory.path().join("pairings.json"),
            crate::pairing::SetupCode::parse("518-26-003").unwrap(),
            Some("AA:BB:CC:DD:EE:FF".into()),
        )
        .unwrap();
        (directory, Arc::new(store))
    }

    #[test]
    fn apple_hap_srp_3072_sha512_session_key_vector() {
        let decode = |value: &str| {
            value
                .as_bytes()
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect::<Vec<_>>()
        };
        let salt = decode("BEB25379D1A8581EB5A727673A2441EE");
        let a = decode("60975527035CF2AD1989806F0407210BC81EDC04E2762A56AFD529DDDA2D4393");
        let b = decode("E487CB59D31AC550471E81F00F6928E01DDA08E974A004F49E61F5D105284D20");
        let expected_key = decode(
            "5CBC219DB052138EE1148C71CD4498963D682549CE91CA24F098468F06015BEB\
             6AF245C2093F98C3651BCA83AB8CAB2B580BBF02184FEFDF26142F73DF95AC50",
        );
        let client = ClientG3072::<Sha512>::new();
        let server = ServerG3072::<Sha512>::new_with_options(false);
        let verifier = client.compute_verifier(b"alice", b"password123", &salt);
        let a_public = client.compute_public_ephemeral(&a);
        let b_public = server.compute_public_ephemeral(&b, &verifier);
        let client_state = client
            .process_reply(&a, b"alice", b"password123", &salt, &b_public)
            .unwrap();
        let server_state = server
            .process_reply(b"alice", &salt, &b, &verifier, &a_public)
            .unwrap();
        assert_eq!(
            server_state.verify_client(client_state.proof()).unwrap(),
            expected_key
        );
        assert_eq!(
            client_state.verify_server(server_state.proof()).unwrap(),
            expected_key
        );
    }

    #[test]
    fn full_m1_through_m6_persists_only_authenticated_controller() {
        let (_directory, store) = setup();
        let mut server = PairSetup::new(store.clone());
        let m1 = encode_items([(TLV_STATE, [1].as_slice()), (TLV_METHOD, [0].as_slice())]);
        let m2 = Tlv8::parse(&server.handle(&m1).unwrap().body).unwrap();
        let salt = m2.get(TLV_SALT).unwrap();
        let server_public = m2.get(TLV_PUBLIC_KEY).unwrap();
        let client = ClientG3072::<Sha512>::new();
        let client_secret = [0x31; 48];
        let client_state = client
            .process_reply(
                &client_secret,
                SRP_USERNAME,
                b"518-26-003",
                salt,
                server_public,
            )
            .unwrap();
        let client_public = client.compute_public_ephemeral(&client_secret);
        let m3 = encode_items([
            (TLV_STATE, [3].as_slice()),
            (TLV_PUBLIC_KEY, client_public.as_slice()),
            (TLV_PROOF, client_state.proof()),
        ]);
        let m4 = Tlv8::parse(&server.handle(&m3).unwrap().body).unwrap();
        let session_key = client_state
            .verify_server(m4.get(TLV_PROOF).unwrap())
            .unwrap();

        let controller_signing = SigningKey::from_bytes(&[0x22; 32]);
        let controller_public = controller_signing.verifying_key().to_bytes();
        let controller_id = b"deterministic-controller";
        let controller_x = hkdf_sha512(
            b"Pair-Setup-Controller-Sign-Salt",
            session_key,
            b"Pair-Setup-Controller-Sign-Info",
        )
        .unwrap();
        let mut info = Vec::new();
        info.extend_from_slice(&controller_x);
        info.extend_from_slice(controller_id);
        info.extend_from_slice(&controller_public);
        let signature = controller_signing.sign(&info).to_bytes();
        let sub_tlv = encode_items([
            (TLV_IDENTIFIER, controller_id.as_slice()),
            (TLV_PUBLIC_KEY, controller_public.as_slice()),
            (TLV_SIGNATURE, signature.as_slice()),
        ]);
        let encryption_key = hkdf_sha512(
            b"Pair-Setup-Encrypt-Salt",
            session_key,
            b"Pair-Setup-Encrypt-Info",
        )
        .unwrap();
        let encrypted = seal_labeled(&encryption_key, b"PS-Msg05", &sub_tlv).unwrap();
        let m5 = encode_items([
            (TLV_STATE, [5].as_slice()),
            (TLV_ENCRYPTED_DATA, encrypted.as_slice()),
        ]);
        let result = server.handle(&m5).unwrap();
        assert!(result.paired);
        let m6 = Tlv8::parse(&result.body).unwrap();
        assert!(open_labeled(
            &encryption_key,
            b"PS-Msg06",
            m6.get(TLV_ENCRYPTED_DATA).unwrap()
        )
        .is_ok());
        assert_eq!(
            store
                .get("deterministic-controller")
                .unwrap()
                .unwrap()
                .public_key,
            controller_public
        );
    }

    #[test]
    fn malformed_replayed_and_wrong_proof_requests_fail_closed() {
        let (_directory, store) = setup();
        let mut server = PairSetup::new(store.clone());
        let m1 = encode_items([(TLV_STATE, [1].as_slice()), (TLV_METHOD, [0].as_slice())]);
        assert!(!server.handle(&m1).unwrap().paired);
        assert!(server.handle(&m1).is_err());
        let bad_m3 = encode_items([
            (TLV_STATE, [3].as_slice()),
            (TLV_PUBLIC_KEY, [1].as_slice()),
            (TLV_PROOF, [0u8; 64].as_slice()),
        ]);
        let response = Tlv8::parse(&server.handle(&bad_m3).unwrap().body).unwrap();
        assert_eq!(
            response.byte(crate::protocol::TLV_ERROR),
            Some(TLV_ERROR_AUTHENTICATION)
        );
        assert!(!store.is_paired().unwrap());
    }
}
