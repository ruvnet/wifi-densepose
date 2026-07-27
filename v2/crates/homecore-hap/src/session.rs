//! Per-connection HAP authentication state.
//!
//! The transition to [`SessionState::Authenticated`] requires an Ed25519
//! signature from a persisted controller. The caller is responsible for
//! supplying the exact Pair-Verify transcript once X25519/HKDF/ChaCha20-
//! Poly1305 transport is implemented; this crate never substitutes a bearer
//! token or plaintext shortcut.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

use crate::error::HapError;
use crate::pairing::PairingStore;

/// Authentication phase of one TCP connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    Connected,
    PairSetup,
    PairVerify,
    Authenticated { controller_id: String, admin: bool },
    Closing,
}

impl SessionState {
    fn name(&self) -> &'static str {
        match self {
            Self::Connected => "connected",
            Self::PairSetup => "pair-setup",
            Self::PairVerify => "pair-verify",
            Self::Authenticated { .. } => "authenticated",
            Self::Closing => "closing",
        }
    }

    /// Whether accessory, characteristic, event, and pairing-management
    /// endpoints may be processed.
    pub fn is_authenticated(&self) -> bool {
        matches!(self, Self::Authenticated { .. })
    }
}

/// Fail-closed state machine for one HAP connection.
#[derive(Debug, Clone)]
pub struct Session {
    state: SessionState,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            state: SessionState::Connected,
        }
    }

    pub fn state(&self) -> &SessionState {
        &self.state
    }

    pub fn begin_pair_setup(&mut self, already_paired: bool) -> Result<(), HapError> {
        if already_paired {
            return Err(HapError::Protocol(
                "Pair-Setup is unavailable after a controller is paired".into(),
            ));
        }
        self.transition(SessionState::PairSetup)
    }

    pub fn begin_pair_verify(&mut self) -> Result<(), HapError> {
        self.transition(SessionState::PairVerify)
    }

    /// Authenticate a completed Pair-Verify transcript with the controller's
    /// persisted Ed25519 long-term public key.
    ///
    /// This method is intentionally not called by the current network server:
    /// the encrypted Pair-Verify transcript is not implemented yet.
    pub fn authenticate_pair_verify(
        &mut self,
        controller_id: &str,
        signed_transcript: &[u8],
        signature: &[u8; 64],
        pairings: &PairingStore,
    ) -> Result<(), HapError> {
        if !matches!(self.state, SessionState::PairVerify) {
            return Err(HapError::InvalidSessionTransition {
                from: self.state.name(),
                to: "authenticated",
            });
        }
        let pairing = pairings
            .get(controller_id)?
            .ok_or_else(|| HapError::PairingNotFound(controller_id.to_owned()))?;
        let key = VerifyingKey::from_bytes(&pairing.public_key)
            .map_err(|_| HapError::InvalidPairingRecord("invalid Ed25519 public key".into()))?;
        let signature = Signature::from_bytes(signature);
        key.verify(signed_transcript, &signature)
            .map_err(|_| HapError::Protocol("Pair-Verify controller signature rejected".into()))?;
        self.state = SessionState::Authenticated {
            controller_id: pairing.controller_id,
            admin: pairing.admin,
        };
        Ok(())
    }

    pub fn reset_pairing(&mut self) -> Result<(), HapError> {
        match self.state {
            SessionState::PairSetup | SessionState::PairVerify => {
                self.state = SessionState::Connected;
                Ok(())
            }
            _ => Err(HapError::InvalidSessionTransition {
                from: self.state.name(),
                to: "connected",
            }),
        }
    }

    pub fn close(&mut self) {
        self.state = SessionState::Closing;
    }

    #[cfg(all(test, feature = "hap-server"))]
    pub(crate) fn authenticated_for_test(admin: bool) -> Self {
        Self {
            state: SessionState::Authenticated {
                controller_id: "test-controller".into(),
                admin,
            },
        }
    }

    fn transition(&mut self, next: SessionState) -> Result<(), HapError> {
        if matches!(self.state, SessionState::Connected) {
            self.state = next;
            Ok(())
        } else {
            Err(HapError::InvalidSessionTransition {
                from: self.state.name(),
                to: next.name(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairing::ControllerPairing;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn authenticated_transition_requires_persisted_valid_signature() {
        let directory = tempfile::tempdir().unwrap();
        let store = PairingStore::open(directory.path().join("pairings.json")).unwrap();
        let signing_key = SigningKey::from_bytes(&[42; 32]);
        store
            .add(ControllerPairing {
                controller_id: "controller".into(),
                public_key: signing_key.verifying_key().to_bytes(),
                admin: true,
            })
            .unwrap();

        let transcript = b"pair-verify transcript supplied by protocol implementation";
        let signature = signing_key.sign(transcript).to_bytes();
        let mut session = Session::new();
        session.begin_pair_verify().unwrap();
        session
            .authenticate_pair_verify("controller", transcript, &signature, &store)
            .unwrap();
        assert!(session.state().is_authenticated());
    }

    #[test]
    fn bad_signature_and_skipped_verify_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let store = PairingStore::open(directory.path().join("pairings.json")).unwrap();
        let signing_key = SigningKey::from_bytes(&[9; 32]);
        store
            .add(ControllerPairing {
                controller_id: "controller".into(),
                public_key: signing_key.verifying_key().to_bytes(),
                admin: true,
            })
            .unwrap();

        let mut session = Session::new();
        assert!(session
            .authenticate_pair_verify("controller", b"x", &[0; 64], &store)
            .is_err());
        session.begin_pair_verify().unwrap();
        assert!(session
            .authenticate_pair_verify("controller", b"x", &[0; 64], &store)
            .is_err());
        assert!(!session.state().is_authenticated());
    }

    #[test]
    fn pairing_phases_cannot_overlap() {
        let mut session = Session::new();
        session.begin_pair_setup(false).unwrap();
        assert!(session.begin_pair_verify().is_err());
        session.reset_pairing().unwrap();
        session.begin_pair_verify().unwrap();
    }
}
