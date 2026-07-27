//! Per-connection HAP authentication state.
#![cfg_attr(not(feature = "hap-server"), allow(dead_code))]

use crate::crypto::SessionKeys;
use crate::error::HapError;

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

    pub fn is_authenticated(&self) -> bool {
        matches!(self, Self::Authenticated { .. })
    }
}

/// Fail-closed state machine for one HAP connection.
pub struct Session {
    state: SessionState,
    pending_keys: Option<SessionKeys>,
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
            pending_keys: None,
        }
    }

    pub fn state(&self) -> &SessionState {
        &self.state
    }

    pub fn begin_pair_setup(&mut self) -> Result<(), HapError> {
        self.transition(SessionState::PairSetup)
    }

    pub fn begin_pair_verify(&mut self) -> Result<(), HapError> {
        self.transition(SessionState::PairVerify)
    }

    pub(crate) fn authenticate(
        &mut self,
        controller_id: String,
        admin: bool,
        keys: SessionKeys,
    ) -> Result<(), HapError> {
        if !matches!(self.state, SessionState::PairVerify) {
            return Err(HapError::InvalidSessionTransition {
                from: self.state.name(),
                to: "authenticated",
            });
        }
        self.state = SessionState::Authenticated {
            controller_id,
            admin,
        };
        self.pending_keys = Some(keys);
        Ok(())
    }

    pub(crate) fn take_session_keys(&mut self) -> Option<SessionKeys> {
        self.pending_keys.take()
    }

    pub fn reset_pairing(&mut self) -> Result<(), HapError> {
        match self.state {
            SessionState::PairSetup | SessionState::PairVerify => {
                self.state = SessionState::Connected;
                self.pending_keys = None;
                Ok(())
            }
            _ => Err(HapError::InvalidSessionTransition {
                from: self.state.name(),
                to: "connected",
            }),
        }
    }

    pub fn close(&mut self) {
        self.pending_keys = None;
        self.state = SessionState::Closing;
    }

    pub(crate) fn controller_id(&self) -> Option<&str> {
        match &self.state {
            SessionState::Authenticated { controller_id, .. } => Some(controller_id),
            _ => None,
        }
    }

    #[cfg(all(test, feature = "hap-server"))]
    pub(crate) fn authenticated_for_test(admin: bool) -> Self {
        Self {
            state: SessionState::Authenticated {
                controller_id: "test-controller".into(),
                admin,
            },
            pending_keys: None,
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

    #[test]
    fn authentication_requires_pair_verify_and_yields_keys_once() {
        let mut session = Session::new();
        let keys = SessionKeys::derive(&[7; 32]).unwrap();
        assert!(session
            .authenticate("controller".into(), true, keys)
            .is_err());
        session.begin_pair_verify().unwrap();
        session
            .authenticate(
                "controller".into(),
                true,
                SessionKeys::derive(&[7; 32]).unwrap(),
            )
            .unwrap();
        assert!(session.state().is_authenticated());
        assert!(session.take_session_keys().is_some());
        assert!(session.take_session_keys().is_none());
    }

    #[test]
    fn pairing_phases_cannot_overlap() {
        let mut session = Session::new();
        session.begin_pair_setup().unwrap();
        assert!(session.begin_pair_verify().is_err());
        session.reset_pairing().unwrap();
        session.begin_pair_verify().unwrap();
    }
}
