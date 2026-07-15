//! Responder-side setup registry for the 802.11bf sensing model — enforces
//! the setup-ID-collision and capacity rejection paths a single session
//! cannot see on its own (ADR-153 acceptance: duplicate setup ID rejected).
//! Both entry points — direct setups ([`SessionTable::handle_setup_request`])
//! and sensing-by-proxy ([`SessionTable::handle_sbp_request`]) — share the
//! same guards and the same per-setup session storage.

use std::collections::BTreeMap;

use super::messages::{
    SbpRequest, SbpResponse, SbpStatus, SensingMeasurementSetupRequest,
    SensingMeasurementSetupResponse,
};
use super::session::{Action, SensingSession, SessionConfig, SessionEvent, SessionState};
use super::types::{BfError, MeasurementSetupId, SetupStatus};

/// Responder-side registry of sensing sessions keyed by setup ID.
///
/// Enforces the setup-ID-collision and capacity rejection paths the single
/// session cannot see on its own.
#[derive(Debug)]
pub struct SessionTable {
    config: SessionConfig,
    sessions: BTreeMap<u8, SensingSession>,
    /// Events dropped because no session owned the setup ID (see
    /// [`Self::handle_for`]).
    unknown_setup_drops: u64,
}

impl SessionTable {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            sessions: BTreeMap::new(),
            unknown_setup_drops: 0,
        }
    }

    /// Number of setups not in Idle.
    pub fn active_setups(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state() != SessionState::Idle)
            .count()
    }

    pub fn session(&self, setup_id: MeasurementSetupId) -> Option<&SensingSession> {
        self.sessions.get(&setup_id.value())
    }

    /// Count of events dropped by [`Self::handle_for`] because the setup ID
    /// was unknown — lets an AP spot peers addressing setups it never
    /// accepted without turning stray frames into errors.
    pub fn unknown_setup_drops(&self) -> u64 {
        self.unknown_setup_drops
    }

    /// Route an inbound setup request, rejecting setup-ID collisions and
    /// capacity overruns before delegating to a responder session.
    pub fn handle_setup_request(
        &mut self,
        req: SensingMeasurementSetupRequest,
    ) -> Result<Vec<Action>, BfError> {
        let reject = |setup_id, status| {
            Ok(vec![Action::SendSetupResponse(
                SensingMeasurementSetupResponse { setup_id, status },
            )])
        };
        if self.is_collision(req.setup_id) {
            return reject(req.setup_id, SetupStatus::RejectedSetupIdCollision);
        }
        if self.at_capacity() {
            return reject(req.setup_id, SetupStatus::RejectedCapacity);
        }
        let key = req.setup_id.value();
        let mut session = SensingSession::new_responder(self.config.clone());
        let actions = session.handle(SessionEvent::SetupRequestReceived(req))?;
        self.sessions.insert(key, session);
        Ok(actions)
    }

    /// Route an inbound SBP request, rejecting proxy-setup-ID collisions and
    /// capacity overruns before delegating to a (new) proxy session — the
    /// SBP mirror of [`Self::handle_setup_request`], so a table-driven AP
    /// accepts SBP end-to-end instead of silently dropping it.
    pub fn handle_sbp_request(&mut self, sbp: SbpRequest) -> Result<Vec<Action>, BfError> {
        let reject = |proxy_setup_id, status| {
            Ok(vec![Action::SendSbpResponse(SbpResponse {
                proxy_setup_id,
                status,
            })])
        };
        if self.is_collision(sbp.proxy_setup_id) {
            return reject(sbp.proxy_setup_id, SbpStatus::RejectedSetupIdCollision);
        }
        if self.at_capacity() {
            return reject(sbp.proxy_setup_id, SbpStatus::RejectedCapacity);
        }
        let key = sbp.proxy_setup_id.value();
        let mut session = SensingSession::new_responder(self.config.clone());
        let actions = session.handle(SessionEvent::SbpRequestReceived(sbp))?;
        self.sessions.insert(key, session);
        Ok(actions)
    }

    /// Route any other event to the session owning `setup_id`.
    ///
    /// Frames addressing an unknown setup are dropped *by design* (stray
    /// frames are ignored, not errors), but the drop is observable through
    /// [`Self::unknown_setup_drops`].
    pub fn handle_for(
        &mut self,
        setup_id: MeasurementSetupId,
        event: SessionEvent,
    ) -> Result<Vec<Action>, BfError> {
        match self.sessions.get_mut(&setup_id.value()) {
            Some(session) => session.handle(event),
            None => {
                self.unknown_setup_drops = self.unknown_setup_drops.saturating_add(1);
                Ok(vec![])
            }
        }
    }

    /// A non-Idle session already owns this setup ID.
    fn is_collision(&self, setup_id: MeasurementSetupId) -> bool {
        self.sessions
            .get(&setup_id.value())
            .is_some_and(|existing| existing.state() != SessionState::Idle)
    }

    /// The active-setup budget is exhausted.
    fn at_capacity(&self) -> bool {
        self.active_setups() >= self.config.capabilities.max_active_setups as usize
    }
}
