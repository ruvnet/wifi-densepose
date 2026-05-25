//! `RecorderListener` — subscribes to `StateMachine` broadcasts and writes
//! every `StateChangedEvent` to the `Recorder`.
//!
//! Spawned via `tokio::spawn`. Runs until the broadcast sender is dropped
//! (i.e. the `StateMachine` is shut down) or until a `Lagged` error occurs
//! (subscriber fell more than 4,096 events behind).
//!
//! On `Lagged`, the listener logs a warning and reconnects; it does not crash
//! because dropping a listener would silently stop persistence.
//!
//! ## Subscription ordering
//!
//! The `broadcast::Receiver` is created inside `new()` (not inside the spawned
//! task), so any events fired between `new()` and `spawn()` are enqueued in
//! the receiver buffer and will be drained when the task starts.

use tokio::sync::broadcast;
use tracing::{debug, warn};

use homecore::event::StateChangedEvent;
use homecore::state::StateMachine;

use crate::db::Recorder;

/// A background task that records every state change.
///
/// Call [`RecorderListener::new`] then [`RecorderListener::spawn`].
/// The subscription starts at construction time so no events are missed
/// between `new()` and `spawn()`.
pub struct RecorderListener {
    recorder: Recorder,
    rx: broadcast::Receiver<StateChangedEvent>,
}

impl RecorderListener {
    /// Create a listener. Subscribes to the broadcast channel immediately so
    /// events fired before `spawn()` are buffered in the receiver.
    pub fn new(state_machine: &StateMachine, recorder: Recorder) -> Self {
        let rx = state_machine.subscribe();
        Self { recorder, rx }
    }

    /// Spawn the listener onto the Tokio runtime.
    ///
    /// Returns a `JoinHandle`. Abort it on graceful shutdown:
    /// ```ignore
    /// let handle = listener.spawn();
    /// // … on shutdown:
    /// handle.abort();
    /// ```
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move { self.run().await })
    }

    async fn run(mut self) {
        loop {
            match self.rx.recv().await {
                Ok(event) => {
                    debug!(entity_id = %event.entity_id, "recording state change");
                    if let Err(e) = self.recorder.record_state(&event).await {
                        warn!(error = %e, "failed to record state change");
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(
                        lagged_by = n,
                        "recorder listener lagged — some state changes were not persisted"
                    );
                    // Continue processing from the next available event.
                }
                Err(broadcast::error::RecvError::Closed) => {
                    debug!("state machine shut down; recorder listener exiting");
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use homecore::entity::EntityId;
    use homecore::event::Context;

    fn eid(s: &str) -> EntityId {
        EntityId::parse(s).unwrap()
    }

    #[tokio::test]
    async fn listener_records_state_changes() {
        let sm = StateMachine::new();
        let recorder = Recorder::open("sqlite::memory:").await.unwrap();

        let listener = RecorderListener::new(&sm, recorder.clone());
        let _handle = listener.spawn();

        // Fire two state changes.
        sm.set(eid("light.hall"), "on", serde_json::json!({}), Context::new());
        sm.set(eid("light.hall"), "off", serde_json::json!({}), Context::new());

        // Give the background task a moment to flush.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let since = chrono::Utc::now() - chrono::Duration::seconds(10);
        let until = chrono::Utc::now() + chrono::Duration::seconds(10);
        let rows = recorder
            .get_state_history(&eid("light.hall"), since, until)
            .await
            .unwrap();

        assert_eq!(rows.len(), 2, "listener must have persisted both events");
        assert_eq!(rows[0].state, "on");
        assert_eq!(rows[1].state, "off");
    }
}
