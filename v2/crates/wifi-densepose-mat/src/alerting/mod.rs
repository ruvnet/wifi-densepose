//! Alerting module for emergency notifications.

mod dispatcher;
mod generator;
mod triage_service;

pub use dispatcher::{AlertConfig, AlertDispatcher};
pub use generator::AlertGenerator;
pub use triage_service::{PriorityCalculator, TriageService};
