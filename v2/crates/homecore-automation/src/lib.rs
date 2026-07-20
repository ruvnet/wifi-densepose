//! homecore-automation — ADR-129 HOMECORE-AUTO
//!
//! Automation engine, trigger evaluator, MiniJinja template evaluator, and
//! script action executor for the HOMECORE Home Assistant port.
//!
//! ## Layout
//!
//! - [`automation`] — `Automation` struct: id, alias, mode, triggers, conditions, actions
//! - [`trigger`] — `Trigger` enum + `EvaluateTrigger` trait
//! - [`condition`] — `Condition` enum + async `evaluate` method + `EvalContext`
//! - [`action`] — `Action` enum + async `execute` method + `ExecutionContext`
//! - [`template`] — MiniJinja environment with HA-compat globals (states, state_attr, is_state, now)
//! - [`engine`] — `AutomationEngine`: subscribes to event bus, drives trigger→condition→action pipeline
//! - [`error`] — crate-wide `AutomationError`

pub mod automation;
pub mod trigger;
pub mod condition;
pub mod action;
pub mod template;
pub mod engine;
pub mod runmode;
pub mod error;

pub use automation::{Automation, RunMode};
pub use trigger::{EvaluateTrigger, Trigger, TriggerContext};
pub use condition::{Condition, EvalContext};
pub use action::{Action, ExecutionContext};
pub use template::TemplateEnvironment;
pub use engine::AutomationEngine;
pub use error::AutomationError;
