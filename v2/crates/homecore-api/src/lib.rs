//! HOMECORE-API — wire-compat Axum REST + WebSocket port of HA's API (ADR-130).
pub mod app;
pub mod auth;
pub mod error;
pub mod rest;
pub mod state;
pub mod tokens;
pub mod ws;

pub use app::{router, AppState};
pub use error::{ApiError, ApiResult};
pub use state::SharedState;
pub use tokens::LongLivedTokenStore;

pub const DEFAULT_PORT: u16 = 8123;
