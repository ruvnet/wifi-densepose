//! Service registry stub.
//!
//! Mirrors `homeassistant.core.ServiceRegistry`. P1 ships the public
//! surface + a simple direct-dispatch `call` so downstream ADRs can
//! depend on it; ADR-127 P2 replaces direct dispatch with the
//! mpsc-router pattern described in §2.3.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use crate::event::Context;

/// Service name within a domain. e.g. `light.turn_on` → domain
/// `"light"`, service `"turn_on"`.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ServiceName {
    pub domain: String,
    pub service: String,
}

impl ServiceName {
    pub fn new(domain: impl Into<String>, service: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            service: service.into(),
        }
    }
}

/// Inbound service-call payload. Mirrors HA's `service_data` dict
/// plus the originating `Context`.
#[derive(Clone, Debug)]
pub struct ServiceCall {
    pub name: ServiceName,
    pub data: serde_json::Value,
    pub context: Context,
}

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("service not registered: {domain}.{service}")]
    NotRegistered { domain: String, service: String },
    #[error("service handler returned error: {0}")]
    HandlerFailed(String),
}

/// Handler trait. Integration code implements this and registers via
/// [`ServiceRegistry::register`]. P2 will add schema validation via
/// `serde` `Deserialize<'_>`.
#[async_trait]
pub trait ServiceHandler: Send + Sync + 'static {
    async fn call(&self, call: ServiceCall) -> Result<serde_json::Value, ServiceError>;
}

/// Direct closure adapter so simple handlers don't need a struct.
pub struct FnHandler<F>(pub F);

#[async_trait]
impl<F, Fut> ServiceHandler for FnHandler<F>
where
    F: Fn(ServiceCall) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<serde_json::Value, ServiceError>> + Send + 'static,
{
    async fn call(&self, call: ServiceCall) -> Result<serde_json::Value, ServiceError> {
        (self.0)(call).await
    }
}

#[derive(Clone)]
pub struct ServiceRegistry {
    handlers: Arc<RwLock<HashMap<ServiceName, Arc<dyn ServiceHandler>>>>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register<H: ServiceHandler>(&self, name: ServiceName, handler: H) {
        self.handlers.write().await.insert(name, Arc::new(handler));
    }

    pub async fn remove(&self, name: &ServiceName) {
        self.handlers.write().await.remove(name);
    }

    pub async fn has(&self, name: &ServiceName) -> bool {
        self.handlers.read().await.contains_key(name)
    }

    /// Call a service. P1 direct dispatch; P2 routes through the
    /// event bus per ADR-127 §2.3.
    pub async fn call(&self, call: ServiceCall) -> Result<serde_json::Value, ServiceError> {
        let handler = {
            let guard = self.handlers.read().await;
            guard.get(&call.name).cloned()
        };
        match handler {
            Some(h) => h.call(call).await,
            None => Err(ServiceError::NotRegistered {
                domain: call.name.domain.clone(),
                service: call.name.service.clone(),
            }),
        }
    }

    pub async fn registered_services(&self) -> Vec<ServiceName> {
        self.handlers.read().await.keys().cloned().collect()
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Suppress unused-import warning when no consumer of Pin/Box uses them yet
#[allow(dead_code)]
type _UnusedFutureType = Pin<Box<dyn Future<Output = ()> + Send>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_and_call_returns_handler_value() {
        let reg = ServiceRegistry::new();
        reg.register(
            ServiceName::new("light", "turn_on"),
            FnHandler(|call: ServiceCall| async move {
                Ok(serde_json::json!({"called_with": call.data}))
            }),
        )
        .await;

        let resp = reg
            .call(ServiceCall {
                name: ServiceName::new("light", "turn_on"),
                data: serde_json::json!({"brightness": 200}),
                context: Context::new(),
            })
            .await
            .unwrap();
        assert_eq!(resp["called_with"]["brightness"], 200);
    }

    #[tokio::test]
    async fn unregistered_service_returns_error() {
        let reg = ServiceRegistry::new();
        let err = reg
            .call(ServiceCall {
                name: ServiceName::new("light", "turn_on"),
                data: serde_json::json!({}),
                context: Context::new(),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ServiceError::NotRegistered { .. }));
    }
}
