use homecore::HomeCore;
use std::sync::Arc;

use crate::tokens::LongLivedTokenStore;

#[derive(Clone)]
pub struct SharedState {
    inner: Arc<SharedStateInner>,
}

struct SharedStateInner {
    pub homecore: HomeCore,
    pub homecore_version: String,
    pub location_name: String,
    pub tokens: LongLivedTokenStore,
}

impl SharedState {
    /// New SharedState with a default empty token store. Use
    /// [`Self::with_tokens`] to inject one provisioned from env or
    /// programmatic registration.
    pub fn new(homecore: HomeCore) -> Self {
        Self::with_metadata(homecore, "Home", env!("CARGO_PKG_VERSION"))
    }

    pub fn with_metadata(
        homecore: HomeCore,
        location_name: impl Into<String>,
        homecore_version: impl Into<String>,
    ) -> Self {
        // Fail closed by default. Tests and explicitly insecure local
        // development must opt into `allow_any_non_empty()` themselves.
        Self::with_tokens(
            homecore,
            location_name,
            homecore_version,
            LongLivedTokenStore::empty(),
        )
    }

    pub fn with_tokens(
        homecore: HomeCore,
        location_name: impl Into<String>,
        homecore_version: impl Into<String>,
        tokens: LongLivedTokenStore,
    ) -> Self {
        Self {
            inner: Arc::new(SharedStateInner {
                homecore,
                homecore_version: homecore_version.into(),
                location_name: location_name.into(),
                tokens,
            }),
        }
    }

    pub fn homecore(&self) -> &HomeCore {
        &self.inner.homecore
    }
    pub fn version(&self) -> &str {
        &self.inner.homecore_version
    }
    pub fn location_name(&self) -> &str {
        &self.inner.location_name
    }
    pub fn tokens(&self) -> &LongLivedTokenStore {
        &self.inner.tokens
    }
}
