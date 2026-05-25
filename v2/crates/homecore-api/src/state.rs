use std::sync::Arc;
use homecore::HomeCore;

#[derive(Clone)]
pub struct SharedState {
    inner: Arc<SharedStateInner>,
}

struct SharedStateInner {
    pub homecore: HomeCore,
    pub homecore_version: String,
    pub location_name: String,
}

impl SharedState {
    pub fn new(homecore: HomeCore) -> Self {
        Self::with_metadata(homecore, "Home", env!("CARGO_PKG_VERSION"))
    }

    pub fn with_metadata(
        homecore: HomeCore,
        location_name: impl Into<String>,
        homecore_version: impl Into<String>,
    ) -> Self {
        Self {
            inner: Arc::new(SharedStateInner {
                homecore,
                homecore_version: homecore_version.into(),
                location_name: location_name.into(),
            }),
        }
    }

    pub fn homecore(&self) -> &HomeCore { &self.inner.homecore }
    pub fn version(&self) -> &str { &self.inner.homecore_version }
    pub fn location_name(&self) -> &str { &self.inner.location_name }
}
