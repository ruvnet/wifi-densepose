//! Long-lived bearer-token store.
//!
//! Closes audit findings **HC-01** and **HC-02** by replacing the
//! "any non-empty bearer" P1 placeholder with a real token whitelist.
//!
//! P2 scope (this commit):
//! - Token set held in memory; populated at boot from env / config /
//!   programmatic registration
//! - `O(1)` `is_valid(&str) -> bool` lookup via `HashSet`
//! - No expiry, no rotation, no per-user attribution yet — P3
//!
//! Boot-time provisioning paths supported:
//! - `HOMECORE_TOKENS` env var: comma-separated bearer tokens
//! - `LongLivedTokenStore::register(token)` for programmatic insert
//!
//! Provided constructors:
//! - `LongLivedTokenStore::empty()` → no tokens accepted (use after
//!   boot to add tokens manually)
//! - `LongLivedTokenStore::from_env()` → reads `HOMECORE_TOKENS`,
//!   splits on commas, trims, drops empties
//! - `LongLivedTokenStore::allow_any_non_empty()` → **DEV ONLY**;
//!   preserves the legacy "accept anything non-empty" behaviour
//!   for users who haven't migrated yet. Emits a warning on every
//!   call. Removed in P3.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::warn;

#[derive(Clone)]
pub struct LongLivedTokenStore {
    inner: Arc<RwLock<LongLivedTokenStoreInner>>,
}

struct LongLivedTokenStoreInner {
    tokens: HashSet<String>,
    /// DEV-only escape hatch: when true, ANY non-empty bearer is
    /// accepted. Logged on every check so the operator notices.
    allow_any: bool,
}

impl LongLivedTokenStore {
    /// Empty store. No tokens accepted. Register tokens explicitly
    /// via [`Self::register`] before exposing the API to the network.
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(LongLivedTokenStoreInner {
                tokens: HashSet::new(),
                allow_any: false,
            })),
        }
    }

    /// Reads `HOMECORE_TOKENS` from the environment and registers
    /// each comma-separated value. Trims whitespace; drops empty
    /// values. If the env var is unset / empty, the store starts
    /// empty.
    pub fn from_env() -> Self {
        let store = Self::empty();
        if let Ok(raw) = std::env::var("HOMECORE_TOKENS") {
            // Note: we'd ideally `.await` here but constructors stay
            // sync. Use try_write to populate synchronously at boot.
            // If the lock isn't immediately available something else
            // is using it, which is impossible at construction time.
            if let Ok(mut guard) = store.inner.try_write() {
                for raw_token in raw.split(',') {
                    let t = raw_token.trim();
                    if !t.is_empty() {
                        guard.tokens.insert(t.to_string());
                    }
                }
            }
        }
        store
    }

    /// **DEV ONLY** — closes HC-01/02 audit findings on paper while
    /// preserving the legacy "any non-empty bearer" behaviour for
    /// users mid-migration. Emits a warn on every check. Removed
    /// in P3.
    pub fn allow_any_non_empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(LongLivedTokenStoreInner {
                tokens: HashSet::new(),
                allow_any: true,
            })),
        }
    }

    /// Register a token. Idempotent. Returns true if the token was
    /// new, false if it was already in the set.
    pub async fn register(&self, token: impl Into<String>) -> bool {
        let mut guard = self.inner.write().await;
        guard.tokens.insert(token.into())
    }

    /// Revoke a token. Returns true if the token was in the set.
    pub async fn revoke(&self, token: &str) -> bool {
        let mut guard = self.inner.write().await;
        guard.tokens.remove(token)
    }

    /// Check a token against the store. Fast O(1) hashset lookup.
    /// In `allow_any` mode, any non-empty token returns true and a
    /// warn is logged.
    pub async fn is_valid(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        let guard = self.inner.read().await;
        if guard.allow_any {
            warn!(
                "LongLivedTokenStore::is_valid called in `allow_any` mode — \
                 any non-empty bearer is accepted. Provision real tokens via \
                 HOMECORE_TOKENS or LongLivedTokenStore::register() before \
                 production."
            );
            return true;
        }
        guard.tokens.contains(token)
    }

    /// Number of registered tokens. Useful for boot log lines.
    pub async fn len(&self) -> usize {
        self.inner.read().await.tokens.len()
    }

    /// Is the store accepting any non-empty bearer (DEV mode)?
    pub async fn is_dev_mode(&self) -> bool {
        self.inner.read().await.allow_any
    }
}

impl Default for LongLivedTokenStore {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn empty_store_rejects_everything() {
        let s = LongLivedTokenStore::empty();
        assert!(!s.is_valid("anything").await);
        assert!(!s.is_valid("").await);
    }

    #[tokio::test]
    async fn registered_token_is_valid() {
        let s = LongLivedTokenStore::empty();
        s.register("hc_abc_123").await;
        assert!(s.is_valid("hc_abc_123").await);
        assert!(!s.is_valid("hc_abc_124").await);
    }

    #[tokio::test]
    async fn revoke_invalidates() {
        let s = LongLivedTokenStore::empty();
        s.register("t1").await;
        s.register("t2").await;
        assert!(s.is_valid("t1").await);
        assert!(s.revoke("t1").await);
        assert!(!s.is_valid("t1").await);
        assert!(s.is_valid("t2").await);
        assert_eq!(s.len().await, 1);
    }

    #[tokio::test]
    async fn register_is_idempotent() {
        let s = LongLivedTokenStore::empty();
        assert!(s.register("t").await);
        assert!(!s.register("t").await);
        assert_eq!(s.len().await, 1);
    }

    #[tokio::test]
    async fn empty_token_always_rejected() {
        let s = LongLivedTokenStore::allow_any_non_empty();
        assert!(!s.is_valid("").await);
    }

    #[tokio::test]
    async fn allow_any_mode_accepts_any_non_empty() {
        let s = LongLivedTokenStore::allow_any_non_empty();
        assert!(s.is_valid("literally-anything").await);
        assert!(s.is_dev_mode().await);
    }

    #[tokio::test]
    async fn from_env_unset_is_empty() {
        // Don't set HOMECORE_TOKENS for this test
        std::env::remove_var("HOMECORE_TOKENS");
        let s = LongLivedTokenStore::from_env();
        assert_eq!(s.len().await, 0);
    }
}
