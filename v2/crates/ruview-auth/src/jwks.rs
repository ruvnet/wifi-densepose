//! JWKS fetch + cache, keyed by `kid`.
//!
//! Ported from `cognitum-one/dashboard` `services/identity/src/jwks.rs` (the
//! same team that signs these tokens), with the unknown-`kid` forced refetch
//! from `meta-llm/src/auth/oauthBearer.ts`. Like both, this is
//! `jsonwebtoken` + `DecodingKey` only — nothing here hand-rolls signature math.
//!
//! ## Offline behaviour is a feature, not an oversight
//!
//! RuView runs on Raspberry-Pi-class hardware that loses WAN. On a refetch
//! failure we keep serving the keys we already have and log a warning, because
//! a signing key that verified a minute ago has not stopped being valid because
//! our network blipped — and failing closed there would log every user out of
//! their own sensing server whenever their internet wobbled.
//!
//! We fail closed in exactly one case: **we have never successfully fetched a
//! key set.** Then there is nothing to reason with, and admitting a request
//! would mean admitting an unverified token.
//!
//! ## The lock is never held across the network call
//!
//! [`JwksCache::decoding_key_for`] reads the cache under the lock, RELEASES it,
//! does any HTTP, then re-takes the lock only to install the result.
//!
//! This is not tidiness. An earlier revision held a `std::sync::Mutex` across a
//! blocking `ureq` call made from inside async middleware. `Mutex::lock()` in an
//! async fn is a real blocking syscall, not a yield point — so one slow or
//! unreachable JWKS fetch (up to the 3s timeout, longer if the link is dead)
//! blocked EVERY concurrent request on that mutex, including requests carrying
//! already-cached, perfectly valid tokens, and parked the tokio worker threads
//! they were running on. On Pi-class hardware with few workers that stalls the
//! whole server, and it fires on the routine 300s TTL rollover whenever the
//! network is degraded — precisely the offline-tolerance case this module
//! exists to handle.
//!
//! The cost of releasing the lock is that two callers can fetch concurrently
//! during a rollover. That is a harmless duplicated idempotent GET, and it is
//! strictly better than serialising every request behind one socket.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use jsonwebtoken::DecodingKey;
use serde::Deserialize;

/// How long a fetched key set is trusted before a routine re-fetch.
/// Identity uses 300 s for the same job; matching it keeps staleness bounded
/// without putting an outbound request on every verify.
pub const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(300);

/// Floor between fetch ATTEMPTS — every attempt, not just the unknown-`kid`
/// forced refetch, and regardless of whether the attempt succeeded.
///
/// Without this, two things go wrong. A stream of tokens bearing a bogus `kid`
/// becomes an outbound request amplifier pointed at the identity service. And,
/// more damagingly, once the cache goes stale (`fetched_at` only advances on
/// success) *every* request performs its own fetch — so a lost WAN link turns
/// into a self-inflicted stall rather than the graceful degradation this module
/// promises.
pub const FETCH_MIN_INTERVAL: Duration = Duration::from_secs(30);

/// Wire timeout for a single JWKS fetch. meta-llm uses 3 s; a verify path must
/// never be able to hang on a slow upstream.
pub const DEFAULT_FETCH_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, thiserror::Error)]
pub enum JwksError {
    #[error("JWKS fetch failed: {0}")]
    Fetch(String),
    #[error("JWKS document malformed: {0}")]
    Malformed(String),
    #[error("JWKS document contained no usable EC keys")]
    NoUsableKeys,
    #[error("no key in JWKS matches kid {0:?}")]
    UnknownKid(String),
    #[error("token header has no kid")]
    MissingKid,
    /// Never fetched successfully — fail closed.
    #[error("JWKS unavailable and no key set has ever been cached")]
    NeverFetched,
}

/// How the key set is retrieved. Abstracted so tests run with no network and so
/// a host that already owns an HTTP client can supply it.
pub trait JwksFetcher: Send + Sync {
    /// Return the raw JWKS document body.
    fn fetch(&self, url: &str) -> Result<String, JwksError>;
}

/// One JWK. Only EC P-256 is accepted: identity signs with ES256 and nothing
/// else, so parsing RSA here would add a key type we would then have to be
/// careful never to verify with.
#[derive(Debug, Deserialize)]
struct Jwk {
    kid: Option<String>,
    kty: String,
    crv: Option<String>,
    x: Option<String>,
    y: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JwksDocument {
    keys: Vec<Jwk>,
}

struct CacheState {
    keys: HashMap<String, DecodingKey>,
    fetched_at: Option<Instant>,
    last_attempt_at: Option<Instant>,
    last_forced_refetch: Option<Instant>,
}

/// `kid`-indexed JWKS cache.
pub struct JwksCache {
    url: String,
    ttl: Duration,
    fetcher: Box<dyn JwksFetcher>,
    state: Mutex<CacheState>,
}

impl JwksCache {
    pub fn new(url: impl Into<String>, fetcher: Box<dyn JwksFetcher>) -> Self {
        Self::with_ttl(url, fetcher, DEFAULT_CACHE_TTL)
    }

    pub fn with_ttl(url: impl Into<String>, fetcher: Box<dyn JwksFetcher>, ttl: Duration) -> Self {
        Self {
            url: url.into(),
            ttl,
            fetcher,
            state: Mutex::new(CacheState {
                keys: HashMap::new(),
                fetched_at: None,
                last_attempt_at: None,
                last_forced_refetch: None,
            }),
        }
    }

    /// Fetch once up front so a misconfigured `jwks_uri` fails at startup rather
    /// than on a user's first request. Call this from server boot: it is what
    /// turns "OAuth is misconfigured" into a refusal to serve instead of a
    /// confusing 401 much later.
    pub fn warm(&self) -> Result<usize, JwksError> {
        let fresh = self.fetch_and_parse()?;
        let n = fresh.len();
        let mut state = self.state.lock().expect("jwks cache poisoned");
        state.keys = fresh;
        state.fetched_at = Some(Instant::now());
        Ok(n)
    }

    /// Resolve the verification key for a token header's `kid`.
    pub fn decoding_key_for(&self, kid: &str) -> Result<DecodingKey, JwksError> {
        // ---- Phase 1: answer from cache, holding the lock only to read. ----
        let (fresh, have_any, may_force, may_attempt, stale_fallback) = {
            let state = self.state.lock().expect("jwks cache poisoned");
            let fresh = state
                .fetched_at
                .map_or(false, |at| at.elapsed() < self.ttl);
            let cached = state.keys.get(kid).cloned();
            // A fresh cache that HAS the key is the overwhelmingly common path
            // and answers without touching anything else.
            if fresh {
                if let Some(key) = cached {
                    return Ok(key);
                }
            }
            let may_force = state
                .last_forced_refetch
                .map_or(true, |at| at.elapsed() >= FETCH_MIN_INTERVAL);
            let may_attempt = state
                .last_attempt_at
                .map_or(true, |at| at.elapsed() >= FETCH_MIN_INTERVAL);
            // When the cache is fresh but lacks this kid there is nothing stale
            // worth serving — identity may have rotated, and a refetch is the
            // whole point. When it is stale, a previously-valid key beats an
            // error if we are rate-limited.
            let stale_fallback = if fresh { None } else { cached };
            (
                fresh,
                state.fetched_at.is_some(),
                may_force,
                may_attempt,
                stale_fallback,
            )
        };
        // Lock released. Everything below may take milliseconds-to-seconds and
        // MUST NOT hold it — see the module docs.

        // TWO independent rate limiters, because they solve different problems.
        // Merging them looks tidy and is wrong: a routine refetch would then
        // suppress the unknown-`kid` path for 30s, delaying pickup of a key
        // rotation that happened inside the TTL.
        if fresh {
            // Fresh cache, unknown kid: identity may have rotated. One forced
            // refetch per floor, so a flood of junk-`kid` tokens cannot become
            // an outbound request amplifier pointed at identity.
            if !may_force {
                return Err(JwksError::UnknownKid(kid.to_owned()));
            }
            self.state
                .lock()
                .expect("jwks cache poisoned")
                .last_forced_refetch = Some(Instant::now());
        } else if !may_attempt {
            // Stale cache and we refetched recently. Serve what we have.
            //
            // This branch is the fix. `fetched_at` advances only on SUCCESS, so
            // once the TTL elapsed after the last successful fetch, `fresh` was
            // permanently false — and the ONLY limiter was gated behind
            // `if fresh`. Every request therefore performed its own blocking
            // fetch. On a Pi that loses WAN, the documented deployment, that
            // turned into a self-inflicted stall 300s after the network went
            // away, with no attacker involved.
            //
            // Serving the stale key is deliberate: one that verified a minute
            // ago has not stopped being valid because our network blipped.
            return match stale_fallback {
                Some(key) => Ok(key),
                None if have_any => Err(JwksError::UnknownKid(kid.to_owned())),
                None => Err(JwksError::NeverFetched),
            };
        }

        // Recorded BEFORE the fetch and regardless of its outcome. Recording it
        // after, or only on success, is precisely the bug described above.
        self.state
            .lock()
            .expect("jwks cache poisoned")
            .last_attempt_at = Some(Instant::now());

        // ---- Phase 2: network, WITHOUT the lock held. ----
        let fetched = self.fetch_and_parse();

        // ---- Phase 3: install, holding the lock only to write. ----
        let mut state = self.state.lock().expect("jwks cache poisoned");
        match fetched {
            Ok(keys) => {
                state.keys = keys;
                state.fetched_at = Some(Instant::now());
            }
            Err(e) => {
                // A key that verified a minute ago has not stopped being valid
                // because the network blipped.
                if !have_any {
                    return Err(JwksError::NeverFetched);
                }
                tracing::warn!(
                    url = %self.url,
                    error = %e,
                    "JWKS refresh failed; continuing with the previously cached key set"
                );
            }
        }
        state
            .keys
            .get(kid)
            .cloned()
            .ok_or_else(|| JwksError::UnknownKid(kid.to_owned()))
    }

    fn fetch_and_parse(&self) -> Result<HashMap<String, DecodingKey>, JwksError> {
        let body = self.fetcher.fetch(&self.url)?;
        parse_jwks(&body)
    }
}

/// Parse a JWKS document into `kid` → `DecodingKey`, skipping entries we cannot
/// or should not use.
fn parse_jwks(body: &str) -> Result<HashMap<String, DecodingKey>, JwksError> {
    let doc: JwksDocument =
        serde_json::from_str(body).map_err(|e| JwksError::Malformed(e.to_string()))?;

    let mut out = HashMap::new();
    for jwk in doc.keys {
        // EC P-256 only. Anything else is skipped rather than rejected, so a
        // future key type appearing in the document does not break verification
        // with the ES256 key sitting next to it.
        if jwk.kty != "EC" {
            tracing::debug!(kty = %jwk.kty, "skipping non-EC JWK");
            continue;
        }
        if jwk.crv.as_deref() != Some("P-256") {
            tracing::debug!(crv = ?jwk.crv, "skipping EC JWK that is not P-256");
            continue;
        }
        let (Some(kid), Some(x), Some(y)) = (jwk.kid, jwk.x, jwk.y) else {
            tracing::debug!("skipping EC JWK missing kid/x/y");
            continue;
        };
        match DecodingKey::from_ec_components(&x, &y) {
            Ok(key) => {
                out.insert(kid, key);
            }
            Err(e) => tracing::debug!(kid = %kid, error = %e, "skipping unparseable EC JWK"),
        }
    }

    if out.is_empty() {
        return Err(JwksError::NoUsableKeys);
    }
    Ok(out)
}

/// Blocking `ureq` transport.
///
/// Blocking on purpose: the sensing server already runs its outbound registry
/// fetch inside `tokio::task::spawn_blocking` for the same reason, and an async
/// client here would pull in a second HTTP stack.
#[cfg(feature = "ureq-transport")]
pub struct UreqFetcher {
    agent: ureq::Agent,
}

#[cfg(feature = "ureq-transport")]
impl UreqFetcher {
    pub fn new() -> Self {
        Self::with_timeout(DEFAULT_FETCH_TIMEOUT)
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout_connect(timeout)
                .timeout_read(timeout)
                .build(),
        }
    }
}

#[cfg(feature = "ureq-transport")]
impl Default for UreqFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "ureq-transport")]
impl JwksFetcher for UreqFetcher {
    fn fetch(&self, url: &str) -> Result<String, JwksError> {
        let resp = self
            .agent
            .get(url)
            .call()
            .map_err(|e| JwksError::Fetch(e.to_string()))?;
        resp.into_string()
            .map_err(|e| JwksError::Fetch(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// The live production key, captured 2026-07-22. Public key material — a
    /// JWKS document is served anonymously to the internet by design.
    const LIVE_KID: &str = "_jQ62WD8cCiIGkKNQB8Hg4El2TNU5rHIITV4h_ba4YM";
    const LIVE_JWKS: &str = r#"{"keys":[{"alg":"ES256","crv":"P-256","kid":"_jQ62WD8cCiIGkKNQB8Hg4El2TNU5rHIITV4h_ba4YM","kty":"EC","use":"sig","x":"ixOcTyD66hYA52GE3NeLjMsUhPTVYl1_u6DimRKmxzU","y":"KQw2gxzKBk-FTGpioh0XKcIuaxh5No-Sn_qPbw3BH1M"}]}"#;

    /// Shared handle so a test can swap the served document or take the
    /// upstream offline *after* the fetcher has been moved into the cache.
    #[derive(Clone)]
    struct StubControl {
        body: Arc<Mutex<String>>,
        calls: Arc<AtomicUsize>,
        offline: Arc<Mutex<bool>>,
    }

    impl StubControl {
        fn new(body: &str) -> Self {
            Self {
                body: Arc::new(Mutex::new(body.to_owned())),
                calls: Arc::new(AtomicUsize::new(0)),
                offline: Arc::new(Mutex::new(false)),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
        fn serve(&self, body: &str) {
            *self.body.lock().unwrap() = body.to_owned();
        }
        fn go_offline(&self) {
            *self.offline.lock().unwrap() = true;
        }
        fn fetcher(&self) -> Box<StubFetcher> {
            Box::new(StubFetcher(self.clone()))
        }
    }

    struct StubFetcher(StubControl);

    impl JwksFetcher for StubFetcher {
        fn fetch(&self, _url: &str) -> Result<String, JwksError> {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            if *self.0.offline.lock().unwrap() {
                return Err(JwksError::Fetch("stub offline".into()));
            }
            Ok(self.0.body.lock().unwrap().clone())
        }
    }

    #[test]
    fn parses_the_live_production_jwks() {
        let keys = parse_jwks(LIVE_JWKS).expect("live JWKS parses");
        assert_eq!(keys.len(), 1);
        assert!(keys.contains_key(LIVE_KID));
    }

    #[test]
    fn rejects_a_document_with_no_usable_keys() {
        let rsa_only = r#"{"keys":[{"kty":"RSA","kid":"r1","n":"AQAB","e":"AQAB"}]}"#;
        assert!(matches!(
            parse_jwks(rsa_only),
            Err(JwksError::NoUsableKeys)
        ));
    }

    #[test]
    fn skips_a_non_p256_ec_key_rather_than_failing_the_whole_document() {
        let mixed = r#"{"keys":[
            {"kty":"EC","crv":"P-384","kid":"wrong-curve","x":"AA","y":"AA"},
            {"alg":"ES256","crv":"P-256","kid":"_jQ62WD8cCiIGkKNQB8Hg4El2TNU5rHIITV4h_ba4YM","kty":"EC","use":"sig","x":"ixOcTyD66hYA52GE3NeLjMsUhPTVYl1_u6DimRKmxzU","y":"KQw2gxzKBk-FTGpioh0XKcIuaxh5No-Sn_qPbw3BH1M"}
        ]}"#;
        let keys = parse_jwks(mixed).expect("parses");
        assert_eq!(keys.len(), 1, "only the P-256 key is usable");
        assert!(!keys.contains_key("wrong-curve"));
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(matches!(parse_jwks("{not json"), Err(JwksError::Malformed(_))));
    }

    #[test]
    fn a_cached_key_is_served_without_refetching() {
        let ctl = StubControl::new(LIVE_JWKS);
        let cache = JwksCache::new("https://stub/jwks.json", ctl.fetcher());

        cache.decoding_key_for(LIVE_KID).expect("first resolves");
        cache.decoding_key_for(LIVE_KID).expect("second resolves");

        assert_eq!(ctl.calls(), 1, "second call hit the cache");
    }

    #[test]
    fn never_fetched_plus_unreachable_upstream_fails_closed() {
        let ctl = StubControl::new(LIVE_JWKS);
        ctl.go_offline();
        let cache = JwksCache::new("https://stub/jwks.json", ctl.fetcher());

        assert!(matches!(
            cache.decoding_key_for(LIVE_KID),
            Err(JwksError::NeverFetched)
        ));
    }

    #[test]
    fn a_previously_cached_key_survives_an_upstream_outage() {
        // The offline-tolerance property RuView's edge deployment depends on:
        // a WAN blip must not log every user out of their own sensing server.
        let ctl = StubControl::new(LIVE_JWKS);
        let cache = JwksCache::with_ttl(
            "https://stub/jwks.json",
            ctl.fetcher(),
            Duration::from_millis(0), // every lookup treats the cache as stale
        );

        cache.decoding_key_for(LIVE_KID).expect("warm the cache");
        ctl.go_offline();

        cache
            .decoding_key_for(LIVE_KID)
            .expect("known kid still resolves while upstream is unreachable");
    }

    #[test]
    fn unknown_kid_triggers_exactly_one_forced_refetch_then_rate_limits() {
        let ctl = StubControl::new(LIVE_JWKS);
        let cache = JwksCache::new("https://stub/jwks.json", ctl.fetcher());

        cache.decoding_key_for(LIVE_KID).expect("warm the cache");
        assert_eq!(ctl.calls(), 1);

        // First unknown kid: one forced refetch, since rotation may have
        // happened inside the TTL.
        assert!(matches!(
            cache.decoding_key_for("bogus-kid"),
            Err(JwksError::UnknownKid(_))
        ));
        assert_eq!(ctl.calls(), 2, "one forced refetch");

        // Subsequent unknown kids inside the floor must NOT amplify: otherwise
        // a flood of junk-kid tokens becomes a DoS aimed at identity.
        for _ in 0..20 {
            let _ = cache.decoding_key_for("bogus-kid");
        }
        assert_eq!(
            ctl.calls(),
            2,
            "rate limiter prevented an outbound request per token"
        );
    }

    #[test]
    fn a_stale_cache_with_a_dead_upstream_does_not_refetch_on_every_request() {
        // THE BUG THIS GUARDS. `fetched_at` advances only on SUCCESS, and the
        // only rate limiter used to sit behind `if fresh`. So once the TTL
        // elapsed after the last successful fetch, `fresh` was permanently
        // false, the limiter was never consulted, and EVERY request performed
        // its own blocking 3s-timeout fetch. On a Pi that loses WAN that is a
        // self-inflicted stall with no attacker present — and an attacker could
        // force the same state by flooding tokens with an unknown `kid`.
        //
        // Before the fix the burst makes 25 further fetches (26 total). After
        // it, zero: the warm-up's attempt timestamp still covers the burst,
        // because the limiter now applies to the stale path too.
        let ctl = StubControl::new(LIVE_JWKS);
        let cache = JwksCache::with_ttl(
            "https://stub/jwks.json",
            ctl.fetcher(),
            Duration::from_millis(1),
        );

        cache.decoding_key_for(LIVE_KID).expect("warm the cache");
        assert_eq!(ctl.calls(), 1);

        ctl.go_offline();
        std::thread::sleep(Duration::from_millis(10)); // TTL elapses

        for i in 0..25 {
            // Still answered from the stale cache: a key that verified a moment
            // ago has not stopped being valid because the network went away.
            cache
                .decoding_key_for(LIVE_KID)
                .unwrap_or_else(|e| panic!("request {i} lost its cached key: {e}"));
        }
        assert_eq!(
            ctl.calls(),
            1,
            "the burst must add NO outbound fetches; only the warm-up fetched"
        );
    }

    #[test]
    fn a_rotated_key_is_picked_up_inside_the_ttl() {
        let ctl = StubControl::new(
            r#"{"keys":[{"kty":"EC","crv":"P-256","kid":"old","x":"ixOcTyD66hYA52GE3NeLjMsUhPTVYl1_u6DimRKmxzU","y":"KQw2gxzKBk-FTGpioh0XKcIuaxh5No-Sn_qPbw3BH1M"}]}"#,
        );
        let cache = JwksCache::new("https://stub/jwks.json", ctl.fetcher());

        cache.decoding_key_for("old").expect("old key resolves");

        // Identity rotates. The TTL has NOT expired, so only the unknown-kid
        // forced-refetch path can recover — which is exactly what it is for.
        ctl.serve(LIVE_JWKS);

        cache
            .decoding_key_for(LIVE_KID)
            .expect("rotation picked up without waiting out the TTL");
    }
}
