//! Short-lived, single-use WebSocket tickets (ADR-272).
//!
//! # Why this exists
//!
//! A browser's `WebSocket` constructor cannot set an `Authorization` header on
//! the upgrade request. That limitation is why `/ws/sensing`,
//! `/ws/introspection` and `/api/v1/stream/pose` have been exempt from
//! [`crate::bearer_auth`] — which means that on a server with auth switched
//! ON, an unauthenticated caller can still complete a WebSocket handshake to
//! the **live sensing stream**. The REST control plane is locked; the data
//! plane is open.
//!
//! A ticket closes that without pretending browsers can do something they
//! cannot: the page makes an ordinary authenticated `POST /api/v1/ws-ticket`
//! (a normal request, where it *can* set headers), gets an opaque string, and
//! passes it as `?ticket=…` on the upgrade.
//!
//! # Why a query parameter is acceptable here, when it usually is not
//!
//! Putting a credential in a URL is normally a mistake: URLs land in access
//! logs, `Referer` headers and browser history. Three properties keep this one
//! bounded, and all three are load-bearing:
//!
//! 1. **Single use.** Consumed on the first upgrade attempt. A ticket in a log
//!    is already spent.
//! 2. **Seconds, not hours.** [`TICKET_TTL`] is 30s — long enough for a page to
//!    open a socket, far too short to be worth harvesting.
//! 3. **It is not the credential.** It authorizes one WebSocket connection.
//!    It cannot be replayed against `/api/v1/*`, cannot be refreshed, and
//!    carries no user identity a thief could reuse elsewhere.
//!
//! Native clients — the Python client, the Rust CLI, the TS MCP client — are
//! **not** browsers and must send a normal `Authorization` header on the
//! upgrade instead. Routing them through tickets would add a round-trip and a
//! second credential path for no benefit.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::RngCore;

/// How long a ticket is valid. Deliberately tiny — a page opens its socket
/// immediately after fetching one, so anything longer is only useful to
/// someone who found the URL later.
pub const TICKET_TTL: Duration = Duration::from_secs(30);

/// Global cap on outstanding tickets.
const MAX_OUTSTANDING: usize = 512;

/// Per-principal cap.
///
/// The global cap alone is not enough: one authenticated `sensing:read` caller
/// looping on `POST /api/v1/ws-ticket` could occupy all 512 slots for 30
/// seconds and 503 every other user — a denial of service by an ordinary,
/// lowest-privilege account. A page needs a handful of concurrent sockets, so
/// this is generous while making one caller unable to starve the rest.
///
/// Tickets issued to the legacy static token share the `None` bucket, since
/// that credential carries no subject to attribute them to.
const MAX_PER_PRINCIPAL: usize = 16;

/// What a redeemed ticket authorizes.
///
/// The scopes are captured at issue time from the authenticated request, so a
/// WebSocket inherits exactly the authority of the credential that asked for
/// it — a `sensing:read` session cannot obtain a ticket that outranks itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TicketGrant {
    /// Space-separated scopes held by the issuing principal, or `None` when the
    /// issuer was the legacy static token (which predates scopes and carries
    /// full authority).
    pub scopes: Option<String>,
    /// `sub` of the issuing principal, for logging. `None` for the static token.
    pub subject: Option<String>,
}

struct Entry {
    grant: TicketGrant,
    expires_at: Instant,
}

/// In-memory ticket store.
///
/// `Debug` deliberately reports only a count, never ticket values — a ticket in
/// a debug log is a live credential for as long as it is unspent.
///
/// In-memory is correct rather than merely convenient: tickets live for
/// seconds, and a ticket surviving a restart would be a ticket outliving the
/// server that vouched for it.
#[derive(Clone, Default)]
pub struct TicketStore {
    inner: Arc<Mutex<HashMap<String, Entry>>>,
}

impl std::fmt::Debug for TicketStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.inner.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("TicketStore").field("outstanding", &n).finish()
    }
}

impl TicketStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a ticket for an authenticated caller.
    ///
    /// Returns `None` if too many tickets are outstanding — refusing to issue
    /// is the correct failure here; the alternative is unbounded growth driven
    /// by a caller who is authenticated but misbehaving.
    pub fn issue(&self, grant: TicketGrant) -> Option<String> {
        let mut map = self.inner.lock().expect("ticket store poisoned");
        prune(&mut map);
        if map.len() >= MAX_OUTSTANDING {
            tracing::warn!(
                outstanding = map.len(),
                "refusing to issue a WebSocket ticket: global cap reached"
            );
            return None;
        }
        // Per-principal cap, so one caller cannot starve every other user.
        let held_by_this_principal = map
            .values()
            .filter(|e| e.grant.subject == grant.subject)
            .count();
        if held_by_this_principal >= MAX_PER_PRINCIPAL {
            tracing::warn!(
                subject = ?grant.subject,
                held = held_by_this_principal,
                "refusing to issue a WebSocket ticket: per-principal cap reached"
            );
            return None;
        }
        let mut bytes = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        let ticket = hex(&bytes);
        map.insert(
            ticket.clone(),
            Entry {
                grant,
                expires_at: Instant::now() + TICKET_TTL,
            },
        );
        Some(ticket)
    }

    /// Redeem a ticket. **Removes it** — a ticket is valid exactly once, so a
    /// replay of the same URL fails even within the TTL.
    pub fn consume(&self, ticket: &str) -> Option<TicketGrant> {
        let mut map = self.inner.lock().expect("ticket store poisoned");
        prune(&mut map);
        let entry = map.remove(ticket)?;
        // Belt and braces: prune already dropped expired entries, but an entry
        // expiring between the two would otherwise slip through.
        if entry.expires_at <= Instant::now() {
            return None;
        }
        Some(entry.grant)
    }

    #[cfg(test)]
    fn outstanding(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

fn prune(map: &mut HashMap<String, Entry>) {
    let now = Instant::now();
    map.retain(|_, e| e.expires_at > now);
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Extract `ticket` from a raw query string.
fn ticket_from_query(query: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some(v) = pair.strip_prefix("ticket=") {
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Extract `ticket` from a request URI's query, if present.
pub fn ticket_from_uri(uri: &axum::http::Uri) -> Option<String> {
    ticket_from_query(uri.query()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grant() -> TicketGrant {
        TicketGrant {
            scopes: Some("sensing:read".into()),
            subject: Some("user-1".into()),
        }
    }

    #[test]
    fn a_ticket_round_trips_once() {
        let store = TicketStore::new();
        let t = store.issue(grant()).expect("issued");
        assert_eq!(store.consume(&t), Some(grant()));
    }

    #[test]
    fn a_ticket_cannot_be_used_twice() {
        // The property that makes a credential-in-a-URL tolerable: by the time
        // it reaches a log, it is spent.
        let store = TicketStore::new();
        let t = store.issue(grant()).unwrap();
        assert!(store.consume(&t).is_some(), "first use succeeds");
        assert!(store.consume(&t).is_none(), "replay must fail");
    }

    #[test]
    fn an_unknown_ticket_is_refused() {
        let store = TicketStore::new();
        assert!(store.consume("deadbeef").is_none());
    }

    #[test]
    fn consuming_removes_the_entry_rather_than_marking_it() {
        let store = TicketStore::new();
        let t = store.issue(grant()).unwrap();
        assert_eq!(store.outstanding(), 1);
        store.consume(&t);
        assert_eq!(store.outstanding(), 0, "spent tickets must not accumulate");
    }

    #[test]
    fn an_expired_ticket_is_refused_and_pruned() {
        let store = TicketStore::new();
        let t = store.issue(grant()).unwrap();
        // Force expiry without sleeping.
        {
            let mut map = store.inner.lock().unwrap();
            map.get_mut(&t).unwrap().expires_at = Instant::now() - Duration::from_secs(1);
        }
        assert!(store.consume(&t).is_none(), "expired ticket must be refused");
        assert_eq!(store.outstanding(), 0, "and must not linger");
    }

    #[test]
    fn tickets_are_unpredictable_and_distinct() {
        let store = TicketStore::new();
        let a = store.issue(grant()).unwrap();
        let b = store.issue(grant()).unwrap();
        assert_ne!(a, b);
        // 32 bytes hex — guessing is not a strategy.
        assert_eq!(a.len(), 64, "expected 256 bits of ticket");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn the_grant_records_the_issuing_principals_scopes() {
        // A sensing:read session must not be able to mint a ticket that
        // outranks it — the WebSocket inherits the issuer's authority.
        let store = TicketStore::new();
        let g = TicketGrant {
            scopes: Some("sensing:read".into()),
            subject: Some("u".into()),
        };
        let t = store.issue(g.clone()).unwrap();
        assert_eq!(store.consume(&t).unwrap().scopes.as_deref(), Some("sensing:read"));
    }

    #[test]
    fn one_principal_cannot_starve_the_global_pool() {
        // The reported DoS: an ordinary sensing:read caller looping on
        // /api/v1/ws-ticket used to be able to occupy every slot and 503
        // everyone else.
        let store = TicketStore::new();
        let noisy = TicketGrant {
            scopes: Some("sensing:read".into()),
            subject: Some("noisy-user".into()),
        };
        for _ in 0..MAX_PER_PRINCIPAL {
            assert!(store.issue(noisy.clone()).is_some());
        }
        assert!(
            store.issue(noisy).is_none(),
            "one principal must hit its own cap"
        );
        // ...and a different user is entirely unaffected.
        let other = TicketGrant {
            scopes: Some("sensing:read".into()),
            subject: Some("quiet-user".into()),
        };
        assert!(
            store.issue(other).is_some(),
            "another principal must still be served"
        );
        assert!(
            store.outstanding() < MAX_OUTSTANDING,
            "the global pool was never exhausted"
        );
    }

    #[test]
    fn issuing_is_refused_once_too_many_are_outstanding() {
        let store = TicketStore::new();
        for i in 0..MAX_OUTSTANDING {
            // Distinct subjects, so this exercises the GLOBAL cap and not the
            // per-principal one.
            let g = TicketGrant {
                scopes: Some("sensing:read".into()),
                subject: Some(format!("user-{i}")),
            };
            assert!(store.issue(g).is_some());
        }
        assert!(
            store
                .issue(TicketGrant {
                    scopes: Some("sensing:read".into()),
                    subject: Some("one-more".into())
                })
                .is_none(),
            "the global cap must still hold"
        );
    }

    #[test]
    fn expired_tickets_free_capacity_again() {
        let store = TicketStore::new();
        for i in 0..MAX_OUTSTANDING {
            store.issue(TicketGrant {
                scopes: Some("sensing:read".into()),
                subject: Some(format!("user-{i}")),
            });
        }
        assert!(store
            .issue(TicketGrant {
                scopes: Some("sensing:read".into()),
                subject: Some("blocked".into())
            })
            .is_none());
        {
            let mut map = store.inner.lock().unwrap();
            for e in map.values_mut() {
                e.expires_at = Instant::now() - Duration::from_secs(1);
            }
        }
        assert!(
            store.issue(grant()).is_some(),
            "the cap must be self-healing, not a permanent wedge"
        );
    }

    #[test]
    fn parses_a_ticket_from_a_query_string() {
        assert_eq!(ticket_from_query("ticket=abc123").as_deref(), Some("abc123"));
        assert_eq!(
            ticket_from_query("foo=1&ticket=abc123&bar=2").as_deref(),
            Some("abc123")
        );
    }

    #[test]
    fn an_absent_or_empty_ticket_parameter_yields_none() {
        assert!(ticket_from_query("foo=1").is_none());
        assert!(ticket_from_query("ticket=").is_none());
        assert!(ticket_from_query("").is_none());
    }

    #[test]
    fn a_parameter_merely_ending_in_ticket_is_not_a_ticket() {
        // `?myticket=x` must not be read as `?ticket=x`.
        assert!(ticket_from_query("myticket=abc").is_none());
    }
}
