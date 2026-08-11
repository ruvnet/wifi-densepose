//! ADR-296: sensor data-plane bind hardening — UDP bind scope + source allowlist.
//!
//! The CSI UDP receiver historically bound `0.0.0.0` unconditionally, with no
//! equivalent of the HTTP `--bind-addr` flag, no source allowlist, and no
//! message authentication. Any host that could reach the port could inject a
//! valid-shaped frame and influence presence/vital/automation outputs.
//!
//! This module supplies the *step one* controls: a pure, socket-free bind
//! decision (loopback default, fail-closed on an unguarded routable bind,
//! mirroring the HTTP/OAuth refusal pattern) and a dependency-free source
//! IP/CIDR allowlist with a drop counter.
//!
//! # Threat model & residual risk
//!
//! An IP/CIDR allowlist restricts *which addresses* may deliver frames; it does
//! **not** authenticate the sender. On a trusted LAN an attacker who can spoof a
//! source address, or who controls an allowlisted host, can still inject frames.
//! The safe default is therefore loopback-only. Binding to a routable address
//! is an explicit operator choice and requires either `--udp-allow` (restrict
//! sources) or `--udp-insecure-lan` (accept the residual risk explicitly).
//!
//! **Deferred to a follow-up ADR (step two):** per-device provisioned keys,
//! MAC/AEAD, device identifiers, monotonic sequence numbers, a freshness
//! window, and replay rejection. Until those land the UDP data plane is NOT
//! authenticated. See the crate `SECURITY.md`.

use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};

/// Upper bound on parsed allowlist entries. The list comes from operator CLI
/// input, but we cap it anyway to keep allocation bounded at the boundary.
const MAX_ENTRIES: usize = 4096;

/// Outcome of evaluating the requested UDP bind scope against the allowlist and
/// override flags. Pure and socket-free so the decision can be unit-tested
/// without binding a real socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdpBindDecision {
    /// Bind is loopback-only (`127.0.0.1` / `::1`); not reachable off-host.
    Loopback,
    /// Bind is routable and a source allowlist is enforced.
    RoutableAllowlisted,
    /// Bind is routable with no allowlist, explicitly accepted via
    /// `--udp-insecure-lan`.
    RoutableInsecure,
}

/// Decide whether the requested UDP bind is permitted.
///
/// Fail-closed, mirroring the HTTP/OAuth boot refusals: a routable bind with no
/// source allowlist is rejected unless the operator passes the explicit
/// `--udp-insecure-lan` override. Loopback is always fine.
pub fn decide_udp_bind(
    bind: IpAddr,
    has_allowlist: bool,
    insecure_lan: bool,
) -> Result<UdpBindDecision, String> {
    if bind.is_loopback() {
        return Ok(UdpBindDecision::Loopback);
    }
    if has_allowlist {
        return Ok(UdpBindDecision::RoutableAllowlisted);
    }
    if insecure_lan {
        return Ok(UdpBindDecision::RoutableInsecure);
    }
    Err(format!(
        "Refusing to bind the UDP CSI receiver to routable address {bind} with no \
         source allowlist. Pass --udp-allow <IP/CIDR,...> to restrict sources, or \
         --udp-insecure-lan to accept the LAN-spoofing risk explicitly. The default \
         is loopback (127.0.0.1); see the crate SECURITY.md (ADR-296)."
    ))
}

/// One-line startup security summary describing the resolved bind scope and
/// allowlist state, for the boot log.
pub fn startup_summary(
    decision: UdpBindDecision,
    bind: IpAddr,
    port: u16,
    allowlist: &UdpSourceAllowlist,
) -> String {
    let scope = match decision {
        UdpBindDecision::Loopback => "loopback-only (not reachable off-host)",
        UdpBindDecision::RoutableAllowlisted => "ROUTABLE, source allowlist enforced",
        UdpBindDecision::RoutableInsecure => {
            "ROUTABLE, NO allowlist (--udp-insecure-lan; data plane is UNAUTHENTICATED)"
        }
    };
    if allowlist.is_active() {
        format!(
            "bind {bind}:{port} — {scope}; allowlist active ({} entr{}), loopback always allowed",
            allowlist.len(),
            if allowlist.len() == 1 { "y" } else { "ies" }
        )
    } else {
        format!("bind {bind}:{port} — {scope}; no source allowlist")
    }
}

/// A single parsed IP or CIDR allowlist entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CidrEntry {
    network: IpAddr,
    prefix_len: u8,
}

impl CidrEntry {
    /// Whether `ip` falls inside this network. Family mismatches never match.
    fn contains(&self, ip: IpAddr) -> bool {
        match (self.network, ip) {
            (IpAddr::V4(net), IpAddr::V4(addr)) => {
                let mask = v4_mask(self.prefix_len);
                (u32::from(net) & mask) == (u32::from(addr) & mask)
            }
            (IpAddr::V6(net), IpAddr::V6(addr)) => {
                let mask = v6_mask(self.prefix_len);
                (u128::from(net) & mask) == (u128::from(addr) & mask)
            }
            _ => false,
        }
    }
}

fn v4_mask(prefix: u8) -> u32 {
    match prefix {
        0 => 0,
        p if p >= 32 => u32::MAX,
        p => u32::MAX << (32 - p),
    }
}

fn v6_mask(prefix: u8) -> u128 {
    match prefix {
        0 => 0,
        p if p >= 128 => u128::MAX,
        p => u128::MAX << (128 - p),
    }
}

/// Parse one `IP` or `IP/prefix` entry. Never panics on malformed input.
fn parse_entry(raw: &str) -> Result<CidrEntry, String> {
    let s = raw.trim();
    if let Some((ip_str, pfx_str)) = s.split_once('/') {
        let ip: IpAddr = ip_str
            .trim()
            .parse()
            .map_err(|_| format!("invalid IP address in allowlist entry '{s}'"))?;
        let max = if ip.is_ipv4() { 32u8 } else { 128u8 };
        let pfx: u8 = pfx_str
            .trim()
            .parse()
            .map_err(|_| format!("invalid prefix length in allowlist entry '{s}'"))?;
        if pfx > max {
            return Err(format!(
                "prefix /{pfx} exceeds maximum /{max} for {ip} in allowlist entry '{s}'"
            ));
        }
        Ok(CidrEntry {
            network: ip,
            prefix_len: pfx,
        })
    } else {
        let ip: IpAddr = s
            .parse()
            .map_err(|_| format!("invalid IP address in allowlist entry '{s}'"))?;
        let prefix_len = if ip.is_ipv4() { 32 } else { 128 };
        Ok(CidrEntry {
            network: ip,
            prefix_len,
        })
    }
}

/// Source IP/CIDR allowlist for inbound UDP CSI frames.
///
/// When no entries are configured the allowlist is *inactive* and accepts every
/// source (the bind decision, not this filter, gates an unguarded routable
/// bind). When entries are present, only loopback and matching sources are
/// allowed; everything else is dropped and counted.
#[derive(Debug, Default)]
pub struct UdpSourceAllowlist {
    entries: Vec<CidrEntry>,
    dropped: AtomicU64,
}

impl UdpSourceAllowlist {
    /// Parse an allowlist from CLI/env specs. Each item may itself be a
    /// comma-separated list; whitespace and empty items are ignored. Returns an
    /// error on the first malformed entry rather than silently dropping it.
    pub fn parse<I, S>(specs: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut entries: Vec<CidrEntry> = Vec::new();
        for spec in specs {
            for part in spec.as_ref().split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                if entries.len() >= MAX_ENTRIES {
                    return Err(format!(
                        "too many allowlist entries (limit {MAX_ENTRIES})"
                    ));
                }
                entries.push(parse_entry(part)?);
            }
        }
        Ok(Self {
            entries,
            dropped: AtomicU64::new(0),
        })
    }

    /// Whether any allowlist entries are configured (i.e. filtering is on).
    pub fn is_active(&self) -> bool {
        !self.entries.is_empty()
    }

    /// Number of configured entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the allowlist has no configured entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `src` is permitted. Loopback is always allowed; an inactive
    /// allowlist accepts everything. Does not touch the drop counter.
    pub fn is_allowed(&self, src: IpAddr) -> bool {
        if src.is_loopback() {
            return true;
        }
        if self.entries.is_empty() {
            return true;
        }
        self.entries.iter().any(|e| e.contains(src))
    }

    /// Like [`is_allowed`](Self::is_allowed) but records a drop when the source
    /// is rejected. Use this on the hot receive path.
    pub fn admit(&self, src: IpAddr) -> bool {
        let ok = self.is_allowed(src);
        if !ok {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        ok
    }

    /// Total frames dropped due to a non-matching source.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    #[test]
    fn default_bind_is_loopback() {
        let d = decide_udp_bind(IpAddr::V4(Ipv4Addr::LOCALHOST), false, false).unwrap();
        assert_eq!(d, UdpBindDecision::Loopback);
        // IPv6 loopback too.
        let d6 = decide_udp_bind(IpAddr::V6(Ipv6Addr::LOCALHOST), false, false).unwrap();
        assert_eq!(d6, UdpBindDecision::Loopback);
    }

    #[test]
    fn routable_bind_without_allowlist_is_refused() {
        // 0.0.0.0 is not loopback → refused with no allowlist and no override.
        let err = decide_udp_bind(IpAddr::V4(Ipv4Addr::UNSPECIFIED), false, false).unwrap_err();
        assert!(err.contains("Refusing"), "{err}");
        let err_lan = decide_udp_bind(v4(192, 168, 1, 10), false, false).unwrap_err();
        assert!(err_lan.contains("--udp-allow"), "{err_lan}");
    }

    #[test]
    fn routable_bind_allowed_with_allowlist_or_override() {
        assert_eq!(
            decide_udp_bind(v4(0, 0, 0, 0), true, false).unwrap(),
            UdpBindDecision::RoutableAllowlisted
        );
        assert_eq!(
            decide_udp_bind(v4(0, 0, 0, 0), false, true).unwrap(),
            UdpBindDecision::RoutableInsecure
        );
    }

    #[test]
    fn loopback_bind_ignores_missing_allowlist() {
        // Even without allowlist/override, loopback is never refused.
        assert_eq!(
            decide_udp_bind(IpAddr::V4(Ipv4Addr::LOCALHOST), false, false).unwrap(),
            UdpBindDecision::Loopback
        );
    }

    #[test]
    fn allowlist_accept_drop_and_count() {
        let a = UdpSourceAllowlist::parse(["192.168.1.0/24"]).unwrap();
        assert!(a.is_active());
        assert!(a.admit(v4(192, 168, 1, 42)));
        assert!(!a.admit(v4(10, 0, 0, 1)));
        assert!(!a.admit(v4(192, 168, 2, 1)));
        assert_eq!(a.dropped(), 2);
        // Accepts did not touch the counter.
        assert!(a.admit(v4(192, 168, 1, 200)));
        assert_eq!(a.dropped(), 2);
    }

    #[test]
    fn loopback_source_always_allowed() {
        let a = UdpSourceAllowlist::parse(["10.0.0.0/8"]).unwrap();
        assert!(a.admit(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert!(a.admit(IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert_eq!(a.dropped(), 0);
        // A non-loopback outside the list is still dropped.
        assert!(!a.admit(v4(192, 168, 0, 1)));
        assert_eq!(a.dropped(), 1);
    }

    #[test]
    fn inactive_allowlist_accepts_everything() {
        let a = UdpSourceAllowlist::parse(Vec::<String>::new()).unwrap();
        assert!(!a.is_active());
        assert!(a.admit(v4(203, 0, 113, 7)));
        assert_eq!(a.dropped(), 0);
    }

    #[test]
    fn exact_ip_entry_matches_only_itself() {
        let a = UdpSourceAllowlist::parse(["10.0.0.5"]).unwrap();
        assert!(a.is_allowed(v4(10, 0, 0, 5)));
        assert!(!a.is_allowed(v4(10, 0, 0, 6)));
    }

    #[test]
    fn comma_and_multi_spec_parsing() {
        let a = UdpSourceAllowlist::parse(["192.168.1.0/24, 10.0.0.5", "172.16.0.0/12"]).unwrap();
        assert_eq!(a.len(), 3);
        assert!(a.is_allowed(v4(172, 20, 5, 5)));
        assert!(a.is_allowed(v4(10, 0, 0, 5)));
        assert!(!a.is_allowed(v4(8, 8, 8, 8)));
    }

    #[test]
    fn ipv6_cidr_matching() {
        let a = UdpSourceAllowlist::parse(["2001:db8::/32"]).unwrap();
        assert!(a.is_allowed("2001:db8:1234::1".parse().unwrap()));
        assert!(!a.is_allowed("2001:dead::1".parse().unwrap()));
        // v4 source never matches a v6 entry.
        assert!(!a.is_allowed(v4(192, 168, 1, 1)));
    }

    #[test]
    fn malformed_entries_error_without_panic() {
        assert!(UdpSourceAllowlist::parse(["not-an-ip"]).is_err());
        assert!(UdpSourceAllowlist::parse(["192.168.1.0/33"]).is_err());
        assert!(UdpSourceAllowlist::parse(["10.0.0.0/x"]).is_err());
        assert!(UdpSourceAllowlist::parse(["::1/129"]).is_err());
    }

    #[test]
    fn prefix_zero_matches_all_of_family() {
        let a = UdpSourceAllowlist::parse(["0.0.0.0/0"]).unwrap();
        assert!(a.is_allowed(v4(1, 2, 3, 4)));
        assert!(a.is_allowed(v4(203, 0, 113, 9)));
        // But not IPv6 — different family.
        assert!(!a.is_allowed("2001:db8::1".parse().unwrap()));
    }

    #[test]
    fn startup_summary_mentions_scope_and_allowlist() {
        let a = UdpSourceAllowlist::parse(["192.168.1.0/24"]).unwrap();
        let s = startup_summary(
            UdpBindDecision::RoutableAllowlisted,
            v4(0, 0, 0, 0),
            5005,
            &a,
        );
        assert!(s.contains("allowlist active"), "{s}");
        assert!(s.contains("loopback always allowed"), "{s}");

        let empty = UdpSourceAllowlist::default();
        let loop_s = startup_summary(
            UdpBindDecision::Loopback,
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            5005,
            &empty,
        );
        assert!(loop_s.contains("loopback-only"), "{loop_s}");
        assert!(loop_s.contains("no source allowlist"), "{loop_s}");
    }
}
