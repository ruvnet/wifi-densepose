//! Generic, leak-free error responses for the sensing-server HTTP API.
//!
//! ## ADR-080 finding #2 — leaked internal errors in responses
//!
//! Several handlers historically serialized the *internal* error `Display`
//! (`format!("{e}")`, `err.to_string()`, a panicked `JoinError`) straight into
//! the JSON response body. That leaks server internals to any client: OS error
//! strings can carry filesystem paths, a `JoinError` carries the panic message
//! (`task … panicked`), and an upstream-fetch error can carry an internal URL.
//! ADR-080 flagged this HIGH (CWE-209: Generation of Error Message Containing
//! Sensitive Information). The HOMECORE/M7 sweep (ADR-161) covered
//! `homecore-server`, **not** this crate, so the finding stayed open.
//!
//! ## Contract
//!
//! [`internal_error`] logs the full detail **server-side only** (at `error`
//! level, tagged with a correlation id) and returns a *generic* body to the
//! client:
//!
//! ```json
//! { "error": "internal_error", "correlation_id": "a1b2c3d4e5f60718", "success": false }
//! ```
//!
//! The correlation id lets an operator grep the server log for the matching
//! detail line without ever shipping that detail to the client. The body
//! deliberately contains no `Display`/`Debug` of the underlying error, no file
//! paths, and never the word `panicked`.
//!
//! Handlers that previously returned `Json<serde_json::Value>` keep doing so via
//! [`internal_error_json`]; handlers that return `(StatusCode, Json<…>)` use
//! [`internal_error`]. A "service unavailable" flavor ([`upstream_unavailable`])
//! exists for the 503 upstream-fetch path so it, too, stops leaking the raw
//! upstream error.

use std::fmt::Display;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::{http::StatusCode, response::Json};
use serde_json::json;

/// Monotonic component of the correlation id, so two errors in the same
/// nanosecond still get distinct ids. Wraps harmlessly.
static CORRELATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a short, opaque correlation id (16 lowercase hex chars). Built from
/// a nanosecond timestamp XORed with a monotonic counter — unique enough to tie
/// a client-visible id back to a single server-side log line without pulling in
/// a UUID dependency. It is **not** a security token; it is only an opaque
/// log-join key, so a non-cryptographic source is fine.
pub fn correlation_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seq = CORRELATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Mix the counter into the high bits so concurrent calls in the same
    // nanosecond don't collide.
    let mixed = nanos ^ seq.rotate_left(40);
    format!("{mixed:016x}")
}

/// Build a generic internal-error response **and log the real detail
/// server-side**. The client sees only `{"error":"internal_error",
/// "correlation_id":…,"success":false}` with a `500` status; the detail is
/// written to the `error`-level log tagged with the same correlation id.
///
/// `context` is a short, *static* description of where the error happened
/// (e.g. `"model delete"`); it is safe to log but is **not** sent to the
/// client.
pub fn internal_error(context: &str, detail: impl Display) -> (StatusCode, Json<serde_json::Value>) {
    let cid = correlation_id();
    // Server-side only — this is where the real detail lives.
    tracing::error!(
        correlation_id = %cid,
        context = context,
        detail = %detail,
        "internal error (detail logged server-side only; client received a generic body)"
    );
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": "internal_error",
            "correlation_id": cid,
            "success": false,
        })),
    )
}

/// Same as [`internal_error`] but returns a bare `Json` body (HTTP `200` at the
/// transport layer) for the legacy handlers that are typed
/// `-> Json<serde_json::Value>` and signal failure via `"success": false`
/// rather than an HTTP status code. The detail is still logged server-side and
/// never reaches the client.
pub fn internal_error_json(context: &str, detail: impl Display) -> Json<serde_json::Value> {
    let cid = correlation_id();
    tracing::error!(
        correlation_id = %cid,
        context = context,
        detail = %detail,
        "internal error (detail logged server-side only; client received a generic body)"
    );
    Json(json!({
        "error": "internal_error",
        "correlation_id": cid,
        "success": false,
    }))
}

/// Generic `503 Service Unavailable` for an upstream dependency that failed,
/// without leaking the raw upstream error (which can carry an internal URL or
/// connection detail). Detail is logged server-side with a correlation id.
pub fn upstream_unavailable(
    context: &str,
    detail: impl Display,
) -> (StatusCode, Json<serde_json::Value>) {
    let cid = correlation_id();
    tracing::warn!(
        correlation_id = %cid,
        context = context,
        detail = %detail,
        "upstream unavailable (detail logged server-side only; client received a generic body)"
    );
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": "upstream_unavailable",
            "correlation_id": cid,
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A "detail" string carrying the kind of internal information the old
    /// `format!("{e}")` path would have leaked: a filesystem path, an OS error,
    /// and the word `panicked`.
    const LEAKY_DETAIL: &str =
        "task 42 panicked at 'C:\\Users\\ruv\\secret\\models\\foo.rvf': No such file or directory (os error 2)";

    /// Recursively collect every string value in a JSON document, so a test can
    /// assert no leaky substring appears *anywhere* in the body (not just in a
    /// single known field).
    fn all_strings(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::String(s) => out.push(s.clone()),
            serde_json::Value::Array(a) => a.iter().for_each(|x| all_strings(x, out)),
            serde_json::Value::Object(o) => o.values().for_each(|x| all_strings(x, out)),
            _ => {}
        }
    }

    fn body_strings(body: &Json<serde_json::Value>) -> Vec<String> {
        let mut out = Vec::new();
        all_strings(&body.0, &mut out);
        out
    }

    /// REGRESSION (ADR-080 #2): the response body must NOT contain the panic
    /// message, the filesystem path, or the OS error string. The pre-fix code
    /// returned `format!("{e}")` / `join_err.to_string()` directly, so the body
    /// *did* contain `panicked`, the path, and `os error 2` — this test fails
    /// on that old behavior.
    #[test]
    fn internal_error_body_does_not_leak_detail() {
        let (status, body) = internal_error("unit-test", LEAKY_DETAIL);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        for s in body_strings(&body) {
            assert!(
                !s.contains("panicked"),
                "response body leaked the panic message: {s:?}"
            );
            assert!(
                !s.contains("secret"),
                "response body leaked a filesystem path: {s:?}"
            );
            assert!(
                !s.contains("os error"),
                "response body leaked an OS error string: {s:?}"
            );
            assert!(
                !s.contains(".rvf"),
                "response body leaked a file name/path: {s:?}"
            );
        }
    }

    /// The generic body still carries a correlation id so an operator can join
    /// the client report to the server log line that *does* hold the detail.
    #[test]
    fn internal_error_body_is_generic_with_correlation_id() {
        let (_status, body) = internal_error("unit-test", LEAKY_DETAIL);
        assert_eq!(body.0["error"], "internal_error");
        assert_eq!(body.0["success"], false);
        let cid = body.0["correlation_id"]
            .as_str()
            .expect("correlation_id must be a string");
        assert_eq!(cid.len(), 16, "correlation id should be 16 hex chars");
        assert!(
            cid.chars().all(|c| c.is_ascii_hexdigit()),
            "correlation id should be hex: {cid:?}"
        );
    }

    /// Same leak guarantee for the bare-`Json` (legacy "success: false")
    /// variant used by handlers that don't return an HTTP status.
    #[test]
    fn internal_error_json_does_not_leak_detail() {
        let body = internal_error_json("unit-test", LEAKY_DETAIL);
        assert_eq!(body.0["error"], "internal_error");
        assert_eq!(body.0["success"], false);
        for s in body_strings(&body) {
            assert!(!s.contains("panicked"), "leaked panic message: {s:?}");
            assert!(!s.contains("secret"), "leaked filesystem path: {s:?}");
            assert!(!s.contains("os error"), "leaked OS error: {s:?}");
        }
    }

    /// The 503 upstream flavor must likewise not echo the raw upstream error
    /// (which can carry an internal URL / connection string).
    #[test]
    fn upstream_unavailable_does_not_leak_detail() {
        let (status, body) = upstream_unavailable(
            "edge-registry",
            "https://internal-host.local:9000/app-registry.json: connection refused",
        );
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        for s in body_strings(&body) {
            assert!(
                !s.contains("internal-host"),
                "leaked internal upstream host: {s:?}"
            );
            assert!(
                !s.contains("connection refused"),
                "leaked upstream connection detail: {s:?}"
            );
        }
        assert_eq!(body.0["error"], "upstream_unavailable");
        assert!(body.0["correlation_id"].is_string());
    }

    /// Correlation ids are unique across rapid successive calls (so two errors
    /// can be told apart in the log even under load).
    #[test]
    fn correlation_ids_are_unique() {
        let a = correlation_id();
        let b = correlation_id();
        assert_ne!(a, b, "successive correlation ids must differ: {a} == {b}");
    }
}
