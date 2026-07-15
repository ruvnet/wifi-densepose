//! Parser for HA `secrets.yaml`.
//!
//! `secrets.yaml` is a flat YAML key→value map at the root of the HA
//! config directory (NOT inside `.storage/`). Example:
//!
//! ```yaml
//! mqtt_password: hunter2
//! latitude: 51.5074
//! longitude: -0.1278
//! ```
//!
//! Values are always strings in HA (even numeric-looking ones are quoted in
//! practice). We parse all values as strings to avoid type-mismatch errors.
//!
//! `!secret <name>` reference resolution (i.e., checking that every secret
//! referenced in other YAML files exists here) is deferred to P2.

use std::collections::HashMap;
use std::path::Path;

use crate::MigrateError;

/// Read `secrets.yaml` from `path` and return a `name → value` map.
///
/// Returns an empty map if the file is empty (HA allows that).
pub fn read_secrets(path: &Path) -> Result<HashMap<String, String>, MigrateError> {
    let raw = std::fs::read_to_string(path).map_err(|e| MigrateError::Io {
        path: path.display().to_string(),
        source: e,
    })?;

    if raw.trim().is_empty() {
        return Ok(HashMap::new());
    }

    // SECURITY: do NOT use `MigrateError::YamlParse` here. serde_yaml error
    // messages can quote the offending scalar verbatim (a typed-tag coercion
    // error renders `invalid value: string "<the-secret-value>"`), and that
    // message would be printed to stderr by the CLI — leaking a secret value.
    // `MigrateError::SecretsParse` carries only the path + line/column.
    let parsed: serde_yaml::Value = serde_yaml::from_str(&raw).map_err(|e| {
        let loc = e.location();
        MigrateError::SecretsParse {
            path: path.display().to_string(),
            line: loc.as_ref().map_or(0, |l| l.line()),
            column: loc.as_ref().map_or(0, |l| l.column()),
        }
    })?;

    let map = match parsed {
        serde_yaml::Value::Mapping(m) => m,
        _ => {
            return Err(MigrateError::MissingField {
                field: "<root mapping>".into(),
                context: path.display().to_string(),
            })
        }
    };

    let mut result = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let key = match k {
            serde_yaml::Value::String(s) => s,
            other => format!("{other:?}"),
        };
        let value = match v {
            serde_yaml::Value::String(s) => s,
            serde_yaml::Value::Number(n) => n.to_string(),
            serde_yaml::Value::Bool(b) => b.to_string(),
            serde_yaml::Value::Null => String::new(),
            other => serde_yaml::to_string(&other)
                .unwrap_or_else(|_| "<unparseable>".into())
                .trim()
                .to_string(),
        };
        result.insert(key, value);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parses_simple_key_value_map() {
        let yaml = "mqtt_password: hunter2\nlatitude: 51.5074\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();
        let secrets = read_secrets(f.path()).unwrap();
        assert_eq!(secrets.get("mqtt_password").map(String::as_str), Some("hunter2"));
        assert_eq!(secrets.get("latitude").map(String::as_str), Some("51.5074"));
    }

    #[test]
    fn empty_secrets_file_returns_empty_map() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"").unwrap();
        let secrets = read_secrets(f.path()).unwrap();
        assert!(secrets.is_empty());
    }

    /// SECURITY regression (fails on the pre-fix `YamlParse` path): a malformed
    /// `secrets.yaml` whose offending scalar is a secret value must NOT have that
    /// value rendered in the returned error. serde_yaml's own error message for a
    /// typed-tag coercion failure embeds the scalar verbatim
    /// (`invalid value: string "<secret>"`); the old code wrapped that message
    /// into `MigrateError::YamlParse { source }`, so `Display` leaked the secret.
    #[test]
    fn malformed_secrets_error_never_contains_secret_value() {
        // `!!int` forces integer coercion of a string scalar; serde_yaml reports
        // the scalar text in its message. The scalar here is a stand-in secret.
        let yaml = "api_port: !!int s3cr3t_TOKEN_VALUE\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();

        let err = read_secrets(f.path()).unwrap_err();
        let rendered = err.to_string();

        // The secret VALUE must never appear in the error output...
        assert!(
            !rendered.contains("s3cr3t_TOKEN_VALUE"),
            "secret value leaked into error: {rendered}"
        );
        // ...and the full chain (with #[source]) must also be clean, since the
        // CLI/anyhow prints the source chain too.
        let mut source = std::error::Error::source(&err);
        while let Some(s) = source {
            assert!(
                !s.to_string().contains("s3cr3t_TOKEN_VALUE"),
                "secret value leaked into error source chain: {s}"
            );
            source = s.source();
        }

        // It should still be a structured, locatable error (fail-closed).
        assert!(
            matches!(err, MigrateError::SecretsParse { .. }),
            "expected SecretsParse, got: {err:?}"
        );
    }

    /// A secret KEY name is non-sensitive context and is fine to surface, but the
    /// redacting error must still help the user locate the problem (line/column).
    #[test]
    fn malformed_secrets_error_reports_location() {
        let yaml = "api_port: !!int notanumber\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();
        let err = read_secrets(f.path()).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("line"), "should report a line: {rendered}");
        assert!(rendered.contains("redacted"), "should signal redaction: {rendered}");
    }

    #[test]
    fn secret_count_is_correct() {
        let yaml = "a: 1\nb: 2\nc: 3\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(yaml.as_bytes()).unwrap();
        let secrets = read_secrets(f.path()).unwrap();
        assert_eq!(secrets.len(), 3);
    }
}
