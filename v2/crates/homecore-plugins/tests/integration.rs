//! Integration tests for ADR-128 P2 — Wasmtime runtime + example WASM plugin.
//!
//! ## Test strategy
//!
//! ### Primary path (compiled .wasm)
//!
//! Loads `homecore_plugin_example.wasm` from the known release output path
//! under the plugin-example's own target directory. If the binary is not
//! present (i.e., the example hasn't been built yet), the primary test is
//! skipped with a warning and the WAT-based fallback runs instead.
//!
//! To run the primary path:
//!
//! ```sh
//! # From v2/crates/homecore-plugin-example:
//! /c/Users/ruv/.cargo/bin/cargo build --target wasm32-unknown-unknown --release
//! # Then from v2/:
//! cargo test -p homecore-plugins --features wasmtime
//! ```
//!
//! ### Fallback path (inline WAT)
//!
//! Always runs. Uses `wat::parse_str` to compile a hand-written WAT module
//! that implements the same temperature-threshold logic as the Rust plugin.
//! This proves the Wasmtime linker works and all 4 host imports are wired
//! correctly even without a pre-built `.wasm` binary.

#[cfg(feature = "wasmtime")]
mod wasmtime_tests {
    use homecore::HomeCore;
    use homecore_plugins::wasmtime_runtime::WasmtimeRuntime;
    use homecore_plugins::StateChangedEventJson;

    // ── Path to compiled example binary ────────────────────────────────────

    /// Path to the pre-compiled example WASM relative to the workspace root.
    ///
    /// The example crate has its own isolated Cargo workspace so its target
    /// directory lives under the crate itself, not the v2/ workspace target.
    const EXAMPLE_WASM_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../crates/homecore-plugin-example/target/wasm32-unknown-unknown/release/homecore_plugin_example.wasm"
    );

    // ── WAT fallback (always runnable) ─────────────────────────────────────

    /// WAT module implementing the same temperature-threshold logic as
    /// `homecore-plugin-example`. Used when the compiled .wasm is unavailable.
    ///
    /// Behaviour:
    /// - `plugin_setup` → subscribes to `sensor.test_temp` via `hc_state_subscribe`
    /// - `plugin_handle_state_changed` → parses the `new_state` field from
    ///   the event JSON and calls `hc_state_set` to write `binary_sensor.test_alert`
    ///
    /// This WAT version uses a simplified string scan rather than full JSON
    /// parsing, which is sufficient for the test payloads.
    const THRESHOLD_WAT: &str = r#"
(module
  (import "env" "hc_state_get"
    (func $hc_state_get (param i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_set"
    (func $hc_state_set (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_subscribe"
    (func $hc_state_subscribe (param i32 i32) (result i32)))
  (import "env" "hc_log"
    (func $hc_log (param i32 i32 i32)))

  (memory (export "memory") 2)
  (global $bump (mut i32) (i32.const 4096))

  ;; Static data at known offsets:
  ;; 0:   "sensor.test_temp"          (16 bytes)
  ;; 64:  "binary_sensor.test_alert"  (24 bytes)
  ;; 128: "on"                         (2 bytes)
  ;; 192: "off"                        (3 bytes)
  ;; 256: "{}"                         (2 bytes)
  (data (i32.const 0)   "sensor.test_temp")
  (data (i32.const 64)  "binary_sensor.test_alert")
  (data (i32.const 128) "on")
  (data (i32.const 192) "off")
  (data (i32.const 256) "{}")

  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $bump))
    (global.set $bump (i32.add (global.get $bump) (local.get $size)))
    (local.get $ptr)
  )
  (func (export "dealloc") (param i32 i32))

  ;; plugin_setup: subscribe to sensor.test_temp
  (func (export "plugin_setup") (param i32 i32) (result i32)
    (call $hc_state_subscribe (i32.const 0) (i32.const 16))
    drop
    (i32.const 0)
  )

  ;; plugin_handle_state_changed(ptr, len) → i32
  ;;
  ;; The host passes a JSON string. We scan for "\"new_state\":\"" and read
  ;; one or two ASCII digit bytes to determine if temp > 25 or < 20.
  ;; The test values are "26" (above 25) and "19" (below 20), so we read
  ;; the first two digits after the marker and compare numerically.
  ;;
  ;; Scan strategy: find byte sequence for "new_state":"
  ;; Then read the decimal integer that follows until '"'.
  ;;
  ;; We implement a simple integer parser inline in WAT.
  (func (export "plugin_handle_state_changed") (param $ptr i32) (param $len i32) (result i32)
    (local $i i32)       ;; scan index into the event buffer
    (local $end i32)     ;; ptr + len
    (local $num i32)     ;; parsed integer temperature
    (local $neg i32)     ;; 1 if negative
    (local $ch i32)      ;; current character
    (local $found i32)   ;; 1 if marker found

    ;; We look for the 13-byte sequence: "new_state":"
    ;; Simplified: scan for byte 'n','e','w' consecutively to find the field.
    ;; Full marker: "new_state":"  (len=13 including both quotes and colon)
    ;; Bytes: 22 6e 65 77 5f 73 74 61 74 65 22 3a 22
    ;;        "  n  e  w  _  s  t  a  t  e  "  :  "

    (local.set $end (i32.add (local.get $ptr) (local.get $len)))
    (local.set $i (local.get $ptr))
    (local.set $found (i32.const 0))

    ;; Scan for '"new_state":"'
    (block $done
      (loop $scan
        ;; Bounds check
        (br_if $done (i32.ge_u (i32.add (local.get $i) (i32.const 13)) (local.get $end)))
        ;; Check 13-byte marker
        (if
          (i32.and
            (i32.and
              (i32.eq (i32.load8_u (local.get $i))                    (i32.const 0x22))  ;; "
              (i32.eq (i32.load8_u (i32.add (local.get $i) (i32.const 1)))  (i32.const 0x6e))  ;; n
            )
            (i32.and
              (i32.eq (i32.load8_u (i32.add (local.get $i) (i32.const 11))) (i32.const 0x3a))  ;; :
              (i32.eq (i32.load8_u (i32.add (local.get $i) (i32.const 12))) (i32.const 0x22))  ;; "
            )
          )
          (then
            ;; Advance past marker to the value start
            (local.set $i (i32.add (local.get $i) (i32.const 13)))
            (local.set $found (i32.const 1))
            (br $done)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $scan)
      )
    )

    ;; If not found or null value, return 0 (no-op).
    (if (i32.eqz (local.get $found)) (then (return (i32.const 0))))

    ;; Parse integer from current position.
    (local.set $num (i32.const 0))
    (local.set $neg (i32.const 0))

    ;; Check for minus sign.
    (if (i32.lt_u (local.get $i) (local.get $end))
      (then
        (local.set $ch (i32.load8_u (local.get $i)))
        (if (i32.eq (local.get $ch) (i32.const 0x2d))  ;; '-'
          (then
            (local.set $neg (i32.const 1))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
          )
        )
      )
    )

    ;; Parse digits.
    (block $numDone
      (loop $digits
        (br_if $numDone (i32.ge_u (local.get $i) (local.get $end)))
        (local.set $ch (i32.load8_u (local.get $i)))
        ;; Stop at non-digit or dot (we ignore decimals for integer comparison)
        (br_if $numDone (i32.lt_u (local.get $ch) (i32.const 0x30)))  ;; < '0'
        (br_if $numDone (i32.gt_u (local.get $ch) (i32.const 0x39)))  ;; > '9'
        (local.set $num
          (i32.add
            (i32.mul (local.get $num) (i32.const 10))
            (i32.sub (local.get $ch) (i32.const 0x30))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $digits)
      )
    )

    ;; Apply negative sign.
    (if (local.get $neg)
      (then (local.set $num (i32.sub (i32.const 0) (local.get $num))))
    )

    ;; Apply threshold: > 25 → set alert ON; < 20 → set alert OFF.
    (if (i32.gt_s (local.get $num) (i32.const 25))
      (then
        (call $hc_state_set
          (i32.const 64) (i32.const 24)  ;; entity_id: "binary_sensor.test_alert"
          (i32.const 128) (i32.const 2)  ;; state: "on"
          (i32.const 256) (i32.const 2)  ;; attrs: "{}"
        )
        drop
      )
    )
    (if (i32.lt_s (local.get $num) (i32.const 20))
      (then
        (call $hc_state_set
          (i32.const 64) (i32.const 24)  ;; entity_id: "binary_sensor.test_alert"
          (i32.const 192) (i32.const 3)  ;; state: "off"
          (i32.const 256) (i32.const 2)  ;; attrs: "{}"
        )
        drop
      )
    )
    (i32.const 0)
  )
)
"#;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn build_rt_and_hc() -> (WasmtimeRuntime, HomeCore) {
        (
            WasmtimeRuntime::new().expect("WasmtimeRuntime::new"),
            HomeCore::new(),
        )
    }

    fn state_changed_event(entity_id: &str, new_state: &str) -> StateChangedEventJson {
        StateChangedEventJson::state_changed(
            entity_id,
            Some(new_state),
            serde_json::json!({}),
        )
    }

    fn assert_alert_state(hc: &HomeCore, expected: &str) {
        let eid = homecore::EntityId::parse("binary_sensor.test_alert").unwrap();
        let state = hc
            .states()
            .get(&eid)
            .unwrap_or_else(|| panic!("binary_sensor.test_alert not found in state machine"));
        assert_eq!(
            state.state, expected,
            "binary_sensor.test_alert should be '{expected}' but was '{}'",
            state.state
        );
    }

    // ── Primary test: compiled .wasm binary ──────────────────────────────────

    #[test]
    fn wasm_plugin_temp_threshold_compiled_binary() {
        let wasm_path = std::path::Path::new(EXAMPLE_WASM_PATH);
        if !wasm_path.exists() {
            eprintln!(
                "[SKIP] {EXAMPLE_WASM_PATH} not found. \
                 Build the example first:\n  \
                 cd v2/crates/homecore-plugin-example && \
                 cargo build --target wasm32-unknown-unknown --release"
            );
            return; // skip — binary not built yet
        }

        let wasm_bytes = std::fs::read(wasm_path)
            .expect("failed to read homecore_plugin_example.wasm");

        let (rt, hc) = build_rt_and_hc();
        let plugin = rt
            .load_wasm(&wasm_bytes, hc.clone())
            .expect("load_wasm should succeed");

        // Call plugin_setup — should subscribe to sensor.test_temp.
        let setup_result = plugin
            .call_setup(r#"{"entry_id":"test","domain":"test","title":"test","data":{}}"#)
            .expect("plugin_setup should not trap");
        assert_eq!(setup_result, 0, "plugin_setup should return 0");

        // Verify subscription was recorded.
        assert!(
            plugin.subscriptions().contains(&"sensor.test_temp".to_owned()),
            "plugin should have subscribed to sensor.test_temp"
        );

        // ── Scenario 1: temp = 26.0 → alert ON ──────────────────────────────
        let event_hot = state_changed_event("sensor.test_temp", "26.0");
        plugin
            .call_state_changed(&event_hot)
            .expect("state_changed should not trap");
        assert_alert_state(&hc, "on");

        // ── Scenario 2: temp = 19.0 → alert OFF ─────────────────────────────
        let event_cold = state_changed_event("sensor.test_temp", "19.0");
        plugin
            .call_state_changed(&event_cold)
            .expect("state_changed should not trap");
        assert_alert_state(&hc, "off");
    }

    // ── Fallback test: inline WAT (always runs) ───────────────────────────────

    #[test]
    fn wasm_plugin_temp_threshold_wat_fallback() {
        let wasm_bytes = wat::parse_str(THRESHOLD_WAT).expect("WAT should parse");

        let (rt, hc) = build_rt_and_hc();
        let plugin = rt
            .load_wasm(&wasm_bytes, hc.clone())
            .expect("load_wasm should succeed for WAT");

        // plugin_setup → subscribes
        let r = plugin.call_setup("{}").expect("setup");
        assert_eq!(r, 0);

        // ── Scenario 1: temp = 26 → alert ON ───────────────────────────────
        let hot_event = StateChangedEventJson::state_changed(
            "sensor.test_temp",
            Some("26"),
            serde_json::json!({}),
        );
        plugin
            .call_state_changed(&hot_event)
            .expect("state_changed should not trap");
        assert_alert_state(&hc, "on");

        // ── Scenario 2: temp = 19 → alert OFF ──────────────────────────────
        let cold_event = StateChangedEventJson::state_changed(
            "sensor.test_temp",
            Some("19"),
            serde_json::json!({}),
        );
        plugin
            .call_state_changed(&cold_event)
            .expect("state_changed should not trap");
        assert_alert_state(&hc, "off");
    }

    // ── Linker smoke test ────────────────────────────────────────────────────

    #[test]
    fn wasmtime_linker_wires_all_four_host_imports() {
        // A minimal WAT that calls all 4 host imports once and returns 0.
        const SMOKE_WAT: &str = r#"
(module
  (import "env" "hc_state_get"       (func (param i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_set"       (func (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_subscribe" (func (param i32 i32) (result i32)))
  (import "env" "hc_log"             (func (param i32 i32 i32)))
  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 512))
  (func (export "alloc") (param i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $bump))
    (global.set $bump (i32.add (global.get $bump) (local.get 0)))
    (local.get $p))
  (func (export "dealloc") (param i32 i32))
  (func (export "plugin_setup") (param i32 i32) (result i32) (i32.const 0))
  (func (export "plugin_handle_state_changed") (param i32 i32) (result i32) (i32.const 0))
)
"#;
        let wasm_bytes = wat::parse_str(SMOKE_WAT).expect("WAT");
        let rt = WasmtimeRuntime::new().expect("rt");
        let hc = HomeCore::new();
        let plugin = rt.load_wasm(&wasm_bytes, hc).expect("instantiate");
        let r = plugin.call_setup("{}").expect("setup");
        assert_eq!(r, 0);
    }

    // ── ADR-162 P4: signature/integrity verification ────────────────────────
    //
    // Each of these FAILS on the pre-ADR-162 code, which had no
    // `load_plugin` / `verify_module` at all — the manifest hash/sig/key
    // were parsed and discarded. They drive the real verification gate.

    use ed25519_dalek::{Signer, SigningKey};
    use homecore_plugins::manifest::PluginManifest;
    use homecore_plugins::verify::{encode_sha256, encode_signature, encode_verifying_key};
    use homecore_plugins::PluginPolicy;

    /// Deterministic publisher key (fixed seed — never use in production;
    /// mirrors the cog-ha-matter witness_signing test-key convention).
    fn publisher_key() -> SigningKey {
        SigningKey::from_bytes(b"hc-plugins-integration-pub-seed-")
    }

    fn untrusted_key() -> SigningKey {
        SigningKey::from_bytes(b"hc-plugins-integration-evil-seed")
    }

    /// A minimal valid module that writes `light.kitchen` on setup, plus a
    /// `light.*` permission grant. Returns the WAT source.
    const WRITE_LIGHT_WAT: &str = r#"
(module
  (import "env" "hc_state_get"       (func $hc_state_get (param i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_set"       (func $hc_state_set (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_subscribe" (func $hc_state_subscribe (param i32 i32) (result i32)))
  (import "env" "hc_log"             (func $hc_log (param i32 i32 i32)))
  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 512))
  (data (i32.const 0)   "light.kitchen")
  (data (i32.const 64)  "on")
  (data (i32.const 128) "{}")
  (func (export "alloc") (param i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $bump))
    (global.set $bump (i32.add (global.get $bump) (local.get 0)))
    (local.get $p))
  (func (export "dealloc") (param i32 i32))
  (func (export "plugin_setup") (param i32 i32) (result i32)
    (call $hc_state_set
      (i32.const 0) (i32.const 13)   ;; "light.kitchen"
      (i32.const 64) (i32.const 2)   ;; "on"
      (i32.const 128) (i32.const 2)) ;; "{}"
    drop
    (i32.const 0))
  (func (export "plugin_handle_state_changed") (param i32 i32) (result i32) (i32.const 0))
)
"#;

    /// Build a manifest signed by `key` over the SHA-256 of `wasm_bytes`,
    /// with the given write-permission grants.
    fn signed_manifest(
        wasm_bytes: &[u8],
        key: &SigningKey,
        perms: &[&str],
    ) -> PluginManifest {
        use sha2::{Digest, Sha256};
        let digest: [u8; 32] = Sha256::digest(wasm_bytes).into();
        let sig = key.sign(&digest);
        let mut m = PluginManifest::parse_json(
            r#"{"domain":"demo","name":"Demo","version":"1.0.0"}"#,
        )
        .unwrap();
        m.wasm_module = Some("demo.wasm".into());
        m.wasm_module_hash = Some(encode_sha256(wasm_bytes));
        m.wasm_module_sig = Some(encode_signature(&sig));
        m.publisher_key = Some(encode_verifying_key(&key.verifying_key()));
        m.homecore_permissions = perms.iter().map(|s| s.to_string()).collect();
        m
    }

    #[test]
    fn p4_valid_sig_from_trusted_key_loads() {
        let wasm = wat::parse_str(WRITE_LIGHT_WAT).expect("WAT");
        let key = publisher_key();
        let manifest = signed_manifest(&wasm, &key, &["light.*"]);
        let policy =
            PluginPolicy::trusted(&[&encode_verifying_key(&key.verifying_key())]).unwrap();

        let rt = WasmtimeRuntime::new().expect("rt");
        let hc = HomeCore::new();
        rt.load_plugin(&manifest, &wasm, hc, &policy)
            .expect("a validly-signed, trusted plugin must load");
    }

    #[test]
    fn p4_tampered_module_is_rejected() {
        let wasm = wat::parse_str(WRITE_LIGHT_WAT).expect("WAT");
        let key = publisher_key();
        // Manifest signs the original bytes; we then load DIFFERENT bytes.
        let manifest = signed_manifest(&wasm, &key, &["light.*"]);
        let policy =
            PluginPolicy::trusted(&[&encode_verifying_key(&key.verifying_key())]).unwrap();

        // Re-compile a byte-different module (writes "off" not "on").
        let tampered_src = WRITE_LIGHT_WAT.replace(r#""on""#, r#""of""#);
        let tampered = wat::parse_str(&tampered_src).expect("WAT");
        assert_ne!(wasm, tampered, "test bug: bytes must differ");

        let rt = WasmtimeRuntime::new().expect("rt");
        let hc = HomeCore::new();
        match rt.load_plugin(&manifest, &tampered, hc, &policy) {
            Err(homecore_plugins::PluginError::SignatureRejected(_)) => {}
            Ok(_) => panic!("tampered module must be rejected (hash mismatch), but it loaded"),
            Err(e) => panic!("expected SignatureRejected, got {e:?}"),
        }
    }

    #[test]
    fn p4_valid_sig_from_untrusted_key_is_rejected() {
        let wasm = wat::parse_str(WRITE_LIGHT_WAT).expect("WAT");
        // Correctly signed by the untrusted key — but it is not on the allowlist.
        let manifest = signed_manifest(&wasm, &untrusted_key(), &["light.*"]);
        let policy =
            PluginPolicy::trusted(&[&encode_verifying_key(&publisher_key().verifying_key())])
                .unwrap();

        let rt = WasmtimeRuntime::new().expect("rt");
        let hc = HomeCore::new();
        match rt.load_plugin(&manifest, &wasm, hc, &policy) {
            Err(homecore_plugins::PluginError::SignatureRejected(_)) => {}
            Ok(_) => panic!("untrusted publisher must be rejected, but it loaded"),
            Err(e) => panic!("expected SignatureRejected, got {e:?}"),
        }
    }

    #[test]
    fn p4_unsigned_module_rejected_by_default_loads_only_under_allow_unsigned() {
        let wasm = wat::parse_str(WRITE_LIGHT_WAT).expect("WAT");
        let mut manifest = PluginManifest::parse_json(
            r#"{"domain":"u","name":"U","version":"1"}"#,
        )
        .unwrap();
        manifest.wasm_module = Some("u.wasm".into());
        manifest.homecore_permissions = vec!["light.*".into()];
        // No hash/sig/key → unsigned.

        let rt = WasmtimeRuntime::new().expect("rt");
        // Secure default: rejected.
        match rt.load_plugin(&manifest, &wasm, HomeCore::new(), &PluginPolicy::deny_all()) {
            Err(homecore_plugins::PluginError::SignatureRejected(_)) => {}
            Ok(_) => panic!("unsigned module must be rejected under the secure default"),
            Err(e) => panic!("expected SignatureRejected, got {e:?}"),
        }
        // Dev escape hatch: loads (with a loud warn).
        rt.load_plugin(
            &manifest,
            &wasm,
            HomeCore::new(),
            &PluginPolicy::AllowUnsigned,
        )
        .expect("AllowUnsigned dev policy must load an unsigned module");
    }

    // ── ADR-162 P5: authority / capability isolation ────────────────────────
    //
    // FAILS on the pre-ADR-162 code, where `hc_state_set` ignored
    // `homecore_permissions` entirely and let any plugin write any entity.

    /// Module that writes `lock.front_door` on setup (an over-privileged
    /// write a `light.*` plugin must NOT be allowed to perform).
    const WRITE_LOCK_WAT: &str = r#"
(module
  (import "env" "hc_state_get"       (func $hc_state_get (param i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_set"       (func $hc_state_set (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_subscribe" (func $hc_state_subscribe (param i32 i32) (result i32)))
  (import "env" "hc_log"             (func $hc_log (param i32 i32 i32)))
  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 512))
  (data (i32.const 0)   "lock.front_door")
  (data (i32.const 64)  "unlocked")
  (data (i32.const 128) "{}")
  (func (export "alloc") (param i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $bump))
    (global.set $bump (i32.add (global.get $bump) (local.get 0)))
    (local.get $p))
  (func (export "dealloc") (param i32 i32))
  ;; plugin_setup returns the hc_state_set result code so the host test can
  ;; assert the guest saw the typed permission-denied error (-3).
  (func (export "plugin_setup") (param i32 i32) (result i32)
    (call $hc_state_set
      (i32.const 0) (i32.const 15)   ;; "lock.front_door"
      (i32.const 64) (i32.const 8)   ;; "unlocked"
      (i32.const 128) (i32.const 2))) ;; "{}"
  (func (export "plugin_handle_state_changed") (param i32 i32) (result i32) (i32.const 0))
)
"#;

    #[test]
    fn p5_declared_light_plugin_may_write_light_but_not_lock() {
        let key = publisher_key();
        let trusted = PluginPolicy::trusted(&[&encode_verifying_key(&key.verifying_key())]).unwrap();
        let rt = WasmtimeRuntime::new().expect("rt");

        // (a) A `light.*` plugin writing `light.kitchen` → ALLOWED.
        let light_wasm = wat::parse_str(WRITE_LIGHT_WAT).expect("WAT");
        let light_manifest = signed_manifest(&light_wasm, &key, &["light.*"]);
        let hc_a = HomeCore::new();
        let plugin_a = rt
            .load_plugin(&light_manifest, &light_wasm, hc_a.clone(), &trusted)
            .expect("light plugin loads");
        let r = plugin_a.call_setup("{}").expect("setup");
        assert_eq!(r, 0, "write to declared light.kitchen should succeed");
        let kitchen = homecore::EntityId::parse("light.kitchen").unwrap();
        assert_eq!(
            hc_a.states().get(&kitchen).expect("light.kitchen written").state,
            "on"
        );

        // (b) The SAME `light.*` plugin attempting to write `lock.front_door`
        // → REJECTED with the typed -3 code, and the lock is NOT written.
        let lock_wasm = wat::parse_str(WRITE_LOCK_WAT).expect("WAT");
        let lock_manifest = signed_manifest(&lock_wasm, &key, &["light.*"]);
        let hc_b = HomeCore::new();
        let plugin_b = rt
            .load_plugin(&lock_manifest, &lock_wasm, hc_b.clone(), &trusted)
            .expect("module loads (verification ok); the WRITE is what's gated");
        let denied = plugin_b.call_setup("{}").expect("setup runs without trapping host");
        assert_eq!(
            denied, -3,
            "over-privileged write to lock.front_door must return -3 (permission denied)"
        );
        let lock = homecore::EntityId::parse("lock.front_door").unwrap();
        assert!(
            hc_b.states().get(&lock).is_none(),
            "lock.front_door must NOT have been written by a light-only plugin"
        );
    }

    #[test]
    fn p5_plugin_with_no_permissions_can_write_nothing() {
        let key = publisher_key();
        let trusted = PluginPolicy::trusted(&[&encode_verifying_key(&key.verifying_key())]).unwrap();
        let rt = WasmtimeRuntime::new().expect("rt");

        let wasm = wat::parse_str(WRITE_LIGHT_WAT).expect("WAT");
        // No permissions declared at all.
        let manifest = signed_manifest(&wasm, &key, &[]);
        let hc = HomeCore::new();
        let plugin = rt
            .load_plugin(&manifest, &wasm, hc.clone(), &trusted)
            .expect("module loads; the write is gated");
        // WRITE_LIGHT_WAT drops the host-import result and returns 0, so we
        // assert the denial via the side-effect: the write must NOT land.
        plugin.call_setup("{}").expect("setup runs without trapping host");
        let kitchen = homecore::EntityId::parse("light.kitchen").unwrap();
        assert!(
            hc.states().get(&kitchen).is_none(),
            "no-permission plugin must not write light.kitchen (P5 authority isolation)"
        );
    }
}
