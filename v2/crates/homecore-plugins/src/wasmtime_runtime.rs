//! `WasmtimeRuntime` — Cranelift JIT WASM plugin runtime (ADR-128 P2).
//!
//! # Design
//!
//! Each `.wasm` binary is compiled once per process by a shared [`Engine`].
//! Every call to [`WasmtimeRuntime::load_wasm`] creates a new [`Store`] so
//! plugins are fully isolated — one plugin cannot read another's linear memory.
//!
//! The 4 host imports the WASM module receives are registered via a [`Linker`]:
//!
//! | Import | Signature | Description |
//! |--------|-----------|-------------|
//! | `hc_state_get` | `(i32,i32,i32,i32)→i32` | Read entity state into guest buffer |
//! | `hc_state_set` | `(i32,i32,i32,i32,i32,i32)→i32` | Write entity state from guest buffer |
//! | `hc_state_subscribe` | `(i32,i32)→i32` | Subscribe to state-changed events |
//! | `hc_log` | `(i32,i32,i32)→()` | Structured log output from plugin |
//!
//! WASI is **not** imported — plugins have no filesystem or network access.
//!
//! # Memory convention
//!
//! The guest exports `alloc(size: i32) → i32` and `dealloc(ptr: i32, size: i32)`.
//! The host calls `alloc` before writing a buffer into guest memory, then calls
//! `dealloc` when done. See [`host_abi`] for the full ABI spec.

use std::sync::{Arc, Mutex};

use homecore::HomeCore;
use wasmtime::{Engine, Linker, Module, Store};

use crate::error::PluginError;
use crate::host_abi::{LogLevel, StateChangedEventJson, MAX_ABI_BUFFER_BYTES};
use crate::manifest::PluginManifest;
use crate::permissions::PermissionSet;
use crate::verify::{verify_module, PluginPolicy};

// ── Store data ─────────────────────────────────────────────────────────────

/// Per-plugin state stored inside the Wasmtime [`Store`].
///
/// Wasmtime's `Store<T>` exposes `T` to host functions via `caller.data()`.
/// We store the `HomeCore` handle, a list of subscribed entity IDs, and the
/// plugin's write-permission set (ADR-162 P5 authority isolation).
pub struct PluginStoreData {
    pub hc: HomeCore,
    pub subscriptions: Vec<String>,
    /// Entity-write authority distilled from the manifest's
    /// `homecore_permissions`. Consulted by `hc_state_set`. The
    /// permission-free [`WasmtimeRuntime::load_wasm`] path installs an
    /// all-allowing set for backward compatibility; the
    /// [`WasmtimeRuntime::load_plugin`] path installs the manifest's
    /// declared set.
    pub permissions: PermissionSet,
}

// ── WasmtimeRuntime ────────────────────────────────────────────────────────

/// Wasmtime-backed WASM plugin runtime (Cranelift JIT on Pi 5 and x86_64).
///
/// One `Engine` is shared across all plugins for module caching. Each plugin
/// gets its own isolated `Store`.
pub struct WasmtimeRuntime {
    engine: Engine,
}

impl WasmtimeRuntime {
    /// Create a new runtime with default Cranelift config.
    pub fn new() -> Result<Self, PluginError> {
        let engine = Engine::default();
        Ok(Self { engine })
    }

    /// Compile and instantiate a WASM plugin from raw bytes, **without**
    /// signature verification or permission gating (the plugin gets
    /// all-write authority).
    ///
    /// Retained for the legacy/test path and first-party trusted modules.
    /// Production plugin loading should go through [`Self::load_plugin`],
    /// which verifies the module (ADR-162 P4) and scopes its write
    /// authority to the manifest (P5).
    pub fn load_wasm(
        &self,
        wasm_bytes: &[u8],
        hc: HomeCore,
    ) -> Result<WasmPlugin, PluginError> {
        self.instantiate(wasm_bytes, hc, PermissionSet::allow_all())
    }

    /// Verify and instantiate a WASM plugin from its manifest + raw bytes.
    ///
    /// This is the secure load path (ADR-162):
    ///   1. **P4** — [`verify_module`] checks the SHA-256 module hash and
    ///      Ed25519 signature against the manifest under `policy`. A
    ///      tampered module, bad/forged signature, untrusted publisher, or
    ///      (under the secure default) an unsigned module is rejected
    ///      **before** any guest code runs.
    ///   2. **P5** — the plugin's `homecore_permissions` are distilled into
    ///      a [`PermissionSet`] installed in the store, so `hc_state_set`
    ///      can only write entities the plugin declared.
    pub fn load_plugin(
        &self,
        manifest: &PluginManifest,
        wasm_bytes: &[u8],
        hc: HomeCore,
        policy: &PluginPolicy,
    ) -> Result<WasmPlugin, PluginError> {
        // P4: verify before instantiation.
        verify_module(manifest, wasm_bytes, policy)?;
        // P5: scope write authority to the manifest's declared permissions.
        let permissions = PermissionSet::from_manifest(manifest);
        self.instantiate(wasm_bytes, hc, permissions)
    }

    /// Shared compile + instantiate, installing the given permission set.
    fn instantiate(
        &self,
        wasm_bytes: &[u8],
        hc: HomeCore,
        permissions: PermissionSet,
    ) -> Result<WasmPlugin, PluginError> {
        let module = Module::new(&self.engine, wasm_bytes)
            .map_err(|e| PluginError::RuntimeError(format!("WASM compile: {e}")))?;

        let mut linker: Linker<PluginStoreData> = Linker::new(&self.engine);
        register_host_imports(&mut linker)?;

        let store_data = PluginStoreData {
            hc,
            subscriptions: Vec::new(),
            permissions,
        };
        let mut store = Store::new(&self.engine, store_data);

        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| PluginError::RuntimeError(format!("WASM instantiate: {e}")))?;

        Ok(WasmPlugin {
            inner: Arc::new(Mutex::new((store, instance))),
        })
    }
}

impl Default for WasmtimeRuntime {
    fn default() -> Self {
        Self::new().expect("default Wasmtime engine should not fail")
    }
}

// ── Host import registration ───────────────────────────────────────────────

/// Register the 4 host imports every HOMECORE plugin can call.
fn register_host_imports(
    linker: &mut Linker<PluginStoreData>,
) -> Result<(), PluginError> {
    register_hc_state_get(linker)?;
    register_hc_state_set(linker)?;
    register_hc_state_subscribe(linker)?;
    register_hc_log(linker)?;
    Ok(())
}

/// `hc_state_get(key_ptr: i32, key_len: i32, out_ptr: i32, out_cap: i32) → i32`
///
/// Reads the current state for the entity whose UTF-8 ID is in the guest
/// buffer at `[key_ptr, key_ptr+key_len)`. Writes the JSON-encoded state
/// into `[out_ptr, out_ptr+out_cap)`. Returns the number of bytes written,
/// or -1 if the entity is not found, or -2 if `out_cap` is too small.
fn register_hc_state_get(
    linker: &mut Linker<PluginStoreData>,
) -> Result<(), PluginError> {
    linker
        .func_wrap(
            "env",
            "hc_state_get",
            |mut caller: wasmtime::Caller<'_, PluginStoreData>,
             key_ptr: i32,
             key_len: i32,
             out_ptr: i32,
             out_cap: i32|
             -> i32 {
                // Phase 1: read the entity key from guest memory.
                let key: String = {
                    let mem = match caller.get_export("memory") {
                        Some(wasmtime::Extern::Memory(m)) => m,
                        _ => return -1,
                    };
                    match read_str(mem.data(&caller), key_ptr, key_len) {
                        Some(k) => k.to_owned(),
                        None => return -1,
                    }
                };

                // Phase 2: look up state and build JSON (no borrow on caller).
                let entity_id = match homecore::EntityId::parse(&key) {
                    Ok(id) => id,
                    Err(_) => return -1,
                };
                let json_bytes: Vec<u8> = {
                    let state_arc = match caller.data().hc.states().get(&entity_id) {
                        Some(s) => s,
                        None => return -1,
                    };
                    match serde_json::to_vec(&*state_arc) {
                        Ok(v) => v,
                        Err(_) => return -1,
                    }
                };

                if json_bytes.len() > out_cap as usize {
                    return -2;
                }

                // Phase 3: write JSON back into guest memory.
                let mem = match caller.get_export("memory") {
                    Some(wasmtime::Extern::Memory(m)) => m,
                    _ => return -1,
                };
                let end = out_ptr as usize + json_bytes.len();
                let out = match mem.data_mut(&mut caller).get_mut(out_ptr as usize..end) {
                    Some(s) => s,
                    None => return -1,
                };
                out.copy_from_slice(&json_bytes);
                json_bytes.len() as i32
            },
        )
        .map_err(|e| PluginError::RuntimeError(format!("register hc_state_get: {e}")))?;
    Ok(())
}

/// `hc_state_set(eid_ptr,eid_len,state_ptr,state_len,attrs_ptr,attrs_len) → i32`
///
/// Sets the state for the entity whose UTF-8 ID is at `[eid_ptr,eid_ptr+eid_len)`.
/// The new state string is at `[state_ptr,state_ptr+state_len)`.
/// The attributes JSON is at `[attrs_ptr,attrs_ptr+attrs_len)`.
/// Returns 0 on success, negative on error: -1 (bad memory/args), -2
/// (invalid entity id), -3 (permission denied — entity not in the
/// plugin's declared `homecore_permissions`, ADR-162 P5).
fn register_hc_state_set(
    linker: &mut Linker<PluginStoreData>,
) -> Result<(), PluginError> {
    linker
        .func_wrap(
            "env",
            "hc_state_set",
            |mut caller: wasmtime::Caller<'_, PluginStoreData>,
             eid_ptr: i32,
             eid_len: i32,
             state_ptr: i32,
             state_len: i32,
             attrs_ptr: i32,
             attrs_len: i32|
             -> i32 {
                // Read all strings from guest memory in one borrow.
                let (eid, new_state, attrs_str) = {
                    let mem = match caller.get_export("memory") {
                        Some(wasmtime::Extern::Memory(m)) => m,
                        _ => return -1,
                    };
                    let data = mem.data(&caller);
                    let eid = match read_str(data, eid_ptr, eid_len) {
                        Some(s) => s.to_owned(),
                        None => return -1,
                    };
                    let new_state = match read_str(data, state_ptr, state_len) {
                        Some(s) => s.to_owned(),
                        None => return -1,
                    };
                    let attrs_str = read_str(data, attrs_ptr, attrs_len)
                        .unwrap_or("{}")
                        .to_owned();
                    (eid, new_state, attrs_str)
                };

                let entity_id = match homecore::EntityId::parse(&eid) {
                    Ok(id) => id,
                    Err(_) => return -2,
                };

                // ── P5 authority isolation (ADR-162) ──────────────────────
                // Reject a write to an entity the plugin did not declare in
                // `homecore_permissions`. Return a typed error code to the
                // guest (-3); do NOT panic the host.
                if !caller.data().permissions.may_write(entity_id.as_str()) {
                    eprintln!(
                        "[PLUGIN WARN] denied hc_state_set on `{}` — not in plugin's declared \
                         homecore_permissions (P5 authority isolation)",
                        entity_id.as_str()
                    );
                    return -3;
                }

                let attrs: serde_json::Value =
                    serde_json::from_str(&attrs_str).unwrap_or(serde_json::json!({}));

                caller
                    .data()
                    .hc
                    .states()
                    .set(entity_id, new_state, attrs, homecore::Context::new());
                0
            },
        )
        .map_err(|e| PluginError::RuntimeError(format!("register hc_state_set: {e}")))?;
    Ok(())
}

/// `hc_state_subscribe(eid_ptr: i32, eid_len: i32) → i32`
///
/// Records a subscription so the host will call `receive_event` on future
/// state changes for this entity. Returns 0 on success, -1 on invalid entity.
fn register_hc_state_subscribe(
    linker: &mut Linker<PluginStoreData>,
) -> Result<(), PluginError> {
    linker
        .func_wrap(
            "env",
            "hc_state_subscribe",
            |mut caller: wasmtime::Caller<'_, PluginStoreData>,
             eid_ptr: i32,
             eid_len: i32|
             -> i32 {
                let eid: String = {
                    let mem = match caller.get_export("memory") {
                        Some(wasmtime::Extern::Memory(m)) => m,
                        _ => return -1,
                    };
                    match read_str(mem.data(&caller), eid_ptr, eid_len) {
                        Some(s) => s.to_owned(),
                        None => return -1,
                    }
                };
                caller.data_mut().subscriptions.push(eid);
                0
            },
        )
        .map_err(|e| PluginError::RuntimeError(format!("register hc_state_subscribe: {e}")))?;
    Ok(())
}

/// `hc_log(level: i32, msg_ptr: i32, msg_len: i32) → ()`
///
/// Structured log output from the plugin. `level`: 0=debug 1=info 2=warn 3=error.
fn register_hc_log(
    linker: &mut Linker<PluginStoreData>,
) -> Result<(), PluginError> {
    linker
        .func_wrap(
            "env",
            "hc_log",
            |mut caller: wasmtime::Caller<'_, PluginStoreData>,
             level: i32,
             msg_ptr: i32,
             msg_len: i32| {
                let mem = match caller.get_export("memory") {
                    Some(wasmtime::Extern::Memory(m)) => m,
                    _ => return,
                };
                let msg = read_str(mem.data(&caller), msg_ptr, msg_len)
                    .unwrap_or("(invalid utf8)")
                    .to_owned();
                let lvl = LogLevel::from_i32(level);
                eprintln!("[PLUGIN {}] {}", lvl.as_str(), msg);
            },
        )
        .map_err(|e| PluginError::RuntimeError(format!("register hc_log: {e}")))?;
    Ok(())
}

// ── WasmPlugin ─────────────────────────────────────────────────────────────

/// A loaded WASM plugin instance. Wraps a Wasmtime `Store` + `Instance`.
///
/// The `Arc<Mutex<_>>` allows the handle to be `Clone` + `Send` while
/// maintaining exclusive access for calls into the WASM module.
pub struct WasmPlugin {
    pub inner: Arc<Mutex<(Store<PluginStoreData>, wasmtime::Instance)>>,
}

impl WasmPlugin {
    /// Return a snapshot of the entity IDs this plugin has subscribed to.
    pub fn subscriptions(&self) -> Vec<String> {
        self.inner
            .lock()
            .map(|g| g.0.data().subscriptions.clone())
            .unwrap_or_default()
    }

    /// Call the `plugin_setup` export with the given config-entry JSON.
    pub fn call_setup(&self, config_entry_json: &str) -> Result<i32, PluginError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| PluginError::RuntimeError(format!("lock: {e}")))?;
        let (store, instance) = &mut *guard;
        call_export_str(store, instance, "plugin_setup", config_entry_json)
    }

    /// Call `plugin_handle_state_changed` with a [`StateChangedEventJson`].
    pub fn call_state_changed(
        &self,
        event: &StateChangedEventJson,
    ) -> Result<i32, PluginError> {
        let json = serde_json::to_string(event)
            .map_err(|e| PluginError::RuntimeError(format!("serialize event: {e}")))?;
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| PluginError::RuntimeError(format!("lock: {e}")))?;
        let (store, instance) = &mut *guard;
        call_export_str(store, instance, "plugin_handle_state_changed", &json)
    }
}

// ── Memory helpers ─────────────────────────────────────────────────────────

/// Read a UTF-8 string from guest linear memory.
fn read_str(mem: &[u8], ptr: i32, len: i32) -> Option<&str> {
    if len < 0 || len as usize > MAX_ABI_BUFFER_BYTES {
        return None;
    }
    let ptr = ptr as usize;
    let len = len as usize;
    let slice = mem.get(ptr..ptr + len)?;
    std::str::from_utf8(slice).ok()
}

/// Allocate a guest buffer via `alloc`, write `payload`, call `export_fn(ptr, len)`,
/// then free via `dealloc`. Returns the i32 result of the guest export.
fn call_export_str(
    store: &mut Store<PluginStoreData>,
    instance: &wasmtime::Instance,
    export_fn: &str,
    payload: &str,
) -> Result<i32, PluginError> {
    let payload_bytes = payload.as_bytes().to_vec(); // owned copy avoids reborrow issues
    let payload_len = payload_bytes.len() as i32;

    // 1. Allocate guest buffer.
    let alloc = instance
        .get_typed_func::<i32, i32>(&mut *store, "alloc")
        .map_err(|e| PluginError::RuntimeError(format!("get alloc: {e}")))?;
    let ptr = alloc
        .call(&mut *store, payload_len)
        .map_err(|e| PluginError::RuntimeError(format!("call alloc: {e}")))?;

    // 2. Write payload into guest memory.
    {
        let mem = instance
            .get_memory(&mut *store, "memory")
            .ok_or_else(|| PluginError::RuntimeError("no memory export".into()))?;
        let guest_slice = mem
            .data_mut(&mut *store)
            .get_mut(ptr as usize..ptr as usize + payload_bytes.len())
            .ok_or_else(|| PluginError::RuntimeError("guest memory OOB".into()))?;
        guest_slice.copy_from_slice(&payload_bytes);
    }

    // 3. Call the guest export.
    let func = instance
        .get_typed_func::<(i32, i32), i32>(&mut *store, export_fn)
        .map_err(|e| PluginError::RuntimeError(format!("get {export_fn}: {e}")))?;
    let result = func
        .call(&mut *store, (ptr, payload_len))
        .map_err(|e| PluginError::RuntimeError(format!("call {export_fn}: {e}")))?;

    // 4. Free the guest buffer.
    let dealloc = instance
        .get_typed_func::<(i32, i32), ()>(&mut *store, "dealloc")
        .map_err(|e| PluginError::RuntimeError(format!("get dealloc: {e}")))?;
    dealloc
        .call(&mut *store, (ptr, payload_len))
        .map_err(|e| PluginError::RuntimeError(format!("call dealloc: {e}")))?;

    Ok(result)
}

// ── Unit tests (using inline WAT) ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal WAT module that implements all host imports as no-ops and
    /// exports `alloc` / `dealloc` / `plugin_setup` /
    /// `plugin_handle_state_changed`. Compiled at test time via `wat::parse_str`.
    ///
    /// The `hc_state_set` call in the test plugin writes back a hard-coded
    /// entity via the host import (the host import will actually call back into
    /// the HomeCore state machine via `caller.data()`).
    const TEST_WAT: &str = r#"
(module
  ;; Host imports
  (import "env" "hc_state_get"
    (func $hc_state_get (param i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_set"
    (func $hc_state_set (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_subscribe"
    (func $hc_state_subscribe (param i32 i32) (result i32)))
  (import "env" "hc_log"
    (func $hc_log (param i32 i32 i32)))

  ;; Linear memory: 1 page = 64 KiB
  (memory (export "memory") 1)

  ;; Simple bump allocator state
  (global $bump (mut i32) (i32.const 1024))

  ;; alloc(size) → ptr
  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $bump))
    (global.set $bump (i32.add (global.get $bump) (local.get $size)))
    (local.get $ptr)
  )

  ;; dealloc(ptr, size) — no-op in bump allocator
  (func (export "dealloc") (param i32 i32))

  ;; plugin_setup(ptr, len) → 0
  (func (export "plugin_setup") (param i32 i32) (result i32)
    (i32.const 0)
  )

  ;; plugin_handle_state_changed(ptr, len) → 0
  ;; Calls hc_log with a fixed message so we can observe the import works.
  (func (export "plugin_handle_state_changed") (param i32 i32) (result i32)
    ;; log "ok" at INFO level — offset 0 in memory, write "ok" there first
    (i32.store8 (i32.const 0) (i32.const 111)) ;; 'o'
    (i32.store8 (i32.const 1) (i32.const 107)) ;; 'k'
    (call $hc_log (i32.const 1) (i32.const 0) (i32.const 2))
    (i32.const 0)
  )
)
"#;

    #[test]
    fn wasmtime_runtime_compiles_and_instantiates_wat() {
        let wasm_bytes = wat::parse_str(TEST_WAT).expect("WAT should parse");
        let rt = WasmtimeRuntime::new().expect("engine should init");
        let hc = HomeCore::new();
        let plugin = rt.load_wasm(&wasm_bytes, hc).expect("should instantiate");

        // call plugin_setup — expect 0
        let r = plugin
            .call_setup(r#"{"entry_id":"test","domain":"test","title":"test","data":{}}"#)
            .expect("setup should not error");
        assert_eq!(r, 0, "plugin_setup should return 0");
    }

    #[test]
    fn hc_state_set_round_trip_via_wat() {
        /// WAT plugin that calls hc_state_set to write "on" for binary_sensor.test_alert
        const SET_WAT: &str = r#"
(module
  (import "env" "hc_state_get"
    (func $hc_state_get (param i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_set"
    (func $hc_state_set (param i32 i32 i32 i32 i32 i32) (result i32)))
  (import "env" "hc_state_subscribe"
    (func $hc_state_subscribe (param i32 i32) (result i32)))
  (import "env" "hc_log"
    (func $hc_log (param i32 i32 i32)))

  (memory (export "memory") 1)
  (global $bump (mut i32) (i32.const 2048))

  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $bump))
    (global.set $bump (i32.add (global.get $bump) (local.get $size)))
    (local.get $ptr)
  )
  (func (export "dealloc") (param i32 i32))

  ;; Strings stored at known offsets in memory:
  ;; offset 0: "binary_sensor.test_alert" (24 bytes)
  ;; offset 64: "on" (2 bytes)
  ;; offset 128: "{}" (2 bytes)
  (data (i32.const 0) "binary_sensor.test_alert")
  (data (i32.const 64) "on")
  (data (i32.const 128) "{}")

  ;; plugin_setup: call hc_state_set to write "on"
  (func (export "plugin_setup") (param i32 i32) (result i32)
    (call $hc_state_set
      (i32.const 0)   ;; eid_ptr
      (i32.const 24)  ;; eid_len  = len("binary_sensor.test_alert")
      (i32.const 64)  ;; state_ptr
      (i32.const 2)   ;; state_len = len("on")
      (i32.const 128) ;; attrs_ptr
      (i32.const 2)   ;; attrs_len = len("{}")
    )
    drop
    (i32.const 0)
  )

  (func (export "plugin_handle_state_changed") (param i32 i32) (result i32)
    (i32.const 0)
  )
)
"#;
        let wasm_bytes = wat::parse_str(SET_WAT).expect("WAT should parse");
        let rt = WasmtimeRuntime::new().expect("engine");
        let hc = HomeCore::new();
        let plugin = rt.load_wasm(&wasm_bytes, hc.clone()).expect("instantiate");

        // Call plugin_setup — the WAT calls hc_state_set inside.
        plugin.call_setup("{}").expect("setup");

        // Verify the host state machine saw the write.
        let eid = homecore::EntityId::parse("binary_sensor.test_alert").unwrap();
        let state = hc.states().get(&eid).expect("state should exist");
        assert_eq!(
            state.state, "on",
            "hc_state_set via host import should write 'on'"
        );
    }
}
