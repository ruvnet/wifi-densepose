//! Guest-side ABI helpers — matching `homecore-plugins/src/host_abi.rs`.
//!
//! # Memory model
//!
//! The host allocates into the guest's linear memory via the exported
//! `alloc` / `dealloc` functions. The guest calls host imports with
//! (ptr: i32, len: i32) pairs pointing into its own linear memory.
//!
//! # Allocator
//!
//! A simple bump allocator backed by a static mutable pointer. Suitable
//! only for the WASM guest context where the host drives all allocations
//! and deallocations synchronously (no concurrency inside a WASM module).
//!
//! # Wire format
//!
//! All host↔guest transfers use **UTF-8 JSON** (see host_abi.rs §Wire types).
//! Maximum buffer: 65,536 bytes.

/// Maximum ABI buffer size — mirrors `MAX_ABI_BUFFER_BYTES` on the host.
pub const MAX_ABI_BUFFER_BYTES: usize = 65_536;

// ── Bump allocator ─────────────────────────────────────────────────────────

/// Start of heap area (bump pointer). Placed after the 64 KiB stack.
static mut BUMP: usize = 0x1_0000; // 64 KiB

/// Allocate `size` bytes from the bump heap. Returns the pointer.
///
/// # Safety
/// The caller must not write past `ptr + size`.
#[no_mangle]
pub unsafe extern "C" fn alloc(size: i32) -> i32 {
    if size <= 0 {
        return 0;
    }
    let size = size as usize;
    // Align to 8 bytes.
    let aligned = (BUMP + 7) & !7;
    BUMP = aligned + size;
    aligned as i32
}

/// Deallocate a buffer. No-op for the bump allocator — caller is the host,
/// which drives the alloc/dealloc lifecycle and calls this after each call.
#[no_mangle]
pub unsafe extern "C" fn dealloc(_ptr: i32, _size: i32) {
    // Bump allocator: no-op. For a real plugin, replace with a proper allocator.
}

// ── Host import declarations ───────────────────────────────────────────────

extern "C" {
    /// Read the current state for an entity. See host_abi.rs §hc_state_get.
    /// Returns bytes written into `out_ptr`, or -1 (not found), -2 (too small).
    pub fn hc_state_get(
        key_ptr: i32,
        key_len: i32,
        out_ptr: i32,
        out_cap: i32,
    ) -> i32;

    /// Write state for an entity. Returns 0 on success, negative on error.
    pub fn hc_state_set(
        eid_ptr: i32,
        eid_len: i32,
        state_ptr: i32,
        state_len: i32,
        attrs_ptr: i32,
        attrs_len: i32,
    ) -> i32;

    /// Subscribe to state changes for an entity. Returns 0 on success.
    pub fn hc_state_subscribe(eid_ptr: i32, eid_len: i32) -> i32;

    /// Log a message. level: 0=debug 1=info 2=warn 3=error.
    pub fn hc_log(level: i32, msg_ptr: i32, msg_len: i32);
}

// ── ABI helpers ────────────────────────────────────────────────────────────

/// Write entity state via `hc_state_set`.
///
/// Returns the result of `hc_state_set` (0 = ok).
///
/// # Safety
/// `entity_id`, `state`, and `attrs` must be valid UTF-8 strings.
pub fn set_state(entity_id: &str, state: &str, attrs: &str) -> i32 {
    unsafe {
        hc_state_set(
            entity_id.as_ptr() as i32,
            entity_id.len() as i32,
            state.as_ptr() as i32,
            state.len() as i32,
            attrs.as_ptr() as i32,
            attrs.len() as i32,
        )
    }
}

/// Emit a log message at INFO level.
pub fn log_info(msg: &str) {
    unsafe {
        hc_log(1, msg.as_ptr() as i32, msg.len() as i32);
    }
}
