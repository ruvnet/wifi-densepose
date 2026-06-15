//! Checkpoint-loading robustness tests for `crate::model::load_safetensors`.
//!
//! Security review (Milestone #9, crate 4/4). These tests pin the behaviour of
//! the SafeTensors weight-loading path against malformed / degenerate
//! checkpoints — the only externally-controlled file-input surface in the crate.
//!
//! The headline regression is the **int32 dtype-widening byte-size bug**
//! (`security/occworld-candle` finding #1): `model.rs` mapped
//! `safetensors::Dtype::I32` → `candle_core::DType::I64` and then handed the
//! raw *int32* byte buffer (4 bytes/elem) to `Tensor::from_raw_buffer(.., I64,
//! shape, ..)`. Candle's `from_raw_buffer` computes `elem_count =
//! data.len() / 8`, producing a tensor whose declared shape claims twice as
//! many elements as the backing storage actually holds — a silent
//! shape/storage inconsistency on attacker-supplied checkpoints.
//!
//! `build_safetensors` hand-assembles the binary container
//! (`<u64 LE header_len><JSON header><raw data>`) so the test states exactly
//! what bytes reach the loader, independent of the `safetensors` writer API.

use candle_core::Device;
use wifi_densepose_occworld_candle::model::load_safetensors;

/// Hand-build a single-tensor SafeTensors buffer.
///
/// `dtype` is the safetensors dtype string (e.g. `"I32"`, `"F32"`).
/// `shape` is the declared shape. `data` is the raw little-endian tensor bytes
/// — the caller is responsible for making `data.len()` consistent with
/// `shape × dtype_size` (safetensors itself validates this, so an inconsistent
/// pair is rejected before reaching the candle conversion).
fn build_safetensors(name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_json: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let header = format!(
        "{{\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[0,{}]}}}}",
        shape_json.join(","),
        data.len()
    );
    let header_bytes = header.into_bytes();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(data);
    buf
}

fn write_temp(bytes: &[u8], stem: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "occworld_ckpt_{stem}_{}_{}.safetensors",
        std::process::id(),
        // nanosecond-ish disambiguator so parallel tests never collide
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&p, bytes).expect("write temp checkpoint");
    p
}

/// REGRESSION (finding #1): an int32 tensor in a checkpoint must load into a
/// tensor whose element count matches its declared shape.
///
/// On the OLD code (`I32 -> DType::I64`) the 6-element int32 tensor below was
/// handed to `from_raw_buffer(.., I64, [2,3], ..)`, which derived
/// `elem_count = 24 bytes / 8 = 3` and built a 3-element storage carrying a
/// shape claiming 6 elements — reading it panicked with a slice-OOB
/// (`range end index 6 out of range for slice of length 3`). On the FIXED code
/// (`I32 -> DType::I32`) the tensor round-trips: dtype I32, 6 elements,
/// values `[1,2,3,4,5,6]`.
#[test]
fn int32_tensor_loads_with_consistent_shape_and_values() {
    let device = Device::Cpu;
    let shape = [2usize, 3];
    let vals: [i32; 6] = [1, 2, 3, 4, 5, 6];
    let mut data = Vec::with_capacity(24);
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let bytes = build_safetensors("quantize.embedding.weight", "I32", &shape, &data);
    let path = write_temp(&bytes, "i32");

    let map = load_safetensors(&path, &device).expect("int32 checkpoint must load");
    let t = map
        .get("quantize.embedding.weight")
        .expect("mapped key present");

    // The declared shape's element count MUST equal the storage's element
    // count. On the old code these disagreed (6 vs 3).
    assert_eq!(
        t.dims(),
        &[2, 3],
        "int32 tensor must preserve its declared shape"
    );
    assert_eq!(
        t.elem_count(),
        6,
        "element count must match shape — storage/shape consistency"
    );

    // The dtype must be I32 — the int32 byte buffer is interpreted as int32,
    // not reinterpreted as half as many int64 lanes.
    assert_eq!(
        t.dtype(),
        candle_core::DType::I32,
        "int32 checkpoint tensor must load as DType::I32"
    );

    // And the values must be exactly recovered (no reinterpretation of two
    // int32 lanes as one int64). This is the strongest proof the dtype is
    // handled correctly end-to-end.
    let flat = t.flatten_all().expect("flatten");
    let got: Vec<i32> = flat.to_vec1::<i32>().expect("to_vec i32");
    assert_eq!(
        got,
        vec![1i32, 2, 3, 4, 5, 6],
        "int32 values must be recovered exactly"
    );

    let _ = std::fs::remove_file(&path);
}

/// A well-formed F32 tensor must round-trip unchanged (control case — proves
/// the fix does not regress the common float path).
#[test]
fn f32_tensor_round_trips() {
    let device = Device::Cpu;
    let shape = [4usize];
    let vals: [f32; 4] = [0.5, -1.0, 2.25, 3.0];
    let mut data = Vec::with_capacity(16);
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let bytes = build_safetensors("post_quant_conv.bias", "F32", &shape, &data);
    let path = write_temp(&bytes, "f32");

    let map = load_safetensors(&path, &device).expect("f32 checkpoint must load");
    let t = map.get("post_quant_conv.bias").expect("key present");
    assert_eq!(t.dims(), &[4]);
    let got: Vec<f32> = t.to_vec1::<f32>().expect("to_vec f32");
    assert_eq!(got, vec![0.5, -1.0, 2.25, 3.0]);

    let _ = std::fs::remove_file(&path);
}

/// A truncated / corrupt header must produce a parse error, never a panic.
/// (Defense-in-depth: the loader is fed an untrusted file.)
#[test]
fn corrupt_checkpoint_errors_cleanly() {
    let device = Device::Cpu;
    // Garbage that is not a valid SafeTensors container.
    let bytes = vec![0xFFu8; 32];
    let path = write_temp(&bytes, "corrupt");

    let result = load_safetensors(&path, &device);
    assert!(
        result.is_err(),
        "corrupt checkpoint must error, got Ok: {result:?}"
    );

    let _ = std::fs::remove_file(&path);
}

/// An int64 tensor must still load correctly (proves the fix narrows only the
/// I32 mapping and leaves the genuine I64 path intact).
#[test]
fn int64_tensor_round_trips() {
    let device = Device::Cpu;
    let shape = [3usize];
    let vals: [i64; 3] = [10, -20, 30];
    let mut data = Vec::with_capacity(24);
    for v in vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let bytes = build_safetensors("transformer.output_head.bias", "I64", &shape, &data);
    let path = write_temp(&bytes, "i64");

    let map = load_safetensors(&path, &device).expect("i64 checkpoint must load");
    let t = map.get("transformer.output_head.bias").expect("key present");
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.elem_count(), 3);
    let got: Vec<i64> = t.to_vec1::<i64>().expect("to_vec i64");
    assert_eq!(got, vec![10, -20, 30]);

    let _ = std::fs::remove_file(&path);
}
