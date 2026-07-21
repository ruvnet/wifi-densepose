//! Safe Rust ownership boundary for the optional depth-anything.cpp C ABI.
//!
//! The native engine and model are runtime-provided. RuView neither links the
//! engine at build time nor bundles third-party model weights.

use crate::depth::{CameraIntrinsics, DepthEstimate};
use anyhow::{bail, Context, Result};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;

const REQUIRED_ABI_VERSION: i32 = 4;
const MAX_OUTPUT_PIXELS: usize = 16_777_216;
const NATIVE_THREAD_STACK_BYTES: usize = 16 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

type AbiVersionFn = unsafe extern "C" fn() -> i32;
type LoadFn = unsafe extern "C" fn(*const c_char, i32) -> *mut c_void;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type LastErrorFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type DepthDenseFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *mut i32,
    *mut i32,
    *mut *mut f32,
    *mut *mut f32,
    *mut *mut f32,
    *mut f32,
    *mut f32,
    *mut i32,
) -> i32;
type FreeFloatsFn = unsafe extern "C" fn(*mut f32);

#[derive(Clone, Copy)]
struct Api {
    abi_version: AbiVersionFn,
    load: LoadFn,
    free: FreeFn,
    last_error: LastErrorFn,
    depth_dense: DepthDenseFn,
    free_floats: FreeFloatsFn,
}

#[derive(Clone, Debug)]
pub struct DepthAnythingConfig {
    pub library_path: PathBuf,
    pub model_path: PathBuf,
    pub model_license: String,
    pub model_sha256: Option<String>,
    pub threads: i32,
    pub allow_noncommercial: bool,
    pub min_confidence: f32,
}

impl DepthAnythingConfig {
    pub fn from_env() -> Result<Self> {
        let library_path = required_env("RUVIEW_DA_LIBRARY")?.into();
        let model_path = required_env("RUVIEW_DA_MODEL")?.into();
        let model_license = required_env("RUVIEW_DA_MODEL_LICENSE")?;
        let model_sha256 = std::env::var("RUVIEW_DA_MODEL_SHA256")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let threads = std::env::var("RUVIEW_DA_THREADS")
            .ok()
            .map(|value| {
                value
                    .parse::<i32>()
                    .context("RUVIEW_DA_THREADS must be an integer")
            })
            .transpose()?
            .unwrap_or_else(default_threads);
        if threads <= 0 {
            bail!("RUVIEW_DA_THREADS must be greater than zero");
        }
        let min_confidence = std::env::var("RUVIEW_DA_MIN_CONFIDENCE")
            .ok()
            .map(|value| {
                value
                    .parse::<f32>()
                    .context("RUVIEW_DA_MIN_CONFIDENCE must be a finite number")
            })
            .transpose()?
            .unwrap_or(1.0);
        if !min_confidence.is_finite() {
            bail!("RUVIEW_DA_MIN_CONFIDENCE must be finite");
        }

        Ok(Self {
            library_path,
            model_path,
            model_license,
            model_sha256,
            threads,
            allow_noncommercial: env_truthy("RUVIEW_DA_ALLOW_NONCOMMERCIAL"),
            min_confidence,
        })
    }

    pub fn validate(&self) -> Result<()> {
        validate_model_license(&self.model_license, self.allow_noncommercial)?;
        if !self.library_path.is_file() {
            bail!(
                "Depth Anything shared library not found: {}",
                self.library_path.display()
            );
        }
        if !self.model_path.is_file() {
            bail!(
                "Depth Anything model not found: {}",
                self.model_path.display()
            );
        }
        if let Some(expected) = &self.model_sha256 {
            verify_sha256(&self.model_path, expected)?;
        }
        Ok(())
    }
}

struct NativeDepthAnything {
    // Function pointers remain valid only while this library is alive.
    _library: Option<Library>,
    api: Api,
    ctx: *mut c_void,
    min_confidence: f32,
}

impl NativeDepthAnything {
    fn load(config: &DepthAnythingConfig) -> Result<Self> {
        // SAFETY: loading a user-configured native library is the explicit opt-in
        // of this backend. Symbols are checked immediately and the Library is kept
        // alive in the returned value for longer than every copied function pointer.
        let library = unsafe { Library::new(&config.library_path) }.with_context(|| {
            format!(
                "failed to load Depth Anything library {}",
                config.library_path.display()
            )
        })?;
        let api = unsafe { load_api(&library)? };
        let actual_abi = unsafe { (api.abi_version)() };
        if actual_abi != REQUIRED_ABI_VERSION {
            bail!(
                "unsupported depth-anything.cpp ABI {actual_abi}; required {REQUIRED_ABI_VERSION}"
            );
        }

        let model = path_to_cstring(&config.model_path)?;
        // SAFETY: model is NUL-terminated for the duration of the call; threads
        // was validated positive; a null context is handled as a load failure.
        let ctx = unsafe { (api.load)(model.as_ptr(), config.threads) };
        if ctx.is_null() {
            bail!(
                "depth-anything.cpp rejected model {}",
                config.model_path.display()
            );
        }

        Ok(Self {
            _library: Some(library),
            api,
            ctx,
            min_confidence: config.min_confidence,
        })
    }

    pub fn infer_rgb(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<DepthEstimate> {
        let image = TemporaryPpm::create(rgb, width, height)?;
        let image_path = path_to_cstring(image.path())?;

        let mut out_h = 0i32;
        let mut out_w = 0i32;
        let mut depth = ptr::null_mut();
        let mut confidence = ptr::null_mut();
        let mut sky = ptr::null_mut();
        let mut extrinsics = [0.0f32; 12];
        let mut intrinsics = [0.0f32; 9];
        let mut is_metric = 0i32;

        // SAFETY: every output pointer references live Rust storage. The context
        // is non-null and exclusively borrowed. Native float buffers are copied
        // before being released with the matching upstream deallocator.
        let rc = unsafe {
            (self.api.depth_dense)(
                self.ctx,
                image_path.as_ptr(),
                &mut out_h,
                &mut out_w,
                &mut depth,
                &mut confidence,
                &mut sky,
                extrinsics.as_mut_ptr(),
                intrinsics.as_mut_ptr(),
                &mut is_metric,
            )
        };
        if rc != 0 {
            unsafe { self.free_outputs(depth, confidence, sky) };
            bail!("depth-anything.cpp inference failed: {}", self.last_error());
        }

        let result = (|| {
            if out_h <= 0 || out_w <= 0 {
                bail!("depth-anything.cpp returned invalid dimensions {out_w}x{out_h}");
            }
            let len = (out_h as usize)
                .checked_mul(out_w as usize)
                .filter(|len| *len <= MAX_OUTPUT_PIXELS)
                .context("depth-anything.cpp output dimensions exceed safety limit")?;
            if depth.is_null() {
                bail!("depth-anything.cpp returned a null depth buffer");
            }

            // SAFETY: successful ABI v4 calls promise H*W floats for each non-null
            // dense output. The safety limit above also bounds the copy allocation.
            let depth_vec = unsafe { std::slice::from_raw_parts(depth, len).to_vec() };
            let confidence_vec = if confidence.is_null() {
                None
            } else {
                Some(unsafe { std::slice::from_raw_parts(confidence, len).to_vec() })
            };

            if is_metric != 1 {
                bail!(
                    "Depth Anything model returned relative depth; RuView point-cloud fusion requires metric depth"
                );
            }
            if depth_vec.iter().any(|value| !value.is_finite()) {
                bail!("Depth Anything returned non-finite depth values");
            }

            let output_width = out_w as u32;
            let output_height = out_h as u32;
            let camera =
                CameraIntrinsics::from_matrix_or_default(&intrinsics, output_width, output_height);
            let camera_pose = if extrinsics.iter().any(|value| *value != 0.0) {
                Some(extrinsics)
            } else {
                None
            };
            Ok(DepthEstimate {
                depth: depth_vec,
                confidence: confidence_vec,
                width: output_width,
                height: output_height,
                intrinsics: camera,
                extrinsics: camera_pose,
                is_metric: true,
                backend: "depth-anything",
                min_confidence: self.min_confidence,
            })
        })();

        unsafe { self.free_outputs(depth, confidence, sky) };
        result
    }

    fn last_error(&self) -> String {
        // SAFETY: the ABI documents this pointer as context-owned and valid until
        // the next context operation. Copy it immediately and tolerate null/UTF-8.
        let ptr = unsafe { (self.api.last_error)(self.ctx) };
        if ptr.is_null() {
            return "unknown native error".into();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    unsafe fn free_outputs(&self, depth: *mut f32, confidence: *mut f32, sky: *mut f32) {
        for output in [depth, confidence, sky] {
            if !output.is_null() {
                // SAFETY: each pointer came from this ABI call and is released at
                // most once through the exact deallocator exported by that ABI.
                unsafe { (self.api.free_floats)(output) };
            }
        }
    }

    #[cfg(test)]
    fn from_test_api(api: Api, min_confidence: f32) -> Result<Self> {
        let abi = unsafe { (api.abi_version)() };
        if abi != REQUIRED_ABI_VERSION {
            bail!("unsupported depth-anything.cpp ABI {abi}; required {REQUIRED_ABI_VERSION}");
        }
        let model = CString::new("mock.gguf").unwrap();
        let ctx = unsafe { (api.load)(model.as_ptr(), 1) };
        if ctx.is_null() {
            bail!("mock model load failed");
        }
        Ok(Self {
            _library: None,
            api,
            ctx,
            min_confidence,
        })
    }
}

/// Verify that a runtime library exposes the expected C symbols and return its
/// ABI version without loading model weights.
pub fn probe_library(path: &Path) -> Result<i32> {
    // SAFETY: the library stays alive until after all symbol calls complete.
    let library = unsafe { Library::new(path) }
        .with_context(|| format!("failed to load Depth Anything library {}", path.display()))?;
    let api = unsafe { load_api(&library)? };
    let version = unsafe { (api.abi_version)() };
    if version != REQUIRED_ABI_VERSION {
        bail!("unsupported depth-anything.cpp ABI {version}; required {REQUIRED_ABI_VERSION}");
    }
    Ok(version)
}

impl Drop for NativeDepthAnything {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: ctx was returned by api.load, remains owned by this wrapper,
            // and Drop is its single destruction point while the library is alive.
            unsafe { (self.api.free)(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

enum WorkerRequest {
    Infer {
        rgb: Vec<u8>,
        width: u32,
        height: u32,
        response: SyncSender<std::result::Result<DepthEstimate, String>>,
    },
    Shutdown,
}

/// Serialized, thread-affine handle to the native Depth Anything context.
///
/// The dedicated stack is required by the upstream Windows model loader and
/// also prevents a native context from moving between Tokio worker threads.
pub struct DepthAnythingBackend {
    requests: SyncSender<WorkerRequest>,
    worker: Option<JoinHandle<()>>,
}

impl DepthAnythingBackend {
    pub fn load(config: &DepthAnythingConfig) -> Result<Self> {
        config.validate()?;
        let config = config.clone();
        let (requests, incoming): (SyncSender<WorkerRequest>, Receiver<WorkerRequest>) =
            mpsc::sync_channel(1);
        let (initialized, initialization) = mpsc::sync_channel(0);
        let worker = std::thread::Builder::new()
            .name("ruview-depth-anything".into())
            .stack_size(NATIVE_THREAD_STACK_BYTES)
            .spawn(move || {
                let mut native = match NativeDepthAnything::load(&config) {
                    Ok(native) => {
                        let _ = initialized.send(Ok(()));
                        native
                    }
                    Err(error) => {
                        let _ = initialized.send(Err(format!("{error:#}")));
                        return;
                    }
                };
                while let Ok(request) = incoming.recv() {
                    match request {
                        WorkerRequest::Infer {
                            rgb,
                            width,
                            height,
                            response,
                        } => {
                            let result = native
                                .infer_rgb(&rgb, width, height)
                                .map_err(|error| format!("{error:#}"));
                            let _ = response.send(result);
                        }
                        WorkerRequest::Shutdown => {
                            #[cfg(windows)]
                            {
                                // The upstream ABI-v4 CPU DLL built with exported
                                // symbols currently access-violates in Engine's
                                // destructor after successful DA2 inference. Keep
                                // the process-singleton context and DLL loaded until
                                // Windows reclaims them at process termination.
                                std::mem::forget(native);
                                break;
                            }
                            #[cfg(not(windows))]
                            break;
                        }
                    }
                }
            })
            .context("failed to spawn Depth Anything native worker")?;

        match initialization.recv() {
            Ok(Ok(())) => Ok(Self {
                requests,
                worker: Some(worker),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                bail!("Depth Anything worker initialization failed: {error}")
            }
            Err(_) => {
                let _ = worker.join();
                bail!("Depth Anything worker terminated during initialization")
            }
        }
    }

    pub fn infer_rgb(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<DepthEstimate> {
        let (response, result) = mpsc::sync_channel(0);
        self.requests
            .send(WorkerRequest::Infer {
                rgb: rgb.to_vec(),
                width,
                height,
                response,
            })
            .context("Depth Anything worker is unavailable")?;
        result
            .recv()
            .context("Depth Anything worker stopped before returning inference")?
            .map_err(anyhow::Error::msg)
    }
}

impl Drop for DepthAnythingBackend {
    fn drop(&mut self) {
        let _ = self.requests.send(WorkerRequest::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

unsafe fn load_api(library: &Library) -> Result<Api> {
    // SAFETY: each symbol name and signature mirrors include/da_capi.h ABI v4.
    unsafe {
        Ok(Api {
            abi_version: *library.get(b"da_capi_abi_version\0")?,
            load: *library.get(b"da_capi_load\0")?,
            free: *library.get(b"da_capi_free\0")?,
            last_error: *library.get(b"da_capi_last_error\0")?,
            depth_dense: *library.get(b"da_capi_depth_dense\0")?,
            free_floats: *library.get(b"da_capi_free_floats\0")?,
        })
    }
}

struct TemporaryPpm {
    path: PathBuf,
}

impl TemporaryPpm {
    fn create(rgb: &[u8], width: u32, height: u32) -> Result<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .context("RGB dimensions overflow")?;
        if width == 0 || height == 0 || rgb.len() != expected {
            bail!(
                "RGB buffer length {} does not match {width}x{height} RGB8 ({expected})",
                rgb.len()
            );
        }

        for _ in 0..32 {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ruview-depth-{}-{sequence}.ppm",
                std::process::id()
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    if let Err(error) = write_ppm(&mut file, rgb, width, height) {
                        drop(file);
                        let _ = std::fs::remove_file(&path);
                        return Err(error);
                    }
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error).context("failed to create temporary PPM"),
            }
        }
        bail!("failed to allocate a unique temporary PPM path")
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryPpm {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn write_ppm(mut writer: impl Write, rgb: &[u8], width: u32, height: u32) -> Result<()> {
    write!(writer, "P6\n{width} {height}\n255\n")?;
    writer.write_all(rgb)?;
    writer.flush()?;
    Ok(())
}

fn validate_model_license(value: &str, allow_noncommercial: bool) -> Result<()> {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "apache-2.0" | "apache2" | "apache-2" => Ok(()),
        "cc-by-nc-4.0" | "cc-by-nc-4" if allow_noncommercial => Ok(()),
        "cc-by-nc-4.0" | "cc-by-nc-4" => bail!(
            "CC-BY-NC-4.0 model requires RUVIEW_DA_ALLOW_NONCOMMERCIAL=1; do not redistribute it as a general RuView asset"
        ),
        _ => bail!(
            "unsupported or unknown Depth Anything model license '{value}'; expected Apache-2.0"
        ),
    }
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let expected = expected.trim().to_ascii_lowercase();
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("RUVIEW_DA_MODEL_SHA256 must contain exactly 64 hexadecimal characters");
    }
    let mut file = File::open(path)
        .with_context(|| format!("failed to open model for hashing: {}", path.display()))?;
    let mut hasher = Sha256::new();
    // Heap-allocate: the default Windows main-thread stack is 1 MiB, so a
    // 1 MiB local array can overflow before the native worker is even started.
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        bail!("Depth Anything model SHA-256 mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .with_context(|| format!("path contains an interior NUL: {}", path.display()))
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .with_context(|| format!("{name} is required for the Depth Anything backend"))
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|value| value.get().min(i32::MAX as usize) as i32)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static CONTEXT_FREE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static ERROR: &[u8] = b"mock failure\0";

    unsafe extern "C" fn abi_v4() -> i32 {
        4
    }
    unsafe extern "C" fn abi_v3() -> i32 {
        3
    }
    unsafe extern "C" fn mock_load(_: *const c_char, _: i32) -> *mut c_void {
        Box::into_raw(Box::new(7u8)).cast()
    }
    unsafe extern "C" fn mock_free(ctx: *mut c_void) {
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(ctx.cast::<u8>()) });
            CONTEXT_FREE_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }
    unsafe extern "C" fn mock_error(_: *mut c_void) -> *const c_char {
        ERROR.as_ptr().cast()
    }
    unsafe extern "C" fn mock_free_floats(ptr: *mut f32) {
        if !ptr.is_null() {
            drop(unsafe { Vec::from_raw_parts(ptr, 4, 4) });
            FREE_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn leak_four(values: [f32; 4]) -> *mut f32 {
        let mut values = Vec::from(values);
        let pointer = values.as_mut_ptr();
        std::mem::forget(values);
        pointer
    }
    unsafe extern "C" fn mock_dense(
        _: *mut c_void,
        _: *const c_char,
        out_h: *mut i32,
        out_w: *mut i32,
        depth: *mut *mut f32,
        confidence: *mut *mut f32,
        sky: *mut *mut f32,
        ext: *mut f32,
        intr: *mut f32,
        metric: *mut i32,
    ) -> i32 {
        unsafe {
            *out_h = 2;
            *out_w = 2;
            *depth = leak_four([1.0, 2.0, 3.0, 4.0]);
            *confidence = leak_four([0.5, 1.0, 1.5, 2.0]);
            *sky = ptr::null_mut();
            std::slice::from_raw_parts_mut(ext, 12).copy_from_slice(&[1.0; 12]);
            std::slice::from_raw_parts_mut(intr, 9)
                .copy_from_slice(&[100.0, 0.0, 1.0, 0.0, 110.0, 1.0, 0.0, 0.0, 1.0]);
            *metric = 1;
        }
        0
    }

    fn mock_api(abi_version: AbiVersionFn) -> Api {
        Api {
            abi_version,
            load: mock_load,
            free: mock_free,
            last_error: mock_error,
            depth_dense: mock_dense,
            free_floats: mock_free_floats,
        }
    }

    #[test]
    fn license_policy_is_fail_closed() {
        assert!(validate_model_license("Apache-2.0", false).is_ok());
        assert!(validate_model_license("CC-BY-NC-4.0", false).is_err());
        assert!(validate_model_license("CC-BY-NC-4.0", true).is_ok());
        assert!(validate_model_license("unknown", true).is_err());
    }

    #[test]
    fn ppm_is_binary_rgb_and_self_cleaning() {
        let path = {
            let image = TemporaryPpm::create(&[255, 0, 0, 0, 255, 0], 2, 1).unwrap();
            let path = image.path().to_path_buf();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(&bytes[..11], b"P6\n2 1\n255\n");
            assert_eq!(&bytes[11..], &[255, 0, 0, 0, 255, 0]);
            path
        };
        assert!(!path.exists());
    }

    #[test]
    fn ppm_rejects_mismatched_buffer() {
        assert!(TemporaryPpm::create(&[0; 5], 2, 1).is_err());
    }

    #[test]
    fn sha256_verification_accepts_exact_digest() {
        let image = TemporaryPpm::create(&[1, 2, 3], 1, 1).unwrap();
        let bytes = std::fs::read(image.path()).unwrap();
        let digest = format!("{:x}", Sha256::digest(bytes));
        assert!(verify_sha256(image.path(), &digest).is_ok());
        assert!(verify_sha256(image.path(), &"0".repeat(64)).is_err());
    }

    #[test]
    fn abi_version_is_gated() {
        assert!(NativeDepthAnything::from_test_api(mock_api(abi_v3), 1.0).is_err());
    }

    #[test]
    fn mock_abi_copies_and_frees_native_outputs() {
        FREE_COUNT.store(0, Ordering::SeqCst);
        CONTEXT_FREE_COUNT.store(0, Ordering::SeqCst);
        {
            let mut backend = NativeDepthAnything::from_test_api(mock_api(abi_v4), 1.0).unwrap();
            let result = backend.infer_rgb(&[10, 20, 30], 1, 1).unwrap();
            assert_eq!(result.depth, vec![1.0, 2.0, 3.0, 4.0]);
            assert_eq!(result.confidence.unwrap(), vec![0.5, 1.0, 1.5, 2.0]);
            assert_eq!((result.width, result.height), (2, 2));
            assert!(result.is_metric);
            assert_eq!(result.intrinsics.fx, 100.0);
            assert_eq!(FREE_COUNT.load(Ordering::SeqCst), 2);
        }
        assert_eq!(CONTEXT_FREE_COUNT.load(Ordering::SeqCst), 1);
    }
}
