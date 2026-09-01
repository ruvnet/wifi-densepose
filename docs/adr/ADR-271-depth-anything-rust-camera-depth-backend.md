# ADR-271: Depth Anything Rust Camera-Depth Backend

- **Status**: Accepted — phase 1 implemented
- **Date**: 2026-07-20
- **Deciders**: ruv
- **Tags**: rust, depth-anything, ggml, camera, point-cloud, ffi, model-licensing

## Context

The `wifi-densepose-pointcloud` crate fuses camera-derived depth with WiFi CSI
occupancy. Its current camera depth path calls an optional loopback MiDaS service
and otherwise produces a luminance/edge heuristic. The fallback is deterministic
and useful for demonstrations, but it is not a learned or metrically trustworthy
depth estimator. It also provides neither confidence nor camera pose.

`mudler/depth-anything.cpp` is a C++17/ggml inference engine for Depth Anything
2 and 3. It offers quantized GGUF models, CPU/CUDA/Metal/Vulkan execution, and a
flat C ABI that returns dense depth, confidence, intrinsics, extrinsics, and a
metric-output flag. Reimplementing the transformer, DPT heads, quantization, and
ggml backends in Rust would duplicate a large parity-sensitive codebase and make
RuView responsible for maintaining several hardware backends.

The upstream ABI currently accepts image paths rather than in-memory RGB frames.
RuView captures RGB8 buffers directly, so phase 1 needs a bounded file bridge
until an upstream RGB-buffer ABI is available.

Model licensing is not uniform. The upstream engine is MIT, while official DA3
weights include Apache-2.0 and CC-BY-NC-4.0 variants. A permissively licensed
runtime must not make non-commercial weights look generally redistributable.

## Decision

RuView will integrate Depth Anything as an **optional native backend behind a
safe Rust boundary**, not as a pure-Rust model rewrite and not as a required
dependency of camera-free sensing.

1. Load the upstream shared library dynamically at runtime. The RuView build
   remains Rust-only and succeeds without CMake, ggml, a model, or a camera.
2. Isolate the ABI and all unsafe operations in `depth_anything.rs`. Keep the
   native library alive for the lifetime of its function pointers and context,
   serialize calls through Rust mutable ownership, copy native output into Rust
   vectors, and release every upstream allocation through its matching function.
3. Require C ABI version 4. Fail closed on any other version.
4. Configure the backend with environment variables:
   - `RUVIEW_DEPTH_BACKEND=auto|depth-anything|midas|heuristic`
   - `RUVIEW_DA_LIBRARY`, `RUVIEW_DA_MODEL`, `RUVIEW_DA_MODEL_LICENSE`
   - optional `RUVIEW_DA_MODEL_SHA256` and `RUVIEW_DA_THREADS`
   - `RUVIEW_DA_ALLOW_NONCOMMERCIAL=1` for an explicit local-only override
5. Permit Apache-2.0 weights by default. Reject unknown licenses and reject
   CC-BY-NC-4.0 unless the explicit non-commercial override is present. RuView
   will not bundle model weights.
6. When a SHA-256 is configured, verify it before native model loading.
7. Bridge RGB8 frames through a unique, create-new PPM file with an RAII cleanup
   guard. Never reuse a predictable existing file. Replace this bridge when the
   upstream project exposes an in-memory RGB ABI.
8. Accept only metric Depth Anything output in the point-cloud fusion path.
   Relative depth must not be labelled or fused as metres.
9. Preserve the existing MiDaS and heuristic paths. `auto` attempts configured
   Depth Anything first, then MiDaS, then the heuristic. An explicitly selected
   Depth Anything backend reports configuration/load/inference errors instead of
   silently claiming that heuristic output came from the model.
10. Carry backend identity, metric status, confidence, and model-provided camera
    geometry into the Rust `DepthEstimate`. Confidence becomes point intensity;
    low-confidence pixels are excluded before voxel fusion.
11. Camera depth remains an optional point-cloud feature. It does not enter the
    RuView/RuField camera-free sensing proof path.
12. Run the native context on a dedicated 16 MiB-stack worker. On Windows, the
    validated upstream ABI-v4 build currently access-violates while destroying a
    successfully used DA2 context; RuView therefore retains its process-singleton
    context and DLL until OS process teardown. Other platforms call `da_capi_free`.

## Consequences

### Positive

- RuView gains local metric monocular depth without Python or PyTorch inference.
- The Rust application owns configuration, policy, provenance, memory safety,
  fallback behavior, and multimodal fusion.
- Confidence and calibrated intrinsics improve point-cloud quality over the
  current fixed-intrinsics heuristic.
- Dynamic loading avoids a mandatory C++ toolchain and preserves portable
  default builds.
- Explicit licensing and digest gates make model provenance auditable.

### Negative

- Production Depth Anything use still requires building or obtaining a compatible
  native library and separately obtaining model weights.
- The phase-1 PPM bridge performs bounded temporary-file I/O per inferred frame.
- The upstream Windows build needs `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON`, and its
  current DA2 destructor requires a process-lifetime context workaround.
- Metric models are substantially larger than the current heuristic and are not
  suitable for ESP32 firmware.
- GPU backend quality and performance remain an upstream responsibility.

### Neutral

- MiDaS compatibility and the deterministic heuristic remain available.
- Motion gating in the point-cloud server continues to limit expensive inference.
- This decision does not authorize redistribution of third-party model weights.

## Validation and acceptance gates

1. Default builds and tests pass with no native library installed.
2. License policy accepts Apache-2.0, rejects unknown licenses, and gates
   CC-BY-NC-4.0 behind an explicit override.
3. Digest mismatch prevents model loading.
4. The PPM bridge validates buffer dimensions, uses create-new semantics, and
   removes its file on success and failure.
5. A mock ABI proves version gating, native-buffer copying/freeing, metric
   enforcement, confidence propagation, and context destruction.
6. Backprojection consumes model dimensions/intrinsics and confidence without
   out-of-bounds access.
7. `cargo test -p wifi-densepose-pointcloud` and Clippy pass.
8. A real upstream Windows/MSVC ABI-v4 build passes `depth-check --abi-only` and
   synthetic inference with the Apache-2.0 DA2 metric Hypersim small Q4_K model
   (`decd99f5c756b564aae5ca1a1612f896e7f76889060e1d25ba610549bbc39b52`).
   Hardware camera validation remains pending receipt of the target hardware.

## Alternatives considered

### Pure Rust DA3 rewrite

Rejected for phase 1. It would duplicate ggml kernels, quantization, model
conversion, and multiple GPU backends while delaying usable integration.

### Statically vendor depth-anything.cpp and ggml

Rejected as the default. It would force the C++ build and backend dependency tree
onto every RuView build, including camera-free and ESP32-oriented workflows.

### Continue with only the MiDaS service

Rejected as the sole production path because it requires an external process and
does not provide the same confidence, pose, quantization, or native deployment
surface.

### Treat relative depth as metres

Rejected because it creates geometrically false point clouds and contaminates
WiFi/mmWave fusion.

## Links

- [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [ADR-058](ADR-058-ruvector-wasm-browser-pose-example.md)
- [ADR-094](ADR-094-pointcloud-github-pages-deployment.md)
- [ADR-260](ADR-260-rufield-mfs.md)
