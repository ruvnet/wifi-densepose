//! Build script: when binding libtorch from the system PyTorch install
//! (`LIBTORCH_USE_PYTORCH=1`), embed an rpath to the PyTorch `lib` directory so
//! the produced binaries and test executables can locate `libtorch_cpu` at
//! runtime. torch-sys only adds the rpath to its own cc-built static library,
//! which does not propagate to downstream binaries on macOS/Linux.
fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");
    if std::env::var_os("LIBTORCH_USE_PYTORCH").is_none() {
        return;
    }
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" && target_os != "linux" {
        return;
    }
    let output = std::process::Command::new("python3")
        .args([
            "-c",
            "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))",
        ])
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            let lib_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !lib_dir.is_empty() {
                // Plain rustc-link-arg (not -bins/-tests): it is the only
                // variant that also reaches the lib unit-test binary.
                println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
            }
        }
    }
}
