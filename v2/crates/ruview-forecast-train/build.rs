use std::path::PathBuf;
use std::process::Command;

use sha2::{Digest, Sha256};

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    for name in [
        "RUVIEW_SOURCE_COMMIT",
        "RUVIEW_CONTAINER_DIGEST",
        "RUVIEW_WORKER_BUILD_ID",
        "RUVIEW_BUILD_MANIFEST_SHA256",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let version = std::env::var_os("RUSTC")
        .and_then(|rustc| Command::new(rustc).arg("--version").output().ok())
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| "UNVERIFIED".to_owned());
    println!("cargo:rustc-env=RUVIEW_RUSTC_VERSION={version}");

    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo provides CARGO_MANIFEST_DIR"),
    );
    let lockfile = manifest_dir.join("../../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lockfile.display());
    let lock_digest = std::fs::read(&lockfile)
        .map(|bytes| hex_lower(&Sha256::digest(bytes)))
        .unwrap_or_else(|_| "UNVERIFIED".to_owned());
    println!("cargo:rustc-env=RUVIEW_CARGO_LOCK_SHA256={lock_digest}");

    let target = std::env::var("TARGET").unwrap_or_else(|_| "UNVERIFIED".to_owned());
    println!("cargo:rustc-env=RUVIEW_BUILD_TARGET={target}");
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}
