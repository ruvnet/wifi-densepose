// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::{env, fs, io};

const TORCH_VERSION: &str = "2.11.0";
const PYTHON_PRINT_PYTORCH_DETAILS: &str = r"
import torch
from torch.utils import cpp_extension
print('LIBTORCH_VERSION:', torch.__version__.split('+')[0])
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI)
for include_path in cpp_extension.include_paths():
  print('LIBTORCH_INCLUDE:', include_path)
for library_path in cpp_extension.library_paths():
  print('LIBTORCH_LIB:', library_path)
";

const PYTHON_PRINT_INCLUDE_PATH: &str = r"
import sysconfig
print('PYTHON_INCLUDE:', sysconfig.get_path('include'))
";

const NO_DOWNLOAD_ERROR_MESSAGE: &str = r"
Cannot find a libtorch install, you can either:
- Install libtorch manually and set the LIBTORCH environment variable to appropriate path.
- Use a system wide install in /usr/lib/libtorch.so.
- Use a Python environment with PyTorch installed by setting LIBTORCH_USE_PYTORCH=1

See the readme for more details:
https://github.com/LaurentMazare/tch-rs/blob/main/README.md
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkType {
    Dynamic,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Os {
    Linux,
    Macos,
    Windows,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SystemInfo {
    os: Os,
    python_interpreter: PathBuf,
    cxx11_abi: String,
    libtorch_include_dirs: Vec<PathBuf>,
    libtorch_lib_dir: PathBuf,
    link_type: LinkType,
}

#[cfg(feature = "ureq")]
fn download<P: AsRef<Path>>(source_url: &str, target_file: P) -> anyhow::Result<()> {
    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let response = ureq::get(source_url).call()?;
    let response_code = response.status();
    if response_code != 200 {
        anyhow::bail!("Unexpected response code {} for {}", response_code, source_url)
    }
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut writer)?;
    Ok(())
}

#[cfg(not(feature = "ureq"))]
fn download<P: AsRef<Path>>(_source_url: &str, _target_file: P) -> anyhow::Result<()> {
    anyhow::bail!("cannot use download without the ureq feature")
}

#[cfg(not(feature = "download-libtorch"))]
fn get_pypi_wheel_url_for_aarch64_macosx() -> anyhow::Result<String> {
    anyhow::bail!("cannot get pypi wheel url without the ureq feature")
}

#[cfg(feature = "download-libtorch")]
#[derive(serde::Deserialize, Debug)]
struct PyPiPackageUrl {
    url: String,
    filename: String,
}
#[cfg(feature = "download-libtorch")]
#[derive(serde::Deserialize, Debug)]
struct PyPiPackage {
    urls: Vec<PyPiPackageUrl>,
}
#[cfg(feature = "download-libtorch")]
fn get_pypi_wheel_url_for_aarch64_macosx() -> anyhow::Result<String> {
    let pypi_url = format!("https://pypi.org/pypi/torch/{TORCH_VERSION}/json");
    let response = ureq::get(pypi_url.as_str()).call()?;
    let response_code = response.status();
    if response_code != 200 {
        anyhow::bail!("Unexpected response code {} for {}", response_code, pypi_url)
    }
    let pypi_package: PyPiPackage = response.into_json()?;
    let urls = pypi_package.urls;
    let expected_filename = format!("torch-{TORCH_VERSION}-cp311-none-macosx_11_0_arm64.whl");
    let url = urls.iter().find_map(|pypi_url: &PyPiPackageUrl| {
        if pypi_url.filename == expected_filename {
            Some(pypi_url.url.clone())
        } else {
            None
        }
    });
    url.context("Failed to find arm64 macosx wheel from pypi")
}

fn extract<P: AsRef<Path>>(filename: P, outpath: P) -> anyhow::Result<()> {
    let file = fs::File::open(&filename)?;
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf)?;
    // ZIP 9's checked extraction rejects path/symlink escapes and unsafe permissions.
    archive.extract(outpath.as_ref())?;

    // This is if we're unzipping a python wheel.
    if outpath.as_ref().join("torch").exists() {
        fs::rename(outpath.as_ref().join("torch"), outpath.as_ref().join("libtorch"))?;
    }
    Ok(())
}

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name)
}

fn version_check(version: &str) -> Result<()> {
    if env_var_rerun("LIBTORCH_BYPASS_VERSION_CHECK").is_ok() {
        return Ok(());
    }
    let version = version.trim();
    // Typical version number is 2.11.0+cpu or 2.11.0+cu129
    let version = match version.split_once('+') {
        None => version,
        Some((version, _)) => version,
    };
    if version != TORCH_VERSION {
        anyhow::bail!("this tch version expects PyTorch {TORCH_VERSION}, got {version}, this check can be bypassed by setting the LIBTORCH_BYPASS_VERSION_CHECK environment variable")
    }
    Ok(())
}

impl SystemInfo {
    fn new() -> Result<Self> {
        let os = match env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS").as_str() {
            "linux" => Os::Linux,
            "windows" => Os::Windows,
            "macos" => Os::Macos,
            os => anyhow::bail!("unsupported TARGET_OS '{os}'"),
        };
        // Locate the currently active Python binary, similar to:
        // https://github.com/PyO3/maturin/blob/243b8ec91d07113f97a6fe74d9b2dcb88086e0eb/src/target.rs#L547
        let python_interpreter = match os {
            Os::Windows => PathBuf::from("python.exe"),
            Os::Linux | Os::Macos => {
                if env::var_os("VIRTUAL_ENV").is_some() {
                    PathBuf::from("python")
                } else {
                    PathBuf::from("python3")
                }
            }
        };
        let mut libtorch_include_dirs = vec![];
        if cfg!(feature = "python-extension") {
            let output = std::process::Command::new(&python_interpreter)
                .arg("-c")
                .arg(PYTHON_PRINT_INCLUDE_PATH)
                .output()
                .with_context(|| format!("error running {python_interpreter:?}"))?;
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(path) = line.strip_prefix("PYTHON_INCLUDE: ") {
                    libtorch_include_dirs.push(PathBuf::from(path))
                }
            }
        }
        let mut libtorch_lib_dir = None;
        let cxx11_abi = if env_var_rerun("LIBTORCH_USE_PYTORCH").is_ok() {
            let output = std::process::Command::new(&python_interpreter)
                .arg("-c")
                .arg(PYTHON_PRINT_PYTORCH_DETAILS)
                .output()
                .with_context(|| format!("error running {python_interpreter:?}"))?;
            let mut cxx11_abi = None;
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some(version) = line.strip_prefix("LIBTORCH_VERSION: ") {
                    version_check(version)?
                }
                match line.strip_prefix("LIBTORCH_CXX11: ") {
                    Some("True") => cxx11_abi = Some("1".to_owned()),
                    Some("False") => cxx11_abi = Some("0".to_owned()),
                    _ => {}
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                    libtorch_include_dirs.push(PathBuf::from(path))
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                    libtorch_lib_dir = Some(PathBuf::from(path))
                }
            }
            match cxx11_abi {
                Some(cxx11_abi) => cxx11_abi,
                None => anyhow::bail!("no cxx11 abi returned by python {output:?}"),
            }
        } else {
            let libtorch = Self::prepare_libtorch_dir(os)?;
            let includes = env_var_rerun("LIBTORCH_INCLUDE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| libtorch.clone());
            let lib = env_var_rerun("LIBTORCH_LIB")
                .map(PathBuf::from)
                .unwrap_or_else(|_| libtorch.clone());
            let mut version_file = libtorch;
            version_file.push("build-version");
            if version_file.exists() {
                if let Ok(version) = std::fs::read_to_string(&version_file) {
                    version_check(&version)?
                }
            }
            libtorch_include_dirs.push(includes.join("include"));
            libtorch_include_dirs.push(includes.join("include/torch/csrc/api/include"));
            libtorch_lib_dir = Some(lib.join("lib"));
            env_var_rerun("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_owned())
        };
        let libtorch_lib_dir = libtorch_lib_dir.expect("no libtorch lib dir found");
        let link_type = match env_var_rerun("LIBTORCH_STATIC").as_deref() {
            Err(_) | Ok("0") | Ok("false") | Ok("FALSE") => LinkType::Dynamic,
            Ok(_) => LinkType::Static,
        };
        Ok(Self {
            os,
            python_interpreter,
            cxx11_abi,
            libtorch_include_dirs,
            libtorch_lib_dir,
            link_type,
        })
    }

    fn check_system_location(os: Os) -> Option<PathBuf> {
        match os {
            Os::Linux => Path::new("/usr/lib/libtorch.so").exists().then(|| PathBuf::from("/usr")),
            _ => None,
        }
    }

    fn prepare_libtorch_dir(os: Os) -> Result<PathBuf> {
        if let Ok(libtorch) = env_var_rerun("LIBTORCH") {
            Ok(PathBuf::from(libtorch))
        } else if let Some(pathbuf) = Self::check_system_location(os) {
            Ok(pathbuf)
        } else {
            if !cfg!(feature = "download-libtorch") {
                anyhow::bail!(NO_DOWNLOAD_ERROR_MESSAGE)
            }

            let device = match env_var_rerun("TORCH_CUDA_VERSION") {
                Ok(cuda_env) => match os {
                    Os::Linux | Os::Windows => cuda_env
                        .trim()
                        .to_lowercase()
                        .trim_start_matches("cu")
                        .split('.')
                        .take(2)
                        .fold("cu".to_owned(), |mut acc, curr| {
                            acc += curr;
                            acc
                        }),
                    os => anyhow::bail!(
                        "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                 binaries with CUDA are only available for Linux and Windows, not: {os:?}.",
                    ),
                },
                Err(_) => "cpu".to_owned(),
            };

            let libtorch_dir =
                PathBuf::from(env::var("OUT_DIR").context("OUT_DIR variable not set")?)
                    .join("libtorch");
            if !libtorch_dir.exists() {
                fs::create_dir(&libtorch_dir).unwrap_or_default();
                let libtorch_url = match os {
                Os::Linux => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu118" => "%2Bcu118",
                        "cu121" => "%2Bcu121",
                        "cu124" => "%2Bcu124",
                        "cu126" => "%2Bcu126",
                        "cu128" => "%2Bcu128",
                        _ => anyhow::bail!("unsupported device {device}, TORCH_CUDA_VERSION may be set incorrectly?"),
                    }
                ),
                Os::Macos => {
                    if env::var("CARGO_CFG_TARGET_ARCH") == Ok(String::from("aarch64")) {
                        get_pypi_wheel_url_for_aarch64_macosx().expect(
                            "Failed to retrieve torch from pypi.  Pre-built version of libtorch for apple silicon are not available.
                            You can install torch manually following the indications from https://github.com/LaurentMazare/tch-rs/issues/629
                            pip3 install torch=={TORCH_VERSION}
                            Then update the following environment variables:
                            export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
                            export DYLD_LIBRARY_PATH=${{LIBTORCH}}/lib
                            ")
                    } else {
                        format!("https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-{TORCH_VERSION}.zip")
                    }
                },
                Os::Windows => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() {
                        "cpu" => "%2Bcpu",
                        "cu118" => "%2Bcu118",
                        "cu121" => "%2Bcu121",
                        "cu124" => "%2Bcu124",
                        "cu126" => "%2Bcu126",
                        "cu128" => "%2Bcu128",
                        _ => ""
                    }),
            };

                let filename = libtorch_dir.join(format!("v{TORCH_VERSION}.zip"));
                download(&libtorch_url, &filename)?;
                extract(&filename, &libtorch_dir)?;
            }
            Ok(libtorch_dir.join("libtorch"))
        }
    }

    fn make(&self) {
        println!("cargo:rerun-if-changed=libtch/torch_python.cpp");
        println!("cargo:rerun-if-changed=libtch/torch_python.h");
        println!("cargo:rerun-if-changed=libtch/torch_api_generated.cpp");
        println!("cargo:rerun-if-changed=libtch/torch_api_generated.h");
        println!("cargo:rerun-if-changed=libtch/torch_api.cpp");
        println!("cargo:rerun-if-changed=libtch/torch_api.h");
        println!("cargo:rerun-if-changed=libtch/stb_image_write.h");
        println!("cargo:rerun-if-changed=libtch/stb_image_resize.h");
        println!("cargo:rerun-if-changed=libtch/stb_image.h");
        let mut c_files = vec!["libtch/torch_api.cpp", "libtch/torch_api_generated.cpp"];
        if cfg!(feature = "python-extension") {
            c_files.push("libtch/torch_python.cpp")
        }

        match self.os {
            Os::Linux | Os::Macos => {
                // Pass the libtorch lib dir to crates that use torch-sys. This will be available
                // as DEP_TORCH_SYS_LIBTORCH_LIB, see:
                // https://doc.rust-lang.org/cargo/reference/build-scripts.html#the-links-manifest-key
                println!("cargo:libtorch_lib={}", self.libtorch_lib_dir.display());
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .includes(&self.libtorch_include_dirs)
                    .flag(format!("-Wl,-rpath={}", self.libtorch_lib_dir.display()))
                    .flag("-std=c++17")
                    .flag(format!("-D_GLIBCXX_USE_CXX11_ABI={}", self.cxx11_abi))
                    .flag("-DGLOG_USE_GLOG_EXPORT")
                    .files(&c_files)
                    .compile("tch");
            }
            Os::Windows => {
                // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
                //       Not yet supported by cc=rs.
                //       https://github.com/alexcrichton/cc-rs/issues/323
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .includes(&self.libtorch_include_dirs)
                    .flag("/std:c++17")
                    .flag("/p:DefineConstants=GLOG_USE_GLOG_EXPORT")
                    .files(&c_files)
                    .compile("tch");
            }
        };
    }

    fn link(&self, lib_name: &str) {
        match self.link_type {
            LinkType::Dynamic => println!("cargo:rustc-link-lib={lib_name}"),
            LinkType::Static => {
                // TODO: whole-archive might only be necessary for libtorch_cpu?
                println!("cargo:rustc-link-lib=static:+whole-archive,-bundle={lib_name}")
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    if !cfg!(feature = "doc-only") {
        let system_info = SystemInfo::new()?;
        // use_cuda is a hacky way to detect whether cuda is available and
        // if it's the case link to it by explicitly depending on a symbol
        // from the torch_cuda library.
        // It would be better to use -Wl,--no-as-needed but there is no way
        // to specify arbitrary linker flags at the moment.
        //
        // Once https://github.com/rust-lang/cargo/pull/8441 is available
        // we should switch to using rustc-link-arg instead e.g. with the
        // following flags:
        //   -Wl,--no-as-needed -Wl,--copy-dt-needed-entries -ltorch
        //
        // This will be available starting from cargo 1.50 but will be a nightly
        // only option to start with.
        // https://github.com/rust-lang/cargo/blob/master/CHANGELOG.md
        //
        // Update: The above doesn't seem to propagate to the downstream binaries
        // so doesn't really help, the comment has been kept though to keep track
        // if this issue.
        // TODO: Try out the as-needed native link modifier when it lands.
        // https://github.com/rust-lang/rust/issues/99424
        //
        // Update: it seems that the dummy dependency is not necessary anymore, so just
        // removing it and keeping this comment around for legacy.
        let si_lib = &system_info.libtorch_lib_dir;
        let use_cuda =
            si_lib.join("libtorch_cuda.so").exists() || si_lib.join("torch_cuda.dll").exists();
        let use_cuda_cu = si_lib.join("libtorch_cuda_cu.so").exists()
            || si_lib.join("torch_cuda_cu.dll").exists();
        let use_cuda_cpp = si_lib.join("libtorch_cuda_cpp.so").exists()
            || si_lib.join("torch_cuda_cpp.dll").exists();
        let use_hip =
            si_lib.join("libtorch_hip.so").exists() || si_lib.join("torch_hip.dll").exists();
        println!("cargo:rustc-link-search=native={}", si_lib.display());

        system_info.make();

        println!("cargo:rustc-link-lib=static=tch");
        if use_cuda {
            system_info.link("torch_cuda")
        }
        if use_cuda_cu {
            system_info.link("torch_cuda_cu")
        }
        if use_cuda_cpp {
            system_info.link("torch_cuda_cpp")
        }
        if use_hip {
            system_info.link("torch_hip")
        }
        if cfg!(feature = "python-extension") {
            system_info.link("torch_python")
        }
        if system_info.link_type == LinkType::Static {
            // TODO: this has only be tried out on the cpu version. Check that it works
            // with cuda too and maybe just try linking all available files?
            system_info.link("asmjit");
            system_info.link("clog");
            system_info.link("cpuinfo");
            system_info.link("dnnl");
            system_info.link("dnnl_graph");
            system_info.link("fbgemm");
            system_info.link("gloo");
            system_info.link("kineto");
            system_info.link("nnpack");
            system_info.link("onnx");
            system_info.link("onnx_proto");
            system_info.link("protobuf");
            system_info.link("pthreadpool");
            system_info.link("pytorch_qnnpack");
            system_info.link("sleef");
            system_info.link("tensorpipe");
            system_info.link("tensorpipe_uv");
            system_info.link("XNNPACK");
        }
        system_info.link("torch_cpu");
        system_info.link("torch");
        system_info.link("c10");
        if use_hip {
            system_info.link("c10_hip");
        }

        let target = env::var("TARGET").context("TARGET variable not set")?;

        if !target.contains("msvc") && !target.contains("apple") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }
    Ok(())
}
