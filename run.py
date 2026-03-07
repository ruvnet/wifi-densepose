#!/usr/bin/env python3
"""
RuView - WiFi DensePose Launcher (Windows-compatible)

Usage:
    python run.py                  Interactive menu
    python run.py verify           Run pipeline verification
    python run.py server           Start Python API server
    python run.py viz              Serve the Three.js visualization UI
    python run.py all              Start API + visualization together
    python run.py docker           Run via Docker
    python run.py rust-test        Run Rust workspace tests
    python run.py rust-build       Build Rust workspace (release)
    python run.py install          Install Python dependencies
    python run.py install --lock   Install pinned (lock) dependencies only
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "v1"
PROOF_DIR = V1 / "data" / "proof"
RUST_DIR = ROOT / "rust-port" / "wifi-densepose-rs"
UI_DIR = ROOT / "ui"
VENV_DIR = ROOT / ".venv"


# -- Helpers ------------------------------------------------------------------

def run(cmd, cwd=None, check=True, shell=False, env=None):
    """Run a command, printing it first."""
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
        shell = True
    print(f"\n> {display}\n")
    run_env = None
    if env:
        run_env = {**os.environ, **env}
    return subprocess.run(cmd, cwd=cwd or ROOT, check=check, shell=shell, env=run_env)


def find_tool(name):
    """Check if a CLI tool is available on PATH."""
    return shutil.which(name)


def venv_python():
    """Return the Python executable inside the .venv."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def ensure_venv():
    """Create a .venv with uv (Python 3.12) if it doesn't exist."""
    if (VENV_DIR / ("Scripts" if platform.system() == "Windows" else "bin")).exists():
        return True

    uv = find_tool("uv")
    if not uv:
        print("ERROR: uv not found on PATH.")
        print("Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False

    print("Creating .venv with Python 3.12 via uv...")
    result = run([uv, "venv", str(VENV_DIR), "--python", "3.12"], check=False)
    if result.returncode != 0:
        print("ERROR: Failed to create .venv")
        return False
    return True


def python_exe():
    """Return the best python executable — prefers .venv, falls back to system."""
    vp = venv_python()
    if Path(vp).exists():
        return vp
    return sys.executable


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def normalized_bool_env(name, default=False):
    """Read an env var and normalize to 'true'/'false' for Pydantic bool settings."""
    raw = os.environ.get(name)
    if raw is None:
        return "true" if default else "false"

    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return "true"
    if value in {"0", "false", "f", "no", "n", "off"}:
        return "false"

    fallback = "true" if default else "false"
    print(f"WARN: Invalid {name}={raw!r}; using {fallback}.")
    return fallback


def open_url(url):
    """Open URL in browser; prefers xdg-open when available."""
    try:
        xdg = find_tool("xdg-open")
        if xdg:
            subprocess.Popen([xdg, url])
            return True

        system = platform.system()
        if system == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        if system == "Darwin":
            subprocess.Popen(["open", url])
            return True
    except Exception as exc:
        print(f"WARN: Failed to open browser automatically: {exc}")
        return False

    print("WARN: No browser opener found (xdg-open/open/os.startfile).")
    return False


def terminate_process(proc, name):
    """Terminate a subprocess gracefully, then force kill if needed."""
    if not proc or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"{name} did not exit in time; killing...")
        proc.kill()
    except Exception as exc:
        print(f"WARN: Failed to terminate {name}: {exc}")


# -- Commands -----------------------------------------------------------------

def cmd_verify(args):
    """Run the deterministic pipeline verification."""
    header("Pipeline Verification (Trust Kill Switch)")

    verify_py = PROOF_DIR / "verify.py"
    if not verify_py.exists():
        print(f"ERROR: {verify_py} not found.")
        return 1

    if not Path(venv_python()).exists():
        print("No .venv found. Running install --lock first...")
        ns = argparse.Namespace(lock=True)
        rc = cmd_install(ns)
        if rc != 0:
            return rc

    cmd = [python_exe(), str(verify_py)]
    if getattr(args, "verbose", False):
        cmd.append("--verbose")
    if getattr(args, "generate_hash", False):
        cmd.append("--generate-hash")

    result = run(cmd, check=False)
    if result.returncode == 0:
        print("\nVERDICT: PASS")
    elif result.returncode == 2:
        print("\nVERDICT: SKIP (no expected hash file)")
        print("  Run: python run.py verify --generate-hash")
    else:
        print("\nVERDICT: FAIL")
    return result.returncode


def cmd_install(args):
    """Install Python dependencies using uv into .venv."""
    header("Installing Python Dependencies")

    if not ensure_venv():
        return 1

    if getattr(args, "lock", False):
        req = V1 / "requirements-lock.txt"
        print("Installing pinned (lock) dependencies for verification...")
    else:
        req = ROOT / "requirements.txt"
        print("Installing full dependencies...")

    if not req.exists():
        print(f"ERROR: {req} not found.")
        return 1

    uv = find_tool("uv")
    if uv:
        return run([uv, "pip", "install", "-r", str(req), "--python", venv_python()], check=False).returncode
    return run([python_exe(), "-m", "pip", "install", "-r", str(req)], check=False).returncode


def cmd_server(args):
    """Start the Python FastAPI API server."""
    header("Starting Python API Server")

    if not Path(venv_python()).exists():
        print("No .venv found. Running install first...")
        ns = argparse.Namespace(lock=False)
        rc = cmd_install(ns)
        if rc != 0:
            return rc

    host = getattr(args, "host", "0.0.0.0")
    port = str(getattr(args, "port", 8000))
    reload_flag = getattr(args, "reload", False)

    # The app uses `from src.config...` so cwd must be v1/
    cmd = [
        python_exe(), "-m", "uvicorn",
        "src.api.main:app",
        "--host", host,
        "--port", port,
    ]
    if reload_flag:
        cmd.extend(["--reload", "--reload-dir", str(V1 / "src")])

    print(f"API docs:   http://localhost:{port}/docs")
    print(f"Health:     http://localhost:{port}/health/live")
    print(f"WebSocket:  ws://localhost:{port}/ws/pose/stream")

    # Set dev defaults so the server starts without a .env file.
    # Real-only mode: mock flags default to false.
    env = {
        "PYTHONPATH": str(V1),
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-local-secret-key"),
        "ENVIRONMENT": os.environ.get("ENVIRONMENT", "development"),
        "DEBUG": normalized_bool_env("DEBUG", default=False),
        "MOCK_HARDWARE": os.environ.get("MOCK_HARDWARE", "false"),
        "MOCK_POSE_DATA": os.environ.get("MOCK_POSE_DATA", "false"),
    }
    if env["MOCK_HARDWARE"].lower() == "true" or env["MOCK_POSE_DATA"].lower() == "true":
        print("ERROR: Mock mode is disabled in this build. Set MOCK_HARDWARE=false and MOCK_POSE_DATA=false.")
        return 1
    return run(cmd, cwd=str(V1), check=False, env=env).returncode


def cmd_viz(args):
    """Serve the Three.js visualization UI via Python HTTP server."""
    header("Serving Three.js Visualization")

    if not UI_DIR.exists():
        print(f"ERROR: UI directory not found at {UI_DIR}")
        return 1

    port = str(getattr(args, "port", 3000))
    print(f"Open http://localhost:{port}/viz.html in your browser")
    print(f"Connect API server on port 8000 for live data.\n")

    return run(
        [python_exe(), "-m", "http.server", port, "--directory", str(UI_DIR)],
        check=False,
    ).returncode


def cmd_all(args):
    """Start API + visualization together and open Observatory."""
    header("Starting API + Visualization")

    if not Path(venv_python()).exists():
        print("No .venv found. Running install first...")
        ns = argparse.Namespace(lock=False)
        rc = cmd_install(ns)
        if rc != 0:
            return rc

    api_host = getattr(args, "host", "0.0.0.0")
    api_port = int(getattr(args, "api_port", 8000))
    viz_port = int(getattr(args, "viz_port", 3000))
    reload_flag = bool(getattr(args, "reload", False))
    no_open = bool(getattr(args, "no_open", False))
    open_delay = float(getattr(args, "open_delay", 2.0))

    env = {
        "PYTHONPATH": str(V1),
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-local-secret-key"),
        "ENVIRONMENT": os.environ.get("ENVIRONMENT", "development"),
        "DEBUG": normalized_bool_env("DEBUG", default=False),
        "MOCK_HARDWARE": os.environ.get("MOCK_HARDWARE", "false"),
        "MOCK_POSE_DATA": os.environ.get("MOCK_POSE_DATA", "false"),
    }
    if env["MOCK_HARDWARE"].lower() == "true" or env["MOCK_POSE_DATA"].lower() == "true":
        print("ERROR: Mock mode is disabled in this build. Set MOCK_HARDWARE=false and MOCK_POSE_DATA=false.")
        return 1

    api_cmd = [
        python_exe(), "-m", "uvicorn",
        "src.api.main:app",
        "--host", str(api_host),
        "--port", str(api_port),
    ]
    if reload_flag:
        api_cmd.extend(["--reload", "--reload-dir", str(V1 / "src")])

    viz_cmd = [
        python_exe(), "-m", "http.server",
        str(viz_port),
        "--directory", str(UI_DIR),
    ]

    observatory_url = f"http://localhost:{viz_port}/observatory.html"
    print("Starting services:")
    print(f"  API: http://localhost:{api_port}/docs")
    print(f"  Viz: http://localhost:{viz_port}/viz.html")
    print(f"  Observatory: {observatory_url}")
    print("Press Ctrl+C to stop both.\n")

    api_proc = None
    viz_proc = None
    exit_code = 0

    try:
        api_proc = subprocess.Popen(
            api_cmd,
            cwd=str(V1),
            env={**os.environ, **env},
        )
        viz_proc = subprocess.Popen(viz_cmd, cwd=str(ROOT))

        if not no_open:
            time.sleep(max(0.0, open_delay))
            opened = open_url(observatory_url)
            if not opened:
                print(f"Open manually: {observatory_url}")

        while True:
            api_rc = api_proc.poll()
            viz_rc = viz_proc.poll()

            if api_rc is not None:
                print(f"API process exited with code {api_rc}")
                exit_code = api_rc
                break
            if viz_rc is not None:
                print(f"Viz process exited with code {viz_rc}")
                exit_code = viz_rc
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping services...")
        exit_code = 0
    except Exception as exc:
        print(f"ERROR: Failed to run all services: {exc}")
        exit_code = 1
    finally:
        terminate_process(viz_proc, "Viz process")
        terminate_process(api_proc, "API process")

    return exit_code


def cmd_docker(args):
    """Run via Docker."""
    header("Docker Deployment")

    docker = find_tool("docker")
    if not docker:
        print("ERROR: docker not found on PATH.")
        print("Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        return 1

    port = str(getattr(args, "port", 3000))
    image = "ruvnet/wifi-densepose:latest"

    run([docker, "pull", image], check=False)
    print(f"\nStarting container on port {port}...")
    print(f"Open http://localhost:{port}\n")
    return run(
        [docker, "run", "--rm", "-p", f"{port}:3000", image],
        check=False,
    ).returncode


def cmd_rust_test(args):
    """Run Rust workspace tests."""
    header("Rust Workspace Tests")

    cargo = find_tool("cargo")
    if not cargo:
        print("ERROR: cargo not found. Install Rust via https://rustup.rs/")
        return 1

    if not RUST_DIR.exists():
        print(f"ERROR: Rust workspace not found at {RUST_DIR}")
        return 1

    return run(
        [cargo, "test", "--workspace", "--no-default-features"],
        cwd=str(RUST_DIR),
        check=False,
    ).returncode


def cmd_rust_build(args):
    """Build Rust workspace in release mode."""
    header("Rust Workspace Build (Release)")

    cargo = find_tool("cargo")
    if not cargo:
        print("ERROR: cargo not found. Install Rust via https://rustup.rs/")
        return 1

    if not RUST_DIR.exists():
        print(f"ERROR: Rust workspace not found at {RUST_DIR}")
        return 1

    return run(
        [cargo, "build", "--release", "--workspace"],
        cwd=str(RUST_DIR),
        check=False,
    ).returncode


def cmd_menu(args):
    """Interactive menu for choosing what to run."""
    header("RuView - WiFi DensePose")

    print(f"  Platform:  {platform.system()} {platform.release()}")
    print(f"  Python:    {python_exe()}")
    print(f"  uv:        {'found' if find_tool('uv') else 'not found'}")
    print(f"  .venv:     {'ready' if Path(venv_python()).exists() else 'not created'}")
    print(f"  Cargo:     {'found' if find_tool('cargo') else 'not found'}")
    print(f"  Docker:    {'found' if find_tool('docker') else 'not found'}")
    print()

    options = [
        ("1", "Verify pipeline (no hardware needed)", cmd_verify),
        ("2", "Install Python dependencies", cmd_install),
        ("3", "Install pinned deps only (for verification)", None),
        ("4", "Start Python API server (port 8000)", cmd_server),
        ("5", "Serve Three.js visualization (port 3000)", cmd_viz),
        ("6", "Run via Docker (port 3000)", cmd_docker),
        ("7", "Run Rust workspace tests", cmd_rust_test),
        ("8", "Build Rust workspace (release)", cmd_rust_build),
        ("9", "Start API + Viz + open Observatory", cmd_all),
        ("q", "Quit", None),
    ]

    for key, label, _ in options:
        print(f"  [{key}] {label}")
    print()

    try:
        choice = input("Choose an option: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return 0

    # Build a minimal args namespace for the selected command
    ns = argparse.Namespace()

    if choice == "1":
        return cmd_verify(ns)
    elif choice == "2":
        return cmd_install(ns)
    elif choice == "3":
        ns.lock = True
        return cmd_install(ns)
    elif choice == "4":
        ns.host = "0.0.0.0"
        ns.port = 8000
        ns.reload = True
        return cmd_server(ns)
    elif choice == "5":
        ns.port = 3000
        return cmd_viz(ns)
    elif choice == "6":
        ns.port = 3000
        return cmd_docker(ns)
    elif choice == "7":
        return cmd_rust_test(ns)
    elif choice == "8":
        return cmd_rust_build(ns)
    elif choice == "9":
        ns.host = "0.0.0.0"
        ns.api_port = 8000
        ns.viz_port = 3000
        ns.reload = False
        ns.no_open = False
        ns.open_delay = 2.0
        return cmd_all(ns)
    elif choice == "q":
        return 0
    else:
        print(f"Unknown option: {choice}")
        return 1


# -- Argument parser ----------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="RuView - WiFi DensePose launcher (Windows-compatible)",
    )
    sub = parser.add_subparsers(dest="command")

    # verify
    p_verify = sub.add_parser("verify", help="Run pipeline verification")
    p_verify.add_argument("--verbose", action="store_true")
    p_verify.add_argument("--generate-hash", action="store_true",
                          help="Regenerate the expected hash file")

    # install
    p_install = sub.add_parser("install", help="Install Python dependencies")
    p_install.add_argument("--lock", action="store_true",
                           help="Install pinned lock deps only (numpy, scipy)")

    # server
    p_server = sub.add_parser("server", help="Start Python API server")
    p_server.add_argument("--host", default="0.0.0.0")
    p_server.add_argument("--port", type=int, default=8000)
    p_server.add_argument("--reload", action="store_true",
                          help="Enable auto-reload for development")

    # viz
    p_viz = sub.add_parser("viz", help="Serve Three.js visualization")
    p_viz.add_argument("--port", type=int, default=3000)

    # all
    p_all = sub.add_parser("all", help="Start API + visualization and open Observatory")
    p_all.add_argument("--host", default="0.0.0.0",
                       help="API host (default: 0.0.0.0)")
    p_all.add_argument("--api-port", type=int, default=8000,
                       help="API port (default: 8000)")
    p_all.add_argument("--viz-port", type=int, default=3000,
                       help="Visualization port (default: 3000)")
    p_all.add_argument("--reload", action="store_true",
                       help="Enable API auto-reload")
    p_all.add_argument("--no-open", action="store_true",
                       help="Do not open browser automatically")
    p_all.add_argument("--open-delay", type=float, default=2.0,
                       help="Seconds to wait before opening Observatory URL")

    # docker
    p_docker = sub.add_parser("docker", help="Run via Docker")
    p_docker.add_argument("--port", type=int, default=3000)

    # rust
    sub.add_parser("rust-test", help="Run Rust workspace tests")
    sub.add_parser("rust-build", help="Build Rust workspace (release)")

    return parser


# -- Main ---------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "verify": cmd_verify,
        "install": cmd_install,
        "server": cmd_server,
        "viz": cmd_viz,
        "all": cmd_all,
        "docker": cmd_docker,
        "rust-test": cmd_rust_test,
        "rust-build": cmd_rust_build,
    }

    if args.command:
        sys.exit(commands[args.command](args))
    else:
        sys.exit(cmd_menu(args))


if __name__ == "__main__":
    main()
