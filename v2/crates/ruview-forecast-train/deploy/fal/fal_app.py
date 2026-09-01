"""Minimal fal Direct Server launcher; all training logic is Rust."""

import os
import re
import subprocess

import fal


IMAGE = fal.ContainerImage.from_dockerfile(
    "v2/crates/ruview-forecast-train/deploy/fal/Dockerfile"
)

worker_build_id = os.environ.get("RUVIEW_WORKER_BUILD_ID", "")
build_manifest_sha256 = os.environ.get("RUVIEW_BUILD_MANIFEST_SHA256", "")
if not re.fullmatch(r"ruview-[0-9a-f]{40}(?:[0-9a-f]{24})?", worker_build_id):
    raise RuntimeError("RUVIEW_WORKER_BUILD_ID must come from deploy.py's tracked source")
if not re.fullmatch(r"[0-9a-f]{64}", build_manifest_sha256):
    raise RuntimeError(
        "RUVIEW_BUILD_MANIFEST_SHA256 must come from deploy.py's tracked source"
    )


@fal.function(
    image=IMAGE,
    machine_type="GPU-A100",
    exposed_port=8000,
    max_multiplexing=1,
    request_timeout=3600,
)
def run_server():
    subprocess.run(
        [
            "/usr/local/bin/ruforecast",
            "serve",
            "--bind",
            "0.0.0.0:8000",
            "--output",
            "/data/ruview-forecast/artifacts",
        ],
        close_fds=True,
        check=True,
    )
