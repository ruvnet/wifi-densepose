#!/usr/bin/env python3
"""Run fal from an allowlisted, tracked-only git archive build context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import time


FORECAST_SCOPE = [
    "v2/Cargo.lock",
    "v2/crates/ruview-forecast-core",
    "v2/crates/ruview-forecast-model",
    "v2/crates/ruview-forecast-train",
]
ARCHIVE_SCOPE = [
    "v2/Cargo.lock",
    "v2/crates/ruview-forecast-core/Cargo.toml",
    "v2/crates/ruview-forecast-core/src",
    "v2/crates/ruview-forecast-model/Cargo.toml",
    "v2/crates/ruview-forecast-model/src",
    "v2/crates/ruview-forecast-train/Cargo.toml",
    "v2/crates/ruview-forecast-train/build.rs",
    "v2/crates/ruview-forecast-train/src",
    "v2/crates/ruview-forecast-train/deploy/fal/Dockerfile",
    "v2/crates/ruview-forecast-train/deploy/fal/fal_app.py",
    "v2/crates/ruview-forecast-train/deploy/fal/workspace.Cargo.toml",
]
FAL_APP_NAME = "ruforecast"
FAL_FUNCTION_REF = (
    "v2/crates/ruview-forecast-train/deploy/fal/fal_app.py::run_server"
)
ALLOWED_SUFFIXES = {".rs"}
ALLOWED_SOURCE_ROOTS = {
    "v2/crates/ruview-forecast-core/src",
    "v2/crates/ruview-forecast-model/src",
    "v2/crates/ruview-forecast-train/src",
}
ALLOWED_EXACT_FILES = set(ARCHIVE_SCOPE) - ALLOWED_SOURCE_ROOTS


def fal_command(fal: str, mode: str) -> list[str]:
    """Build the only allowed fal invocation.

    fal 1.80 requires an explicit ``file.py::symbol`` function reference.
    ``fal run`` otherwise defaults to public authentication, so both the
    ephemeral and deployed aliases are explicitly private here.
    """
    if mode not in {"run", "deploy"}:
        raise ValueError("fal mode must be run or deploy")
    command = [
        fal,
        mode,
        FAL_FUNCTION_REF,
        "--app-name",
        FAL_APP_NAME,
        "--auth",
        "private",
    ]
    if mode == "deploy":
        # fal 1.80 otherwise inherits the previous deployment's machine and
        # concurrency settings instead of applying the reviewed decorator.
        command.append("--reset-scale")
    return command


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def safe_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "v2":
        return False
    name = path.as_posix().rstrip("/")
    if member.isdir():
        build_inputs = ALLOWED_EXACT_FILES | ALLOWED_SOURCE_ROOTS
        return any(item == name or item.startswith(f"{name}/") for item in build_inputs)
    if not member.isfile():
        return False
    if name in ALLOWED_EXACT_FILES:
        return True
    return path.suffix.lower() in ALLOWED_SUFFIXES and any(
        name.startswith(f"{root}/") for root in ALLOWED_SOURCE_ROOTS
    )


def self_test() -> None:
    dockerfile = tarfile.TarInfo(
        "v2/crates/ruview-forecast-train/deploy/fal/Dockerfile"
    )
    dockerfile.type = tarfile.REGTYPE
    traversal = tarfile.TarInfo("v2/crates/ruview-forecast-train/../../secret")
    traversal.type = tarfile.REGTYPE
    symlink = tarfile.TarInfo("v2/crates/ruview-forecast-train/src/link.rs")
    symlink.type = tarfile.SYMTYPE
    readme = tarfile.TarInfo("v2/crates/ruview-forecast-train/README.md")
    readme.type = tarfile.REGTYPE
    test_source = tarfile.TarInfo("v2/crates/ruview-forecast-train/tests/secret.rs")
    test_source.type = tarfile.REGTYPE
    model_source = tarfile.TarInfo("v2/crates/ruview-forecast-model/src/lib.rs")
    model_source.type = tarfile.REGTYPE
    if (
        not safe_member(dockerfile)
        or not safe_member(model_source)
        or safe_member(traversal)
        or safe_member(symlink)
        or safe_member(readme)
        or safe_member(test_source)
    ):
        raise SystemExit("deployment archive allowlist self-test failed")
    for mode in ("run", "deploy"):
        command = fal_command("/usr/bin/fal", mode)
        if command[2] != FAL_FUNCTION_REF:
            raise SystemExit("deployment command self-test failed")
        if command[command.index("--auth") + 1] != "private":
            raise SystemExit("deployment authentication self-test failed")
        if command[command.index("--app-name") + 1] != FAL_APP_NAME:
            raise SystemExit("deployment app-name self-test failed")
        if ("--reset-scale" in command) != (mode == "deploy"):
            raise SystemExit("deployment scale-policy self-test failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "deploy", "self-test"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    if args.mode == "self-test":
        self_test()
        return 0
    if args.receipt is None:
        parser.error("--receipt is required for run and deploy")

    repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    dirty = git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *FORECAST_SCOPE,
    )
    if dirty.strip():
        raise SystemExit("forecast deployment scope is dirty or untracked; commit and review it first")

    commit = git(repo, "rev-parse", "HEAD").strip()
    tree = git(repo, "rev-parse", "HEAD^{tree}").strip()
    if len(tree) not in (40, 64) or any(char not in "0123456789abcdef" for char in tree):
        raise SystemExit("unexpected git tree identity")
    lock_bytes = git(repo, "show", "HEAD:v2/Cargo.lock", text=False)
    lock_digest = hashlib.sha256(lock_bytes).hexdigest()
    worker_build_id = f"ruview-{tree}"

    with tempfile.TemporaryDirectory(prefix="ruforecast-fal-") as temporary:
        context = Path(temporary) / "context"
        context.mkdir(mode=0o700)
        archive = Path(temporary) / "source.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                "HEAD",
                *ARCHIVE_SCOPE,
            ],
            check=True,
        )
        with tarfile.open(archive, "r:") as source:
            members = source.getmembers()
            rejected = [member.name for member in members if not safe_member(member)]
            if rejected:
                raise SystemExit(f"deployment archive contains non-source paths: {rejected[:5]}")
            source.extractall(context, members=members)
        minimal_workspace = (
            context
            / "v2/crates/ruview-forecast-train/deploy/fal/workspace.Cargo.toml"
        )
        shutil.copyfile(minimal_workspace, context / "v2/Cargo.toml")
        (context / ".ruview-build-tree").write_text(tree + "\n", encoding="ascii")

        receipt = {
            "schema_version": 1,
            "created_at_unix_ms": time.time_ns() // 1_000_000,
            "git_commit": commit,
            "git_tree": tree,
            "cargo_lock_sha256": lock_digest,
            "worker_build_id": worker_build_id,
            "build_manifest_sha256": lock_digest,
            "source": "git_archive_head_allowlisted_v1",
            "mode": args.mode,
            "fal_app_name": FAL_APP_NAME,
            "fal_function_ref": FAL_FUNCTION_REF,
            "auth": "private",
            "status": "SUCCEEDED",
        }

        fal = shutil.which("fal")
        if fal is None:
            raise SystemExit("fal CLI is not installed")
        environment = os.environ.copy()
        environment["RUVIEW_WORKER_BUILD_ID"] = worker_build_id
        environment["RUVIEW_BUILD_MANIFEST_SHA256"] = lock_digest
        subprocess.run(
            fal_command(fal, args.mode), cwd=context, env=environment, check=True
        )

        # A deployment receipt is evidence of a successful CLI operation, not
        # merely an attempted build. Do not leave a success-shaped receipt when
        # fal rejects the function or the remote build fails.
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.chmod(args.receipt, 0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
