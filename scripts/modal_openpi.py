"""Deploy the OpenPI policy server as a Modal WebSocket endpoint."""

import os
import shlex
import subprocess

import modal

APP_NAME = os.getenv("OPENPI_APP_NAME", "openpi")
OPENPI_REPOSITORY = os.getenv("OPENPI_REPOSITORY", "https://github.com/itsperini/openpi.git")
OPENPI_GIT_REF = os.getenv("OPENPI_GIT_REF", "15a9616a00943ada6c20a0f158e3adb39df2ccac")
OPENPI_SERVER_ARGS_VALUE = os.getenv("OPENPI_SERVER_ARGS", "--env=DROID")
OPENPI_SERVER_ARGS = shlex.split(OPENPI_SERVER_ARGS_VALUE)
OPENPI_GPU = os.getenv("OPENPI_GPU", "L40S")
OPENPI_MIN_CONTAINERS = int(os.getenv("OPENPI_MIN_CONTAINERS", "0"))
OPENPI_CACHE_PATH = "/openpi-cache"

app = modal.App(APP_NAME)
checkpoint_cache = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)

repository = shlex.quote(OPENPI_REPOSITORY)
git_ref = shlex.quote(OPENPI_GIT_REF)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "git-lfs",
        "linux-headers-generic",
        "build-essential",
        "clang",
    )
    .pip_install("uv==0.11.12")
    .run_commands(
        f"git clone {repository} /openpi",
        f"cd /openpi && git checkout --detach {git_ref}",
        "cd /openpi && git submodule update --init --recursive",
        "cd /openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev",
    )
    # The module is imported again inside the remote container. Bake the policy
    # selection into its environment so the runtime import sees the deployed overlay.
    .env(
        {
            "OPENPI_DATA_HOME": OPENPI_CACHE_PATH,
            "OPENPI_SERVER_ARGS": OPENPI_SERVER_ARGS_VALUE,
        }
    )
    .workdir("/openpi")
)


@app.function(
    image=image,
    gpu=OPENPI_GPU,
    cpu=4,
    memory=32768,
    volumes={OPENPI_CACHE_PATH: checkpoint_cache},
    min_containers=OPENPI_MIN_CONTAINERS,
    max_containers=1,
    scaledown_window=20 * 60,
    timeout=24 * 60 * 60,
)
@modal.web_server(
    8000,
    startup_timeout=20 * 60,
    requires_proxy_auth=True,
)
def openpi_server() -> None:
    subprocess.Popen(
        [
            "/openpi/.venv/bin/python",
            "/openpi/scripts/serve_policy.py",
            *OPENPI_SERVER_ARGS,
        ],
        cwd="/openpi",
    )
