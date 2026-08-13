"""Run the LIBERO Franka simulator natively on Apple Silicon."""

# ruff: noqa: E402, I001 -- imports below must follow the Mac runtime bootstrap.

import ctypes
import logging
import os
import pathlib
import sys

import tyro


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT / "packages/openpi-client/src"))
sys.path.insert(0, str(_REPO_ROOT / "third_party/libero"))

os.environ.setdefault("LIBERO_CONFIG_PATH", str(pathlib.Path(__file__).parent / "config.macos"))
os.environ.setdefault("MPLCONFIGDIR", str(_REPO_ROOT / ".matplotlib-cache"))
os.environ.setdefault("MUJOCO_GL", "cgl")
# LIBERO's init-state files are trusted assets from the pinned submodule.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# MuJoCo 2.3.7 uses Apple's old OpenGL framework path. Redirect only that
# lookup on newer macOS versions and delegate every other library unchanged.
_original_cdll = ctypes.CDLL


def _macos_cdll(name, *args, **kwargs):
    old_opengl_path = "/System/Library/OpenGL.framework/OpenGL"
    current_opengl_path = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
    if name == old_opengl_path and not pathlib.Path(name).exists():
        name = current_opengl_path
    return _original_cdll(name, *args, **kwargs)


ctypes.CDLL = _macos_cdll

import main as libero_main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(libero_main.eval_libero)
