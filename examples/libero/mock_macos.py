"""Animate a Franka Panda in LIBERO without a policy server."""

# ruff: noqa: E402, I001 -- imports below must follow the Mac runtime bootstrap.

import ctypes
import dataclasses
import logging
import os
import pathlib
import sys
import time

import imageio
import numpy as np
import tyro


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(_REPO_ROOT)
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
    if name == "/System/Library/OpenGL.framework/OpenGL" and not pathlib.Path(name).exists():
        name = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
    return _original_cdll(name, *args, **kwargs)


ctypes.CDLL = _macos_cdll

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


@dataclasses.dataclass
class Args:
    task_id: int = 0
    steps: int = 100
    seed: int = 7
    display: bool = True
    video_path: str = "data/libero/videos/franka_mock_trajectory.mp4"


def _mock_action(step: int) -> np.ndarray:
    """Return a smooth, bounded OSC pose command with gripper motion."""
    phase = 2.0 * np.pi * step / 80.0
    action = np.zeros(7, dtype=np.float32)
    action[0] = 0.20 * np.sin(phase)
    action[1] = 0.14 * np.sin(phase / 2.0)
    action[2] = 0.08 * np.sin(phase + np.pi / 2.0)
    action[5] = 0.10 * np.sin(phase / 2.0)
    action[6] = 1.0 if (step // 60) % 2 else -1.0
    return action


def run(args: Args) -> None:
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {suite.n_tasks - 1}]")

    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    env.reset()
    obs = env.set_init_state(initial_states[args.seed % len(initial_states)])

    cv2 = None
    if args.display:
        import cv2 as _cv2

        cv2 = _cv2

    frames = []
    logging.info("Animating Franka with a local mock trajectory: %s", task.language)
    try:
        for step in range(args.steps):
            # Let objects settle before moving the robot.
            action = np.array([0.0] * 6 + [-1.0]) if step < 15 else _mock_action(step - 15)
            obs, _, _, _ = env.step(action.tolist())
            frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            frames.append(frame)

            if cv2 is not None:
                cv2.imshow("OpenPI - Franka mock trajectory (no Modal)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(50) & 0xFF in (27, ord("q")):
                    logging.info("Preview stopped from the display window")
                    break
            else:
                time.sleep(0.05)
    finally:
        env.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    output = pathlib.Path(args.video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output, frames, fps=20)
    logging.info("Saved %d frames to %s", len(frames), output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(run)
