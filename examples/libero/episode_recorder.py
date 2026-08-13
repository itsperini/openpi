"""Synchronized episode artifacts for the LIBERO visualizer."""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any

import imageio
import numpy as np

ACTION_LABELS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
JOINT_LABELS = [f"joint {index}" for index in range(1, 8)]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


class EpisodeRecorder:
    def __init__(
        self,
        output_root: str | pathlib.Path,
        *,
        task_suite: str,
        task_id: int,
        trial_id: int,
        instruction: str,
        control_hz: int,
        replan_steps: int,
        server_metadata: dict[str, Any],
    ) -> None:
        created_at = datetime.datetime.now(datetime.UTC)
        self.episode_id = f"episode_{created_at:%Y%m%dT%H%M%SZ}_{task_id:03d}_{trial_id:02d}"
        self.output_root = pathlib.Path(output_root)
        self.episode_dir = self.output_root / self.episode_id
        self.created_at = created_at.isoformat()
        self.task_suite = task_suite
        self.task_id = task_id
        self.trial_id = trial_id
        self.instruction = instruction
        self.control_hz = control_hz
        self.replan_steps = replan_steps
        self.server_metadata = _json_value(server_metadata)
        self.records: list[dict[str, Any]] = []
        self.frames: dict[str, list[np.ndarray]] = {"mujoco": [], "agent": [], "wrist": []}

    def record(
        self,
        *,
        step: int,
        observation: dict[str, Any],
        mujoco_frame: np.ndarray,
        agent_frame: np.ndarray,
        wrist_frame: np.ndarray,
        executed_action: np.ndarray,
        chunk_id: int,
        chunk_step: int,
        action_chunk: np.ndarray | None,
        inference_ms: float | None,
        server_timing: dict[str, Any] | None,
        policy_timing: dict[str, Any] | None,
        reward: float,
        done: bool,
    ) -> None:
        self.frames["mujoco"].append(np.asarray(mujoco_frame))
        self.frames["agent"].append(np.asarray(agent_frame))
        self.frames["wrist"].append(np.asarray(wrist_frame))
        self.records.append(
            {
                "step": step,
                "timestamp": round(step / self.control_hz, 4),
                "observation": {
                    "joint_position": _json_value(observation["robot0_joint_pos"]),
                    "joint_velocity": _json_value(observation["robot0_joint_vel"]),
                    "ee_position": _json_value(observation["robot0_eef_pos"]),
                    "ee_quaternion": _json_value(observation["robot0_eef_quat"]),
                    "gripper_position": _json_value(observation["robot0_gripper_qpos"]),
                    "gripper_velocity": _json_value(observation["robot0_gripper_qvel"]),
                },
                "policy": {
                    "executed_action": _json_value(executed_action),
                    "chunk_id": chunk_id,
                    "chunk_step": chunk_step,
                    "action_chunk": _json_value(action_chunk) if action_chunk is not None else None,
                    "inference_ms": inference_ms,
                    "server_timing": _json_value(server_timing or {}),
                    "policy_timing": _json_value(policy_timing or {}),
                },
                "result": {"reward": float(reward), "done": bool(done)},
            }
        )

    def finalize(self, *, success: bool, error: str | None = None) -> pathlib.Path:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        videos = {
            "mujoco": "mujoco.mp4",
            "agent": "camera_agent.mp4",
            "wrist": "camera_wrist.mp4",
        }
        for name, filename in videos.items():
            if self.frames[name]:
                imageio.mimwrite(self.episode_dir / filename, self.frames[name], fps=self.control_hz)

        action_horizon = 0
        inference_count = 0
        inference_values = []
        for record in self.records:
            policy = record["policy"]
            if policy["action_chunk"] is not None:
                action_horizon = max(action_horizon, len(policy["action_chunk"]))
                inference_count += 1
            if policy["inference_ms"] is not None:
                inference_values.append(float(policy["inference_ms"]))

        duration = len(self.records) / self.control_hz
        metadata = {
            "schema_version": 1,
            "episode_id": self.episode_id,
            "created_at": self.created_at,
            "task": {
                "suite": self.task_suite,
                "id": self.task_id,
                "trial": self.trial_id,
                "instruction": self.instruction,
            },
            "result": {"success": success, "error": error},
            "timeline": {
                "control_hz": self.control_hz,
                "frame_count": len(self.records),
                "duration_seconds": round(duration, 4),
            },
            "policy": {
                "name": "π0.5 LIBERO",
                "checkpoint": "pi05_libero",
                "replan_steps": self.replan_steps,
                "action_horizon": action_horizon,
                "inference_count": inference_count,
                "mean_inference_ms": round(float(np.mean(inference_values)), 3) if inference_values else None,
                "server_metadata": self.server_metadata,
            },
            "series": {"joints": JOINT_LABELS, "actions": ACTION_LABELS},
            "artifacts": {"trajectory": "trajectory.jsonl", "videos": videos},
        }
        (self.episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (self.episode_dir / "trajectory.jsonl").open("w") as trajectory_file:
            for record in self.records:
                trajectory_file.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._update_index(metadata)
        return self.episode_dir

    def _update_index(self, metadata: dict[str, Any]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        index_path = self.output_root / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {"schema_version": 1, "episodes": []}

        summary = {
            "episode_id": metadata["episode_id"],
            "created_at": metadata["created_at"],
            "instruction": metadata["task"]["instruction"],
            "success": metadata["result"]["success"],
            "duration_seconds": metadata["timeline"]["duration_seconds"],
            "metadata": f"{metadata['episode_id']}/metadata.json",
        }
        episodes = [item for item in index.get("episodes", []) if item["episode_id"] != self.episode_id]
        episodes.append(summary)
        episodes.sort(key=lambda item: item["created_at"], reverse=True)
        index["episodes"] = episodes
        index["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        index_path.write_text(json.dumps(index, indent=2) + "\n")
