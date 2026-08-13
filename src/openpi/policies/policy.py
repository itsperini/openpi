from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        debug_metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            debug_metadata: Model and transform details returned only for traced inference.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._debug_metadata = debug_metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._sample_actions_with_trace = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_with_trace = getattr(model, "sample_actions_with_trace", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if trace_method := getattr(model, "sample_actions_with_trace", None):
                self._sample_actions_with_trace = nnx_utils.module_jit(trace_method)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        debug_requested = bool(inputs.pop("_openpi_debug", False))
        raw_inputs = jax.tree.map(lambda x: x, inputs) if debug_requested else None
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        debug_inputs = _summarize_debug_inputs(raw_inputs, inputs) if raw_inputs is not None else None
        start_time = time.monotonic()
        debug_trace = None
        if debug_requested and self._sample_actions_with_trace is not None:
            trace_kwargs = {key: value for key, value in sample_kwargs.items() if key == "noise"}
            sampled_actions, debug_trace = self._sample_actions_with_trace(
                sample_rng_or_pytorch_device, observation, **trace_kwargs
            )
        else:
            sampled_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": sampled_actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            if debug_trace is not None:
                debug_trace = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), debug_trace)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            if debug_trace is not None:
                debug_trace = jax.tree.map(lambda x: np.asarray(x[0, ...]), debug_trace)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if debug_requested:
            outputs["debug_trace"] = {
                "supported": debug_trace is not None,
                "model": self._debug_metadata,
                "input": debug_inputs,
                "flow": debug_trace,
                "output_action_dim": int(np.asarray(outputs["actions"]).shape[-1]),
            }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return np.asarray(value.detach().cpu())
    return np.asarray(value)


def _summarize_debug_inputs(raw_inputs: dict | None, model_inputs: dict) -> dict[str, Any]:
    if raw_inputs is None:
        return {}

    raw_images = {
        key.removeprefix("observation/"): {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for key, value in raw_inputs.items()
        if key.startswith("observation/") and "image" in key
    }
    prompt_mask = _to_numpy(model_inputs["tokenized_prompt_mask"])[0].astype(bool)
    prompt_tokens = _to_numpy(model_inputs["tokenized_prompt"])[0]
    model_images = {}
    for key, value in model_inputs["image"].items():
        image = _to_numpy(value)[0]
        model_images[key] = {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "range": [float(np.min(image)), float(np.max(image))],
            "mask": bool(_to_numpy(model_inputs["image_mask"][key])[0]),
        }

    raw_state = np.asarray(raw_inputs.get("observation/state", []))
    model_state = _to_numpy(model_inputs["state"])[0]
    return {
        "raw": {
            "prompt": str(raw_inputs.get("prompt", "")),
            "state": raw_state,
            "state_shape": list(raw_state.shape),
            "images": raw_images,
        },
        "model": {
            "state": model_state,
            "state_shape": list(model_state.shape),
            "prompt_tokens": prompt_tokens[prompt_mask],
            "prompt_token_count": int(np.sum(prompt_mask)),
            "prompt_token_capacity": int(prompt_mask.size),
            "images": model_images,
        },
    }


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
