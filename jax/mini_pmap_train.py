"""Minimal JAX + pmap example.

This file is designed to be:

- Structurally correct and easy to read for a reviewer
- Executable on CPU for a lightweight check
- TPU-ready when run in an appropriate environment (e.g., Colab TPU)

The core idea is to show a simple data-parallel training step with `pmap`.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import jax
import jax.numpy as jnp


def create_toy_batch(batch_size: int, dim: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create a simple linear regression batch: y = 2x + 1 + noise."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, dim))
    noise = 0.01 * jax.random.normal(key, (batch_size, 1))
    y = 2.0 * x.sum(axis=-1, keepdims=True) + 1.0 + noise
    return x, y


def init_params(dim: int) -> dict:
    key = jax.random.PRNGKey(42)
    w = jax.random.normal(key, (dim, 1))
    b = jnp.zeros((1,))
    return {"w": w, "b": b}


def loss_fn(params: dict, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    x, y = batch
    preds = x @ params["w"] + params["b"]
    return jnp.mean((preds - y) ** 2)


def train_step(params: dict, batch: Tuple[jnp.ndarray, jnp.ndarray], lr: float) -> dict:
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, batch)
    return {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"],
    }


# Vectorized version for pmap: each device sees a shard of the batch.
p_train_step = jax.pmap(train_step, in_axes=(None, 0, None))


def run_cpu_check() -> None:
    """Run a lightweight structural check on CPU.

    This does not require a TPU; it simply verifies that the pmap graph
    can be constructed and executed on a single host device.
    """
    devices = jax.devices()
    print(f"Available devices: {devices}")
    num_devices = max(1, len(devices))

    dim = 4
    batch_per_device = 2

    # Shape: (num_devices, batch_per_device, dim)
    x, y = create_toy_batch(num_devices * batch_per_device, dim)
    x = x.reshape(num_devices, batch_per_device, dim)
    y = y.reshape(num_devices, batch_per_device, 1)

    params = init_params(dim)
    updated_params = p_train_step(params, (x, y), 0.1)
    print("pmap step completed. Example b:", updated_params["b"][0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-check",
        action="store_true",
        help="Run a CPU-only structural check (no TPU required)."
    )
    args = parser.parse_args()

    if args.cpu_check:
        run_cpu_check()
    else:
        # In a TPU environment (e.g., Colab TPU), you could extend this
        # to run multiple steps; here we keep it minimal.
        run_cpu_check()


if __name__ == "__main__":
    main()