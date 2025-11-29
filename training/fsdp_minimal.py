"""Minimal FSDP-style training example.

This file is designed to:

- Use FSDP when `torch.distributed` and FSDP are available
- Gracefully fall back to a standard nn.Module otherwise
- Run a single training step on CPU

The goal is to show that you understand how FSDP is wired,
without requiring a multi-node GPU cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim


try:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    _HAS_FSDP = True
except Exception:  # pragma: no cover - if distributed is missing
    dist = None  # type: ignore
    FSDP = None  # type: ignore
    _HAS_FSDP = False


@dataclass
class TinyConfig:
    d_model: int = 32
    vocab_size: int = 64
    seq_len: int = 16
    batch_size: int = 4
    lr: float = 1e-3


class TinyTransformerBlock(nn.Module):
    """A very small Transformer-like block (not production grade)."""

    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.linear = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        logits = self.linear(x)
        return logits


def _init_dist_if_possible() -> bool:
    """Initialize a CPU-only single-process group if possible.

    Returns True if FSDP can be used, False otherwise.
    """
    if not _HAS_FSDP:
        return False
    if dist.is_available() and not dist.is_initialized():
        try:
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
            return True
        except Exception:
            return False
    return dist.is_initialized()


def train_one_step() -> float:
    cfg = TinyConfig()
    device = torch.device("cpu")

    model = TinyTransformerBlock(cfg)
    use_fsdp = _init_dist_if_possible()
    if use_fsdp and FSDP is not None:
        model = FSDP(model)  # type: ignore[call-arg]

    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    # Synthetic next-token prediction task.
    x = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.seq_len),
        device=device,
    )
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = 0

    logits = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        logits.view(-1, cfg.vocab_size),
        y.view(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    loss_value = float(loss.detach().cpu().item())

    if use_fsdp and dist is not None and dist.is_initialized():
        dist.destroy_process_group()

    return loss_value


def main() -> None:
    loss = train_one_step()
    print(f"FSDP-style single step completed. Loss = {loss:.4f}")


if __name__ == "__main__":
    main()