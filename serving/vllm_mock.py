"""A CPU-friendly mock of vLLM-style KV cache & batching.

This does *not* implement real attention. Instead, it focuses on:

- Request batching
- Simple KV cache lifetime
- Interfaces that resemble a fast-path inference engine

This is intentionally small so a reviewer can quickly understand the design.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional


RequestId = int


@dataclass
class KVEntry:
    """Represents cached key/value for a single request.

    For a real implementation this would hold tensors; here we use integers
    to keep the example CPU-friendly and focus on structure.
    """

    key_tokens: List[int] = field(default_factory=list)
    value_tokens: List[int] = field(default_factory=list)


class KVCache:
    """Minimal KV cache with create/update semantics."""

    def __init__(self) -> None:
        self._store: Dict[RequestId, KVEntry] = {}

    def get(self, request_id: RequestId) -> Optional[KVEntry]:
        return self._store.get(request_id)

    def update(self, request_id: RequestId, new_tokens: List[int]) -> KVEntry:
        entry = self._store.get(request_id)
        if entry is None:
            entry = KVEntry()
            self._store[request_id] = entry
        # In a real implementation, keys/values are separate; here we just
        # append tokens to both lists for demonstration purposes.
        entry.key_tokens.extend(new_tokens)
        entry.value_tokens.extend(new_tokens)
        return entry

    def size(self) -> int:
        return len(self._store)


@dataclass
class GenerationRequest:
    request_id: RequestId
    prompt: str
    max_new_tokens: int = 16


@dataclass
class GenerationResult:
    request_id: RequestId
    text: str


class SimpleBatcher:
    """Collects incoming requests and forms small batches.

    This is deliberately minimal: in a real system you would have timing-based
    flush conditions etc. For our purposes it's enough to show that the
    batching interface exists and behaves correctly.
    """

    def __init__(self, max_batch_size: int = 8) -> None:
        self.max_batch_size = max_batch_size
        self._queue: List[GenerationRequest] = []

    def add(self, req: GenerationRequest) -> None:
        self._queue.append(req)

    def pop_batch(self) -> List[GenerationRequest]:
        if not self._queue:
            return []
        batch = self._queue[: self.max_batch_size]
        self._queue = self._queue[self.max_batch_size :]
        return batch

    def __len__(self) -> int:  # for testing
        return len(self._queue)


class MockVLLMEngine:
    """A very small, CPU-only 'engine' with KV cache + batching.

    Instead of real LLM inference, this engine:

    - uses the prompt length to generate synthetic tokens
    - updates KV cache entries
    - returns a dummy text that encodes cache length information
    """

    _request_counter = itertools.count(1)

    def __init__(self, max_batch_size: int = 8) -> None:
        self.kv_cache = KVCache()
        self.batcher = SimpleBatcher(max_batch_size=max_batch_size)

    def submit(self, prompt: str, max_new_tokens: int = 16) -> RequestId:
        request_id = next(self._request_counter)
        req = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        self.batcher.add(req)
        return request_id

    def process_next_batch(self) -> List[GenerationResult]:
        batch = self.batcher.pop_batch()
        results: List[GenerationResult] = []
        for req in batch:
            # 'Generate' dummy tokens as a simple function of prompt length.
            num_tokens = min(len(req.prompt), req.max_new_tokens)
            new_tokens = list(range(num_tokens))
            entry = self.kv_cache.update(req.request_id, new_tokens)
            text = f"request={req.request_id}, cached={len(entry.key_tokens)}"
            results.append(GenerationResult(request_id=req.request_id, text=text))
        return results