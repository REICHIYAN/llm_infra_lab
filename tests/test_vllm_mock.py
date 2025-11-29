from serving.vllm_mock import KVCache, MockVLLMEngine


def test_kv_cache_update_and_size() -> None:
    cache = KVCache()
    assert cache.size() == 0
    entry1 = cache.update(1, [1, 2, 3])
    assert cache.size() == 1
    assert entry1.key_tokens == [1, 2, 3]

    entry2 = cache.update(1, [4])
    assert cache.size() == 1  # same request_id
    assert entry2.key_tokens == [1, 2, 3, 4]


def test_engine_batches_and_updates_cache() -> None:
    engine = MockVLLMEngine(max_batch_size=2)
    req1 = engine.submit("hello", max_new_tokens=3)
    req2 = engine.submit("world", max_new_tokens=5)
    assert req1 != req2

    results = engine.process_next_batch()
    assert len(results) == 2
    cached_size_req1 = engine.kv_cache.get(req1).key_tokens  # type: ignore[union-attr]
    cached_size_req2 = engine.kv_cache.get(req2).key_tokens  # type: ignore[union-attr]
    assert len(cached_size_req1) > 0
    assert len(cached_size_req2) > 0

    # Submit another request for req1 and ensure cache grows.
    engine.submit("hello again", max_new_tokens=2)
    engine.process_next_batch()
    updated_entry = engine.kv_cache.get(req1)
    assert updated_entry is not None
    assert len(updated_entry.key_tokens) >= len(cached_size_req1)