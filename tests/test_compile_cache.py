from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from openvdn_comfy.compile_cache import CompilerMonitor, cache_settings, summarize_compilation
from openvdn_comfy.config import Settings
from openvdn_comfy.resident_geometry import GeometryCache
from openvdn_comfy.warmup_history import WarmupHistory


def test_successful_working_set_survives_more_than_eight_geometries():
    resets = []
    cache = GeometryCache(lambda: resets.append(True))
    runtime = SimpleNamespace(sequence_length=123)
    plan = Settings(duration=10, ratio="9:16", resolution=768).render_plan()
    for length in range(100, 120):
        assert cache.prepare(runtime, plan, torch.zeros(length, 3), torch.ones(length), None)
        assert runtime.sequence_length == 0
        cache.commit()
    assert not cache.prepare(runtime, plan, torch.zeros(100, 3), torch.ones(100), None)
    assert not resets
    assert cache.last["geometry_seen"]
    # An abandoned/failed sample never becomes a successful shape.
    assert cache.prepare(runtime, plan, torch.zeros(333, 3), torch.ones(333), None)
    assert cache.prepare(runtime, plan, torch.zeros(333, 3), torch.ones(333), None)


def test_real_dynamo_metrics_separate_first_compile_from_repeated_shape():
    torch._dynamo.reset()
    flex = SimpleNamespace(_MASK_CACHE={}, create_block_mask=lambda: "mask")
    monitor = CompilerMonitor(flex, torch.device("cpu"))
    compiled = torch.compile(lambda x: x.sin() + x * x, backend="eager", dynamic=False)
    x = torch.randn(13)
    before = monitor.snapshot()
    torch.testing.assert_close(compiled(x), x.sin() + x * x)
    cold = monitor.since(before)
    assert cold["unique_graphs"] == 1
    assert cold["dynamo_compile_seconds"] > 0
    before = monitor.snapshot()
    compiled(x + 1)
    hot = monitor.since(before)
    assert hot["unique_graphs"] == hot["dynamo_compile_seconds"] == 0
    # Keep cache lookup and mask construction separate from Dynamo counters.
    before = monitor.snapshot()
    assert "geometry" not in flex._MASK_CACHE
    flex._MASK_CACHE["geometry"] = flex.create_block_mask()
    assert "geometry" in flex._MASK_CACHE
    mask = monitor.since(before)
    assert mask["mask_hits"] == mask["mask_misses"] == 1
    assert mask["mask_build_seconds"] > 0
    summary = summarize_compilation([cold, hot])
    assert summary["dynamo_compile_seconds"] == cold["dynamo_compile_seconds"]
    assert summary["compiled_new_graph"]
    torch._dynamo.reset()


def test_warmup_replays_only_compatible_successful_local_caches(tmp_path):
    settings = asdict(Settings(duration=10, ratio="9:16", resolution=768))
    sources, profile = {"model": "pinned"}, {"fp8": True}
    history = WarmupHistory(tmp_path / "history.json", sources, profile, capacity=2)
    jobs = tmp_path / "jobs"
    for index, status, source in ((0, "complete", sources), (1, "failed", sources),
                                  (2, "complete", {"model": "different"})):
        cached = tmp_path / f"{index}.pt"
        cached.touch()
        job = jobs / str(index)
        job.mkdir(parents=True)
        (job / "result.json").write_text(json.dumps({"status": status, "sources": source,
            "settings": settings, "prompt_file": str(cached), "prompt": "private text"}))
    replay = history.requests(8, jobs)
    assert len(replay) == 1 and replay[0]["prompt_file"].endswith("0.pt")
    assert replay[0]["require_cached_prompt"] and replay[0]["prompt"] == ""
    assert history.requests(0, jobs) == []
    for index in range(3):
        path = tmp_path / f"{index}.pt"
        history.remember(str(index), {"settings": settings, "prompt_file": str(path)})
    restored = WarmupHistory(history.path, sources, profile, capacity=2)
    assert list(restored.entries) == ["1", "2"]
    assert restored.requests(1)[0]["prompt_file"].endswith("2.pt")
    (tmp_path / "2.pt").unlink()
    assert restored.requests(1)[0]["prompt_file"].endswith("1.pt")
    assert not WarmupHistory(history.path, {"model": "new"}, profile).requests(8)
    assert not WarmupHistory(history.path, sources, {"fp8": False}).requests(8)


def test_cache_options_validate_before_launch(monkeypatch):
    monkeypatch.delenv("REF2VA_COMPILE_SHAPES", raising=False)
    monkeypatch.delenv("REF2VA_WARMUP_RECENT", raising=False)
    assert cache_settings() == {"max_shapes": 32, "recompile_limit": 256, "warmup_recent": 8}
    monkeypatch.setenv("REF2VA_COMPILE_SHAPES", "8")
    monkeypatch.setenv("REF2VA_WARMUP_RECENT", "7")
    assert cache_settings()["warmup_recent"] == 7
    monkeypatch.setenv("REF2VA_WARMUP_RECENT", "8")
    with pytest.raises(ValueError, match="REF2VA_WARMUP_RECENT"):
        cache_settings()
