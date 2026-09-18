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
    assert not summary["runtime_graph_reused"]
    assert summarize_compilation([hot])["runtime_graph_reused"]
    before = monitor.snapshot()
    compiled(torch.randn(15))
    changed = monitor.since(before)
    assert changed['unique_graphs'] > 0 and changed['guard_failures'] > 0
    assert any('size mismatch' in entry['reason'] for entry in changed['recompile_reasons'])
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
    monkeypatch.delenv("REF2VA_TOKEN_BUCKET", raising=False)
    monkeypatch.delenv("REF2VA_WARMUP_DURATIONS", raising=False)
    monkeypatch.delenv("REF2VA_WARMUP_VERIFY", raising=False)
    assert cache_settings() == {"max_shapes": 32, "recompile_limit": 256, "warmup_recent": 0,
                                "token_bucket": 0, "warmup_durations": [], "warmup_verify": False}
    monkeypatch.setenv("REF2VA_TOKEN_BUCKET", "1024")
    assert cache_settings()['token_bucket'] == 1024
    monkeypatch.setenv("REF2VA_WARMUP_DURATIONS", "5, 8,10,15,5")
    monkeypatch.setenv("REF2VA_WARMUP_VERIFY", "1")
    assert cache_settings()['warmup_durations'] == [5, 8, 10, 15]
    assert cache_settings()['warmup_verify']
    monkeypatch.setenv("REF2VA_COMPILE_SHAPES", "8")
    monkeypatch.setenv("REF2VA_WARMUP_RECENT", "3")
    assert cache_settings()["warmup_recent"] == 3
    monkeypatch.setenv("REF2VA_WARMUP_RECENT", "4")
    with pytest.raises(ValueError, match="REF2VA_WARMUP_RECENT"):
        cache_settings()


def test_migration_backfills_existing_short_history_for_all_thirteen_cases(tmp_path):
    settings = asdict(Settings(duration=10, ratio="9:16", resolution=768))
    sources, profile = {"model": "pinned"}, {"fp8": True}
    history = WarmupHistory(tmp_path / "history.json", sources, profile)
    jobs = tmp_path / "jobs"
    for index in range(13):
        cached = tmp_path / f"{index}.pt"
        cached.touch()
        record = {"status": "complete", "sources": sources,
                  "settings": settings, "prompt_file": str(cached)}
        folder = jobs / str(index)
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(record))
        if index >= 5:
            history.remember(str(index), record)
    requests = history.requests(27, jobs)
    assert len(requests) == 13
    assert {r["prompt_file"] for r in requests} == {str(tmp_path / f"{i}.pt") for i in range(13)}


@pytest.mark.parametrize("value", ["-1", "2", "1023", "4096", "bad"])
def test_invalid_token_bucket_rejected(monkeypatch, value):
    monkeypatch.setenv("REF2VA_TOKEN_BUCKET", value)
    with pytest.raises(ValueError, match="REF2VA_TOKEN_BUCKET"):
        cache_settings()


@pytest.mark.parametrize('name,value', [('REF2VA_WARMUP_DURATIONS', '5,nan'),
    ('REF2VA_WARMUP_DURATIONS', '3'), ('REF2VA_WARMUP_DURATIONS', '16'),
    ('REF2VA_WARMUP_DURATIONS', '5,,10'), ('REF2VA_WARMUP_DURATIONS', '5,6,7,8,9'),
    ('REF2VA_WARMUP_VERIFY', '2')])
def test_invalid_optional_startup_work_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        cache_settings()
