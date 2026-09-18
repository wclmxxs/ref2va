"""Compiler counters and mask build timing; no changes to attention semantics."""
from collections import OrderedDict
import math
import os
import time


def cache_settings():
    def integer(name, default, low, high):
        raw = os.environ.get(name, str(default))
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None
        if not low <= value <= high:
            raise ValueError(f"{name} must be in [{low}, {high}]")
        return value
    capacity = integer("REF2VA_COMPILE_SHAPES", 32, 8, 64)
    # Padding/score_mod improved reuse but regressed H200 hot denoising latency.
    # Keep native FA4 as the default; bucketing remains an explicit experiment.
    stride = integer("REF2VA_TOKEN_BUCKET", 0, 0, 2048)
    if stride not in (0, 256, 512, 1024, 2048):
        raise ValueError("REF2VA_TOKEN_BUCKET must be 0, 256, 512, 1024 or 2048")
    raw_durations = os.environ.get("REF2VA_WARMUP_DURATIONS", "").strip()
    try:
        durations = list(dict.fromkeys(float(value.strip()) for value in raw_durations.split(','))) if raw_durations else []
    except ValueError:
        raise ValueError("REF2VA_WARMUP_DURATIONS must be empty or comma-separated seconds in [4, 15]") from None
    if len(durations) > 4 or any(not math.isfinite(value) or not 4 <= value <= 15 for value in durations):
        raise ValueError("REF2VA_WARMUP_DURATIONS accepts at most four durations in [4, 15]")
    return {"max_shapes": capacity,
            # Each helper can specialize several times per geometry. Retain the
            # upstream hard-failure policy instead of silently falling back.
            "recompile_limit": capacity * 8,
            "warmup_recent": integer("REF2VA_WARMUP_RECENT", 0, 0, capacity - 1 - len(durations)),
            "warmup_verify": bool(integer("REF2VA_WARMUP_VERIFY", 0, 0, 1)),
            "token_bucket": stride, "warmup_durations": durations}


class ObservedMasks(OrderedDict):
    def __init__(self, previous=()):
        super().__init__(previous)
        self.hits = self.misses = 0

    def __contains__(self, key):
        exists = super().__contains__(key)
        if exists:
            self.hits += 1
        else:
            self.misses += 1
        return exists


class CompilerMonitor:
    def __init__(self, flex_module, device):
        import torch
        from torch._dynamo import utils
        self.utils = utils
        self.device = device
        self.masks = ObservedMasks(flex_module._MASK_CACHE)
        flex_module._MASK_CACHE = self.masks
        self.mask_build_seconds = 0.
        original = flex_module.create_block_mask

        def build(*args, **kwargs):
            # Only misses synchronize. A hot mask lookup remains a dict lookup.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                self.mask_build_seconds += time.perf_counter() - started
        flex_module.create_block_mask = build

    def snapshot(self):
        counters = self.utils.counters
        timing = self.utils.calculate_time_spent()
        return {"dynamo_compile_seconds": float(timing.get("entire_frame_compile", 0.)),
                "unique_graphs": int(counters["stats"]["unique_graphs"]),
                "fxgraph_cache_hits": int(counters["inductor"]["fxgraph_cache_hit"]),
                "fxgraph_cache_misses": int(counters["inductor"]["fxgraph_cache_miss"]),
                "mask_hits": self.masks.hits, "mask_misses": self.masks.misses,
                "mask_build_seconds": self.mask_build_seconds,
                "guard_failure_counts": {code: len(values) for code, values in self.utils.guard_failures.items()}}

    def since(self, before):
        after = self.snapshot()
        result = {key: max(0, value - before[key]) for key, value in after.items()
                  if key != "guard_failure_counts"}
        failures = []
        for code, values in self.utils.guard_failures.items():
            for reason in values[before["guard_failure_counts"].get(code, 0):]:
                failures.append({"function": code.co_name, "reason": str(reason)[:1200]})
        result["guard_failures"] = len(failures)
        result["recompile_reasons"] = failures[-8:]
        return result


def summarize_compilation(records):
    # Ranks compile concurrently; summing their times would overstate latency.
    compiled = any(r["unique_graphs"] for r in records)
    seconds = max(r["dynamo_compile_seconds"] for r in records)
    return {"by_rank": records, "times_overlap_denoise": True,
            "dynamo_compile_seconds": seconds,
            "mask_build_seconds": max(r["mask_build_seconds"] for r in records),
            "compiled_new_graph": compiled,
            "runtime_graph_reused": not compiled and seconds == 0,
            "disk_graph_cache_hits": sum(r["fxgraph_cache_hits"] for r in records),
            "disk_graph_cache_misses": sum(r["fxgraph_cache_misses"] for r in records),
            "scope": "denoise; Dynamo timing includes tracing/cache loading and may overlap mask construction"}
