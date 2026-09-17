"""Compiler counters and mask build timing; no changes to attention semantics."""
from collections import OrderedDict
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
    return {"max_shapes": capacity,
            # Each helper can specialize several times per geometry. Retain the
            # upstream hard-failure policy instead of silently falling back.
            "recompile_limit": capacity * 8,
            "warmup_recent": integer("REF2VA_WARMUP_RECENT", min(8, capacity - 1), 0, capacity - 1)}


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
                "mask_build_seconds": self.mask_build_seconds}

    def since(self, before):
        return {key: max(0, value - before[key]) for key, value in self.snapshot().items()}


def summarize_compilation(records):
    # Ranks compile concurrently; summing their times would overstate latency.
    return {"by_rank": records, "times_overlap_denoise": True,
            "dynamo_compile_seconds": max(r["dynamo_compile_seconds"] for r in records),
            "mask_build_seconds": max(r["mask_build_seconds"] for r in records),
            "compiled_new_graph": any(r["unique_graphs"] for r in records),
            "scope": "denoise; Dynamo timing includes tracing/cache loading and may overlap mask construction"}
