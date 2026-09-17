"""Per-request Ulysses layouts, DBCache, and opt-in CUDA-event profiling."""
from contextlib import contextmanager
from dataclasses import replace
import types

from .cache_dit import CacheConfig, DBCache


BLOCK_LOOP = """    for block in self.transformer_blocks:
        packed = block(packed, temb, local_adaln, local_rotary)"""
BLOCK_REPLACEMENT = """    runtime.profile_end("input_prepare", _ref2va_input_event)
    packed = self._ref2va_blocks.run(
        packed, (temb, local_adaln, local_rotary), video_indices, audio_indices)
    _ref2va_gather_event = runtime.profile_start()"""


def transformer_replacements():
    return [
        ("    runtime.configure(sequence_length, first_attn.num_heads)",
         "    runtime.configure(sequence_length, first_attn.num_heads)\n    _ref2va_input_event = runtime.profile_start()"),
        (BLOCK_LOOP, BLOCK_REPLACEMENT),
        ("    packed = runtime.gather_sequence(packed[0]).unsqueeze(0)",
         "    packed = runtime.gather_sequence(packed[0]).unsqueeze(0)\n"
         '    runtime.profile_end("final_gather", _ref2va_gather_event)\n'
         "    _ref2va_head_event = runtime.profile_start()"),
        ("    if not return_dict:",
         '    runtime.profile_end("output_head", _ref2va_head_event)\n    if not return_dict:')]


def timed_method(method, runtime, name):
    def call(*args, **kwargs):
        if not runtime.profile_enabled:
            return method(*args, **kwargs)
        start = runtime.profile_start()
        try:
            return method(*args, **kwargs)
        finally:
            runtime.profile_end(name, start)
    return call


class DiTRuntime:
    def __init__(self, transformer, runtime, hybrids):
        self.transformer, self.runtime = transformer, runtime
        self.hybrids = list(hybrids)
        self.cache = DBCache(runtime)
        transformer._ref2va_blocks = self
        for block in transformer.transformer_blocks:
            block.ff.forward = timed_method(block.ff.forward, runtime, "ffn")
        # Standard Ulysses does not have the branch path's detailed event scopes.
        for name in ("sequence_to_heads", "heads_to_sequence"):
            method = getattr(runtime, name)
            setattr(runtime, name, timed_method(method, runtime, name))

    def install_attention(self, forwards):
        self.forwards = forwards
        self.select_layout(self.runtime.softmax_ranks)

    def select_layout(self, softmax_ranks):
        runtime = self.runtime
        if type(softmax_ranks) is not int or not 0 <= softmax_ranks < runtime.world_size:
            raise ValueError("softmax_ranks must be in [0, world_size - 1]")
        # Called only between serialized requests, after every prior collective
        # completed. Existing dispatch communicators contain all eight ranks.
        runtime.sequence_length = 0
        runtime.softmax_ranks = softmax_ranks
        name = "_branch_parallel_attention_forward" if softmax_ranks else "_ulysses_attention_forward"
        for attn in self.hybrids:
            method = types.MethodType(self.forwards[name], attn)
            attn.forward = timed_method(method, runtime, "attention")

    @contextmanager
    def request(self, settings, *, warmup=False):
        self.select_layout(settings.softmax_ranks)
        self.runtime.profile_enabled = settings.profile and not warmup
        self.runtime.reset_profile()
        config = CacheConfig.from_settings(settings)
        if warmup:
            # All startup/history parity checks and shape warming always execute
            # every block of all eight NFE, including histories with cache enabled.
            config = replace(config, enabled=False)
        try:
            with self.cache.request(config, total_blocks=len(self.transformer.transformer_blocks)):
                yield self
        finally:
            self.runtime.profile_enabled = False
            self.runtime.reset_profile()

    def run(self, packed, args, video_indices, audio_indices):
        cfg = self.cache.config
        if cfg.enabled and cfg.threshold > 0 and cfg.max_cached_steps > 0:
            layout = self.hybrids[0].layout
            buckets = getattr(self.hybrids[0], "_ref2va_buckets", None)
            self.cache.configure_groups(packed, video_indices, audio_indices,
                                        layout.video_start, layout.video_end,
                                        buckets.current.prefix_tokens if buckets and buckets.active else None)

        def call(block, value, block_args):
            if not self.runtime.profile_enabled:
                return block(value, *block_args)
            start = self.runtime.profile_start()
            result = block(value, *block_args)
            self.runtime.profile_end("blocks", start)
            return result
        return self.cache.run(self.transformer.transformer_blocks, packed, args, call)

    def report(self, denoise_seconds):
        runtime = self.runtime
        totals = runtime.profile_milliseconds() if runtime.profile_enabled else {}
        counts = {name: len(events) for name, events in runtime.profile_events.items()}
        role = runtime.branch_kind if runtime.branch_parallel else "both"
        report = {"rank": runtime.rank, "role": role, "heads": runtime.heads_per_rank,
                  "denoise_seconds": denoise_seconds, "profile_enabled": runtime.profile_enabled,
                  "total_ms": totals, "calls": counts,
                  "ms_per_nfe": {name: ms / max(1, self.cache.step) for name, ms in totals.items()}}
        return {"profile": report, "cache_dit": self.cache.report()}


def summarize_profiles(records):
    """Keep overlapping scopes separate; never sum GPU ranks into latency."""
    enabled = all(record["profile_enabled"] for record in records)
    result = {"enabled": enabled, "by_rank": records, "method": "cuda_events_on_compute_stream",
              "scope": "denoise only; includes enqueue gaps/waits; nested scopes overlap",
              "pure_nccl_kernel_time": False,
              "notes": ["branch_dispatch includes branch_pack and branch_relevant_wait",
                        "output_dispatch includes output_a2a and output_unpack",
                        "blocks includes attention and ffn; never sum these scopes or ranks",
                        "profiling overhead is included; use profile=false for speed comparisons"]}
    if not enabled:
        return result
    result["max_ms_per_nfe"] = {name: max(r["ms_per_nfe"].get(name, 0.) for r in records)
                                for name in {n for r in records for n in r["ms_per_nfe"]}}
    groups = {}
    for role in ("softmax", "linear", "both"):
        ranks = [r for r in records if r["role"] == role]
        if ranks:
            groups[role] = {"ranks": [r["rank"] for r in ranks],
                "max_compute_ms_per_nfe": (max(r["ms_per_nfe"].get(role + "_compute", 0.) for r in ranks)
                                           if role != "both" else None),
                "max_return_dispatch_ms_per_nfe": max(r["ms_per_nfe"].get(
                    "output_dispatch" if role != "both" else "heads_to_sequence", 0.) for r in ranks)}
    result["branches"] = groups
    # Compute-stream spans measure exposed waits, not the asynchronous NCCL
    # kernels' entire duration. Do not label an overlap-heavy sum a comm percent.
    return result


def summarize_cache(records):
    reference = records[0]
    for other in records[1:]:
        for key in ("config", "cached_steps", "executed_blocks", "skipped_blocks", "decisions"):
            if other[key] != reference[key]:
                raise RuntimeError(f"DBCache rank decisions diverged: {key}")
    return {**reference, "all_rank_agreement": True, "world_size": len(records)}
