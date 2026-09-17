"""Request-scoped DBCache adaptation for OpenVDN's packed, sequence-sharded DiT.

An independent implementation of Cache-DiT's Fn/Mn/Bn residual-cache algorithm:
https://github.com/vipshop/cache-dit/tree/main/src/cache_dit/caching/cache_blocks
This is NOT the upstream package/plugin and does not install its model patches.
Differences: modality-separated global error statistics, finite guards, mandatory
last steps, and one shared eight-rank decision before skipping any collectives.
No TaylorSeer, attention replacement, fixed-interval skip or cross-request reuse.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math


FIELDS = ("cache_dit", "cache_dit_threshold", "cache_dit_fn_blocks", "cache_dit_bn_blocks",
          "cache_dit_warmup_steps", "cache_dit_max_consecutive", "cache_dit_max_cached_steps",
          "cache_dit_last_steps")


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = False
    threshold: float = .08
    fn_blocks: int = 8
    bn_blocks: int = 8
    warmup_steps: int = 3
    max_consecutive: int = 1
    max_cached_steps: int = 2
    last_steps: int = 1

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.cache_dit, **{name: getattr(settings, "cache_dit_" + name)
                                         for name in cls.__dataclass_fields__ if name != "enabled"})

    def validate(self, blocks=50, steps=8):
        if type(self.enabled) is not bool:
            raise ValueError("cache_dit must be boolean")
        if (type(self.threshold) not in (int, float) or not math.isfinite(self.threshold)
                or not 0 <= self.threshold <= 1):
            raise ValueError("cache_dit_threshold must be a finite number in [0, 1]; 0 disables reuse")
        for name, low, high in (("fn_blocks", 1, blocks - 1), ("bn_blocks", 0, blocks - 1),
                                ("warmup_steps", 1, steps), ("max_consecutive", 1, steps - 1),
                                ("max_cached_steps", 0, steps - 1), ("last_steps", 0, steps - 1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"cache_dit_{name} must be an integer in [{low}, {high}]")
        if self.fn_blocks + self.bn_blocks >= blocks:
            raise ValueError(f"cache_dit_fn_blocks + cache_dit_bn_blocks must be less than {blocks}")
        if self.warmup_steps + self.last_steps > steps:
            raise ValueError("cache_dit_warmup_steps + cache_dit_last_steps must be <= 8")
        return self


class DBCache:
    def __init__(self, runtime):
        self.runtime = runtime
        self.active_request = False
        self.clear()

    def clear(self):
        self.previous = self.residual = self.residual_finite = None
        self.groups = None

    @contextmanager
    def request(self, config, *, total_blocks=50, total_steps=8):
        if self.active_request:
            raise RuntimeError("DBCache cannot share state across concurrent requests")
        config.validate(total_blocks, total_steps)
        self.config, self.total_blocks, self.total_steps = config, total_blocks, total_steps
        self.step = self.hits = self.consecutive = self.executed_blocks = 0
        self.records = []
        self.clear()
        self.active_request = True
        try:
            yield self
        finally:
            self.clear()
            self.active_request = False

    def configure_groups(self, x, video_indices, audio_indices, video_start, video_end, valid_prefix=None):
        import torch
        if self.groups is not None:
            return
        start, end = self.runtime.local_start, self.runtime.local_end
        # Done once per enabled request, before block execution. These row indices
        # refer to local sequence owners, not the attention head/branch layout.
        kinds = torch.zeros(end - start, dtype=torch.long, device=x.device)
        if valid_prefix is not None:
            lo, hi = max(start, valid_prefix) - start, min(end, video_start) - start
            if hi > lo:
                kinds[lo:hi] = -1  # Padding must not change the RDT decision.
        for value, indices in ((1, video_indices), (3, audio_indices)):
            local = indices[(indices >= start) & (indices < end)] - start
            kinds[local] = value
        lo, hi = max(start, video_start) - start, min(end, video_end) - start
        if hi > lo:
            kinds[lo:hi] = 2
        self.groups = [(kinds == index).nonzero().flatten() for index in range(4)]

    def decision(self, current):
        import torch
        import torch.distributed as dist
        # Every eligible rank participates, even if its local cache is missing or
        # invalid. Sum NUMERATORS / DENOMINATORS, not unequal-shard averages.
        valid = (self.previous is not None and self.residual is not None
                 and self.previous.shape == current.shape and self.previous.dtype == current.dtype
                 and self.previous.device == current.device and self.residual.shape == current.shape
                 and self.residual.dtype == current.dtype and self.residual.device == current.device
                 and self.residual_finite is not None)
        stats = torch.zeros((4, 4), dtype=torch.float32, device=current.device)
        if valid:
            for i, rows in enumerate(self.groups):
                now = current.index_select(1, rows).float()
                old = self.previous.index_select(1, rows).float()
                stats[i, 0] = (now - old).abs().sum()
                stats[i, 1] = old.abs().sum()
                stats[i, 2] = now.numel()
            stats[0, 3] = (~self.residual_finite).float()
        else:
            stats[0, 3] = 1
        if self.runtime.world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        values = stats.cpu().tolist()  # One necessary shared branch decision.
        scores = {}
        for name, (num, den, count, _) in zip(("text", "reference", "video", "audio"), values):
            if count:
                scores[name] = num / max(den, count * 1e-8)
        finite = all(math.isfinite(score) for score in scores.values())
        score = max(scores.values(), default=float("inf"))
        reuse = values[0][3] == 0 and finite and score < self.config.threshold
        # JSON must never contain NaN/Infinity.
        return reuse, {name: value if math.isfinite(value) else None for name, value in scores.items()}

    def run(self, blocks, x, args, call):
        import torch
        if not self.active_request:
            raise RuntimeError("DBCache requires a request context")
        cfg = self.config
        step = self.step
        self.step += 1
        if self.step > self.total_steps:
            raise RuntimeError("Unexpected extra DiT invocation in 8-NFE cache context")

        def compute(first, last, value):
            for index in range(first, last):
                value = call(blocks[index], value, args)
                self.executed_blocks += 1
            return value

        if not cfg.enabled or cfg.threshold == 0 or cfg.max_cached_steps == 0:
            self.records.append({"step": step + 1, "reused": False, "reason": "disabled"})
            return compute(0, len(blocks), x)
        # Snapshot boundaries: both fused and eager implementations are allowed to
        # mutate their input. Never cache an alias into the live residual stream.
        original = x.clone()
        prefix = compute(0, cfg.fn_blocks, x)
        prefix_residual = prefix - original
        del original
        reason = ("warmup" if step < cfg.warmup_steps else
                  "last_steps" if step >= self.total_steps - cfg.last_steps else
                  "max_cached_steps" if self.hits >= cfg.max_cached_steps else
                  "max_consecutive" if self.consecutive >= cfg.max_consecutive else None)
        scores, reuse = {}, False
        if reason is None:
            event = self.runtime.profile_start()
            reuse, scores = self.decision(prefix_residual)
            self.runtime.profile_end("cache_decision", event)
            reason = "threshold_passed" if reuse else "threshold_or_invalid_cache"
        middle_end = len(blocks) - cfg.bn_blocks
        if reuse:
            value = prefix + self.residual
            self.hits += 1
            self.consecutive += 1
        else:
            before = prefix.clone()
            value = compute(cfg.fn_blocks, middle_end, prefix)
            self.residual = (value - before).detach()
            self.residual_finite = torch.isfinite(self.residual).all()
            self.previous = prefix_residual.detach()
            self.consecutive = 0
        self.records.append({"step": step + 1, "reused": reuse, "reason": reason,
                             "relative_l1_by_modality": scores})
        return compute(middle_end, len(blocks), value)

    def report(self):
        return {"implementation": "openvdn_dbcache_adapter_v1", "config": asdict(self.config),
                "enabled": self.config.enabled, "approximate": self.hits > 0,
                "cached_steps": [r["step"] for r in self.records if r["reused"]],
                "full_steps": self.step - self.hits, "cache_hits": self.hits,
                "executed_blocks": self.executed_blocks,
                "skipped_blocks": self.step * self.total_blocks - self.executed_blocks,
                "decisions": list(self.records), "scope": "one request; GPU residuals; all-rank decision",
                "metric": "max modality relative L1 against last fully computed prefix residual"}
