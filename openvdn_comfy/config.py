from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from .render_plan import make_plan
from .hardware import Hardware, runtime_directory

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
UPSTREAM = DEPS / "openvdn"
MODELS = Path(os.environ.get("REF2VA_MODELS", ROOT / "models")).resolve()
RUNTIME = runtime_directory(ROOT)
WORKER_PYTHON = ROOT / ".venv-vdn/bin/python"
CONFIG = "configs/inference/8nfe_tuned_fp8_ulysses_h200.yaml"


def source_lock():
    return json.loads((ROOT / "sources.lock.json").read_text())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass(frozen=True)
class Settings:
    num_frames: int = 345
    seed: int = 42
    reference_short_edge: int = 768
    fp8: bool = True
    inference_kernels: bool = True
    softmax_backend: str = field(default_factory=lambda: Hardware.from_env().softmax_backend)
    softmax_ranks: int = field(default_factory=lambda: Hardware.from_env().softmax_ranks)
    warmup_steps: int = 2
    profile: bool = False
    profile_kernels: bool = False
    duration: float | None = None
    ratio: str | None = None
    resolution: int | None = None
    fast_communication: bool = True
    fused_delta: bool = True
    boundary_scan: bool = True
    fast_softmax: bool = True
    dual_stream: bool | None = None
    linear_stats_chunk_frames: int = 16
    linear_kv_keep_ratio: float = 1.0
    attention_kernel: str = "native"
    isolate_padding: bool = False
    streaming_output: bool = True
    vae_tile_batch_size: int = 4
    vae_compile: bool = True
    cleanup_policy: str = "adaptive"
    cache_dit: bool = True
    cache_dit_threshold: float = 0.25
    cache_dit_fn_blocks: int = 8
    cache_dit_bn_blocks: int = 8
    cache_dit_warmup_steps: int = 3
    cache_dit_max_consecutive: int = 1
    cache_dit_max_cached_steps: int = 2
    cache_dit_last_steps: int = 1

    def __post_init__(self):
        # An explicit branch layout or unfused model must remain usable without
        # specifying the new dual-stream option. Explicit conflicts still fail.
        if self.dual_stream is None:
            object.__setattr__(self, 'dual_stream', self.softmax_ranks == 0 and self.inference_kernels is True)

    def validate(self):
        for name, low, high in (("num_frames", 107, 345), ("seed", 0, 2**63 - 1),
                                ("reference_short_edge", 128, 2048), ("softmax_ranks", 0, Hardware.from_env().world_size - 1),
                                ("warmup_steps", 0, 8)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if (self.num_frames - 5) % 17:
            raise ValueError("num_frames must be 17n+5, e.g. 124 or 345")
        if self.reference_short_edge % 32:
            raise ValueError("reference_short_edge must be a multiple of 32")
        for name in ("fp8", "inference_kernels", "profile", "profile_kernels"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.profile_kernels and not self.profile:
            raise ValueError("profile_kernels requires profile=true")
        if self.softmax_backend not in ("flex", "decomposed", "ref"):
            raise ValueError("Unsupported softmax_backend")
        from .optimization_options import validate
        validate(self)
        from .cache_dit import CacheConfig
        CacheConfig.from_settings(self).validate()
        self.render_plan()
        return self

    def render_plan(self):
        return make_plan(self.num_frames, self.duration, self.ratio, self.resolution)

    def inference_config(self, prompt_file, output):
        self.validate()
        if any("${" in str(path) for path in (MODELS, prompt_file, output)):
            raise ValueError("Paths containing ${ are not supported by the upstream OmegaConf loader")
        return {
            "checkpoint": str(MODELS / "vdn/stage-dmd-step-250"),
            "base_source": str(MODELS / "vdn/h3-base"),
            "vae_source": str(MODELS / "vdn/h3-base"),
            "render": {"prompt_file": str(prompt_file), "out": str(output),
                       "num_frames": self.render_plan().sampling_frames, "num_steps": 8,
                       "warmup_steps": self.warmup_steps, "seed": self.seed,
                       "video_shift": 12.0, "audio_shift": 3.0, "record": True},
            "kernels": {"inference_kernels": self.inference_kernels, "softmax_backend": self.softmax_backend},
            "precision": {"dtype": "bfloat16", "fp8": {"enabled": self.fp8, "skip_end_blocks": 0}},
            "parallel": {"softmax_ranks": self.softmax_ranks, "profile": self.profile},
        }


def merge_request_options(defaults, overrides):
    """Apply request controls without inheriting an incompatible dual-stream default."""
    options = {**defaults, **overrides}
    if 'dual_stream' not in overrides and (options.get('softmax_ranks', 0) != 0 or
                                           options.get('inference_kernels') is False):
        options['dual_stream'] = False
    return options
