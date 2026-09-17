from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
UPSTREAM = DEPS / "openvdn"
MODELS = Path(os.environ.get("REF2VA_MODELS", ROOT / "models")).resolve()
RUNTIME = ROOT / ".runtime"
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
    softmax_backend: str = "flex"
    softmax_ranks: int = 6
    warmup_steps: int = 2
    profile: bool = False

    def validate(self):
        for name, low, high in (("num_frames", 107, 345), ("seed", 0, 2**63 - 1),
                                ("reference_short_edge", 128, 2048), ("softmax_ranks", 0, 7),
                                ("warmup_steps", 0, 8)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if (self.num_frames - 5) % 17:
            raise ValueError("num_frames must be 17n+5, e.g. 124 or 345")
        if self.reference_short_edge % 32:
            raise ValueError("reference_short_edge must be a multiple of 32")
        for name in ("fp8", "inference_kernels", "profile"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.softmax_backend not in ("flex", "decomposed", "ref"):
            raise ValueError("Unsupported softmax_backend")
        return self

    def inference_config(self, prompt_file, output):
        self.validate()
        if any("${" in str(path) for path in (MODELS, prompt_file, output)):
            raise ValueError("Paths containing ${ are not supported by the upstream OmegaConf loader")
        return {
            "checkpoint": str(MODELS / "vdn/stage-dmd-step-250"),
            "base_source": str(MODELS / "vdn/h3-base"),
            "vae_source": str(MODELS / "vdn/h3-base"),
            "render": {"prompt_file": str(prompt_file), "out": str(output),
                       "num_frames": self.num_frames, "num_steps": 8,
                       "warmup_steps": self.warmup_steps, "seed": self.seed,
                       "video_shift": 12.0, "audio_shift": 3.0, "record": True},
            "kernels": {"inference_kernels": self.inference_kernels, "softmax_backend": self.softmax_backend},
            "precision": {"dtype": "bfloat16", "fp8": {"enabled": self.fp8, "skip_end_blocks": 0}},
            "parallel": {"softmax_ranks": self.softmax_ranks, "profile": self.profile},
        }


def inference_command(config):
    # A JSON object is valid YAML; paths in the config avoid dotlist quoting
    # errors for spaces, brackets and commas.
    return [str(WORKER_PYTHON), "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc_per_node=8", "src/inference/infer_ulysses.py",
            "--config", str(config)]


def encode_command(prompt, refs, dest, short_edge):
    args = [str(WORKER_PYTHON), "-m", "src.inference.encode_keyframes" if refs else "src.inference.encode_prompt",
            "--prompt", prompt, "--out", str(dest), "--model_root", str(MODELS / "conditioner"),
            "--device", "cuda:0"]
    if refs:
        args += ["--vae_root", str(MODELS / "vdn/h3-base"), "--ref_size", str(short_edge),
                 "--refs", *map(str, refs)]
    return args
