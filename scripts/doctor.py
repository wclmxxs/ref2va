import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.config import DEPS, MODELS, RUNTIME, UPSTREAM, Settings, atomic_json, source_lock
from openvdn_comfy.runner import gpu_lock, run_process, worker_environment


def check_models():
    stamp = MODELS / "sources.json"
    if not stamp.exists() or json.loads(stamp.read_text()) != source_lock()["models"]:
        raise RuntimeError("Pinned model download incomplete; run ./deploy.sh download")
    components = ["vdn/h3-base/transformer", "vdn/h3-base/vae", "vdn/h3-base/audio_vae",
                  "vdn/stage-dmd-step-250/linear_branch", "vdn/stage-dmd-step-250/adapters/default",
                  "vdn/stage-dmd-step-250/adapters/turbo", "conditioner/text_encoder"]
    for component in components:
        path = MODELS / component
        files = list(path.glob("*.safetensors"))
        if not files or any(p.stat().st_size < 16 for p in files):
            raise RuntimeError(f"Missing/empty weights: {path}; rerun download")
    for index in MODELS.rglob("*.safetensors.index.json"):
        for name in set(json.loads(index.read_text())["weight_map"].values()):
            if not (index.parent / name).is_file():
                raise RuntimeError(f"Missing shard {index.parent / name}; rerun download")
    if not (MODELS / "conditioner/processor/tokenizer_config.json").is_file():
        raise RuntimeError("Missing Qwen3-VL processor; rerun download")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl", action="store_true", help="Run an actual 8-rank NCCL collective probe")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the VDN environment")
    if torch.__version__ != "2.13.0+cu129":
        raise RuntimeError(f"Expected torch 2.13.0+cu129, got {torch.__version__}")
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"Expose exactly 8 H200 GPUs; found {torch.cuda.device_count()}")
    for i in range(8):
        gpu = torch.cuda.get_device_properties(i)
        if "H200" not in gpu.name or gpu.total_memory < 130 * 1024**3:
            raise RuntimeError(f"GPU {i} is not a full H200: {gpu}")
        print(f"GPU {i}: {gpu.name}, {gpu.total_memory / 1024**3:.1f} GiB")
    for package in ("torch", "transformers", "flash-attn-4", "triton", "diffusers"):
        print(f"{package}: {importlib.metadata.version(package)}")
    for name in ("openvdn", "ComfyUI"):
        head = subprocess.check_output(["git", "-C", str(DEPS / name), "rev-parse", "HEAD"], text=True).strip()
        if head != source_lock()["git"][name]["revision"]:
            raise RuntimeError(f"Source revision changed: {name}")
    check_models()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    config_path = RUNTIME / "doctor-inference.json"
    atomic_json(config_path, Settings().inference_config(UPSTREAM / "prompts/reference/example_ref2va.pt",
                                                       ROOT / "output/doctor-not-rendered.mp4"))
    subprocess.run([sys.executable, "-m", "src.inference.infer_ulysses", "--config", str(config_path),
                    "--validate-only"], cwd=UPSTREAM, env=worker_environment(), check=True, timeout=120)
    subprocess.run(["nvidia-smi", "topo", "-m"], check=True)
    if args.nccl:
        log = RUNTIME / "nccl-probe.log"
        print(f"8-rank NCCL probe: NCCL_NVLS_ENABLE={worker_environment()['NCCL_NVLS_ENABLE']}", flush=True)
        with gpu_lock(lambda: None):
            run_process([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                         str(ROOT / "scripts/nccl_probe.py")], log, timeout=180)
        print(log.read_text())
    print("READY: ./deploy.sh start — upload an image and import workflows/openvdn_ref2va_like.json")


if __name__ == "__main__":
    main()
