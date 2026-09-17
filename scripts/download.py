from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.config import MODELS, atomic_json, source_lock


def main():
    from huggingface_hub import snapshot_download
    sources = source_lock()["models"]
    for name, source in sources.items():
        dest = MODELS / name
        print(f"Downloading {source['repo']} @ {source['revision']} -> {dest}", flush=True)
        snapshot_download(source["repo"], revision=source["revision"],
                          allow_patterns=source["patterns"], local_dir=str(dest), max_workers=2)
    atomic_json(MODELS / "sources.json", sources)
    print("Official 8-NFE weights and Qwen3-VL conditioner ready")


if __name__ == "__main__":
    main()
