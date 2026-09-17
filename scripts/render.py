"""CLI uses the same serialization, cancellation and metrics as the ComfyUI node."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.config import Settings
from openvdn_comfy.runner import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="")
    parser.add_argument("--refs", nargs="*", default=[])
    parser.add_argument("--prompt-file", help="Existing official .pt prompt cache; skips Qwen3-VL encoding")
    parser.add_argument("--output")
    parser.add_argument("--num-frames", type=int, default=345)
    parser.add_argument("--duration", type=float, help="Requested seconds, 4–15; overrides --num-frames")
    parser.add_argument("--ratio", help="Output width:height, e.g. 9:16")
    parser.add_argument("--resolution", type=int, help="Output short edge in pixels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-short-edge", type=int, default=768)
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-kernels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--softmax-backend", choices=["flex", "decomposed", "ref"], default="flex")
    parser.add_argument("--softmax-ranks", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = vars(parser.parse_args())
    request = {key: args.pop(key) for key in ("prompt", "refs", "prompt_file", "output")}
    result = generate(**request, settings=Settings(**args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
