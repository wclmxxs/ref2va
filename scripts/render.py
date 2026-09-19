"""CLI uses the same serialization, cancellation and metrics as the ComfyUI node."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.config import Settings
from openvdn_comfy.hardware import Hardware
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
    parser.add_argument("--reference-short-edge", type=int, default=768,
                        help="Reference image short edge, 128–2048 in multiples of 32; independent of output resolution")
    parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-kernels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--softmax-backend", choices=["flex", "decomposed", "ref"], default=Hardware.from_env().softmax_backend)
    parser.add_argument("--softmax-ranks", type=int, default=Hardware.from_env().softmax_ranks)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-kernels", action="store_true", help="Requires --profile; diagnostic CPU/CUDA kernel tracing")
    parser.add_argument("--fast-communication", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-softmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dual-stream", action=argparse.BooleanOptionalAction, default=False,
                        help="Shared raw QKV with overlapped branches; requires --softmax-ranks 0")
    parser.add_argument("--linear-stats-chunk-frames", type=int, choices=[8,16,32], default=16)
    parser.add_argument("--linear-kv-keep-ratio", type=float, choices=[1.0, .5, .25], default=1.0,
                        help="Approximate video K/V statistics only; 1.0 keeps the original computation")
    parser.add_argument("--attention-kernel", choices=["native", "decomposed"], default="native")
    parser.add_argument("--isolate-padding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-policy", choices=["adaptive", "always"], default="adaptive")
    parser.add_argument("--cache-dit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-dit-threshold", type=float, default=.08)
    parser.add_argument("--cache-dit-fn-blocks", type=int, default=8)
    parser.add_argument("--cache-dit-bn-blocks", type=int, default=8)
    parser.add_argument("--cache-dit-warmup-steps", type=int, default=3)
    parser.add_argument("--cache-dit-max-consecutive", type=int, default=1)
    parser.add_argument("--cache-dit-max-cached-steps", type=int, default=2)
    parser.add_argument("--cache-dit-last-steps", type=int, default=1)
    args = vars(parser.parse_args())
    request = {key: args.pop(key) for key in ("prompt", "refs", "prompt_file", "output")}
    result = generate(**request, settings=Settings(**args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
