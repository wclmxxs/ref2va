"""Resident eight-rank OpenVDN; native weights, attention, sampler and decoders."""
from dataclasses import asdict
from datetime import timedelta
import json
import gc
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.backend import BACKEND, PROFILE_FIELDS, read_json
from openvdn_comfy.config import MODELS, UPSTREAM, Settings, atomic_json
from openvdn_comfy.fast_output import decode_and_save
from openvdn_comfy.resident_geometry import GeometryCache
sys.path.insert(0, str(UPSTREAM))


class Conditioner:
    def __init__(self, vae, device):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        self.device, self.vae = device, vae
        self.processor = Qwen3VLProcessor.from_pretrained(str(MODELS / "conditioner"), subfolder="processor")
        # Contiguous layer placement through Accelerate; all weights stay on GPUs.
        # Limit each card's conditioner share so DiT/activations retain headroom.
        self.encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(MODELS / "conditioner"), subfolder="text_encoder", dtype=torch.bfloat16,
            device_map="balanced", max_memory={i: "12GiB" for i in range(8)})
        if any(str(value) in ("cpu", "disk") for value in self.encoder.hf_device_map.values()):
            raise RuntimeError("Conditioner did not fit the GPU budget; CPU/disk offload is disabled")
        self.encoder.eval().requires_grad_(False)
        self.input_device = self.encoder.get_input_embeddings().weight.device
        print(f"Resident Qwen3-VL placement: {self.encoder.hf_device_map}", flush=True)

    def encode(self, prompt, refs, output, short_edge):
        import numpy as np
        import torch
        from PIL import Image
        from src.inference.encode_keyframes import (normalize_references, build_presentation,
            qwen3vl_prompt_embeds, encode_vae_condition, PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED)
        from src.inference.encode_prompt import encode
        if not refs:
            encode(self.processor, self.encoder, prompt, str(output), self.input_device)
            return
        images = []
        for path in refs:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        prepared = normalize_references(images, short_edge)
        ids, tags, vision = build_presentation(self.processor, prompt, prepared)
        embeds = qwen3vl_prompt_embeds(self.encoder, self.processor, ids, vision, self.input_device)
        with torch.no_grad():
            conditions = [encode_vae_condition(
                self.vae, torch.from_numpy(np.array(image)).to(self.device).permute(2, 0, 1)[None, :, None],
                PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED).cpu() for image in prepared]
        torch.save({"prompt": prompt, "prompt_embeds": embeds, "text_token_tags": torch.tensor(tags),
                    "keyframe_anchors": ["ref"] * len(refs), "keyframe_files": refs,
                    "condition_latents": conditions, "height": 768, "width": 1344,
                    "reference_size": short_edge}, output)


def main():
    import torch
    import torch.distributed as dist
    from PIL import Image
    from src.config import load_config
    from src.config.inference import InferenceConfig, validate_ablation, validate_kernels, validate_parallel
    from src.inference.utils.assemble import build_inference_model, render_record
    from src.inference.utils.ulysses import init_ulysses, install_ulysses
    from src.inference import render

    launch = read_json(BACKEND / "launch.json")
    instance = launch["instance"]
    settings = Settings(**launch["settings"]).validate()
    cfg = load_config(InferenceConfig, ["--config", str(BACKEND / "inference.json")],
                      extra_validators=[validate_ablation, validate_kernels, validate_parallel])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    # Idle workers poll local files; this timeout covers startup/long compilation only.
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    runtime = init_ulysses(profile_enabled=settings.profile)
    if runtime.world_size != 8:
        raise RuntimeError("Resident worker requires exactly eight ranks")
    torch.set_grad_enabled(False)
    profile = {name: getattr(settings, name) for name in PROFILE_FIELDS}
    setup_start = time.monotonic()

    def state(status, phase, **extra):
        if runtime.is_main:
            atomic_json(BACKEND / "state.json", {"instance": instance, "status": status, "phase": phase,
                        "profile": profile, "world_size": 8, "metrics_schema_version": 2, "updated_at": time.time(), **extra})

    state("loading", "loading_dit_and_vaes")
    model = build_inference_model(cfg, device, load_decoders=runtime.is_main, log=runtime.is_main)
    if not model.is_hybrid:
        raise RuntimeError("Expected OpenVDN hybrid checkpoint")
    install_ulysses(model.transformer, runtime, softmax_ranks=settings.softmax_ranks)
    runtime.barrier()
    # Release temporary loading buffers before the rank-0 conditioner spans all GPUs.
    torch.cuda.empty_cache()
    runtime.barrier()
    state("loading", "loading_conditioner")
    conditioner = Conditioner(model.vae, device) if runtime.is_main else None
    runtime.barrier()
    # Cache checkpoint metadata once; hashing the checkpoint per job is unnecessary.
    base_record = render_record(cfg, model) if runtime.is_main else None
    from src.models.softmax_attention import flex_attention as flex_module

    def reset_compiler():
        torch._dynamo.reset()
        flex_module._MASK_CACHE.clear()
        gc.collect()
    geometries = GeometryCache(reset_compiler)

    def run(request, warmup=False):
        current = Settings(**request["settings"]).validate()
        if any(getattr(current, name) != profile[name] for name in PROFILE_FIELDS):
            raise ValueError("Request must use the resident model profile")
        plan = current.render_plan()
        render.LATENT_H, render.LATENT_W = plan.generation_height // 16, plan.generation_width // 16
        cache = Path(request["prompt_file"])
        worker_start = time.monotonic()
        encode_start = time.monotonic()
        cache_hit = cache.is_file() and cache.stat().st_size > 0
        if runtime.is_main and not cache_hit:
            state("loading" if warmup else "busy", "encoding_references", token=request.get("token"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".partial.pt")
            try:
                conditioner.encode(request["prompt"], request["references"], temporary, current.reference_short_edge)
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
            # Other ranks use separate CUDA allocators. Return Qwen's temporary
            # allocations to CUDA so the diffusion ranks can use that memory.
            torch.cuda.empty_cache()
        runtime.barrier()
        encode_seconds = time.monotonic() - encode_start
        prepare_start = time.monotonic()
        embeds, tags, conditions = render.load_prompt(str(cache), str(device))
        new_shape = geometries.prepare(runtime, plan, embeds, tags, conditions)
        state("loading" if warmup else "busy", "warming_up" if warmup else "denoising", token=request.get("token"))
        torch.cuda.synchronize(device)
        condition_load_seconds = time.monotonic() - prepare_start
        runtime.reset_profile()
        steps = []
        denoise_start = time.monotonic()
        latents, audio = render.generate_latents(
            model.transformer, embeds, tags, plan.sampling_frames, 8, current.seed, device,
            video_shift=12., audio_shift=3., runtime=runtime, step_seconds=steps, conditions=conditions)
        denoise_seconds = time.monotonic() - denoise_start
        profiles = [None] * 8
        dist.all_gather_object(profiles, {name: ms / 8 for name, ms in runtime.profile_milliseconds().items()})
        decode_start = time.monotonic()
        if runtime.is_main:
            state("loading" if warmup else "busy", "decoding", token=request.get("token"))
            output_timings, encoding = decode_and_save(
                latents, audio, model.vae, model.audio_vae, request["output"], device, plan,
                render.PIXEL_MEAN, render.PIXEL_STD,
                phase=lambda phase: state("loading" if warmup else "busy", phase, token=request.get("token")))
        runtime.barrier()
        timings = {"denoise_seconds": denoise_seconds, "seconds_per_step": denoise_seconds / 8,
                   "step_seconds": steps, "model_setup_seconds": 0,
                   "decode_and_encode_seconds": time.monotonic() - decode_start,
                   "parallel_profile_ms_per_nfe_by_rank": profiles}
        cleanup_start = time.monotonic()
        del latents, audio, embeds, tags, conditions
        gc.collect()
        torch.cuda.empty_cache()
        runtime.barrier()
        cleanup_seconds = time.monotonic() - cleanup_start
        if runtime.is_main:
            timings.update(output_timings)
            timings.update(conditioning_seconds=encode_seconds, condition_load_seconds=condition_load_seconds,
                           cleanup_seconds=cleanup_seconds, worker_wall_seconds=time.monotonic() - worker_start)
            from src.inference.utils.assemble import flex_latch_state
            actual_config = current.inference_config(cache, request["output"])
            actual_config["render"]["warmup_steps"] = 8 if warmup else 0
            record = {**base_record, "overlay": actual_config,
                      "parallel": {"kind": "ulysses_branch_parallel" if runtime.branch_parallel else "ulysses",
                                   "world_size": 8, "softmax_ranks": runtime.softmax_ranks,
                                   "warmup_steps": 8 if warmup else 0}, "timings": timings,
                      "resident": True, "output_encoding": encoding, "new_geometry": new_shape, "flex_backend": flex_latch_state(),
                      "render_plan": plan.metadata()}
            atomic_json(request["output"] + ".inference.json", record)
            return {"conditioning_cache_hit": cache_hit, "encode_seconds": encode_seconds,
                    "inference_process_seconds": denoise_seconds + timings["decode_and_encode_seconds"],
                    "upstream": record}

    # Warm up all eight diffusion steps, conditioner, video/audio VAE and MP4 encoding.
    warm_ref = BACKEND / "warmup-reference.png"
    if runtime.is_main:
        Image.new("RGB", (576, 768), (100, 120, 140)).save(warm_ref)
    runtime.barrier()
    warm_request = {"settings": asdict(settings), "prompt_file": str(BACKEND / f"warmup-{instance}.pt"),
                    "prompt": "The person in <Picture 1> walks through a hallway. Natural ambient sound.",
                    "references": [str(warm_ref)], "output": str(BACKEND / "warmup.mp4")}
    run(warm_request, warmup=True)
    state("ready", "idle", startup_seconds=time.monotonic() - setup_start, warmup_plan=settings.render_plan().metadata(), metrics_schema_version=2)
    print(f"Rank {runtime.rank}: resident models loaded and warmed up", flush=True)
    last_token = None
    while True:
        request = read_json(BACKEND / "command.json", {})
        if request.get("instance") != instance or request.get("token") == last_token:
            time.sleep(.1)
            continue
        last_token = request["token"]
        try:
            metrics = run(request)
            if runtime.is_main:
                state("ready", "idle")
                atomic_json(BACKEND / "results" / f"{last_token}.json", {"ok": True, "metrics": metrics})
        except BaseException as error:
            # A rank error invalidates the whole NCCL group. torchrun/supervisor reaps it.
            if runtime.is_main:
                state("failed", "failed", error=str(error))
                atomic_json(BACKEND / "results" / f"{last_token}.json", {"ok": False, "error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        launch = read_json(BACKEND / "launch.json", {})
        rank = os.environ.get("RANK", "unknown")
        atomic_json(BACKEND / "errors" / f"{launch.get('instance')}-{rank}.json",
                    {"traceback": traceback.format_exc(), "time": time.time()})
        raise
