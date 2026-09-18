"""Resident eight-rank OpenVDN; native weights, attention, sampler and decoders."""
from dataclasses import asdict, replace
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
from openvdn_comfy.backend import BACKEND, PROFILE_FIELDS, REQUEST_DEFAULT_FIELDS, read_json
from openvdn_comfy.config import MODELS, RUNTIME, UPSTREAM, Settings, atomic_json, source_lock
from openvdn_comfy.compile_cache import CompilerMonitor, cache_settings, summarize_compilation
from openvdn_comfy.fast_output import decode_and_save, measured
from openvdn_comfy.exact_runtime import ExactRuntime, enabled
from openvdn_comfy.attention_runtime import AttentionRuntime, summarize_attention
from openvdn_comfy.communication import CommunicationRuntime
from openvdn_comfy.request_cleanup import RequestCleanup
from openvdn_comfy.streaming_output import StreamingMP4
from openvdn_comfy.optimization_options import FIELDS as OPTIMIZATION_FIELDS
from openvdn_comfy.dit_runtime import DiTRuntime, summarize_profiles, summarize_cache
from openvdn_comfy.cache_dit import FIELDS as CACHE_FIELDS
from openvdn_comfy.parallel_vae import decode_parallel
from openvdn_comfy.resident_geometry import GeometryCache
from openvdn_comfy.runner import conditioning_key
from openvdn_comfy.vae_tiles import ClipDecoder
from openvdn_comfy.warmup_history import WarmupHistory
from openvdn_comfy.conditioning import load_conditioning
from openvdn_comfy.token_buckets import BUCKET_POLICY, TokenBuckets, describe_bucket
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
                    "reference_size": short_edge,
                    "reference_metadata": [
                        {"original_size": list(original.size), "normalized_size": list(prepared_image.size),
                         "qwen_grid_thw": vision["image_grid_thw"][index].tolist()}
                        for index, (original, prepared_image) in enumerate(zip(images, prepared))]}, output)


def main():
    import torch
    import torch.distributed as dist
    from PIL import Image
    from src.config import load_config
    from src.config.inference import InferenceConfig, validate_ablation, validate_kernels, validate_parallel
    from src.inference.utils.assemble import build_inference_model, render_record
    from src.inference.utils.ulysses import init_ulysses, install_ulysses
    from src.inference.utils import ulysses
    from src.inference import render

    launch = read_json(BACKEND / "launch.json")
    instance = launch["instance"]
    parallel_vae = launch.get("parallel_vae", False)
    compile_options = launch.get("compile_cache", cache_settings())
    torch._dynamo.config.recompile_limit = max(
        compile_options["recompile_limit"], torch._dynamo.config.recompile_limit)
    torch._dynamo.config.accumulated_recompile_limit = max(
        compile_options["recompile_limit"] * 4, torch._dynamo.config.accumulated_recompile_limit)
    torch._dynamo.config.fail_on_recompile_limit_hit = True
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

    startup_report = {}

    def state(status, phase, **extra):
        if runtime.is_main:
            atomic_json(BACKEND / "state.json", {"instance": instance, "status": status, "phase": phase,
                        "profile": {**profile, **{name: getattr(settings, name) for name in REQUEST_DEFAULT_FIELDS}},
                        "request_options": {"softmax_ranks": list(range(8)), "profile": True,
                                            "optimizations": {name: getattr(settings, name) for name in OPTIMIZATION_FIELDS},
                                            "cache_dit": {name: getattr(settings, name) for name in CACHE_FIELDS}},
                        "active_softmax_ranks": runtime.softmax_ranks,
                        "world_size": 8, "video_vae_world_size": 8 if parallel_vae else 1,
                        "metrics_schema_version": 8, "compile_cache": compile_options,
                        "nccl": {"nvls_enable": os.environ.get("NCCL_NVLS_ENABLE", "NCCL default")},
                        "token_bucket_policy": BUCKET_POLICY if compile_options["token_bucket"] and settings.softmax_backend == "flex" else "native",
                        "startup_warmup": startup_report,
                        "exact_runtime_enabled": enabled("REF2VA_EXACT_RUNTIME"),
                        "sampler_geometry": "request_bound_v1",
                        "async_output_enabled": enabled("REF2VA_ASYNC_OUTPUT"),
                        "updated_at": time.time(), **extra})

    state("loading", "loading_dit_and_vaes")
    model = build_inference_model(cfg, device, load_decoders=runtime.is_main, log=runtime.is_main)
    if not model.is_hybrid:
        raise RuntimeError("Expected OpenVDN hybrid checkpoint")
    install_ulysses(model.transformer, runtime, softmax_ranks=settings.softmax_ranks)
    dit_runtime = DiTRuntime(model.transformer, runtime, ulysses.iter_hybrids(model.transformer))
    # Native decomposed/reference paths remain available, without padded tokens.
    bucket_stride = compile_options["token_bucket"] if settings.softmax_backend == "flex" else 0
    buckets = TokenBuckets(bucket_stride)
    attention_runtime = AttentionRuntime(dit_runtime.hybrids, buckets, ulysses._window_softmax_branch)
    communication = CommunicationRuntime(runtime)
    state("loading", "checking_communication_kernels")
    if settings.fast_communication:
        communication.verify(head_dim=dit_runtime.hybrids[0].head_dim)
    cleanup = RequestCleanup()
    exact_runtime = ExactRuntime(model.transformer, ulysses, render, active=enabled("REF2VA_EXACT_RUNTIME"),
                                 block_runtime=dit_runtime, token_buckets=buckets if bucket_stride else None,
                                 attention_runtime=attention_runtime)
    runtime.barrier()
    if parallel_vae:
        state("loading", "loading_parallel_video_vaes")
        if not runtime.is_main:
            from diffusers import AutoencoderKLMiniMaxH3
            from src.paths import resolve_weights
            # Match rank zero's native fp32 weights and autocast compute. Audio
            # remains on rank zero; each video decoder stays resident after load.
            model.vae = AutoencoderKLMiniMaxH3.from_pretrained(
                resolve_weights(cfg.vae_source), subfolder="vae").to(device).eval().requires_grad_(False)
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
    compiler = CompilerMonitor(flex_module, device)
    clip_decoder = ClipDecoder(model.vae) if parallel_vae else None
    history = WarmupHistory(BACKEND / "warmup-history.json", source_lock(), profile,
                            capacity=compile_options["max_shapes"]) if runtime.is_main else None

    def reset_compiler():
        torch._dynamo.reset()
        # BlockMasks are deterministic tensors, independent of Dynamo's graph
        # lifetime. Keep their existing bounded LRU across a graph rotation.
        gc.collect()
    geometries = GeometryCache(reset_compiler, max_shapes=compile_options["max_shapes"])

    def run(request, warmup=False, denoise_only=False, startup=False):
        run_status = "loading" if warmup or startup else "busy"
        current = Settings(**request["settings"]).validate()
        if any(getattr(current, name) != profile[name] for name in PROFILE_FIELDS):
            raise ValueError("Request must use the resident model profile")
        plan = current.render_plan()
        render.LATENT_H, render.LATENT_W = plan.generation_height // 16, plan.generation_width // 16
        cache = Path(request["prompt_file"])
        worker_start = time.monotonic()
        encode_start = time.monotonic()
        cache_hit = cache.is_file() and cache.stat().st_size > 0
        if request.get("require_cached_prompt") and not cache_hit:
            raise RuntimeError(f"Warmup conditioning cache disappeared: {cache}")
        if not cache_hit:
            # Qwen on rank zero temporarily allocates on all eight GPUs. Return
            # each diffusion process's retained allocator blocks before encoding
            # a new prompt/reference, then keep hot inference requests fast.
            gc.collect()
            torch.cuda.empty_cache()
        runtime.barrier()
        if runtime.is_main and not cache_hit:
            state(run_status, "encoding_references", token=request.get("token"))
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
        embeds, tags, conditions, conditioning = load_conditioning(str(cache), str(device))
        bucket = describe_bucket(embeds, conditions, plan, tuple(model.transformer.config.patch_size),
                                 render.audio_latent_num_frames(plan.sampling_frames) * render.AUDIO_CHANNELS,
                                 render.video_latent_num_frames(plan.sampling_frames, 17, 5), bucket_stride)
        buckets.prepare(bucket, device)
        attention_runtime.select(current)
        if attention_runtime.verification_first_run:
            # All ranks verify once on first opt-in, including linear ranks.
            # Keep the entire check in condition_load_seconds, outside DiT timing.
            runtime.barrier()
        if current.fast_communication and communication.parity is None:
            communication.verify(head_dim=dit_runtime.hybrids[0].head_dim)
        communication.select(current.fast_communication)
        runtime._ref2va_attention_signature = attention_runtime.signature()
        dit_runtime.select_layout(current.softmax_ranks)
        new_shape = geometries.prepare(runtime, plan, embeds, tags, conditions, bucket=bucket)
        phase = "warming_up" if warmup else "verifying_warm_cache" if startup else "denoising"
        state(run_status, phase, token=request.get("token"))
        torch.cuda.synchronize(device)
        condition_load_seconds = time.monotonic() - prepare_start
        steps = []
        compile_before = compiler.snapshot()
        denoise_start = time.monotonic()
        with dit_runtime.request(current, warmup=warmup), exact_runtime.request(verify=warmup):
            latents, audio = exact_runtime.generate(
                model.transformer, embeds, tags, plan.sampling_frames, 8, current.seed, device,
                video_shift=12., audio_shift=3., runtime=runtime, step_seconds=steps, conditions=conditions)
            actual_geometry = plan.validate_latent_shape(latents.shape)
            # Include the sampler's final unpatchify/copies in the stage wall time.
            torch.cuda.synchronize(device)
            exact_report = exact_runtime.report()
            denoise_seconds = time.monotonic() - denoise_start
            dit_report = dit_runtime.report(denoise_seconds)
        compile_records = [None] * runtime.world_size
        dist.all_gather_object(compile_records, {"rank": runtime.rank, **compiler.since(compile_before),
                                               "exact_runtime": exact_report, "dit_runtime": dit_report,
                                               "attention": attention_runtime.report()})
        exact_records = [{"rank": item["rank"], **item.pop("exact_runtime")} for item in compile_records]
        dit_records = [item.pop("dit_runtime") for item in compile_records]
        attention_report = summarize_attention([{'rank': item['rank'], **item.pop('attention')} for item in compile_records])
        parallel_profile = summarize_profiles([item["profile"] for item in dit_records])
        cache_report = summarize_cache([item["cache_dit"] for item in dit_records])
        if runtime.is_main and warmup:
            atomic_json(BACKEND / "exact-runtime-parity.json", {"instance": instance, "by_rank": exact_records})
        if any(item["parity"]["checked"] and item["parity"]["exact"] is not True for item in exact_records):
            raise RuntimeError("Exact runtime startup parity failed; restart with REF2VA_EXACT_RUNTIME=0")
        bucket_report = bucket.metadata()
        if current.isolate_padding and bucket_stride:
            bucket_report.update(policy="prefix_gap_isolated_v3", padding_attention="excluded_keys")
        compilation = {**summarize_compilation(compile_records), **geometries.last,
                       "token_bucket": bucket_report}
        sol_compile = attention_report.get('sol', {})
        compilation['sol_compile_misses'] = sol_compile.get('compile_misses_all_ranks', 0)
        compilation['sol_preprocess_signatures'] = sol_compile.get('preprocess_signatures_all_ranks', 0)
        if compilation['sol_compile_misses'] or compilation['sol_preprocess_signatures']:
            compilation.update(runtime_graph_reused=False, compiled_new_graph=True)
        geometries.commit()
        compilation.update(geometries.last)
        profiles = [item["profile"]["ms_per_nfe"] for item in dit_records]
        decode_start = time.monotonic()
        decoded_video = None
        video_decode_seconds = None
        video_decode_details = {"world_size": 1, "native_temporal_assembly": True}
        output_timings, encoding = {}, {}
        stream_writer = (StreamingMP4(plan, request["output"], model.audio_vae.config.sampling_rate,
                                      render.PIXEL_MEAN, render.PIXEL_STD, verify=warmup)
                         if current.streaming_output and parallel_vae and runtime.is_main and not denoise_only else None)
        try:
            if parallel_vae and not denoise_only:
                state(run_status, "decoding_video_vae", token=request.get("token"))
                decode_timings = {}
                clip_decoder.tile_count = 0
                with measured(decode_timings, "video_vae_decode_seconds", device):
                    mean = torch.tensor(model.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
                    std = torch.tensor(model.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        decoded_video, video_decode_details = decode_parallel(
                            model.vae, latents * std + mean, rank=runtime.rank, world_size=runtime.world_size,
                            verify=warmup, clip_decode=clip_decoder, streaming=current.streaming_output,
                            on_chunk=stream_writer.submit if stream_writer else None)
                tile_counts = [None] * runtime.world_size
                dist.all_gather_object(tile_counts, clip_decoder.tile_count)
                video_decode_details.update(spatial_tiles_by_rank=tile_counts,
                                            stitch="preallocated_native_blending",
                                            wall_time_scope="decode, transport, assembly, and pixel-queue backpressure"
                                            if current.streaming_output else "decode, transport, assembly")
                video_decode_seconds = decode_timings["video_vae_decode_seconds"]
                if warmup and runtime.is_main:
                    atomic_json(BACKEND / "vae-parity.json", {"instance": instance, **video_decode_details})
                    print(f"Parallel video VAE: startup parity passed for {video_decode_details['temporal_clips']} clips", flush=True)
            if runtime.is_main and not denoise_only:
                state(run_status, "decoding", token=request.get("token"))
                output_timings, encoding = decode_and_save(
                    latents, audio, model.vae, model.audio_vae, request["output"], device, plan,
                    render.PIXEL_MEAN, render.PIXEL_STD,
                    phase=lambda phase: state(run_status, phase, token=request.get("token")),
                    decoded_video=decoded_video, video_decode_seconds=video_decode_seconds, verify_output=warmup,
                    stream_writer=stream_writer)
            runtime.barrier()
        except BaseException as error:
            if stream_writer is not None:
                stream_writer.abort(error)
            raise
        timings = {"denoise_seconds": denoise_seconds, "seconds_per_step": denoise_seconds / 8,
                   "step_seconds": steps, "model_setup_seconds": 0,
                   "step_timing_method": exact_report["step_timing_method"],
                   "dynamo_compile_seconds": compilation["dynamo_compile_seconds"],
                   "mask_build_seconds": compilation["mask_build_seconds"],
                   "sol_compile_seconds": sol_compile.get('compile_seconds_max_rank', 0.),
                   "sol_preprocess_cold_seconds": sol_compile.get('preprocess_cold_seconds_max_rank', 0.),
                   "denoise_wall_seconds": denoise_seconds,
                   "hot_denoise_seconds": denoise_seconds if compilation["runtime_graph_reused"] else None,
                   "decode_and_encode_seconds": time.monotonic() - decode_start,
                   "parallel_profile_ms_per_nfe_by_rank": profiles}
        cleanup_start = time.monotonic()
        del latents, audio, embeds, tags, conditions, decoded_video
        cleanup_report = cleanup.run(torch, device, current.cleanup_policy,
                                     compiled=compilation["compiled_new_graph"], warmup=warmup)
        runtime.barrier()
        cleanup_seconds = time.monotonic() - cleanup_start
        if runtime.is_main:
            timings.update(output_timings)
            if video_decode_details.get('streaming'):
                timings['video_vae_compute_max_rank_seconds'] = max(r['compute_seconds'] for r in video_decode_details['by_rank'])
            timings.update(conditioning_seconds=encode_seconds, condition_load_seconds=condition_load_seconds,
                           cleanup_seconds=cleanup_seconds, worker_wall_seconds=time.monotonic() - worker_start)
            from src.inference.utils.assemble import flex_latch_state
            actual_config = current.inference_config(cache, request["output"])
            actual_config["render"]["warmup_steps"] = 8 if warmup else 0
            record = {**base_record, "overlay": actual_config,
                      "parallel": {"kind": "ulysses_branch_parallel" if runtime.branch_parallel else "ulysses",
                                   "world_size": 8, "softmax_ranks": runtime.softmax_ranks,
                                   "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE", "NCCL default"),
                                   "warmup_steps": 8 if warmup else 0}, "timings": timings,
                      "resident": True, "output_encoding": encoding,
                      "optimizations": {"requested": {name: getattr(current, name) for name in OPTIMIZATION_FIELDS},
                                        "communication": {"enabled": communication.active, "parity": communication.parity,
                                                          "pack_launches_per_layer": 2 if communication.active and runtime.branch_parallel else None},
                                        "attention": attention_report, "cleanup": cleanup_report}, "video_vae_decode": video_decode_details,
                      "compilation": compilation,
                      "conditioning": conditioning,
                      "exact_runtime": {"enabled": exact_runtime.active, "by_rank": exact_records},
                      "parallel_profile": parallel_profile, "cache_dit": cache_report,
                      "new_geometry": new_shape, "flex_backend": flex_latch_state(),
                      "render_plan": plan.metadata(), "actual_geometry": actual_geometry}
            if not denoise_only:
                atomic_json(request["output"] + ".inference.json", record)
            if not warmup or denoise_only:
                history.remember(compilation["geometry_id"], request)
            return {"conditioning_cache_hit": cache_hit, "encode_seconds": encode_seconds,
                    "inference_process_seconds": denoise_seconds + timings["decode_and_encode_seconds"],
                    "upstream": record}

    # Warm up all eight diffusion steps, conditioner, video/audio VAE and MP4 encoding.
    warm_ref = BACKEND / "warmup-reference.png"
    if runtime.is_main:
        Image.new("RGB", (576, 768), (100, 120, 140)).save(warm_ref)
    runtime.barrier()
    warm_prompt = "The person in <Picture 1> walks through a hallway. Natural ambient sound."
    warm_key = conditioning_key(warm_prompt, [warm_ref], settings.reference_short_edge)
    warm_request = {"settings": asdict(settings), "prompt_file": str(BACKEND / f"warmup-{warm_key}.pt"),
                    "prompt": warm_prompt,
                    "references": [str(warm_ref)], "output": str(BACKEND / "warmup.mp4")}
    run(warm_request, warmup=True)
    replay_box = [history.requests(compile_options["warmup_recent"], RUNTIME / "jobs") if runtime.is_main else None]
    dist.broadcast_object_list(replay_box, src=0)
    # Additional durations/history are opt-in. Defaults run only the base
    # warmup; new buckets compile lazily on their first actual request.
    common = [{**warm_request, "settings": asdict(replace(settings, duration=duration))}
              for duration in compile_options["warmup_durations"] if duration != settings.duration]
    replays = common + replay_box[0]
    replay_metrics = []
    for index, replay in enumerate(replays):
        replay["output"] = str(BACKEND / f"warmup-history-{index}.mp4")
        state("loading", "warming_up_recent_shapes", warmup_index=index + 1, warmup_count=len(replays))
        result = run(replay, warmup=True, denoise_only=True)
        if runtime.is_main:
            replay_metrics.append({"prompt_file": replay["prompt_file"],
                                   "render_plan": result["upstream"]["render_plan"],
                                   "timings": result["upstream"]["timings"],
                                   "exact_runtime": result["upstream"]["exact_runtime"],
                                   "compilation": result["upstream"]["compilation"]})
            print(f"Startup shape warmup {index + 1}/{len(replays)} complete", flush=True)
    # Optional diagnostic sweep; never delay default startup to prove all shapes
    # are hot. Runtime compiler counters still report misses on actual requests.
    verification = []
    verify_requests = [warm_request, *replays] if compile_options["warmup_verify"] else []
    for index, replay in enumerate(verify_requests):
        state("loading", "verifying_warm_cache", warmup_index=index + 1, warmup_count=1 + len(replays))
        checked = {**replay, "settings": {**replay["settings"], "cache_dit": False, "profile": False}}
        result = run(checked, warmup=False, denoise_only=True, startup=True)
        if runtime.is_main:
            compilation = result["upstream"]["compilation"]
            verification.append({"prompt_file": replay["prompt_file"],
                                 "render_plan": result["upstream"]["render_plan"],
                                 "compilation": compilation, "timings": result["upstream"]["timings"]})
    if runtime.is_main:
        hits = sum(item["compilation"]["runtime_graph_reused"] for item in verification)
        startup_report.update(base_warmup_complete=True, extra_warmup_requests=len(replays),
                              verification_requested=compile_options["warmup_verify"],
                              verified_requests=len(verification), hot_requests=hits,
                              all_runtime_graphs_reused=(hits == len(verification)) if verification else None,
                              complete=not verification or hits == len(verification))
        atomic_json(BACKEND / "warmup-report.json", {"instance": instance, "requests": replay_metrics,
                                                    "verification": verification, "summary": startup_report})
        if hits != len(verification):
            raise RuntimeError("Startup cache verification compiled new graphs; inspect warmup-report.json")
    runtime.barrier()
    state("ready", "idle", startup_seconds=time.monotonic() - setup_start,
          warmup_plan=settings.render_plan().metadata(), replayed_shapes=len(replay_box[0]))
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
