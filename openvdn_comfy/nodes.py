import hashlib
import io
import json
import uuid
import time
import os
from concurrent.futures import ThreadPoolExecutor
import psutil

_OUTPUT_FINALIZERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="api-output")
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image

from .config import RUNTIME, Settings, atomic_json
from .hardware import Hardware
from .runner import generate
from .references import download_references, parse_urls
from .jobs import update_job, read_job


class OpenVDNReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {"previous": ("OPENVDN_REFS",)}}

    RETURN_TYPES = ("OPENVDN_REFS",)
    FUNCTION = "append"
    CATEGORY = "OpenVDN H200"
    DESCRIPTION = "Append one reference image. Chain nodes to assign <Picture 1>, <Picture 2>, etc."

    def append(self, image, previous=None):
        if image.ndim != 4 or len(image) != 1 or image.shape[-1] < 3:
            raise ValueError("Connect one RGB image per reference node")
        height, width = image.shape[1:3]
        if min(height, width) < 1 or max(height, width) > 8192 or not .25 <= width / height <= 4:
            raise ValueError("Reference aspect ratio must be 1:4–4:1, max side 8192")
        refs = list(previous or [])
        if len(refs) >= 9:
            raise ValueError("At most 9 reference images in this wrapper")
        pixels = (image[0, :, :, :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        content = stream.getvalue()
        path = RUNTIME / "references" / f"{hashlib.sha256(content).hexdigest()}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        return ([*refs, str(path)],)


class OpenVDNH200Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "task": (["ref2va_like", "t2va"],),
            "prompt": ("STRING", {"multiline": True, "default": "The person in <Picture 1> walks toward the camera and waves. Natural ambient sound."}),
            "num_frames": ("INT", {"default": 345, "min": 107, "max": 345, "step": 17}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1, "control_after_generate": True}),
            "reference_short_edge": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 32,
                "tooltip": "Reference image short edge: preserves aspect, rounds dimensions to 32 pixels. Independent of video resolution."}),
            "fp8": ("BOOLEAN", {"default": True}),
            "inference_kernels": ("BOOLEAN", {"default": True}),
            "softmax_backend": (list(dict.fromkeys([Hardware.from_env().softmax_backend, "flex", "decomposed", "ref"])),),
            "softmax_ranks": ("INT", {"default": Hardware.from_env().softmax_ranks, "min": 0, "max": Hardware.from_env().world_size - 1}),
            "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 8}),
            "profile": ("BOOLEAN", {"default": False}),
        }, "optional": {"references": ("OPENVDN_REFS",), **cache_inputs(), **optimization_inputs()}}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "metrics_json")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "OpenVDN H200"
    DESCRIPTION = ("Official 8-NFE Ulysses, 1344×768 at 24 fps; uses this API instance's GPU group. Ref2VA-like uses FL2VA weights. "
                   "inference_kernels controls the official fused/compiled kernel bundle; it is not a whole-DiT compile switch. "
                   "Optional approximate DBCache; disabled by default. Models are resident; kernel/precision settings must match /openvdn/health. "
                   "warmup_steps is a legacy field; startup performs 8 NFE once, requests perform no extra warmup.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def generate(self, task, prompt, references=None, **kwargs):
        import folder_paths
        import comfy.model_management as mm
        from comfy_api.input_impl import VideoFromFile

        if task not in ("ref2va_like", "t2va"):
            raise ValueError(f"Unsupported task: {task}")
        if task == "ref2va_like" and not references:
            raise ValueError("Ref2VA-like requires an OpenVDN Reference input")
        if task == "t2va" and references:
            raise ValueError("Disconnect references for T2VA")
        name = f"vdn8_{uuid.uuid4().hex}.mp4"
        output = Path(folder_paths.get_output_directory()) / "openvdn" / name
        result = generate(prompt=prompt, refs=references or [], settings=Settings(**kwargs), output=output,
                          interrupt=mm.throw_exception_if_processing_interrupted)
        return {"ui": {"openvdn_videos": [{"filename": name, "subfolder": "openvdn", "type": "output"}]},
                "result": (VideoFromFile(str(output)), json.dumps(result, ensure_ascii=False, indent=2))}


class OpenVDNH200Request(OpenVDNH200Generate):
    @classmethod
    def INPUT_TYPES(cls):
        legacy = OpenVDNH200Generate.INPUT_TYPES()["required"]
        return {"required": {
            "prompt": legacy["prompt"],
            "duration": ("FLOAT", {"default": 5, "min": 4, "max": 15, "step": .1}),
            "ratio": ("STRING", {"default": "16:9"}),
            "resolution": ("INT", {"default": 720, "min": 256, "max": 1080, "step": 2}),
            "reference_image_urls": ("STRING", {"multiline": True, "default": "", "tooltip": "One HTTP(S) image URL per line, or a JSON array; order is <Picture 1>, <Picture 2>, ..."}),
            **{name: legacy[name] for name in ("seed", "reference_short_edge", "fp8", "inference_kernels", "softmax_backend", "softmax_ranks", "warmup_steps", "profile")},
        }, "optional": {**cache_inputs(), **optimization_inputs()}}

    DESCRIPTION = "Reference-image URLs to video. Duration is in seconds, ratio is width:height, resolution is the output short edge. Uses this instance's GPU group and official Ref2VA-like weights."

    async def generate(self, prompt, duration, ratio, resolution, reference_image_urls, **kwargs):
        return await self._execute(prompt, duration, ratio, resolution, reference_image_urls, **kwargs)

    async def _execute(self, prompt, duration, ratio, resolution, reference_image_urls,
                       *, prepared_references=None, defer_result=False, **kwargs):
        import folder_paths
        import comfy.model_management as mm
        from comfy_api.input_impl import VideoFromFile
        from comfy_execution.utils import get_executing_context
        context = get_executing_context()
        job_id = context.prompt_id if context else None
        started, started_wall = time.monotonic(), time.time()
        job_record = read_job(job_id) if job_id else None
        queue_seconds = max(0., started_wall - job_record.get("queued_at", job_record["created_at"])) if job_record else 0.
        prepared_seconds = job_record.get("reference_prepare_seconds", 0.) if job_record else 0.
        try:
            settings = Settings(duration=duration, ratio=ratio, resolution=resolution, **kwargs).validate()
            update_job(job_id, status="running", phase="downloading_references")
            if prepared_references is None:
                urls = parse_urls(reference_image_urls)
                refs = await download_references(urls, mm.throw_exception_if_processing_interrupted)
            else:
                refs = prepared_references
            download_seconds = prepared_seconds + time.monotonic() - started
            name = f"vdn8_{uuid.uuid4().hex}.mp4"
            output = Path(folder_paths.get_output_directory()) / "openvdn" / name
            import asyncio
            result = await asyncio.to_thread(generate, prompt=prompt, refs=refs, settings=settings, output=output,
                              interrupt=mm.throw_exception_if_processing_interrupted,
                              progress=lambda phase: update_job(job_id, status="running", phase=phase), defer_output=defer_result)
            def finish(result):
                result["timings"].update(reference_download_seconds=download_seconds, api_queue_seconds=queue_seconds,
                                         processing_wall_seconds=prepared_seconds + time.monotonic() - started,
                                         api_wall_seconds=queue_seconds + prepared_seconds + time.monotonic() - started)
                if prepared_references is not None:
                    result["timings"]["input_prepare_seconds"] = prepared_seconds
                # Preserve the worker result's schema; adding API timings must not
                # downgrade schema 6 compilation metrics to the old schema 5 label.
                atomic_json(str(output) + ".metrics.json", result)
                atomic_json(Path(result["log_directory"]) / "result.json", result)
                video = {"filename": name, "subfolder": "openvdn", "type": "output"}
                update_job(job_id, status="succeeded", phase="complete", video_url="/view?" + urlencode(video), metrics=result)
                return result, video
            if defer_result:
                update_job(job_id, status="running", phase="encoding_output",
                           output_owner={"pid": os.getpid(), "created": psutil.Process().create_time()})
                def complete():
                    try:
                        finish(result.finish())
                    except BaseException as error:
                        update_job(job_id, status="failed", phase="failed", error=str(error))
                _OUTPUT_FINALIZERS.submit(complete)
                return {"result": (job_id,)}
            result, video = finish(result)
            return {"ui": {"openvdn_videos": [video]},
                    "result": (VideoFromFile(str(output)), json.dumps(result, ensure_ascii=False, indent=2))}
        except BaseException as error:
            update_job(job_id, status="failed", error=str(error))
            raise


class OpenVDNH200BusinessRequest(OpenVDNH200Request):
    """Private prepared-input bridge; HTTP never accepts local reference paths."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_id": ("STRING",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("job_id",)

    DESCRIPTION = "Internal gateway task. Submit reference images through the business API."

    async def generate(self, job_id):
        from comfy_execution.utils import get_executing_context
        context = get_executing_context()
        if context is None or context.prompt_id != job_id:
            raise ValueError("Prepared request must execute in its original queued task")
        record = read_job(job_id)
        if not record or "settings" not in record or "resolved_references" not in record or record["status"] != "queued":
            raise ValueError("Prepared gateway task is unavailable or already executed")
        request = record["request"]
        return await self._execute(request["prompt"], reference_image_urls="",
                                   prepared_references=record["resolved_references"], defer_result=True, **record["settings"])


def cache_inputs():
    return {
        "cache_dit": ("BOOLEAN", {"default": False, "tooltip": "Approximate DBCache for this request only."}),
        "cache_dit_threshold": ("FLOAT", {"default": .08, "min": 0., "max": 1., "step": .01}),
        "cache_dit_fn_blocks": ("INT", {"default": 8, "min": 1, "max": 49}),
        "cache_dit_bn_blocks": ("INT", {"default": 8, "min": 0, "max": 49}),
        "cache_dit_warmup_steps": ("INT", {"default": 3, "min": 1, "max": 8}),
        "cache_dit_max_consecutive": ("INT", {"default": 1, "min": 1, "max": 7}),
        "cache_dit_max_cached_steps": ("INT", {"default": 2, "min": 0, "max": 7}),
        "cache_dit_last_steps": ("INT", {"default": 1, "min": 0, "max": 7}),
    }


NODE_CLASS_MAPPINGS = {"OpenVDNReference": OpenVDNReference, "OpenVDNH200Generate": OpenVDNH200Generate,
                       "OpenVDNH200Request": OpenVDNH200Request,
                       "OpenVDNH200BusinessRequest": OpenVDNH200BusinessRequest}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenVDNReference": "OpenVDN · Reference Image",
                              "OpenVDNH200Generate": "OpenVDN · 8 NFE (Ref2VA-like)",
                              "OpenVDNH200Request": "OpenVDN · URL References · Duration / Ratio / Resolution",
                              "OpenVDNH200BusinessRequest": "OpenVDN · Gateway Task (internal)"}


def optimization_inputs():
    return {
        "fast_communication": ("BOOLEAN", {"default": True}),
        "attention_kernel": (["native", "decomposed"],),
        "linear_stats_chunk_frames": ([16, 8, 32],),
        "isolate_padding": ("BOOLEAN", {"default": False}),
        "streaming_output": ("BOOLEAN", {"default": True}),
        "cleanup_policy": (["adaptive", "always"],),
    }
