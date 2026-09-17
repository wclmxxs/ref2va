import hashlib
import io
import json
import uuid
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image

from .config import RUNTIME, Settings
from .runner import generate
from .references import download_references, parse_urls
from .jobs import update_job


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
            "reference_short_edge": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 32}),
            "fp8": ("BOOLEAN", {"default": True}),
            "inference_kernels": ("BOOLEAN", {"default": True}),
            "softmax_backend": (["flex", "decomposed", "ref"],),
            "softmax_ranks": ("INT", {"default": 6, "min": 0, "max": 7}),
            "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 8}),
            "profile": ("BOOLEAN", {"default": False}),
        }, "optional": {"references": ("OPENVDN_REFS",)}}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "metrics_json")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "OpenVDN H200"
    DESCRIPTION = ("Official 8-NFE / 8×H200 Ulysses, 1344×768 at 24 fps. Ref2VA-like uses FL2VA weights. "
                   "inference_kernels controls the official fused/compiled kernel bundle; it is not a whole-DiT compile switch. "
                   "No Sol or cross-step DiT cache. Models are resident; kernel/precision settings must match /openvdn/health. "
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
        }}

    DESCRIPTION = "Reference-image URLs to video. Duration is in seconds, ratio is width:height, resolution is the output short edge. Uses all 8 GPUs and official Ref2VA-like weights."

    async def generate(self, prompt, duration, ratio, resolution, reference_image_urls, **kwargs):
        import folder_paths
        import comfy.model_management as mm
        from comfy_api.input_impl import VideoFromFile
        from comfy_execution.utils import get_executing_context
        context = get_executing_context()
        job_id = context.prompt_id if context else None
        try:
            settings = Settings(duration=duration, ratio=ratio, resolution=resolution, **kwargs).validate()
            urls = parse_urls(reference_image_urls)
            update_job(job_id, status="running", phase="downloading_references")
            refs = await download_references(urls, mm.throw_exception_if_processing_interrupted)
            name = f"vdn8_{uuid.uuid4().hex}.mp4"
            output = Path(folder_paths.get_output_directory()) / "openvdn" / name
            import asyncio
            result = await asyncio.to_thread(generate, prompt=prompt, refs=refs, settings=settings, output=output,
                              interrupt=mm.throw_exception_if_processing_interrupted,
                              progress=lambda phase: update_job(job_id, status="running", phase=phase))
            video = {"filename": name, "subfolder": "openvdn", "type": "output"}
            update_job(job_id, status="succeeded", phase="complete", video_url="/view?" + urlencode(video), metrics=result)
            return {"ui": {"openvdn_videos": [video]},
                    "result": (VideoFromFile(str(output)), json.dumps(result, ensure_ascii=False, indent=2))}
        except BaseException as error:
            update_job(job_id, status="failed", error=str(error))
            raise


NODE_CLASS_MAPPINGS = {"OpenVDNReference": OpenVDNReference, "OpenVDNH200Generate": OpenVDNH200Generate,
                       "OpenVDNH200Request": OpenVDNH200Request}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenVDNReference": "OpenVDN · Reference Image",
                              "OpenVDNH200Generate": "OpenVDN · 8×H200 · 8 NFE (Ref2VA-like)",
                              "OpenVDNH200Request": "OpenVDN · URL References · Duration / Ratio / Resolution"}
