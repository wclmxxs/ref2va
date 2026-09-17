"""Native VAE decode with bounded GPU postprocessing and a threaded MP4 writer.

Pixel preparation overlaps CPU H.264 encoding; component times are accumulated
active wall times and must not be summed as the output wall time.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import os
import time


@contextmanager
def measured(timings, name, device=None):
    import torch
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        timings[name] = timings.get(name, 0.) + time.perf_counter() - started


def pixel_chunk(video, start, plan, pixel_mean, pixel_std, timings, chunk_size=8):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        device = video.device
        with measured(timings, "pixel_prepare_seconds", device):
            # Trim before materializing RGB; never allocate a full float32 clip.
            pixels = video[0, :, start:min(start + chunk_size, plan.output_frames)].permute(1, 0, 2, 3).float()
            mean = torch.as_tensor(pixel_mean, device=device).view(1, 3, 1, 1)
            std = torch.as_tensor(pixel_std, device=device).view(1, 3, 1, 1)
            pixels = pixels.mul(std).add_(mean).clamp_(0, 1).mul_(255).round_()
            if tuple(pixels.shape[-2:]) != (plan.height, plan.width):
                pixels = F.interpolate(pixels, size=(plan.height, plan.width), mode="bilinear",
                                       align_corners=False, antialias=True).round_().clamp_(0, 255)
            pixels = pixels.to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        with measured(timings, "device_to_host_seconds", device):
            return pixels.cpu()


def write_mp4(video, audio, sample_rate, plan, output, pixel_mean, pixel_std, timings, phase=lambda _: None):
    import av
    import torch.nn.functional as F
    from diffusers.utils.export_utils import _write_audio, _prepare_audio_stream
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3 or video.shape[2] < plan.output_frames:
        raise ValueError("Unexpected VAE output shape or insufficient frames")
    samples = round(plan.output_frames / plan.fps * sample_rate)
    audio = audio[..., :samples]
    if audio.shape[-1] < samples:
        audio = F.pad(audio, (0, samples - audio.shape[-1]))
    with measured(timings, "device_to_host_seconds", audio.device):
        audio = audio.cpu()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    preset = os.environ.get("REF2VA_X264_PRESET", "veryfast")
    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        raise ValueError("Unsupported REF2VA_X264_PRESET")
    threads = int(os.environ.get("REF2VA_X264_THREADS", "8"))
    if not 1 <= threads <= 64:
        raise ValueError("REF2VA_X264_THREADS must be 1–64")
    phase("encoding_mp4")
    try:
        # One prefetch future bounds staging memory to two RGB chunks. A GPU
        # producer prepares the next chunk while this thread encodes the current.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="output-pixels") as pool:
            with av.open(str(partial), mode="w") as container:
                stream = container.add_stream("libx264", rate=plan.fps)
                stream.width, stream.height, stream.pix_fmt = plan.width, plan.height, "yuv420p"
                stream.codec_context.thread_count = threads
                stream.options = {"preset": preset, "crf": "23"}
                audio_stream = _prepare_audio_stream(container, sample_rate)
                def mux(packets):
                    with measured(timings, "mux_seconds"):
                        for packet in packets:
                            container.mux(packet)
                future = pool.submit(pixel_chunk, video, 0, plan, pixel_mean, pixel_std, timings)
                for start in range(0, plan.output_frames, 8):
                    chunk = future.result()
                    if start + 8 < plan.output_frames:
                        future = pool.submit(pixel_chunk, video, start + 8, plan, pixel_mean, pixel_std, timings)
                    for offset, array in enumerate(chunk.numpy()):
                        with measured(timings, "h264_encode_seconds"):
                            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                            frame.pts, frame.time_base = start + offset, Fraction(1, plan.fps)
                            packets = stream.encode(frame)
                        mux(packets)
                with measured(timings, "h264_encode_seconds"):
                    packets = stream.encode()
                mux(packets)
                with measured(timings, "audio_encode_and_mux_seconds"):
                    _write_audio(container, audio_stream, audio, sample_rate, av)
                with measured(timings, "mux_seconds"):
                    container.close()
        with measured(timings, "output_commit_seconds"):
            partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"video_codec": "libx264", "preset": preset, "crf": 23, "threads": threads,
            "pixel_device": str(video.device), "prefetch_chunks": 1,
            "component_times_overlap": True}


def decode_and_save(latents, audio_latents, vae, audio_vae, output, device, plan, pixel_mean, pixel_std,
                    phase=lambda _: None, decoded_video=None, video_decode_seconds=None):
    import torch
    timings = {}
    started = time.perf_counter()
    with torch.no_grad():
        if decoded_video is None:
            if video_decode_seconds is not None:
                raise ValueError("A predecoded video is required with video_decode_seconds")
            phase("decoding_video_vae")
            with measured(timings, "video_vae_decode_seconds", device):
                mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
                std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    video = vae.decode(latents * std + mean, return_dict=False)[0]
        else:
            if video_decode_seconds is None or video_decode_seconds < 0:
                raise ValueError("Predecoded video requires nonnegative video_decode_seconds")
            video = decoded_video
            timings["video_vae_decode_seconds"] = video_decode_seconds
        phase("decoding_audio_vae")
        with measured(timings, "audio_vae_decode_seconds", device):
            mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
            std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
            audio = audio_vae.decode(audio_latents * std + mean, return_dict=False)[0]
            audio = audio.float().permute(1, 0, 2)[0]
        encoding = write_mp4(video, audio, audio_vae.config.sampling_rate, plan, output,
                             pixel_mean, pixel_std, timings, phase)
    # A distributed decode happened immediately before this call. Include its
    # actual wall time once, preserving the existing end-to-end output metric.
    timings["output_wall_seconds"] = time.perf_counter() - started + (video_decode_seconds or 0.)
    return timings, encoding
