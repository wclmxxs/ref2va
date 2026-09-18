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

from .exact_runtime import enabled


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


def prepare_pixels(video, start, plan, pixel_mean, pixel_std, chunk_size=8):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        # Preserve the previous operation order, rounding and interpolation.
        pixels = video[0, :, start:min(start + chunk_size, plan.output_frames)].permute(1, 0, 2, 3).float()
        mean = torch.as_tensor(pixel_mean, device=video.device).view(1, 3, 1, 1)
        std = torch.as_tensor(pixel_std, device=video.device).view(1, 3, 1, 1)
        pixels = pixels.mul(std).add_(mean).clamp_(0, 1).mul_(255).round_()
        if tuple(pixels.shape[-2:]) != (plan.height, plan.width):
            pixels = F.interpolate(pixels, size=(plan.height, plan.width), mode="bilinear",
                                   align_corners=False, antialias=True).round_().clamp_(0, 255)
        return pixels.to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def pixel_chunk(video, start, plan, pixel_mean, pixel_std, timings, chunk_size=8):
    with measured(timings, "pixel_prepare_seconds", video.device):
        pixels = prepare_pixels(video, start, plan, pixel_mean, pixel_std, chunk_size)
    with measured(timings, "device_to_host_seconds", video.device):
        return pixels.cpu()


class PinnedPixels:
    """Two bounded host buffers; CUDA preparation, D2H and CPU encoding overlap.

The consumer releases a slot by requesting the next chunk, after PyAV has
copied its frames. An event prevents reading pinned memory before D2H finishes.
"""
    def __init__(self, video, plan, mean, std, timings, *, verify=False):
        import torch
        self.video, self.plan, self.mean, self.std = video, plan, mean, std
        self.timings, self.verify = timings, verify
        self.pending = []
        self.checked_chunks = 0
        with torch.cuda.device(video.device):
            self.compute = torch.cuda.Stream(device=video.device)
            self.copy = torch.cuda.Stream(device=video.device)
            self.compute.wait_stream(torch.cuda.current_stream(video.device))
            self.buffers = [torch.empty((8, plan.height, plan.width, 3), dtype=torch.uint8,
                                        device="cpu", pin_memory=True) for _ in range(2)]

    def submit(self, start, slot):
        import torch
        with torch.cuda.device(self.video.device), torch.no_grad():
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            with torch.cuda.stream(self.compute):
                events[0].record()
                pixels = prepare_pixels(self.video, start, self.plan, self.mean, self.std)
                events[1].record()
            host = self.buffers[slot][:pixels.shape[0]]
            with torch.cuda.stream(self.copy):
                self.copy.wait_event(events[1])
                events[2].record()
                host.copy_(pixels, non_blocking=True)
                events[3].record()
            # Keep the GPU source alive until the copy completes, including errors.
            ticket = (start, slot, host, pixels, events)
            self.pending.append(ticket)
            return ticket

    def receive(self, ticket):
        import torch
        start, _, host, _, events = ticket
        waited = time.perf_counter()
        events[3].synchronize()
        self.timings["pixel_prefetch_wait_seconds"] = self.timings.get("pixel_prefetch_wait_seconds", 0.) + time.perf_counter() - waited
        for key, first, last in (("pixel_prepare_seconds", 0, 1), ("device_to_host_seconds", 2, 3)):
            self.timings[key] = self.timings.get(key, 0.) + events[first].elapsed_time(events[last]) / 1000
        self.pending = [p for p in self.pending if p is not ticket]
        if self.verify:
            reference = pixel_chunk(self.video, start, self.plan, self.mean, self.std, {})
            if not torch.equal(host, reference):
                raise RuntimeError("Async output pixel parity failed; restart with REF2VA_ASYNC_OUTPUT=0")
            self.checked_chunks += 1
        return host

    def chunks(self):
        starts = iter(range(0, self.plan.output_frames, 8))
        queue = []
        try:
            for slot in range(2):
                start = next(starts, None)
                if start is not None:
                    queue.append(self.submit(start, slot))
            while queue:
                ticket = queue.pop(0)
                yield ticket[0], self.receive(ticket)
                slot = ticket[1]
                del ticket
                start = next(starts, None)
                if start is not None:
                    queue.append(self.submit(start, slot))
        finally:
            # Only our streams, never a device-wide fence in the hot loop.
            self.compute.synchronize()
            self.copy.synchronize()
            self.pending.clear()


def threaded_pixels(video, plan, mean, std, timings):
    # CPU and explicitly disabled asynchronous-output fallback.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="output-pixels") as pool:
        future = pool.submit(pixel_chunk, video, 0, plan, mean, std, timings)
        for start in range(0, plan.output_frames, 8):
            chunk = future.result()
            if start + 8 < plan.output_frames:
                future = pool.submit(pixel_chunk, video, start + 8, plan, mean, std, timings)
            yield start, chunk


def write_mp4(video, audio, sample_rate, plan, output, pixel_mean, pixel_std, timings, phase=lambda _: None,
              *, verify=False, pixel_chunks=None):
    import av
    import torch.nn.functional as F
    from diffusers.utils.export_utils import _write_audio, _prepare_audio_stream
    if pixel_chunks is None and (video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3 or video.shape[2] < plan.output_frames):
        raise ValueError("Unexpected VAE output shape or insufficient frames")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    preset = os.environ.get("REF2VA_X264_PRESET", "veryfast")
    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        raise ValueError("Unsupported REF2VA_X264_PRESET")
    threads = int(os.environ.get("REF2VA_X264_THREADS", "8"))
    if not 1 <= threads <= 64:
        raise ValueError("REF2VA_X264_THREADS must be 1–64")
    asynchronous = pixel_chunks is None and enabled("REF2VA_ASYNC_OUTPUT") and video.is_cuda
    producer = PinnedPixels(video, plan, pixel_mean, pixel_std, timings, verify=verify) if asynchronous else None
    chunks = pixel_chunks if pixel_chunks is not None else (producer.chunks() if producer else threaded_pixels(video, plan, pixel_mean, pixel_std, timings))
    phase("encoding_mp4")
    try:
        from contextlib import closing
        with closing(chunks):
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
                for start, chunk in chunks:
                    for offset, array in enumerate(chunk.numpy()):
                        with measured(timings, "h264_encode_seconds"):
                            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                            frame.pts, frame.time_base = start + offset, Fraction(1, plan.fps)
                            packets = stream.encode(frame)
                        mux(packets)
                with measured(timings, "h264_encode_seconds"):
                    packets = stream.encode()
                mux(packets)
                audio = audio() if callable(audio) else audio
                samples = round(plan.output_frames / plan.fps * sample_rate)
                audio = audio[..., :samples]
                if audio.shape[-1] < samples:
                    audio = F.pad(audio, (0, samples - audio.shape[-1]))
                with measured(timings, "device_to_host_seconds", audio.device):
                    audio = audio.cpu()
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
            "pixel_device": str(video.device) if video is not None else "streamed", "prefetch_chunks": 2 if asynchronous else 0 if pixel_chunks is not None else 1,
            "async_pinned_output": asynchronous, "host_buffer_slots": 2 if asynchronous else 0,
            "pixel_timing_method": "cuda_events" if asynchronous else "synchronized_wall",
            "pixel_parity": {"checked": bool(producer and verify),
                             "chunks_checked": producer.checked_chunks if producer else 0},
            "component_times_overlap": True, "streaming_vae_output": pixel_chunks is not None}


def decode_and_save(latents, audio_latents, vae, audio_vae, output, device, plan, pixel_mean, pixel_std,
                    phase=lambda _: None, decoded_video=None, video_decode_seconds=None, verify_output=False, stream_writer=None):
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
        if stream_writer is None:
            encoding = write_mp4(video, audio, audio_vae.config.sampling_rate, plan, output,
                                 pixel_mean, pixel_std, timings, phase, verify=verify_output)
        else:
            phase("finishing_mp4")
            streamed_timings, encoding = stream_writer.finish(audio, video)
            timings.update(streamed_timings)
    # A distributed decode happened immediately before this call. Include its
    # actual wall time once, preserving the existing end-to-end output metric.
    timings["output_wall_seconds"] = time.perf_counter() - started + (video_decode_seconds or 0.)
    return timings, encoding
