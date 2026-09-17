"""Apply requested frame count/geometry before the official MP4 encoding step."""
import math


def output_chunks(frames, plan, chunk_size=8):
    import torch.nn.functional as F
    if len(frames) < plan.output_frames:
        raise ValueError("Decoder returned fewer frames than requested")
    for start in range(0, plan.output_frames, chunk_size):
        chunk = frames[start:min(start + chunk_size, plan.output_frames)]
        if tuple(chunk.shape[1:3]) != (plan.height, plan.width):
            chunk = F.interpolate(chunk.permute(0, 3, 1, 2).float(), size=(plan.height, plan.width),
                                  mode="bilinear", align_corners=False, antialias=True)
            chunk = chunk.round().clamp(0, 255).to(frames.dtype).permute(0, 2, 3, 1)
        yield chunk.contiguous()


def requested_encoder(original, plan):
    def encode(frames, fps, output_path, audio=None, audio_sample_rate=None, **kwargs):
        if fps != plan.fps:
            raise ValueError("Unexpected upstream frame rate")
        if audio is not None:
            import torch.nn.functional as F
            samples = round(plan.output_frames / fps * audio_sample_rate)
            audio = audio[..., :samples]
            if audio.shape[-1] < samples:
                audio = F.pad(audio, (0, samples - audio.shape[-1]))
        return original(output_chunks(frames, plan), fps=fps, output_path=output_path,
                        audio=audio, audio_sample_rate=audio_sample_rate,
                        video_chunks_number=math.ceil(plan.output_frames / 8))
    return encode
