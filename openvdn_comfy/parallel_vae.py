"""Distribute the pinned MiniMax H3 VAE's independent temporal clips.

Only _decode_clip runs on other ranks. The native decode() still performs all
padding, temporal blending and trimming on rank zero. Inputs are denormalized
latents, already replicated by the Ulysses sampler. No spatial tiling is added.
"""
from contextlib import contextmanager
import time


def clip_plan(vae, shape):
    if len(shape) != 5 or shape[0] != 1:
        raise ValueError("Parallel video VAE requires one BCTHW sample")
    chunk = vae.tokens_chunk_size
    drop = vae.config.token_drop
    overlap = vae.token_overlap
    if chunk < 1 or drop < 0 or overlap < 0:
        raise ValueError("Unsupported VAE temporal configuration")
    pseudo = shape[2] + drop
    padding = (-pseudo) % chunk
    count = (pseudo + padding) // chunk - int(drop > 0)
    if count < 1:
        raise ValueError("Insufficient latent frames for temporal VAE decode")
    bounds = [(i * chunk, min(i * chunk + chunk + overlap, shape[2] + padding))
              for i in range(count)]
    return padding, bounds


def pad_latents(z, padding):
    import torch
    if padding:
        return torch.cat([z, z[:, :, -1:].repeat(1, 1, padding, 1, 1)], dim=2)
    return z


@contextmanager
def clip_provider(vae, provider):
    # Restore the instance exactly, including whether the method was inherited.
    present = "_decode_clip" in vae.__dict__
    previous = vae.__dict__.get("_decode_clip")
    vae._decode_clip = provider
    try:
        yield
    finally:
        if present:
            vae._decode_clip = previous
        else:
            del vae._decode_clip


def assemble_native(vae, z, bounds, get_clip, verify=False):
    """Feed distributed clips through the original decoder, optionally checking each."""
    import torch
    native = vae._decode_clip
    index = 0
    parity = {"checked": bool(verify), "exact": True if verify else None,
              "clips_checked": 0}

    def provided(clip_z):
        nonlocal index
        if index >= len(bounds):
            raise RuntimeError("Native VAE requested an unexpected extra clip")
        start, end = bounds[index]
        if clip_z.shape[2] != end - start:
            raise RuntimeError("Native VAE temporal geometry changed")
        clip = get_clip(index)
        index += 1
        expected = (z.shape[0], 3, (end - start) * vae.temporal_compression_ratio,
                    z.shape[-2] * vae.spatial_compression_ratio,
                    z.shape[-1] * vae.spatial_compression_ratio)
        if tuple(clip.shape) != expected:
            raise RuntimeError(f"Unexpected decoded clip shape {tuple(clip.shape)} != {expected}")
        if verify:
            reference = native(clip_z)
            # torch.equal alone regards matching infinities as equal. A startup
            # validation must also reject nonfinite decoder output.
            same = bool(torch.isfinite(clip).all()) and torch.equal(clip, reference)
            parity["exact"] = parity["exact"] and same
            parity["clips_checked"] += 1
        return clip

    with clip_provider(vae, provided):
        video = vae.decode(z, return_dict=False)[0]
    if index != len(bounds):
        raise RuntimeError("Native VAE requested fewer clips than planned")
    return video, parity


def decode_parallel(vae, z, *, rank, world_size, verify=False, clip_decode=None):
    """All ranks enter; return the video on rank zero, and diagnostics everywhere.

Each worker buffers only its own clips (2 for a 10-second video on 8 GPUs).
Blocking sends all point to rank zero, which consumes them in temporal order;
there is no rank-to-rank dependency cycle. Startup validation drains every send
before reporting a numerical mismatch to all ranks.
"""
    import torch
    import torch.distributed as dist
    if not 0 <= rank < world_size:
        raise ValueError("Invalid decoder rank")
    padding, bounds = clip_plan(vae, z.shape)
    prepared = pad_latents(z, padding)
    if z.is_cuda:
        torch.cuda.synchronize(z.device)
    compute_start = time.perf_counter()
    local = {}
    decoder = clip_decode or vae._decode_clip
    for index, (start, end) in enumerate(bounds):
        if index % world_size == rank:
            local[index] = decoder(prepared[:, :, start:end]).contiguous()
    if z.is_cuda:
        torch.cuda.synchronize(z.device)
    compute_seconds = time.perf_counter() - compute_start
    # Actual output dtype, rather than assuming what autocast selected.
    metadata = [local[0].dtype if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    gather_start = time.perf_counter()
    if rank == 0:
        def receive(index):
            owner = index % world_size
            if owner == 0:
                return local.pop(index)
            start, end = bounds[index]
            shape = (z.shape[0], 3, (end - start) * vae.temporal_compression_ratio,
                     z.shape[-2] * vae.spatial_compression_ratio,
                     z.shape[-1] * vae.spatial_compression_ratio)
            clip = torch.empty(shape, dtype=metadata[0], device=z.device)
            dist.recv(clip, src=owner)
            return clip
        video, parity = assemble_native(vae, z, bounds, receive, verify=verify)
    else:
        video, parity = None, None
        for index in sorted(local):
            dist.send(local.pop(index), dst=0)
    if z.is_cuda:
        torch.cuda.synchronize(z.device)
    gather_seconds = time.perf_counter() - gather_start
    records = [None] * world_size
    dist.all_gather_object(records, {"rank": rank, "clips": sum(i % world_size == rank for i in range(len(bounds))),
                                    "compute_seconds": compute_seconds, "gather_and_assemble_seconds": gather_seconds})
    parity_box = [parity]
    dist.broadcast_object_list(parity_box, src=0)
    parity = parity_box[0]
    if verify and not parity["exact"]:
        raise RuntimeError("Parallel VAE startup parity failed; restart with REF2VA_VAE_PARALLEL=0")
    return video, {"world_size": world_size, "temporal_clips": len(bounds),
                   "by_rank": records, "parity": parity, "native_temporal_assembly": True}
