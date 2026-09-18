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


def _decode_buffered(vae, z, *, rank, world_size, verify=False, clip_decode=None):
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


@contextmanager
def streamed_assembly(vae, on_chunk):
    """Inject callbacks into the pinned native assembly, keeping its exact math."""
    if on_chunk is None:
        yield
        return
    import hashlib
    import inspect
    import textwrap
    import types
    original = vae._decode
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != NATIVE_DECODE_HASH:
        raise RuntimeError('Native VAE assembly source changed; audit streaming callbacks')
    source = textwrap.dedent(source)
    for value in ('chunk', 'overlap'):
        old = f'decoded_chunks.append({value})'
        assert source.count(old) == 1
        source = source.replace(old, old + f'; _ref2va_emit({value})')
    ns = {**original.__func__.__globals__, '_ref2va_emit': on_chunk}
    exec(compile(source, '<ref2va_streamed_assembly>', 'exec'), ns)
    present, previous = '_decode' in vae.__dict__, vae.__dict__.get('_decode')
    vae._decode = types.MethodType(ns['_decode'], vae)
    try:
        yield
    finally:
        if present:
            vae._decode = previous
        else:
            del vae._decode


NATIVE_DECODE_HASH = '5998c354e65ab25da294df3ebe431be76de890bf11edff07ea5ed64035a30c64'


def decode_parallel(vae, z, *, rank, world_size, verify=False, clip_decode=None,
                    streaming=False, on_chunk=None):
    if not streaming:
        if on_chunk is not None:
            raise ValueError('on_chunk requires streaming decode')
        return _decode_buffered(vae, z, rank=rank, world_size=world_size,
                                verify=verify, clip_decode=clip_decode)
    import torch
    import torch.distributed as dist
    if not 0 <= rank < world_size:
        raise ValueError('Invalid decoder rank')
    padding, bounds = clip_plan(vae, z.shape)
    prepared = pad_latents(z, padding)
    owned = [i for i in range(len(bounds)) if i % world_size == rank]
    decoder = clip_decode or vae._decode_clip
    events, cpu_seconds = [], 0.
    started = time.perf_counter()
    def decode(index):
        nonlocal cpu_seconds
        a, b = bounds[index]
        if z.is_cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        before = time.perf_counter()
        clip = decoder(prepared[:, :, a:b]).contiguous()
        cpu_seconds += time.perf_counter() - before
        if z.is_cuda:
            end.record(); events.append((begin, end))
        return clip
    first = decode(owned[0]) if owned else None
    metadata = [first.dtype if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    if rank == 0:
        def receive(index):
            nonlocal first
            owner = index % world_size
            if index == 0:
                result, first = first, None
                return result
            if owner == 0:
                return decode(index)
            a, b = bounds[index]
            shape = (z.shape[0], 3, (b-a)*vae.temporal_compression_ratio,
                     z.shape[-2]*vae.spatial_compression_ratio, z.shape[-1]*vae.spatial_compression_ratio)
            clip = torch.empty(shape, dtype=metadata[0], device=z.device)
            dist.recv(clip, src=owner)
            return clip
        with streamed_assembly(vae, on_chunk):
            video, parity = assemble_native(vae, z, bounds, receive, verify=verify)
    else:
        video, parity = None, None
        pending = []
        for pos, index in enumerate(owned):
            clip = first if pos == 0 else decode(index)
            pending.append((dist.isend(clip, dst=0), clip))
            first = None
            # Retain each buffer until NCCL finishes reading it. A bounded queue
            # overlaps the first send with the next local clip's decode.
            if len(pending) == 2:
                work, held = pending.pop(0)
                work.wait()
                del held
        for work, held in pending:
            work.wait()
    if z.is_cuda:
        torch.cuda.synchronize(z.device)
    elapsed = time.perf_counter() - started
    compute = sum(a.elapsed_time(b) for a,b in events)/1000 if z.is_cuda else cpu_seconds
    records = [None] * world_size
    dist.all_gather_object(records, {'rank':rank, 'clips':len(owned), 'compute_seconds':compute,
                                    'decode_pipeline_wall_seconds':elapsed,
                                    'compute_timing_method':'cuda_events' if z.is_cuda else 'wall'})
    box = [parity]
    dist.broadcast_object_list(box, src=0)
    if verify and not box[0]['exact']:
        raise RuntimeError('Parallel VAE startup parity failed; restart with REF2VA_VAE_PARALLEL=0')
    return video, {'world_size':world_size, 'temporal_clips':len(bounds), 'by_rank':records,
                   'parity':box[0], 'native_temporal_assembly':True, 'streaming':True,
                   'component_times_overlap':True, 'send_buffer_slots':2}
