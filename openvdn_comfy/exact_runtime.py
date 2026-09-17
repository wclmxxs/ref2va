"""Request-local constant reuse and scheduling changes for the pinned OpenVDN.

This component does not reuse denoising activations across steps. Its cached tensors are
RoPE(position_ids) and token_refiner(context_embedder(prompt_embeds)). Attention,
FP8, model weights, schedulers and the eight denoiser calls remain upstream's.
"""
from contextlib import contextmanager
import hashlib
import inspect
import linecache
import os
import time
import types


def enabled(name):
    value = os.environ.get(name, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


# Refuse a different upstream implementation, even if its edited fragments match.
SOURCE_HASHES = {
    "_ulysses_attention_forward": "0b8b14e5f7ed90b4f990dc65da00fce1562cd8aaa66e8e9e49b6785c5d776927",
    "_branch_parallel_attention_forward": "04644ea5cb324915ac3ed4becf708ebece7b1c9cb4aa01f308da577e0c9d6b22",
    "_ulysses_transformer_forward": "600693368448119ec935936079ab033d2d98196c6bfabd07dd67eaa5b09fb09a",
    "generate_latents": "775c1c97e82d3a3997fd41944a72813b68ad9e0635fc732b1b898e2637d04ff5",
}

PROJECTION_OLD = """        positions = torch.arange(runtime.local_start, runtime.local_end, device=x.device)
        is_video = (positions >= layout.video_start) & (positions < layout.video_end)
        if is_video.any():
            out[is_video] += self.to_out_linear(
                linear_local[is_video].reshape(int(is_video.sum().item()), -1).type_as(x)
            )"""
PROJECTION_NEW = """        out = _ref2va_project(self, out, linear_local, runtime, layout, x)"""
TEXT_OLD = """    text_embeds = self.context_embedder(
        encoder_hidden_states.to(get_parameter_dtype(self.context_embedder))
    )
    text_embeds = self.token_refiner(text_embeds)"""
TEXT_NEW = """    text_embeds = self._ref2va_exact.constant(
        "text", encoder_hidden_states,
        lambda: self.token_refiner(self.context_embedder(
            encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))))"""


def rewrite(function, replacements, extra_globals=None):
    """Keep the entire pinned forward, replacing only audited small fragments.

The generated source remains available to inspect/Dynamo and tracebacks. This
does not edit the managed .deps checkout or globally patch PyTorch/functions.
"""
    original = inspect.unwrap(function)
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_HASHES[original.__name__]:
        raise RuntimeError(f"Pinned OpenVDN source changed: {original.__name__}; exact runtime not installed")
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Unexpected exact-runtime patch boundary: {original.__name__}")
        source = source.replace(old, new)
    path = f"<ref2va-exact/{original.__name__}.py>"
    linecache.cache[path] = (len(source), None, source.splitlines(keepends=True), path)
    namespace = {**original.__globals__, **(extra_globals or {})}
    exec(compile(source, path, "exec"), namespace)
    return namespace[original.__name__]


def tensor_key(tensor):
    import torch
    # New views of the same prompt are created each step; identity alone misses.
    # Hold the input while cached to prevent allocator reuse of this address.
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor._version,
            tensor.dtype, tensor.device, torch.is_autocast_enabled(tensor.device.type),
            torch.get_autocast_dtype(tensor.device.type))


def video_slice(start, end, video_start, video_end):
    return slice(max(0, min(end - start, video_start - start)),
                 max(0, min(end - start, video_end - start)))


def project_video_rows(attn, out, linear_local, runtime, layout, x):
    import torch
    state = attn._ref2va_exact
    rows = video_slice(runtime.local_start, runtime.local_end, layout.video_start, layout.video_end)
    check = state.verify and id(attn) not in state.projected
    reference = None
    if check:
        state.projected.add(id(attn))
        reference = out.clone()
        positions = torch.arange(runtime.local_start, runtime.local_end, device=out.device)
        mask = (positions >= layout.video_start) & (positions < layout.video_end)
        if mask.any():
            reference[mask] += attn.to_out_linear(
                linear_local[mask].reshape(int(mask.sum().item()), -1).type_as(x))
    if rows.stop > rows.start:
        # Native boolean indexing materializes a contiguous [rows, heads, dim]
        # tensor. Preserve that GEMM input layout and dimensions exactly.
        source = linear_local[rows].contiguous().view(rows.stop - rows.start, -1).type_as(x)
        out[rows] += attn.to_out_linear(source)
    state.projection_calls += 1
    if check:
        state.check("video_projection", out, reference)
    return out


class ExactRuntime:
    def __init__(self, transformer, ulysses, render, *, active=True, block_runtime=None):
        self.active = active
        self.generate = render.generate_latents
        self.values = {}
        self.in_request = False
        if not active and block_runtime is None:
            return
        forwards = {}
        for name in ("_ulysses_attention_forward", "_branch_parallel_attention_forward"):
            forwards[name] = (rewrite(getattr(ulysses, name), [(PROJECTION_OLD, PROJECTION_NEW)],
                                      {"_ref2va_project": project_video_rows}) if active else getattr(ulysses, name))
        replacements = [
            ('rotary_emb = self.rope(position_ids)',
             'rotary_emb = self._ref2va_exact.constant("rope", position_ids, lambda: self.rope(position_ids))'),
            (TEXT_OLD, TEXT_NEW)] if active else []
        if block_runtime is not None:
            from .dit_runtime import transformer_replacements
            replacements += transformer_replacements()
        forward = rewrite(ulysses._ulysses_transformer_forward, replacements)
        if active:
            self.generate = rewrite(render.generate_latents, [
                # rewrite() snapshots globals, but the resident worker updates
                # the canvas before every request. Bind those two values at call
                # time, before layout, noise allocation and final unpatchify.
                ("    num_frames = align_num_frames(num_frames, 17, 5)",
                 "    LATENT_H, LATENT_W = _ref2va_render.LATENT_H, _ref2va_render.LATENT_W\n"
                 "    num_frames = align_num_frames(num_frames, 17, 5)"),
                ("step_started = time.perf_counter()", "step_started = _ref2va_step_start(device)"),
                ("            torch.cuda.synchronize(device)\n            step_seconds.append(time.perf_counter() - step_started)",
                 "            step_seconds.append(_ref2va_step_end(step_started))"),
                ("    # Unpatchify (the AfterDenoise step's reshape) and unpack the channel-major audio rows.",
                 "    _ref2va_finish_steps(step_seconds)\n\n    # Unpatchify (the AfterDenoise step's reshape) and unpack the channel-major audio rows.")],
                {"_ref2va_render": render,
                 "_ref2va_step_start": step_start, "_ref2va_step_end": step_end,
                 "_ref2va_finish_steps": finish_steps})
        # Install only after all four source contracts have passed.
        for attn in ulysses.iter_hybrids(transformer):
            name = attn.forward.__func__.__name__
            if name not in forwards:
                raise RuntimeError(f"Unexpected attention owner: {name}")
            attn._ref2va_exact = self
            attn.forward = types.MethodType(forwards[name], attn)
        transformer._ref2va_exact = self
        transformer.forward = types.MethodType(forward, transformer)
        if block_runtime is not None:
            block_runtime.install_attention(forwards)

    @contextmanager
    def request(self, *, verify=False):
        if self.in_request:
            raise RuntimeError("Exact runtime cannot serve concurrent requests")
        self.in_request = True
        self.verify = verify and self.active
        self.values, self.hits, self.misses = {}, {}, {}
        self.checks, self.check_counts, self.checked_constants = [], {}, set()
        self.projected, self.projection_calls = set(), 0
        try:
            yield self
        finally:
            self.values.clear()
            self.checks.clear()
            self.projected.clear()
            self.in_request = False

    def constant(self, name, tensor, compute):
        if not self.in_request:
            raise RuntimeError("Constant reuse requires an active request")
        key = tensor_key(tensor)
        previous = self.values.get(name)
        if previous is None or previous[0] != key:
            value = compute()
            self.values[name] = (key, value, tensor)
            self.misses[name] = self.misses.get(name, 0) + 1
            return value
        self.hits[name] = self.hits.get(name, 0) + 1
        if self.verify and name not in self.checked_constants:
            self.check(name, previous[1], compute())
            self.checked_constants.add(name)
        return previous[1]

    def check(self, name, actual, reference):
        import torch
        if isinstance(actual, (tuple, list)):
            if len(actual) != len(reference):
                raise RuntimeError("Exact runtime result structure changed")
            for a, b in zip(actual, reference):
                self.check(name, a, b)
            return
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            same = torch.zeros((), dtype=torch.bool, device=actual.device)
        else:
            same = torch.isfinite(actual).all() & torch.eq(actual, reference).all()
        # Keep only scalar verdicts. Resolve once after every rank leaves DiT.
        self.checks.append(same)
        self.check_counts[name] = self.check_counts.get(name, 0) + 1

    def report(self):
        import torch
        exact = bool(torch.stack(self.checks).all()) if self.checks else None
        return {"enabled": self.active, "constant_hits": dict(self.hits),
                "constant_misses": dict(self.misses), "projection_calls": self.projection_calls,
                "parity": {"checked": self.verify, "exact": exact, "components": dict(self.check_counts)},
                "step_timing_method": "cuda_events" if self.active else "synchronized_wall"}


def step_start(device):
    import torch
    if torch.device(device).type != "cuda":
        return time.perf_counter()
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device))
    return event


def step_end(start):
    import torch
    if isinstance(start, float):
        return time.perf_counter() - start
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    return start, end


def finish_steps(steps):
    if not steps or isinstance(steps[0], float):
        return
    steps[-1][1].synchronize()
    steps[:] = [start.elapsed_time(end) / 1000 for start, end in steps]
