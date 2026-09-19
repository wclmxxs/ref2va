"""Batch independent native VAE tiles, with lazy regional compilation.

No spatial/temporal padding, attention changes, weight casts or CUDA graphs.
Only identical tile shapes are stacked on the batch axis. Numerical checks
compare each new batch signature against the original, single-tile decoder.
"""
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
import hashlib
import inspect
import time

from .vae_tiles import override_methods

NATIVE_CLIP_HASH = 'ec5a94b8ba303cfdee2fc00ca8aca91aa3e05c37e94db7137ae91fad73fa26f8'
RELATIVE_LIMIT = .005
ABSOLUTE_LIMIT = .05


class TileBatcher:
    def __init__(self, vae, native_clip, *, compile_fn=None):
        if hashlib.sha256(inspect.getsource(native_clip).encode()).hexdigest() != NATIVE_CLIP_HASH:
            raise RuntimeError('Native VAE tile source changed; audit batched tile geometry')
        self.vae = vae
        self.blocks = list(getattr(vae.decoder, 'transformer_blocks', ()))
        self.original_forwards = [block.forward for block in self.blocks]
        self.compiled_forwards = None
        self.compile_fn = compile_fn
        self.verified = OrderedDict()
        self.configure(1, False)

    def configure(self, batch_size, compile_decoder):
        if type(batch_size) is not int or batch_size not in (1, 2, 4, 8):
            raise ValueError('vae_tile_batch_size must be 1, 2, 4 or 8')
        if type(compile_decoder) is not bool:
            raise ValueError('vae_compile must be boolean')
        self.batch_size, self.compile_decoder = batch_size, compile_decoder
        self.calls = self.tiles = self.verification_hits = 0
        self.batch_histogram, self.checks, self.used_checks = {}, [], {}
        self.verification_seconds = 0.
        self.events, self.cpu_times = [], {}

    def reset_compiler(self):
        # A global Dynamo reset invalidates compiled graphs, so validate newly
        # compiled arithmetic again. Eager batch verification remains valid.
        self.verified = OrderedDict((k, v) for k, v in self.verified.items() if not k[-1])

    @contextmanager
    def compiled_blocks(self):
        if not self.compile_decoder:
            yield
            return
        if not self.blocks:
            raise RuntimeError('VAE regional compilation requires decoder.transformer_blocks')
        if self.compiled_forwards is None:
            import torch
            compile_fn = self.compile_fn or torch.compile
            # Regional graphs reuse the repeated block's code instead of tracing
            # the entire 36-layer decoder. Avoid CUDA graph tensor-lifetime rules
            # and exhaustive autotuning on every rank during the first request.
            self.compiled_forwards = [compile_fn(forward, fullgraph=True, dynamic=False,
                                                options={'triton.cudagraphs': False})
                                      for forward in self.original_forwards]
        with ExitStack() as stack:
            for block, forward in zip(self.blocks, self.compiled_forwards):
                stack.enter_context(override_methods(block, forward=forward))
            yield

    @contextmanager
    def measured(self, name, tensor):
        import torch
        if tensor.is_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self.events.append((name, start, end))
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                self.cpu_times[name] = self.cpu_times.get(name, 0.) + time.perf_counter() - start

    def core(self, z):
        return self.vae.decoder(self.vae.post_quant_conv(z))

    def decode_batch(self, tiles):
        import torch
        first = tiles[0]
        with self.measured('tile_decoder_seconds', first):
            # Keep the old strides for the batch=1 rollback path.
            z = torch.cat(tiles, dim=0) if len(tiles) > 1 else first
            with self.compiled_blocks():
                output = self.core(z)
        self.calls += 1
        self.tiles += len(tiles)
        self.batch_histogram[len(tiles)] = self.batch_histogram.get(len(tiles), 0) + 1
        if output.shape[0] != sum(t.shape[0] for t in tiles):
            raise RuntimeError('VAE decoder changed batch size')
        outputs = list(output.split(first.shape[0], dim=0))
        if len(tiles) == 1 and not self.compile_decoder:
            return outputs
        autocast = torch.is_autocast_enabled(first.device.type)
        key = (tuple(z.shape), tuple(z.stride()), str(z.dtype), str(z.device),
               str(torch.get_autocast_dtype(first.device.type)) if autocast else None,
               self.compile_decoder)
        if key in self.verified:
            self.verified.move_to_end(key)
            self.verification_hits += 1
        else:
            # Compile wrappers have been restored; these are the exact native
            # single-tile forwards, including original per-tile input strides.
            if first.is_cuda:
                torch.cuda.synchronize(first.device)
            started = time.perf_counter()
            errors = []
            for tile, result in zip(tiles, outputs):
                reference = self.core(tile)
                if result.shape != reference.shape or result.dtype != reference.dtype:
                    raise RuntimeError('VAE tile batching changed output shape/dtype')
                error = result.float() - reference.float()
                relative = error.norm() / reference.float().norm().clamp_min(1e-12)
                maximum = error.abs().max()
                finite = torch.isfinite(result).all() & torch.isfinite(reference).all()
                errors.append(torch.stack((relative, maximum, finite.float())))
            values = torch.stack(errors).detach().cpu().tolist()
            relative = max(v[0] for v in values)
            maximum = max(v[1] for v in values)
            passed = all(v[2] == 1. and v[0] <= RELATIVE_LIMIT and v[1] <= ABSOLUTE_LIMIT for v in values)
            self.verification_seconds += time.perf_counter() - started
            check = {'shape': list(z.shape), 'stride': list(z.stride()), 'dtype': str(z.dtype), 'autocast_dtype': key[4],
                     'compiled': self.compile_decoder, 'relative_l2': relative,
                     'max_abs': maximum, 'passed': passed, 'tiles_checked': len(tiles)}
            self.checks.append(check)
            if not passed:
                raise RuntimeError(f'VAE batched decoder numerical check failed: {check}; '
                                   'use vae_tile_batch_size=1, vae_compile=false to roll back')
            self.verified[key] = check
            if len(self.verified) > 64:
                self.verified.popitem(last=False)
        self.used_checks[key] = self.verified[key]
        return outputs

    def decode_clip(self, z, stitch):
        if not self.vae.use_tiling:
            return self.decode_batch([z])[0]
        vae, ratio = self.vae, self.vae.spatial_compression_ratio
        yi, yl, yo = vae._split_tiles(z.shape[-2] * ratio, vae.tile_sample_min_height,
                                     vae.tile_sample_min_overlap_height)
        xi, xl, xo = vae._split_tiles(z.shape[-1] * ratio, vae.tile_sample_min_width,
                                     vae.tile_sample_min_overlap_width)
        rows = [[None for _ in xi] for _ in yi]
        pending = []

        def flush():
            if pending:
                outputs = self.decode_batch([item[2] for item in pending])
                for (i, j, _), output in zip(pending, outputs):
                    rows[i][j] = output
                pending.clear()

        for i, (y, h) in enumerate(zip(yi, yl)):
            for j, (x, w) in enumerate(zip(xi, xl)):
                tile = z[..., y // ratio:y // ratio + h // ratio,
                         x // ratio:x // ratio + w // ratio]
                if pending and tile.shape != pending[0][2].shape:
                    flush()
                pending.append((i, j, tile))
                if len(pending) == self.batch_size:
                    flush()
        flush()
        with self.measured('tile_stitch_seconds', z):
            return stitch(rows, yo, xo)

    def report(self):
        # Caller has synchronized the completed VAE stage. No fences in hot tile
        # loops: events include launch gaps/first compilation, not pure busy time.
        times = dict(self.cpu_times)
        for name, start, end in self.events:
            times[name] = times.get(name, 0.) + start.elapsed_time(end) / 1000
        return {'tile_batch_size': self.batch_size, 'compile_enabled': self.compile_decoder,
                'compile_scope': 'decoder.transformer_blocks', 'spatial_tiles': self.tiles,
                'decoder_calls': self.calls, 'batches': dict(self.batch_histogram),
                'decoder_calls_scope': 'output batches; excludes numerical checks and startup reference decode',
                'timing_method': 'cuda_events' if self.events else 'wall',
                'timings': {**times, 'verification_seconds': self.verification_seconds},
                'verification': {'new_checks': self.checks, 'reused_batches': self.verification_hits,
                                 'signatures_used': list(self.used_checks.values()),
                                 'relative_l2_limit': RELATIVE_LIMIT, 'max_abs_limit': ABSOLUTE_LIMIT,
                                 'reference': 'native eager single-tile decoder', 'bitwise_guarantee': False}}
