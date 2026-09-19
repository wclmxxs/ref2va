"""Keep native tiles and blend arithmetic, but avoid row/video concatenation copies."""
from collections import OrderedDict
from contextlib import contextmanager


@contextmanager
def override_methods(module, **methods):
    previous = {name: (name in module.__dict__, module.__dict__.get(name)) for name in methods}
    for name, method in methods.items():
        setattr(module, name, method)
    try:
        yield
    finally:
        for name, (present, value) in previous.items():
            if present:
                setattr(module, name, value)
            else:
                delattr(module, name)


class ClipDecoder:
    def __init__(self, vae):
        self.vae = vae
        # Temporal assembly temporarily replaces vae._decode_clip with its
        # transport provider. Rank zero still decodes later local clips inside
        # that scope, so keep the original bound method instead of reentering
        # the provider. Its tile/blend lookups still use the overrides below.
        self.decode_clip = vae._decode_clip
        self.weights = OrderedDict()
        self.tile_count = 0
        self.batcher = None

    def configure(self, batch_size, compile_decoder):
        from .vae_batch import TileBatcher
        if self.batcher is None:
            self.batcher = TileBatcher(self.vae, self.decode_clip)
        self.batcher.configure(batch_size, compile_decoder)
        self.tile_count = 0

    def reset_compiler(self):
        if self.batcher is not None:
            self.batcher.reset_compiler()

    def report(self):
        return self.batcher.report() if self.batcher is not None else {}

    def blend(self, a, b, extent, dim):
        import torch
        extent = min(a.shape[dim], b.shape[dim], extent)
        key = (extent, b.device, b.dtype)
        if key not in self.weights:
            positions = torch.arange(extent, device=b.device, dtype=b.dtype)
            # Match the native operation order and dtype; no fused add/multiply.
            self.weights[key] = (1 - positions / extent, positions / extent)
            if len(self.weights) > 32:
                self.weights.popitem(last=False)
        self.weights.move_to_end(key)
        shape = [1] * a.ndim
        shape[dim] = extent
        wa, wb = (w.view(shape) for w in self.weights[key])
        sa, sb = [slice(None)] * a.ndim, [slice(None)] * b.ndim
        sa[dim], sb[dim] = slice(-extent, None), slice(0, extent)
        blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
        if extent == b.shape[dim]:
            return blended
        sb[dim] = slice(extent, None)
        return torch.cat([blended, b[tuple(sb)]], dim=dim)

    def stitch(self, tiles, height_overlaps, width_overlaps):
        height = sum(row[0].shape[-2] - (height_overlaps[i] if i < len(tiles) - 1 else 0)
                     for i, row in enumerate(tiles))
        width = sum(tile.shape[-1] - (width_overlaps[j] if j < len(tiles[0]) - 1 else 0)
                    for j, tile in enumerate(tiles[0]))
        output = tiles[0][0].new_empty((*tiles[0][0].shape[:-2], height, width))
        top = 0
        for i, row in enumerate(tiles):
            left = 0
            for j, tile in enumerate(row):
                if i:
                    tile = self.blend(tiles[i - 1][j], tile, height_overlaps[i - 1], -2)
                if j:
                    tile = self.blend(row[j - 1], tile, width_overlaps[j - 1], -1)
                if i < len(tiles) - 1:
                    tile = tile[..., :-height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-width_overlaps[j]]
                h, w = tile.shape[-2:]
                output[..., top:top + h, left:left + w].copy_(tile)
                left += w
                self.tile_count += 1
            top += h
        return output

    def __call__(self, z):
        if self.batcher is not None:
            return self.batcher.decode_clip(z, self.stitch)
        with override_methods(self.vae, _stitch_tiles=self.stitch, _blend=self.blend):
            return self.decode_clip(z)
