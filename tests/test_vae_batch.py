import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from openvdn_comfy.config import Settings
from openvdn_comfy.vae_tiles import ClipDecoder
from openvdn_comfy.vae_batch import TileBatcher
from tests.test_parallel_vae import native_decoder
from openvdn_comfy.parallel_vae import assemble_native, clip_plan, clip_provider, pad_latents


def independent_vae():
    vae = native_decoder(spatial=True)
    class Core(torch.nn.Module):
        def forward(self, z):
            # Per-tile context catches wrong origins/order. No reduction whose
            # rounding could differ between strided and contiguous inputs.
            return (z[:, :3] + z[:, :1, :1, :1, :1]).repeat_interleave(4, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
    vae.decoder = Core()
    return vae


@pytest.mark.parametrize('batch', [1, 2, 4, 8])
@pytest.mark.parametrize('shape', [(3, 4), (6, 7), (8, 12)])
def test_batch_preserves_native_tiling_order_overlap_and_tail(batch, shape):
    vae = independent_vae()
    z = torch.randn(2, 24, 7, *shape)
    expected = vae._decode_clip(z)
    decode = ClipDecoder(vae)
    decode.configure(batch, False)
    assert torch.equal(decode(z), expected)
    report = decode.report()
    assert report['spatial_tiles'] == decode.tile_count
    assert report['decoder_calls'] == (decode.tile_count + batch - 1) // batch
    if decode.tile_count > batch:
        assert batch in report['batches']
    assert '_decode_clip' not in vae.__dict__ and '_stitch_tiles' not in vae.__dict__
    # Different tile values with the same shape reuse verification, never pixels.
    decode.configure(batch, False)
    assert torch.equal(decode(z + .125), vae._decode_clip(z + .125))
    assert not decode.report()['verification']['new_checks']


def test_different_tile_sizes_are_never_padded_or_concatenated_together():
    vae = independent_vae()
    # Also cover a future/native tail layout that returns nonuniform lengths.
    vae._split_tiles = lambda *args: ([0, 6], [8, 6], [2, 0])
    decode = ClipDecoder(vae)
    decode.configure(8, False)
    z = torch.randn(1, 24, 7, 6, 6)
    assert torch.equal(decode(z), vae._decode_clip(z))
    assert decode.report()['batches'] == {1: 4}


def test_untiled_native_layout_and_compiler_reset_validation():
    vae = independent_vae()
    vae.use_tiling = False
    decode = ClipDecoder(vae)
    decode.configure(4, False)
    z = torch.randn(1, 24, 7, 3, 4)
    assert torch.equal(decode(z), vae._decode_clip(z))
    assert decode.report()['decoder_calls'] == 1
    decode.reset_compiler()


@pytest.mark.parametrize('batch', [1, 4])
def test_native_temporal_provider_and_startup_exact_check_still_work(batch):
    vae = independent_vae()
    z = torch.randn(1, 24, 12, 6, 7)
    expected = vae.decode(z, return_dict=False)[0]
    decode = ClipDecoder(vae)
    decode.configure(batch, False)
    padding, bounds = clip_plan(vae, z.shape)
    prepared = pad_latents(z, padding)
    # Decode local clips while _decode_clip is overridden by the transport
    # provider, exactly like rank zero does during streaming assembly.
    video, parity = assemble_native(vae, z, bounds,
                                    lambda i: decode(prepared[:, :, bounds[i][0]:bounds[i][1]]), verify=True)
    assert torch.equal(video, expected) and parity['exact']
    with clip_provider(vae, decode):
        assert torch.equal(vae.decode(z, return_dict=False)[0], expected)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_decoder_values_never_pass_verification(bad):
    vae = independent_vae()
    original = vae.decoder.forward
    vae.decoder.forward = lambda z: original(z) * bad
    decode = ClipDecoder(vae)
    decode.configure(4, False)
    with pytest.raises(RuntimeError, match='numerical check failed'):
        decode(torch.zeros(1, 24, 7, 6, 7))
    assert not decode.batcher.verified


def test_bad_batch_arithmetic_is_rejected_and_not_cached():
    vae = independent_vae()
    original = vae.decoder.forward
    vae.decoder.forward = lambda z: original(z) + (1 if z.shape[0] > 1 else 0)
    decode = ClipDecoder(vae)
    decode.configure(4, False)
    with pytest.raises(RuntimeError, match='numerical check failed'):
        decode(torch.randn(1, 24, 7, 8, 12))
    assert not decode.batcher.verified


def small_vit():
    """Pinned decoder/attention/RoPE with a small CPU FFN/backend dependency shim."""
    import math
    root = Path(__file__).resolve().parents[1]
    path = root/'work/patch-check/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py'
    if not path.exists():
        path = root/'.deps/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py'
    if not path.exists():
        pytest.skip('Pinned Diffusers sources unavailable')
    names = ('MiniMaxH3VideoRotaryPosEmbed', 'MiniMaxH3VideoAttnProcessor', 'MiniMaxH3VideoAttention',
             'MiniMaxH3VideoTransformerBlock', 'MiniMaxH3VideoViTDecoder3d')
    classes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
    class ProcessorMixin:
        def set_processor(self, processor):
            self.processor = processor
    class FFN(torch.nn.Module):
        def __init__(self, dim, mult, **kwargs):
            super().__init__()
            self.up = torch.nn.Linear(dim, 2 * dim * mult)
            self.down = torch.nn.Linear(dim * mult, dim)
        def forward(self, x):
            gate, value = self.up(x).chunk(2, -1)
            return self.down(torch.nn.functional.silu(gate) * value)
    def attention(q, k, v, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                                v.transpose(1, 2)).transpose(1, 2)
    ns = {'torch': torch, 'nn': torch.nn, 'math': math, 'FeedForward': FFN,
          'AttentionModuleMixin': ProcessorMixin, 'dispatch_attention_fn': attention}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), ns)
    core = ns[names[-1]](in_channels=24, patch_size=2, patch_size_t=4, num_layers=2,
                         num_attention_heads=2, attention_head_dim=8).eval()
    with torch.no_grad():
        for block in core.transformer_blocks:
            block.scale1.fill_(.5)
            block.scale2.fill_(.5)
    return core


def test_pinned_vit_batch_attention_and_regional_graph_reuse():
    torch._dynamo.reset()
    vae = independent_vae()
    vae.decoder = small_vit()
    graphs = []
    def backend(gm, inputs, **kwargs):
        graphs.append(gm)
        return gm.forward
    def compile_fn(fn, **kwargs):
        return torch.compile(fn, backend=backend, **kwargs)
    decode = ClipDecoder(vae)
    decode.batcher = TileBatcher(vae, decode.decode_clip, compile_fn=compile_fn)
    z = torch.randn(1, 24, 2, 8, 12)
    with torch.no_grad():
        expected = vae._decode_clip(z)
        decode.configure(4, True)
        actual = decode(z)
        torch.testing.assert_close(actual, expected)
        assert graphs and decode.report()['verification']['new_checks']
        count = len(graphs)
        decode.configure(4, True)
        torch.testing.assert_close(decode(z), expected)
        assert len(graphs) == count
        assert not decode.report()['verification']['new_checks']
        decode.configure(1, False)
        assert torch.equal(decode(z), expected)
        decode.configure(4, True)
        torch.testing.assert_close(decode(z), expected)
        assert len(graphs) == count
        assert all('forward' not in b.__dict__ for b in vae.decoder.transformer_blocks)
        torch._dynamo.reset()
        decode.reset_compiler()
        assert not decode.batcher.verified
        decode.configure(4, True)
        decode(z)
        assert len(graphs) > count and decode.report()['verification']['new_checks']
    torch._dynamo.reset()


def test_compile_failure_restores_original_forwards_and_rollback_works():
    vae = independent_vae()
    vae.decoder = small_vit()
    def fail(*args, **kwargs):
        raise RuntimeError('compiler failed')
    decode = ClipDecoder(vae)
    decode.batcher = TileBatcher(vae, decode.decode_clip, compile_fn=lambda *a, **k: fail)
    decode.configure(4, True)
    with torch.no_grad(), pytest.raises(RuntimeError, match='compiler failed'):
        decode(torch.randn(1, 24, 2, 8, 12))
    assert all('forward' not in b.__dict__ for b in vae.decoder.transformer_blocks)
    decode.configure(1, False)
    z = torch.randn(1, 24, 2, 3, 4)
    with torch.no_grad():
        assert torch.equal(decode(z), vae._decode_clip(z))


@pytest.mark.parametrize('options', [{'vae_tile_batch_size': v} for v in (0, 3, 16, True, '4')]
                         + [{'vae_compile': v} for v in (1, 'true')])
def test_invalid_vae_controls_are_rejected(options):
    with pytest.raises(ValueError):
        replace(Settings(), **options).validate()


def test_changed_upstream_tile_code_rejected():
    vae = independent_vae()
    vae._decode_clip = lambda z: z
    with pytest.raises(RuntimeError, match='source changed'):
        ClipDecoder(vae).configure(4, False)


def test_vae_controls_propagate_from_environment_and_business_defaults(monkeypatch):
    from openvdn_comfy.backend import startup_settings
    from openvdn_comfy.business_contract import normalize_request
    monkeypatch.setenv('REF2VA_VAE_TILE_BATCH_SIZE', '2')
    monkeypatch.setenv('REF2VA_VAE_COMPILE', '0')
    settings = startup_settings()
    assert settings.vae_tile_batch_size == 2 and not settings.vae_compile
    state = {'profile': {'fp8': True, 'inference_kernels': True, 'softmax_backend': 'flex', 'softmax_ranks': 6},
             'request_options': {'optimizations': {'vae_tile_batch_size': 2, 'vae_compile': False}}}
    body = {'model': 'MiniMax-H3', 'content': [{'type': 'text', 'text': 'hello'},
            {'type': 'image_url', 'role': 'reference_image', 'image_url': {'url': 'https://example.com/reference.png'}}],
            'duration': 10, 'ratio': '9:16', 'resolution': '768P'}
    _, inherited, _ = normalize_request(body, state)
    assert inherited.vae_tile_batch_size == 2 and not inherited.vae_compile
    _, explicit, _ = normalize_request({**body, 'optimization': {'vae_tile_batch_size': 8, 'vae_compile': True}}, state)
    assert explicit.vae_tile_batch_size == 8 and explicit.vae_compile
