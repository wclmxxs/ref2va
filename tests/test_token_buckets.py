import ast
from pathlib import Path
import types

import numpy as np
import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from openvdn_comfy.api import normalize_request, prompt_graph
from openvdn_comfy.cache_dit import CacheConfig, DBCache
from openvdn_comfy.conditioning import load_conditioning
from openvdn_comfy.config import Settings
from openvdn_comfy.resident_geometry import GeometryCache
from openvdn_comfy.token_buckets import PrefixBucket, TokenBuckets, describe_bucket


def native_ref_layout():
    """Run pinned Diffusers packing, without loading its model dependencies."""
    root = Path(__file__).resolve().parents[1]
    suffix = 'src/diffusers/modular_pipelines/minimax_h3/before_denoise.py'
    paths = [root / 'work/patch-check/diffusers' / suffix, root / '.deps/diffusers' / suffix]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        pytest.skip('Install pinned Diffusers sources')
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name.startswith(('_spatial_', '_temporal_', '_frame_', '_fill_audio_'))]
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == 'MiniMaxH3Ref2VAPrepareLayoutStep')
    function = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'build_ref2va_packed_sequence')
    function.decorator_list = []
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                 *selected, function]
    ns = {'torch': torch, 'np': np, '_ROPE_FRAME_RESCALE': 5 / 3,
          '_ROPE_FRAMES_PER_LATENT': (1, 4, 4, 4, 4), '_ROPE_SPATIAL_SCALE': 32}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns['build_ref2va_packed_sequence']


@pytest.mark.parametrize('length,refs,stride', [(13, [(4, 6)], 64), (29, [(8, 4), (4, 8)], 64),
                                             (12, [], 0), (12, [], 16)])
def test_native_layout_positions_modalities_and_outputs_survive_padding(length, refs, stride):
    tags = torch.arange(length) % 2
    conditions = [torch.zeros(1, 24, 1, h, w) for h, w in refs]
    native = native_ref_layout()(tags, [types.SimpleNamespace(kind='image') for _ in refs], conditions,
                                [], 3, 8, 4, 2, (1, 2, 2), 2, 2, 0)
    state = TokenBuckets(stride)
    bucket = PrefixBucket(length, native[5], 4, 24, stride)
    state.prepare(bucket, 'cpu')
    packed = state.pack_layout(native)
    # Both latent video/reference and audio output gathers preserve row order,
    # coordinates and modalities, not just the final canvas dimensions.
    for index in (2, 3, 4):
        assert torch.equal(packed[0][packed[index]], native[0][native[index]])
        assert torch.equal(packed[1][packed[index]], native[1][native[index]])
    assert torch.equal(packed[4], torch.arange(length))
    assert len(packed[0]) == bucket.capacity + 24
    assert int(packed[2][native[5]]) == bucket.capacity
    assert torch.equal(packed[2][:native[5]], native[2][:native[5]])
    if stride:
        with pytest.raises(RuntimeError, match='does not match'):
            state.prepare(PrefixBucket(length + 1, native[5], 4, 24, stride), 'cpu')
            state.pack_layout(native)


def window_mask(prefix, frames, spatial, full):
    def mask(b, h, q, k):
        q_video, k_video = q >= prefix, k >= prefix
        qf = torch.div(q - prefix, spatial, rounding_mode='floor')
        kf = torch.div(k - prefix, spatial, rounding_mode='floor')
        return (~(q_video & k_video)) | ((qf - kf).abs() <= (frames if full else 1))
    return mask


@pytest.mark.parametrize('full', [False, True])
def test_real_flex_excludes_gap_even_inside_full_blocks_and_reuses_graph(full):
    torch.manual_seed(7)
    torch._dynamo.reset()
    from torch._dynamo import utils
    state = TokenBuckets(32)
    compiled = torch.compile(flex_attention, backend='eager', dynamic=False)
    # Superset block mask is identical for both valid prefix lengths.
    mask = create_block_mask(window_mask(32, 4, 8, full), None, None, 64, 64,
                             device='cpu', BLOCK_SIZE=128)
    graph_count = None
    for length in (19, 27, 19):
        state.prepare(PrefixBucket(length - 8, 4, 4, 32, 32), 'cpu')
        native = [torch.randn(1, 2, length + 32, 8) for _ in range(3)]
        # Large finite padding values detect leakage; zero padding alone can
        # obscure bugs in masking and softmax denominators.
        padded = [torch.cat((t[:, :, :length], t.new_full((1, 2, 32 - length, 8), 99.),
                             t[:, :, length:]), dim=2) for t in native]
        native_mask = window_mask(length, 4, 8, full)
        positions = torch.arange(length + 32)
        allowed = native_mask(None, None, positions[:, None], positions[None, :])
        expected = torch.nn.functional.scaled_dot_product_attention(*native, attn_mask=allowed)
        actual = compiled(*padded, block_mask=mask, score_mod=state.score_mod)
        actual = torch.cat((actual[:, :, :length], actual[:, :, 32:]), dim=2)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.isfinite().all()
        count = utils.counters['stats']['unique_graphs']
        if graph_count is not None:
            assert count == graph_count  # changing tensor values cannot recompile
        graph_count = count
    torch._dynamo.reset()


def test_padding_never_changes_dbcache_modality_error():
    def result(padded):
        count = 10 if padded else 8
        runtime = types.SimpleNamespace(world_size=1, local_start=0, local_end=count)
        cache = DBCache(runtime)
        with cache.request(CacheConfig(enabled=True)):
            x = torch.ones(1, count, 3)
            video = torch.tensor([2, 3, 6, 7, 8, 9] if padded else [2, 3, 4, 5, 6, 7])
            cache.configure_groups(x, video, torch.tensor([], dtype=torch.long),
                                   6 if padded else 4, count, valid_prefix=4 if padded else None)
            cache.previous = x.clone()
            cache.residual = x.clone()
            cache.residual_finite = torch.tensor(True)
            current = x + .01
            if padded:
                current[:, 4:6] = 1e6
            return cache.decision(current)
    assert result(False) == result(True)


def test_bucket_geometry_reuses_capacity_not_caption_or_reference_aspect():
    plan = Settings(duration=10, ratio='9:16', resolution=768).render_plan()
    cache = GeometryCache(lambda: pytest.fail('unexpected reset'))
    runtime = types.SimpleNamespace(softmax_ranks=6)
    for index, (length, shape) in enumerate([(117, (1, 24, 1, 48, 32)), (165, (1, 24, 1, 32, 48))]):
        embeds = torch.zeros(length, 3)
        conditions = (('ref',), [torch.zeros(shape)])
        bucket = describe_bucket(embeds, conditions, plan, (1, 2, 2), 810, 72, 1024)
        assert bucket.video_tokens == 72 * 24 * 43
        assert cache.prepare(runtime, plan, embeds, torch.ones(length), conditions, bucket) == (index == 0)
        cache.commit()
    runtime.softmax_ranks = 4
    assert cache.prepare(runtime, plan, embeds, torch.ones(length), conditions, bucket)


def test_reference_short_edge_is_independent_and_in_graph():
    request, settings = normalize_request({'prompt': 'Reference test', 'duration': 10, 'ratio': '9:16',
        'resolution': 768, 'reference_short_edge': 512, 'reference_image_urls': ['https://example.com/a.png']})
    assert settings.reference_short_edge == 512
    assert prompt_graph(request)['1']['inputs']['reference_short_edge'] == 512
    assert settings.render_plan().width == 768
    for value in (0, 511, 2049, None, '512', True):
        with pytest.raises(ValueError):
            Settings(reference_short_edge=value).validate()


def test_conditioning_metadata_works_with_new_and_old_cache(tmp_path):
    from PIL import Image
    image = tmp_path / 'ref.png'
    Image.new('RGB', (1080, 1440)).save(image)
    payload = {'prompt_embeds': torch.ones(7, 4), 'text_token_tags': torch.tensor([0, 0, 1, 1, 1, 1, 1]),
               'keyframe_anchors': ['ref'], 'keyframe_files': [str(image)], 'reference_size': 768,
               'condition_latents': [torch.ones(1, 24, 1, 64, 48)]}
    path = tmp_path / 'prompt.pt'
    torch.save(payload, path)
    embeds, tags, conditions, info = load_conditioning(path, 'cpu')
    assert info['references'][0]['original_size'] == [1080, 1440]
    assert info['references'][0]['normalized_size'] == [768, 1024]
    assert info['prompt_tokens'] == 7 and info['vision_tokens'] == 2
    assert embeds.dtype == torch.bfloat16 and conditions[1][0].dtype == torch.float32
    payload['reference_metadata'] = [{'original_size': [1080, 1440], 'qwen_grid_thw': [1, 64, 48]}]
    torch.save(payload, path)
    image.unlink()
    assert load_conditioning(path, 'cpu')[3]['references'][0]['qwen_grid_thw'] == [1, 64, 48]


def test_pinned_forward_rewrites_install_with_and_without_exact_runtime():
    from tests.test_exact_runtime import sources, DummyTransformer
    from openvdn_comfy.exact_runtime import ExactRuntime
    for active in (True, False):
        ulysses, render = sources()
        ulysses._window_softmax_branch = lambda *args: None
        model = DummyTransformer(ulysses)
        state = TokenBuckets()
        ExactRuntime(model, ulysses, render, active=active, token_buckets=state)
        assert model.attn._ref2va_buckets is state


def test_native_linear_branch_receives_only_real_text_and_video_rows():
    from tests.test_exact_runtime import upstream_functions
    shard = upstream_functions('src/inference/utils/ulysses.py', {'_linear_branch_shard'})._linear_branch_shard
    length, prefix, capacity, video = 3, 9, 16, 8
    raw = [torch.randn(prefix + video, 2, 4) for _ in range(3)]
    padded = [torch.cat((t[:prefix], t.new_full((capacity - prefix, 2, 4), 123.), t[prefix:])) for t in raw]
    captured = []
    class Branch:
        head_dim = 4
        short_conv = None
        def __call__(self, x, frames, spatial, bounds, qkv, **kwargs):
            captured.append((qkv, kwargs['text_qkv_raw'], kwargs['text_beta']))
            return qkv[0].reshape(video, -1)
    for start, qkv in ((prefix, raw), (capacity, padded)):
        layout = types.SimpleNamespace(seq_len=start + video, video_start=start, video_end=start + video,
                                      text_range=(0, length), num_frames=2, tokens_per_frame=4)
        attn = types.SimpleNamespace(layout=layout, linear_attention=Branch(), enable_text_state=True,
                                     anchor_frames='none', _bounds=lambda _: [(0, 0), (1, 1)])
        result = shard(attn, qkv, torch.ones(start + video, 2), torch.ones(start + video, 2, 4),
                       torch.zeros(2, 8), 0, 2)
        assert result.shape == (start + video, 2, 4)
    for native, bucketed in zip(captured[0], captured[1]):
        if isinstance(native, tuple):
            assert all(torch.equal(a, b) for a, b in zip(native, bucketed))
        else:
            assert torch.equal(native, bucketed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 requires CUDA / H200')
def test_actual_fa4_bucket_padding_on_cuda():
    import sys
    root = Path(__file__).resolve().parents[1]
    upstream = root / '.deps/openvdn'
    if not upstream.exists():
        pytest.skip('Install pinned OpenVDN dependencies')
    sys.path.insert(0, str(upstream))
    from src.inference.utils.ulysses import _window_softmax_branch
    from openvdn_comfy.token_buckets import verify_cuda_padding
    state = TokenBuckets()
    state.native_window_attention = _window_softmax_branch
    assert verify_cuda_padding(state, torch.device('cuda'))['passed']
