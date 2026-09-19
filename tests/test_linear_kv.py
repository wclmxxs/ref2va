import ast
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openvdn_comfy.api import normalize_request, prompt_graph
from openvdn_comfy.attention_runtime import AttentionRuntime
from openvdn_comfy.config import Settings
from openvdn_comfy.linear_kv import (LinearKVRuntime, kept_tokens, rewrite_readout,
                                    sample_indices, summarize_linear_kv)
from openvdn_comfy.nodes import OpenVDNH200Request
from openvdn_comfy.warmup_history import WarmupHistory


def upstream_path(relative):
    root = Path(__file__).resolve().parents[1]
    for location in ('work/upstream/openvdn', '.deps/openvdn'):
        path = root / location / relative
        if path.is_file():
            return path
    pytest.skip('Pinned OpenVDN sources not installed')


def source_module(name):
    path = upstream_path(f'src/models/linear_attention/{name}.py')
    spec = importlib.util.spec_from_file_location(f'test_linear_kv_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_branch():
    """Run the pinned readout/text/scans on CPU; only compiled helpers are eager.

    Features and epilogue are deliberately small tensor stand-ins, shared by both
    paths. The real readout controls full-query shape, text handling and backend S.
    """
    scan, delta = source_module('scan'), source_module('delta_rule')
    scan._STATS_PREP_CACHE['fn'] = scan._frame_stats_prep_body
    scan._GATHER_CACHE[('alpha', True, torch.float32)] = scan._gather_body
    calls = []

    def statistics(k, v, beta, a_fp32=True, inference=False):
        calls.append((inference, tuple(k.shape)))
        return scan.frame_statistics(k, v, beta, a_fp32=a_fp32, inference=inference)

    def epilogue(readout, weight, gate, eps, **kwargs):
        f, h, s, d = readout.shape
        result = readout.permute(0, 2, 1, 3).reshape(f*s, h, d)
        return (result * torch.rsqrt(result.square().mean(-1, keepdim=True) + eps)
                * weight * gate).reshape(f*s, h*d)

    ns = dict(torch=torch, frame_statistics=statistics, DELTA_BACKENDS=delta.DELTA_BACKENDS,
              _run_scans_inference=scan._run_scans_inference,
              gather_linear_state=scan.gather_linear_state, linear_epilogue=epilogue)
    path = upstream_path('src/models/linear_attention/branch.py')
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'BidirectionalLinearBranch')
    names = {'forward', '_readout_inference', '_delta_backend', '_text_state', '_text_chunk_state'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), ns)
    branch = type('PinnedBranch', (), {name: ns[name] for name in names})()
    branch.num_heads, branch.head_dim = 4, 4
    branch.TEXT_STATE_SCALE, branch.a_fp32, branch.bridge = .5, True, 'alpha'
    branch.delta_rule, branch.backend, branch.text_backend = 'vdn_solve', None, None
    branch.norm = SimpleNamespace(weight=torch.ones(4), eps=1e-6)
    branch._feature_one = lambda x, proj, *args, **kwargs: (
        torch.nn.functional.normalize(x, dim=-1) if proj == 'k' else x)

    def features(qkv, frames, grid, **kwargs):
        q, k, v = qkv
        h, d = q.shape[-2:]
        return (q.reshape(frames, -1, h, d).permute(0, 2, 1, 3),
                torch.nn.functional.normalize(k, dim=-1), v)

    branch._features = features
    branch.alpha = lambda mean, heads=None: torch.full(
        (len(mean), 4 if heads is None else heads.stop-heads.start, 4), .9)
    branch.beta_proj = lambda x: x
    branch.output_gate = lambda x: torch.sigmoid(x)[..., None].expand(-1, -1, 4)
    return branch, calls


def runtime(branches=(), profile=False):
    events = []
    rt = SimpleNamespace(rank=3, profile_enabled=profile,
                         profile_start=lambda: None,
                         profile_end=lambda name, start: events.append(name) if profile else None)
    state = LinearKVRuntime([SimpleNamespace(linear_attention=b) for b in branches], rt)
    return state, events


@pytest.mark.parametrize('ratio', [1, 1.0, .5, .25])
def test_api_ui_and_worker_settings_roundtrip(ratio):
    request, settings = normalize_request(dict(prompt='Reference test', duration=10, resolution=768,
        ratio='9:16', reference_image_urls=['https://example.com/ref.png'], linear_kv_keep_ratio=ratio))
    assert prompt_graph(request)['1']['inputs']['linear_kv_keep_ratio'] == ratio
    assert Settings(**asdict(settings)).validate().linear_kv_keep_ratio == ratio
    assert OpenVDNH200Request.INPUT_TYPES()['optional']['linear_kv_keep_ratio'][0] == [1., .5, .25]


@pytest.mark.parametrize('ratio', [True, False, 0, -.5, .125, .75, 2, '0.5', None, float('nan'), float('inf')])
def test_invalid_ratio_rejected_before_worker(ratio):
    with pytest.raises(ValueError, match='linear_kv_keep_ratio'):
        replace(Settings(), linear_kv_keep_ratio=ratio).validate()


@pytest.mark.parametrize('tokens', [1, 2, 3, 5, 8, 9, 1032, 2064])
@pytest.mark.parametrize('ratio', [1., .5, .25])
def test_sampling_indices_are_unique_bounded_deterministic_and_leave_rng_untouched(tokens, ratio):
    before = torch.random.get_rng_state()
    ids = sample_indices(tokens, ratio, 'cpu')
    assert torch.equal(before, torch.random.get_rng_state())
    assert ids.numel() == kept_tokens(tokens, ratio)
    assert ids.unique().numel() == ids.numel() and 0 <= ids.min() <= ids.max() < tokens
    assert torch.equal(ids, sample_indices(tokens, ratio, 'cpu'))
    if ratio == 1:
        assert torch.equal(ids, torch.arange(tokens))


def outer_statistics(k, v, beta, **kwargs):
    # Independent explicit outer-product reference, no batched matmul/repack.
    a = (beta[..., None, None] * k[..., :, None] * k[..., None, :]).sum(2)
    b = (beta[..., None, None] * v[..., :, None] * k[..., None, :]).sum(2)
    return a, b


@pytest.mark.parametrize('ratio', [.5, .25])
@pytest.mark.parametrize('tokens', [5, 12])
def test_statistics_align_k_v_beta_and_correct_both_sums(ratio, tokens):
    torch.manual_seed(8)
    k, v = [torch.randn(3, 2, tokens, d, dtype=torch.float64) for d in (4, 6)]
    beta = torch.rand(3, 2, tokens, dtype=torch.float64)
    copies = [t.clone() for t in (k, v, beta)]
    state, events = runtime(profile=True)
    state.native_statistics = lambda k, v, b, **kw: (
        torch.einsum('fhsk,fhs,fhsl->fhkl', k, b, k),
        torch.einsum('fhsv,fhs,fhsk->fhvk', v, b, k))
    count = kept_tokens(tokens, ratio)
    ids = [int((i+.5)*tokens/count) for i in range(count)]
    expected = [m * (tokens/count) for m in outer_statistics(k[:, :, ids], v[:, :, ids], beta[:, :, ids])]
    with torch.no_grad(), state.request(ratio):
        actual = state.statistics(k, v, beta, a_fp32=True, inference=True)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        for a, b in zip((k, v, beta), copies):
            assert torch.equal(a, b)
        assert state.report()['shapes'][0]['statistics_weight'] == tokens/count
        assert state.report()['approximate']
    assert events == ['linear_kv_select', 'linear_frame_statistics', 'linear_kv_rescale']
    assert not state.indices and not state.observations


@pytest.mark.parametrize('ratio', [.5, .25])
def test_weight_correction_preserves_identical_spatial_tokens_including_odd_counts(ratio):
    k = torch.ones(2, 3, 5, 4)
    v, beta = k*3, torch.full((2, 3, 5), .5)
    state, _ = runtime()
    state.native_statistics = outer_statistics
    with torch.no_grad(), state.request(ratio):
        for a, b in zip(state.statistics(k, v, beta, a_fp32=True, inference=True), outer_statistics(k, v, beta)):
            torch.testing.assert_close(a, b)


def test_pinned_readout_switch_preserves_full_query_text_and_backend_scaling():
    branch, calls = native_branch()
    native = branch._readout_inference
    state, _ = runtime([branch])
    with pytest.raises(RuntimeError, match='source changed'):
        rewrite_readout(state.bindings[0][2].__func__)
    torch.manual_seed(41)
    f, s, h, d, text = 6, 9, 4, 4, 7
    qkv = tuple(torch.randn(f*s, h, d) for _ in range(3))
    text_qkv = tuple(torch.randn(text, h, d) for _ in range(3))
    x, tx = torch.randn(f*s, h), torch.randn(text, h)
    args = dict(xv=x, num_frames=f, tokens_per_frame=s, bounds=[(i, i) for i in range(f)],
                qkv_raw=qkv, text_x=tx, text_qkv_raw=text_qkv, frame_size=(3, 3), inference=True)
    with torch.no_grad():
        baseline = branch.forward(**args)
        text_state = branch._text_state(tx, text_qkv)
        for ratio in (1., .5, .25, 1.):
            calls.clear()
            with state.request(ratio):
                if ratio == 1:
                    assert branch._readout_inference == native
                out = branch.forward(**args)
                assert out.shape == baseline.shape == (f*s, h*d)
                assert torch.isfinite(out).all()
                assert calls == [(True, (f, h, kept_tokens(s, ratio), d)), (False, (1, h, text, d))]
                assert branch.backend._S == s and branch.text_backend._S == text
                assert torch.equal(branch._text_state(tx, text_qkv), text_state)
                if ratio == 1:
                    assert torch.equal(out, baseline)
                    assert state.report()['statistics_calls'] is None  # native bypass, not measured
                else:
                    assert not torch.equal(out, baseline)  # opt-in approximation is real
                    assert state.report()['statistics_calls'] == 1  # text never intercepted
            assert branch._readout_inference == native


@pytest.mark.parametrize('ratio', [1., .5, .25])
@pytest.mark.parametrize('skip_ends', [False, True])
def test_head_sharding_and_anchor_pruning_keep_full_readout(ratio, skip_ends):
    branch, _ = native_branch()
    state, _ = runtime([branch])
    torch.manual_seed(9)
    f, s, h, d, text = 6, 8, 4, 4, 3
    qkv = tuple(torch.randn(f*s, h, d) for _ in range(3))
    tqkv = tuple(torch.randn(text, h, d) for _ in range(3))
    beta, gate, tbeta = torch.rand(f*s, h), torch.rand(f*s, h, d), torch.rand(text, h)
    args = dict(xv=None, num_frames=f, tokens_per_frame=s, bounds=[(i, i) for i in range(f)],
                frame_size=(2, 4), frame_mean=torch.ones(f, h), skip_ends=skip_ends, inference=True)
    with torch.no_grad(), state.request(ratio):
        full = branch.forward(**args, qkv_raw=qkv, beta=beta, gate=gate, text_qkv_raw=tqkv, text_beta=tbeta)
        parts = [branch.forward(**args, heads=heads, qkv_raw=tuple(x[:, heads] for x in qkv),
                 beta=beta[:, heads], gate=gate[:, heads], text_qkv_raw=tuple(x[:, heads] for x in tqkv),
                 text_beta=tbeta[:, heads]) for heads in (slice(0, 1), slice(1, 4))]
        torch.testing.assert_close(torch.cat(parts, -1), full, rtol=1e-5, atol=1e-6)
        if skip_ends:
            assert not full[:s].any() and not full[-s:].any()
        if ratio < 1:
            assert all(item['frames'] == f-2*skip_ends for item in state.report()['shapes'])


def test_failure_resets_bindings_and_empty_rank_does_not_hide_actual_sampling():
    branch, _ = native_branch()
    state, events = runtime([branch], profile=True)
    native = branch._readout_inference
    k, v, b = torch.ones(2, 4, 5, 4), torch.ones(2, 4, 5, 4), torch.ones(2, 4, 5)
    with pytest.raises(ValueError, match='request failed'), torch.no_grad(), state.request(.25):
        state.statistics(k, v, b, a_fp32=True, inference=True)
        recorded = state.report()
        empty = {**recorded, 'rank': 0, 'statistics_calls': 0, 'shapes': [], 'approximate': False}
        assert summarize_linear_kv([empty, recorded])['approximate']
        with pytest.raises(RuntimeError, match='differs'):
            summarize_linear_kv([empty, {**recorded, 'requested_keep_ratio': .5}])
        with pytest.raises(RuntimeError, match='concurrent'):
            with state.request(1.):
                pass
        raise ValueError('request failed')
    assert not state.active and not state.indices and not state.observations
    assert branch._readout_inference == native
    events.clear()
    with torch.no_grad(), state.request(1.):
        state.statistics(k, v, b, a_fp32=True, inference=True)
        assert not state.report()['approximate']
    assert events == ['linear_frame_statistics']


def test_shape_history_separates_ratio_without_invalidating_original_ids(tmp_path):
    state = AttentionRuntime([], None, None)
    state.linear_chunk, state.linear_kv_keep_ratio = 16, 1.
    assert state.signature() == ('native', False, 16)
    signatures = set()
    history = WarmupHistory(tmp_path/'history.json', {}, {})
    prompt_file = tmp_path/'prompt.pt'
    prompt_file.touch()
    for ratio in (1., .5, .25):
        state.linear_kv_keep_ratio = ratio
        signatures.add(state.signature())
        history.remember(str(ratio), dict(settings=asdict(Settings(linear_kv_keep_ratio=ratio)), prompt_file=str(prompt_file)))
    assert len(signatures) == 3
    assert [r['settings']['linear_kv_keep_ratio'] for r in history.requests(3)] == [1., .5, .25]
    assert history.requests(0) == []  # still lazy; no extra startup warmup
