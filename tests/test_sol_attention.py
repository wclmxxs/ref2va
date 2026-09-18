from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

from openvdn_comfy.config import Settings
from openvdn_comfy.sol_attention import WindowSol
from openvdn_comfy.sol_plan import WindowPlan
from openvdn_comfy.sol_reference import sol_reference


def dense(q, k, v, scale):
    return sdpa(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                scale=scale)[0].transpose(0, 1)


def upstream_window_bounds(frames, radius, chunk):
    root = Path(__file__).resolve().parents[1]
    for folder in ('work/upstream/openvdn', '.deps/openvdn'):
        path = root/folder/'src/models/softmax_attention/window.py'
        if path.exists():
            tree = ast.parse(path.read_text())
            tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'window_bounds']
            ns = {}
            exec(compile(tree, str(path), 'exec'), ns)
            return ns['window_bounds'](frames, radius, chunk)
    pytest.skip('Pinned OpenVDN sources not installed')


@pytest.mark.parametrize('anchors', ['none', 'rows', 'columns', 'both'])
@pytest.mark.parametrize('prefix', [None, 0, 2, 5])
@pytest.mark.parametrize('windows', ['clamped', 'frame', 'chunk', 'full', 'full_unclamped'])
def test_window_adapter_matches_original_restricted_domain(anchors, prefix, windows):
    layout = SimpleNamespace(video_start=5, video_end=26, seq_len=29, num_frames=7, tokens_per_frame=3)
    if windows in ('frame', 'chunk'):
        bounds = upstream_window_bounds(7, 1, 5 if windows == 'chunk' else 0)
        assert bounds[0][0] < 0 and bounds[-1][1] >= 7
    else:
        bounds = {'full': [(0, 6)]*7, 'full_unclamped': [(-9, 15)]*7,
                  'clamped': [(0, 1), (0, 1), (1, 3), (1, 3), (4, 6), (4, 6), (5, 6)]}[windows]
    # Independently construct the original VDN boolean domain.
    mask = torch.zeros((29, 29), dtype=torch.bool)
    for qi in range(29):
        for ki in range(29):
            qf, kf = (qi-5)//3, (ki-5)//3
            keep = qi < 5 or qi >= 26 or ki < 5 or ki >= 26
            if 0 <= qf < 7 and 0 <= kf < 7:
                keep |= bounds[qf][0] <= kf <= bounds[qf][1]
                keep |= anchors in ('rows', 'both') and qf in (0, 6)
                keep |= anchors in ('columns', 'both') and kf in (0, 6)
            mask[qi, ki] = keep and not (prefix is not None and prefix <= ki < 5)
    generator = torch.Generator().manual_seed(17)
    q, k, v = [torch.randn((29, 3, 8), generator=generator, dtype=torch.float64) for _ in range(3)]
    calls = []
    def kernel(qb, kb, vb, *, scale, tau, sink_tokens):
        calls.append((qb.shape, kb.shape, sink_tokens))
        return sdpa(qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2), scale=scale).transpose(1, 2)
    adapter = WindowSol(kernel, dense, staging_bytes=1000)
    result = adapter(q, k, v, layout, bounds, .2, anchors=anchors, tau=1, valid_prefix=prefix)
    expected = sdpa(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                    scale=.2, attn_mask=mask)[0].transpose(0, 1)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    plan = WindowPlan(layout, bounds, anchors, prefix)
    def ids(ranges):
        return [i for a, b in ranges for i in range(a, b)]
    all_queries = ids(plan.dense_queries)
    for batch in plan.batches.values():
        for group in batch:
            queries, keys = ids(group.queries), ids(group.keys)
            assert len(keys) == len(set(keys))
            assert all(set(keys) == set(mask[qi].nonzero().flatten().tolist()) for qi in queries)
            # Every global and anchor K is in the exact prefix of the packed window.
            protected = {i for i in keys if i < 5 or i >= 26 or
                         (anchors in ('columns', 'both') and (i-5)//3 in (0, 6))}
            assert set(keys[:group.sink_tokens]) == protected
            all_queries += queries
    assert sorted(all_queries) == list(range(29))
    assert adapter.report()['kernel_query_rows']+adapter.report()['dense_query_rows'] == 29
    assert any(qshape[1] != kshape[1] for qshape, kshape, _ in calls)
    adapter(q, k, v, layout, bounds, .2, anchors=anchors, tau=.5, valid_prefix=prefix)
    assert adapter.plan_misses == 1


@pytest.mark.parametrize('tq,tk', [(73, 197), (131, 65), (64, 128)])
def test_reference_all_exact_is_dense_attention(tq, tk):
    gen = torch.Generator().manual_seed(1)
    q = torch.randn((2, tq, 3, 8), generator=gen, dtype=torch.bfloat16)
    k, v = [torch.randn((2, tk, 3, 8), generator=gen, dtype=torch.bfloat16) for _ in range(2)]
    actual = sol_reference(q, k, v, scale=.2, tau=1, all_exact=True)
    expected = sdpa(q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2), scale=.2).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_reference_approximation_uses_value_sums_and_real_tail_lengths():
    q = torch.zeros((1, 73, 1, 8), dtype=torch.bfloat16)
    k = torch.zeros((1, 197, 1, 8), dtype=torch.bfloat16)
    v = torch.arange(197, dtype=torch.float32)[None, :, None, None].expand_as(k).to(torch.bfloat16)
    actual = sol_reference(q, k, v, scale=.2, tau=1)
    for block, query in enumerate(actual.split(64, 1)):
        # Near-diagonal blocks are exact even for zero logits. Other blocks
        # contribute BF16-rounded V sums, including the five-token final block.
        total = sum(x.float().sum(1) if abs(i-block) <= 1 else
                    x.float().sum(1).to(torch.bfloat16).float()
                    for i, x in enumerate(v.split(64, 1)))
        torch.testing.assert_close(query, (total/197)[:, None].expand_as(query))


@pytest.mark.parametrize('sink', [0, 193])
def test_reference_preserves_forced_neighbor_blocks_below_threshold(sink):
    import math
    # All block centroids are zero, so none passes the positive threshold.
    # Alternating +/-1 K=V makes exact attention observably different from
    # centroid attention. The expected value follows directly from exp(+/-1).
    q = torch.ones((1, 193, 1, 1), dtype=torch.bfloat16)
    k = torch.zeros((1, 261, 1, 1), dtype=torch.bfloat16)
    k[:, :256:2] = 1
    k[:, 1:256:2] = -1
    result = sol_reference(q, k, k, scale=1., tau=4., sink_tokens=sink)
    for block, query in enumerate(result.split(64, 1)):
        exact_blocks = {i for i in range(4) if abs(i-block) <= 1 or i < (sink+63)//64}
        count = 64*len(exact_blocks)
        expected = count*math.sinh(1)/(count*math.cosh(1)+(261-count))
        torch.testing.assert_close(query, torch.full_like(query, expected), rtol=1e-6, atol=1e-6)


def test_sol_options_reach_api_and_ui():
    from openvdn_comfy.api import normalize_request, prompt_graph
    from openvdn_comfy.nodes import OpenVDNH200Request
    body = dict(prompt='hello', reference_image_url='https://example.com/ref.png', duration=10, ratio='9:16', resolution=768, attention_kernel='sol',
                sol_tau=.5, sol_dense_steps=2, sol_dense_layers=4, isolate_padding=True)
    normalized, settings = normalize_request(body)
    graph = prompt_graph(normalized)['1']['inputs']
    for name in ('attention_kernel', 'sol_tau', 'sol_dense_steps', 'sol_dense_layers'):
        assert graph[name] == body[name] == getattr(settings, name)
        assert name in OpenVDNH200Request.INPUT_TYPES()['optional']
    for fields in ({'sol_tau': float('nan')}, {'sol_tau': True}, {'sol_tau': -1}, {'sol_tau': 4.1},
                   {'sol_dense_steps': 9}, {'sol_dense_layers': 51}, {'sol_dense_layers': False},
                   {'sol_dense_steps': 1.5}, {'inference_kernels': False}, {'softmax_backend': 'ref'}):
        with pytest.raises(ValueError):
            replace(settings, **fields).validate()


def test_sol_step_layer_schedule_and_residual_transition():
    from openvdn_comfy.attention_runtime import AttentionRuntime
    hybrids = [SimpleNamespace() for _ in range(5)]
    state = AttentionRuntime(hybrids, None, None)
    state.kernel, state.dense_steps, state.dense_layers = 'sol', 3, 2
    state.transitions = 0
    state.native_reasons = {'initial_steps': 0, 'initial_layers': 0}
    resets = []
    used = []
    for step in range(8):
        if state.begin_step(step):
            resets.append(step)
        used.append([state.use_sol(a._ref2va_sol_layer) for a in hybrids])
    assert resets == [3]
    assert used[:3] == [[False]*5]*3
    assert used[3:] == [[False, False, True, True, True]]*5
    assert state.native_reasons == {'initial_steps': 15, 'initial_layers': 10}


def test_residual_invalidated_before_group_preparation_and_blocks():
    from openvdn_comfy.dit_runtime import DiTRuntime
    events = []
    attention = SimpleNamespace(begin_step=lambda step: events.append(('step', step)) or step == 3)
    cache = SimpleNamespace(step=3, consecutive=1, config=SimpleNamespace(enabled=True, threshold=.25, max_cached_steps=2),
                            clear=lambda: events.append('clear'),
                            configure_groups=lambda *a: events.append('groups'),
                            run=lambda *a: events.append('blocks') or 'result')
    runtime = DiTRuntime.__new__(DiTRuntime)
    runtime.hybrids = [SimpleNamespace(_ref2va_attention=attention, layout=SimpleNamespace(video_start=2, video_end=10))]
    runtime.cache = cache
    runtime.transformer = SimpleNamespace(transformer_blocks=[])
    assert runtime.run(None, (), None, None) == 'result'
    assert events == [('step', 3), 'clear', 'groups', 'blocks'] and cache.consecutive == 0


@pytest.mark.parametrize('tq,tk,heads', [(73, 197, 9), (131, 65, 10), (128, 4096, 11), (64, 128, 14)])
def test_rectangular_preprocess_dimensions_and_threshold_alignment(monkeypatch, tq, tk, heads):
    import sys
    import types
    from openvdn_comfy.sol_kernel import prepare_rectangular
    # Exercise the actual host allocations and launch arguments without a GPU;
    # GPU arithmetic tests independently check the equation and reads/writes.
    launches = {}
    class Kernel:
        def __init__(self, name):
            self.name = name
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches[self.name] = grid, args
            return launch
    class Descriptor:
        @staticmethod
        def from_tensor(tensor, block_shape):
            return tensor
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(cdiv=lambda x, y: (x+y-1)//y))
    monkeypatch.setitem(sys.modules, 'triton.tools.tensor_descriptor', SimpleNamespace(TensorDescriptor=Descriptor))
    module = types.ModuleType('openvdn_comfy._vendor.sol_attn.preprocess')
    module._reduce_kv = lambda k, v: (torch.empty((2, (tk+63)//64, heads, 128), dtype=k.dtype),)*2
    module._reduce_kc_stats_kernel = Kernel('stats')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, 'openvdn_comfy.sol_preprocess', SimpleNamespace(diagonal_threshold=Kernel('threshold')))
    q = torch.empty((2, tq, heads, 128), dtype=torch.bfloat16)
    k = torch.empty((2, tk, heads, 128), dtype=torch.bfloat16)
    kc, vc, threshold = prepare_rectangular(q, k, k, scale=.1, tau=1.)
    assert kc.shape == vc.shape == (2, (tk+63)//64, heads, 128)
    assert threshold.shape == (2, (tq+63)//64, heads)
    assert all(stride*threshold.element_size() % 16 == 0 for stride in threshold.stride()[:-1])
    assert launches['stats'][1][3] == (tk+63)//64
    assert launches['threshold'][0] == ((tq+63)//64, 2*heads)
    args = launches['threshold'][1]
    assert args[5:9] == (tq, heads, (tq+63)//64, threshold.stride(1))


def test_sol_reporting_aggregates_ranks_without_hiding_linear_only_ranks():
    from openvdn_comfy.attention_runtime import summarize_attention
    def record(rank, launches, misses):
        return dict(rank=rank, kernel='sol', sol=dict(sparse_executed=launches > 0, sparse_kernel_launches=launches,
                    compile=dict(compile_misses=misses, preprocess_signatures=misses, compile_seconds=float(misses))))
    report = summarize_attention([record(0, 3, 1), record(1, 5, 2), record(2, 0, 0)])
    assert report['sol']['sparse_executed']
    assert report['sol']['sparse_kernel_launches_all_ranks'] == 8
    assert report['sol']['compile_seconds_max_rank'] == 2
    assert report['by_rank'][2]['sol']['sparse_executed'] is False


def test_full_cover_keeps_native_path_when_window_hook_is_used():
    from openvdn_comfy.attention_runtime import window_attention
    def forbidden(*a, **kw):
        raise AssertionError('full-cover must not dispatch to Sol')
    state = SimpleNamespace(kernel='sol', native_reasons={'full_cover': 0},
                            use_sol=forbidden, sol=forbidden, isolate_padding=False,
                            native=lambda *a: 'native')
    attn = SimpleNamespace(_ref2va_attention=state, _ref2va_sol_layer=3)
    assert window_attention(attn, None, None, None, SimpleNamespace(num_frames=3), [(0, 2)]*3, .1) == 'native'
    assert state.native_reasons['full_cover'] == 1


def test_benchmark_requires_actual_sparse_execution_and_preserves_cache_settings():
    from scripts.benchmark_sol_attention import variants, validate_result, summarize
    choices = variants([6, 5, 4], [.5, 1.], 1, 2)
    assert len(choices) == 9
    assert all('cache_dit' not in value for value in choices.values())
    req = choices['sol-6+2-tau1']
    up = dict(optimizations=dict(requested=req, attention={'sol': {'sparse_executed': False}}), parallel={'softmax_ranks': 6})
    with pytest.raises(RuntimeError, match='no sparse kernel'):
        validate_result({'metrics': {'upstream': up}}, req)
    up['optimizations']['attention']['sol']['sparse_executed'] = True
    validate_result({'metrics': {'upstream': up}}, req)
    rows = [dict(variant='sol', **{'pass': p}, graph_reused=p != 2, denoise_seconds=100 if p == 0 else 8,
                 output_seconds=3, processing_seconds=11, cached_steps=[4, 6], video_url=str(p)) for p in range(3)]
    assert summarize(rows)[0]['denoise_seconds'] is None
    assert summarize(rows[:2])[0]['denoise_seconds'] == 8


def test_vendored_source_matches_manifest():
    root = Path(__file__).resolve().parents[1]/'openvdn_comfy/_vendor/sol_attn'
    manifest = json.loads((root/'manifest.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((root/entry['path']).read_bytes()).hexdigest() == entry['sha256']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Hopper GPU required')
def test_real_sm90_rectangular_sparse_and_exact_arithmetic():
    from openvdn_comfy.sol_kernel import SolKernel
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 required')
    kernel = SolKernel()
    assert kernel.verify(torch.device('cuda', torch.cuda.current_device()))['passed']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Hopper GPU required')
@pytest.mark.parametrize('heads', [9, 12, 14])
@pytest.mark.parametrize('kind', ['kc', 'vc', 'kv'])
def test_real_sm90_summary_accumulates_before_bf16_rounding(heads, kind):
    from triton.tools.tensor_descriptor import TensorDescriptor
    from openvdn_comfy._vendor.sol_attn import preprocess as p
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 required')
    device = torch.device('cuda', torch.cuda.current_device())
    rng = torch.Generator().manual_seed(928)
    # Binary fractions make all FP32 partial sums and full-block means exact.
    # BF16 intermediate reductions lose low bits before the final store. A
    # constant five-token K tail also has an exactly representable centroid.
    k, v = [(torch.randint(-256, 257, (2, 517, heads, 128), generator=rng).float()/8)
            .to(torch.bfloat16) for _ in range(2)]
    k[:, 512:] = .125
    expected_kc = torch.stack([x.double().mean(1).to(k.dtype) for x in k.split(64, 1)], 1).to(device)
    expected_vc = torch.stack([x.double().sum(1).to(v.dtype) for x in v.split(64, 1)], 1).to(device)
    k, v = k.to(device), v.to(device)
    kd, vd = [TensorDescriptor.from_tensor(x, [1, 64, 1, 128]) for x in (k, v)]
    kc, vc = torch.empty_like(expected_kc), torch.empty_like(expected_vc)
    args = (517, heads, 9, 128, 64, 128)
    for warps in (4, 8):
        kc.fill_(float('nan'))
        vc.fill_(float('nan'))
        # Exercise both reduction layouts directly, regardless of which one
        # the production autotuner selects on this GPU.
        grid = (1, 9, 2*heads)
        config = dict(num_warps=warps, num_stages=1)
        if kind == 'kv':
            p._reduce_kv_kernel.fn[grid](kd, vd, kc, vc, *args, **config)
        elif kind == 'kc':
            p._reduce_kc_kernel.fn[grid](kd, kc, *args, **config)
        else:
            p._reduce_vc_kernel.fn[grid](vd, vc, *args, **config)
        if kind in ('kc', 'kv'):
            torch.testing.assert_close(kc, expected_kc, rtol=0, atol=0)
        if kind in ('vc', 'kv'):
            torch.testing.assert_close(vc, expected_vc, rtol=0, atol=0)
