import ast
from pathlib import Path
import types

import pytest
import torch

from openvdn_comfy.exact_runtime import (ExactRuntime, PROJECTION_NEW, PROJECTION_OLD,
                                        enabled, project_video_rows, rewrite)


def upstream_functions(relative, names):
    root = Path(__file__).resolve().parents[1]
    path = root / 'work/upstream/openvdn' / relative
    if not path.exists():
        path = root / '.deps/openvdn' / relative
    if not path.exists():
        pytest.skip('Pinned OpenVDN sources not installed')
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'torch': torch, 'get_parameter_dtype': lambda m: next(m.parameters()).dtype,
          'MINIMAX_H3_MODALITY_NUM': 3, 'UlyssesRuntime': object,
          'iter_hybrids': lambda model: iter([model.attn])}
    exec(compile(tree, str(path), 'exec'), ns)
    return types.SimpleNamespace(**ns)


def sources():
    ulysses = upstream_functions('src/inference/utils/ulysses.py',
        {'_ulysses_attention_forward', '_branch_parallel_attention_forward', '_ulysses_transformer_forward'})
    render = upstream_functions('src/inference/render.py', {'generate_latents'})
    return ulysses, render


def state_only():
    state = ExactRuntime(None, None, types.SimpleNamespace(generate_latents=None), active=False)
    state.active = True
    return state


@pytest.mark.parametrize('value', ['false', 'yes', '2', ''])
def test_invalid_options_rejected(monkeypatch, value):
    monkeypatch.setenv('REF2VA_EXACT_RUNTIME', value)
    with pytest.raises(ValueError):
        enabled('REF2VA_EXACT_RUNTIME')


def test_all_pinned_source_contracts_and_changed_source_rejection():
    ulysses, _ = sources()
    for name in ('_ulysses_attention_forward', '_branch_parallel_attention_forward'):
        changed = rewrite(getattr(ulysses, name), [(PROJECTION_OLD, PROJECTION_NEW)])
        # A second application must fail rather than silently changing ownership.
        with pytest.raises(RuntimeError, match='source changed'):
            rewrite(changed, [(PROJECTION_OLD, PROJECTION_NEW)])


def test_constants_reuse_views_but_invalidate_mutations_and_requests():
    state = state_only()
    original = torch.arange(12.).reshape(3, 4)
    calls = []
    def compute():
        calls.append(True)
        return original + 2
    with state.request(verify=True):
        expected = state.constant('text', original[None], compute)
        assert torch.equal(state.constant('text', original[None], compute), expected)
        assert state.report()['parity']['exact']
        # One actual computation and one independent parity calculation.
        assert len(calls) == 2
        assert state.constant('text', original[None], compute) is expected
        assert len(calls) == 2
        original.add_(1)
        updated = state.constant('text', original[None], compute)
        assert not torch.equal(updated, expected)
        assert state.report()['constant_misses']['text'] == 2
        state.constant('text', original[:2], lambda: original[:2].clone())
        assert state.report()['constant_misses']['text'] == 3
    assert not state.values
    with state.request():
        state.constant('text', original[None], compute)
        assert len(calls) == 4
    with pytest.raises(RuntimeError, match='active request'):
        state.constant('text', original, compute)


def test_request_scope_cleans_up_after_failure_and_rejects_bad_parity():
    state = state_only()
    with pytest.raises(ValueError):
        with state.request():
            state.constant('text', torch.ones(1), lambda: torch.ones(1))
            raise ValueError('request failed')
    assert not state.values and not state.in_request
    with state.request(verify=True):
        state.check('drift', torch.ones(2), torch.zeros(2))
        assert state.report()['parity']['exact'] is False
    with state.request(verify=True):
        state.check('nonfinite', torch.tensor([float('inf')]), torch.tensor([float('inf')]))
        assert state.report()['parity']['exact'] is False


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('bounds', [(0, 5, 8, 17), (18, 23, 8, 17), (6, 12, 8, 17),
                                   (12, 20, 8, 17), (9, 13, 8, 17), (0, 20, 8, 17)])
def test_projection_exact_for_strided_heads_and_rank_boundaries(dtype, bounds):
    torch.manual_seed(11)
    first, last, low, high = bounds
    count = last - first
    values = torch.randn(count, 3, 8, dtype=dtype)[..., :4]
    assert not values.is_contiguous()
    out = torch.randn(count, 5, dtype=dtype)
    projection = torch.nn.Linear(12, 5).to(dtype).eval()
    state = state_only()
    attn = types.SimpleNamespace(to_out_linear=projection, _ref2va_exact=state)
    runtime = types.SimpleNamespace(local_start=first, local_end=last)
    layout = types.SimpleNamespace(video_start=low, video_end=high)
    positions = torch.arange(first, last)
    mask = (positions >= low) & (positions < high)
    expected = out.clone()
    with torch.no_grad(), state.request(verify=True):
        if mask.any():
            expected[mask] += projection(values[mask].reshape(int(mask.sum()), -1))
        actual = project_video_rows(attn, out.clone(), values, runtime, layout, out)
        assert torch.equal(actual, expected)
        assert state.report()['parity']['exact']
        assert state.report()['parity']['components'] == {'video_projection': 1}


class CountingLinear(torch.nn.Linear):
    calls = 0
    def forward(self, value):
        self.calls += 1
        return super().forward(value)


class DummyTransformer(torch.nn.Module):
    def __init__(self, ulysses):
        super().__init__()
        self.proj_in = torch.nn.Linear(2, 8)
        self.audio_proj_in = torch.nn.Linear(3, 8)
        self.context_embedder = CountingLinear(4, 8)
        self.token_refiner = CountingLinear(8, 8)
        self.time_proj = lambda t: t[:, None]
        self.time_embedder = torch.nn.Linear(1, 8)
        self.proj_out = torch.nn.Linear(8, 2)
        self.audio_proj_out = torch.nn.Linear(8, 3)
        self.rope_calls = 0
        self.norm_out = lambda x, temb, indices: x + temb.mean(0)[None, None] * .02
        self.attn = types.SimpleNamespace(num_heads=56)
        self.attn.forward = types.MethodType(ulysses._branch_parallel_attention_forward, self.attn)
        self._ulysses_runtime = types.SimpleNamespace(local_start=0, local_end=20,
            configure=lambda *a: None, gather_sequence=lambda x: x)
        self.transformer_blocks = [lambda x, temb, indices, rotary:
            x + temb.mean(0)[None, None] * .03 + indices[None, :, None] * .01 + rotary[0][:, :1][None] * .01]
        self.forward = types.MethodType(ulysses._ulysses_transformer_forward, self)
        self.eval()

    def rope(self, positions):
        self.rope_calls += 1
        return positions.float().cos(), positions.float().sin()


def test_actual_pinned_transformer_forward_matches_at_each_step_and_request():
    ulysses, render = sources()
    torch.manual_seed(21)
    model = DummyTransformer(ulysses)
    native = model.forward
    state = ExactRuntime(model, ulysses, render)
    video = torch.randn(1, 10, 2)
    audio = torch.randn(1, 7, 3)
    tags = torch.arange(20) % 3
    with torch.no_grad():
        for request_index in range(2):
            prompt = torch.randn(3, 4) + request_index
            positions = torch.arange(60).reshape(20, 3) + request_index
            model.context_embedder.calls = model.token_refiner.calls = model.rope_calls = 0
            with state.request(verify=True):
                for step in range(8):
                    kwargs = dict(hidden_states=video + step, audio_hidden_states=audio + step / 10,
                        encoder_hidden_states=prompt[None], timestep=torch.tensor([.9, .5]) - step * .01,
                        timestep_indices=torch.arange(20) % 2, token_tags=tags, position_ids=positions,
                        video_indices=torch.arange(3, 13), audio_indices=torch.arange(13, 20),
                        text_indices=torch.arange(3), return_dict=False)
                    # Invoke an untouched copy of the pinned forward, using exactly
                    # the same tensors, weights, sampler step and layouts.
                    expected = native(**kwargs)
                    actual = model(**kwargs)
                    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
                report = state.report()
                assert report['constant_hits'] == {'rope': 7, 'text': 7}
                assert report['constant_misses'] == {'rope': 1, 'text': 1}
                assert report['parity']['exact']
                # Eight native calls + one cached computation + one validation.
                assert model.context_embedder.calls == model.token_refiner.calls == model.rope_calls == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for event timing')
def test_cuda_deferred_step_events():
    from openvdn_comfy.exact_runtime import step_start, step_end, finish_steps
    steps = []
    value = torch.randn(128, 128, device='cuda')
    for _ in range(8):
        started = step_start('cuda')
        value = value @ value
        steps.append(step_end(started))
    finish_steps(steps)
    assert len(steps) == 8 and all(isinstance(t, float) and t >= 0 for t in steps)
